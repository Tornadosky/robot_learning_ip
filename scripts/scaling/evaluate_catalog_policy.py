"""Evaluate a checkpoint on frozen embodiment catalogs.

Every arm - the policy, zero actions, and any comparison checkpoints - sees the
exact same bodies, reset keys, initial motion phases and horizon, because the
catalog assigns bodies by environment slot rather than by random draw.  Results
are reported per body as well as in aggregate, since an average can hide a body
that never stands up.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

WORKSPACE = Path(__file__).resolve().parents[2]
SCRIPTS = WORKSPACE / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from loco_mujoco.algorithms import PPOJax  # noqa: E402
from loco_mujoco.core.wrappers import LogWrapper  # noqa: E402

from scaling.embodiment_catalog import (  # noqa: E402
    EmbodimentCatalog,
    fixed_balanced_assignment,
)
from scaling.morphology_terminal import (  # noqa: E402,F401  (registers handler)
    MorphologyAwareRootPoseTrajTerminalStateHandler,
)
from scaling.online_h1 import MORPHOLOGY_NAMES  # noqa: E402
from scaling.online_h1_train import build_online_env  # noqa: E402
from scaling.urma_networks import URMAPPO  # noqa: E402


def checkpoint_manifest(checkpoint: Path) -> tuple[Path, dict]:
    checkpoint = Path(checkpoint).resolve()
    candidates = [checkpoint.parent / "manifest.json"]
    candidates.extend(parent / "manifest.json" for parent in checkpoint.parents)
    for candidate in candidates:
        if candidate.is_file():
            return candidate, json.loads(candidate.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"No manifest.json found above checkpoint {checkpoint}.")


def build_eval_env(
    manifest: dict,
    catalog_path: Path,
    use_mjwarp: bool,
    tag: str,
    terminal_handler: str | None = None,
):
    """Rebuild the training environment, swapping in the evaluation catalog.

    The morphology bounds stay at the *training* bounds so the descriptor the
    policy reads is on the same scale it was trained with.  Out-of-distribution
    catalogs therefore produce descriptors outside [-1, 1], which is the point.
    """
    frequency = float(manifest["frequency_hz"])
    args = SimpleNamespace(
        clip=str(manifest["clip"]),
        duration=float(manifest["window_frames"]) / frequency,
        start_frame=int(manifest["window_start_frame"]),
        run_tag=tag,
        use_mjwarp=bool(use_mjwarp),
        backbone=str(manifest.get("backbone", "mlp")),
        resample_per_episode=False,
        morph_low=list(manifest["morphology_low"]),
        morph_high=list(manifest["morphology_high"]),
        catalog=catalog_path,
        catalog_mode="fixed_balanced",
        catalog_stride=1,
        keep_morph_bounds=True,
        allow_nontrain_catalog=True,
        terminal_handler=terminal_handler,
    )
    return build_online_env(args)[0]


def rollout(
    env,
    agent_conf,
    agent_state,
    *,
    num_envs: int,
    horizon: int,
    seed: int,
    zero_action: bool = False,
):
    logged = LogWrapper(env)
    keys = jax.random.split(jax.random.PRNGKey(seed), num_envs)
    slots = jnp.arange(num_envs, dtype=jnp.int32)

    variables = None
    if not zero_action:
        variables = {
            "params": agent_state.train_state.params,
            "run_stats": agent_state.train_state.run_stats,
        }

    def action_for(obs):
        if zero_action:
            return jnp.zeros(
                (*obs.shape[:-1], env.info.action_space.shape[0]), dtype=obs.dtype
            )
        (policy, _), _ = agent_conf.network.apply(variables, obs, mutable=["run_stats"])
        return policy.mean()

    vstep = jax.vmap(logged.step, in_axes=(0, 0))

    def step_fn(carry, _):
        obs, state, returns, lengths, completed, fell = carry
        action = action_for(obs)
        next_obs, reward, absorbing, done, _, next_state = vstep(state, action)
        active = ~completed
        returns = returns + reward * active
        lengths = lengths + active.astype(lengths.dtype)
        # LocoMuJoCo sets done = absorbing OR step >= env horizon, so `done`
        # alone cannot tell a fall from running out of episode.  `absorbing` is
        # the terminal-state handler firing, i.e. the actual fall.
        fell = fell | (absorbing.astype(jnp.bool_) & active)
        completed = completed | done.astype(jnp.bool_)
        return (next_obs, next_state, returns, lengths, completed, fell), None

    from loco_mujoco.core.wrappers.mjx import LogEnvState, Metrics

    def reset_and_rollout(keys, slots):
        """Reset and roll out inside one graph.

        A *standalone* ``jit(vmap(mjx_reset_with_slot))`` proved unstable on the
        measured ROCm stack (job 10803152 aborted in it, jobs 10806775-77 hung in
        it).  The same reset is reliable when it is fused into the surrounding
        computation, exactly as PPO does it, so evaluation never materialises it
        on its own.
        """
        env_state = jax.vmap(env.mjx_reset_with_slot, in_axes=(0, 0))(keys, slots)
        observation = env_state.observation
        zeros = jnp.zeros((num_envs,), dtype=jnp.float32)
        izeros = jnp.zeros((num_envs,), dtype=jnp.int32)
        log_state = LogEnvState(
            env_state,
            Metrics(zeros, izeros, zeros, izeros, izeros, jnp.zeros((num_envs,), bool)),
        )
        initial = (
            observation,
            log_state,
            jnp.zeros((num_envs,), dtype=jnp.float32),
            jnp.zeros((num_envs,), dtype=jnp.int32),
            jnp.zeros((num_envs,), dtype=jnp.bool_),
            jnp.zeros((num_envs,), dtype=jnp.bool_),
        )
        (_, _, returns, lengths, completed, fell), _ = jax.lax.scan(
            step_fn, initial, None, horizon
        )
        return (
            env_state.additional_carry.body_index,
            env_state.additional_carry.morphology,
            returns,
            lengths,
            completed,
            fell,
        )

    started = time.perf_counter()
    body_index, morphology, returns, lengths, completed, fell = jax.jit(
        reset_and_rollout
    )(keys, slots)
    jax.block_until_ready(returns)
    seconds = time.perf_counter() - started
    return {
        "body_index": np.asarray(body_index),
        "morphology": np.asarray(morphology),
        "returns": np.asarray(returns),
        "lengths": np.asarray(lengths),
        "terminated": np.asarray(completed),
        "fell": np.asarray(fell),
        "seconds": seconds,
    }


def _summary(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p10": float(np.percentile(values, 10)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }


def _correlation(x, y):
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return None
    value = float(np.corrcoef(x, y)[0, 1])
    return value if np.isfinite(value) else None


def per_body_aggregate(values, body_index, num_bodies) -> np.ndarray:
    sums = np.bincount(body_index, weights=values, minlength=num_bodies)
    counts = np.bincount(body_index, minlength=num_bodies)
    return sums / np.maximum(counts, 1)


def evaluate_catalog(
    checkpoint: Path,
    manifest: dict,
    catalog_path: Path,
    *,
    replicas: int,
    horizon: int,
    seed: int,
    use_mjwarp: bool,
    terminal_handler: str | None = None,
) -> dict:
    catalog = EmbodimentCatalog.load(catalog_path)
    num_envs = catalog.num_bodies * replicas
    backbone = str(manifest.get("backbone", "mlp")).lower()
    algorithm = PPOJax if backbone == "mlp" else URMAPPO
    agent_conf, agent_state = algorithm.load_agent(checkpoint)
    env = build_eval_env(
        manifest,
        catalog_path,
        use_mjwarp,
        f"eval_{catalog_path.stem}_{seed}_{terminal_handler or 'stock'}",
        terminal_handler=terminal_handler,
    )

    policy = rollout(
        env,
        agent_conf,
        agent_state,
        num_envs=num_envs,
        horizon=horizon,
        seed=seed,
    )
    baseline = rollout(
        env,
        agent_conf,
        agent_state,
        num_envs=num_envs,
        horizon=horizon,
        seed=seed,
        zero_action=True,
    )
    if not np.array_equal(policy["body_index"], baseline["body_index"]):
        raise AssertionError("Policy and zero-action arms saw different bodies.")

    expected = fixed_balanced_assignment(num_envs, catalog.num_bodies)
    if not np.array_equal(policy["body_index"], expected):
        raise AssertionError("Evaluation did not use the balanced catalog schedule.")

    body_index = policy["body_index"]
    n = catalog.num_bodies
    body_return = per_body_aggregate(policy["returns"], body_index, n)
    body_length = per_body_aggregate(policy["lengths"].astype(float), body_index, n)
    zero_return = per_body_aggregate(baseline["returns"], body_index, n)
    zero_length = per_body_aggregate(baseline["lengths"].astype(float), body_index, n)
    non_fall = per_body_aggregate((~policy["fell"]).astype(float), body_index, n)
    zero_non_fall = per_body_aggregate((~baseline["fell"]).astype(float), body_index, n)
    reached_horizon = per_body_aggregate(
        (policy["lengths"] >= horizon).astype(float), body_index, n
    )
    beats_zero = body_return > zero_return

    normalized = catalog.normalized(
        low=manifest["morphology_low"], high=manifest["morphology_high"]
    )
    return {
        "catalog": str(catalog_path),
        "catalog_hash": catalog.content_hash,
        "catalog_split": catalog.split,
        "catalog_sampling_method": catalog.sampling_method,
        "num_bodies": n,
        "replicas_per_body": replicas,
        "num_envs": num_envs,
        "horizon": horizon,
        "seed": seed,
        "episode_return": _summary(body_return),
        "episode_length": _summary(body_length),
        "zero_action_episode_return": _summary(zero_return),
        "zero_action_episode_length": _summary(zero_length),
        # A fall is the terminal-state handler firing (`absorbing`), not `done`:
        # `done` also fires when the episode simply reaches the env horizon.
        "non_fall_rate_mean": float(np.mean(non_fall)),
        "zero_action_non_fall_rate_mean": float(np.mean(zero_non_fall)),
        "reached_horizon_rate_mean": float(np.mean(reached_horizon)),
        "episode_length_is_right_censored_at_horizon": bool(
            np.any(policy["lengths"] >= horizon)
        ),
        "env_horizon": int(env.info.horizon),
        "termination_rate_mean": float(np.mean(policy["terminated"])),
        "fraction_bodies_beating_zero_action": float(np.mean(beats_zero)),
        "num_bodies_beating_zero_action": int(np.sum(beats_zero)),
        "mean_return_improvement_over_zero": float(np.mean(body_return - zero_return)),
        "mean_length_improvement_over_zero": float(np.mean(body_length - zero_length)),
        "return_correlation_by_morphology": {
            name: _correlation(normalized[:, i], body_return)
            for i, name in enumerate(MORPHOLOGY_NAMES)
        },
        "max_abs_morphology_correlation": max(
            abs(v)
            for v in (
                _correlation(normalized[:, i], body_return)
                for i in range(len(MORPHOLOGY_NAMES))
            )
            if v is not None
        ),
        "rollout_seconds": policy["seconds"],
        "zero_action_rollout_seconds": baseline["seconds"],
        "per_body": [
            {
                "body_id": int(catalog.body_ids[i]),
                "descriptor": catalog.descriptors[i].tolist(),
                "normalized_descriptor": normalized[i].tolist(),
                "policy_return": float(body_return[i]),
                "policy_length": float(body_length[i]),
                "zero_action_return": float(zero_return[i]),
                "zero_action_length": float(zero_length[i]),
                "non_fall_rate": float(non_fall[i]),
                "reached_horizon_rate": float(reached_horizon[i]),
                "beats_zero_action": bool(beats_zero[i]),
            }
            for i in range(n)
        ],
    }


def main() -> None:
    args = parse_args()
    manifest_path, manifest = checkpoint_manifest(args.checkpoint)
    results = {
        "experiment": "catalog_heldout_evaluation",
        "checkpoint": str(args.checkpoint),
        "training_manifest": str(manifest_path),
        "training_catalog_hash": manifest.get("catalog_hash"),
        "training_catalog_mode": manifest.get("catalog_mode"),
        "terminal_handler_override": args.terminal_handler,
        "training_seed": manifest.get("seed"),
        "training_total_timesteps": manifest.get("total_timesteps"),
        "jax_backend": jax.default_backend(),
        "evaluations": {},
    }
    for catalog_path in args.catalogs:
        name = Path(catalog_path).stem
        print(f"[eval] {name}", flush=True)
        results["evaluations"][name] = evaluate_catalog(
            args.checkpoint,
            manifest,
            Path(catalog_path),
            replicas=args.replicas,
            horizon=args.horizon,
            seed=args.seed,
            use_mjwarp=args.use_mjwarp,
            terminal_handler=args.terminal_handler,
        )
        summary = results["evaluations"][name]
        print(
            f"[eval] {name}: mean return {summary['episode_return']['mean']:.3f} "
            f"(zero {summary['zero_action_episode_return']['mean']:.3f}), "
            f"median length {summary['episode_length']['median']:.1f}, "
            f"non-fall {summary['non_fall_rate_mean']:.1%}, "
            f"beats zero {summary['fraction_bodies_beating_zero_action']:.1%}",
            flush=True,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(results, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"[eval] wrote {args.output}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--catalogs", type=Path, nargs="+", required=True)
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--use-mjwarp", action="store_true")
    parser.add_argument(
        "--terminal-handler",
        default=None,
        help=(
            "Override the terminal-state handler, e.g. "
            "MorphologyAwareRootPoseTrajTerminalStateHandler to stop judging "
            "tall bodies against the reference body's standing height."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
