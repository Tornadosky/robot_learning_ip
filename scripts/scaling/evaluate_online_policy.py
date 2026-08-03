"""Evaluate an online-morphology policy on a disjoint, fixed embodiment batch.

The evaluator reconstructs the training reference and morphology bounds from the
manifest beside a checkpoint.  Each evaluation environment receives a sampled
body at reset and keeps it fixed, even when the training run used per-episode
resampling.  Reported returns and lifetimes are from the first episode only.
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
from loco_mujoco.core.wrappers import LogWrapper, VecEnv  # noqa: E402

from scaling.online_h1 import MORPHOLOGY_NAMES  # noqa: E402
from scaling.online_h1_train import build_online_env  # noqa: E402
from scaling.urma_networks import URMAPPO  # noqa: E402


def _checkpoint_manifest(checkpoint: Path) -> tuple[Path, dict]:
    checkpoint = checkpoint.resolve()
    candidates = [checkpoint.parent / "manifest.json"]
    candidates.extend(parent / "manifest.json" for parent in checkpoint.parents)
    for candidate in candidates:
        if candidate.is_file():
            return candidate, json.loads(candidate.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"No manifest.json found above checkpoint {checkpoint}.")


def _build_env_args(args, manifest):
    frequency = float(manifest["frequency_hz"])
    duration = float(manifest["window_frames"]) / frequency
    return SimpleNamespace(
        clip=str(manifest["clip"]),
        duration=duration,
        start_frame=int(manifest["window_start_frame"]),
        run_tag=f"heldout_eval_{args.seed}",
        use_mjwarp=bool(args.use_mjwarp),
        backbone=str(manifest.get("backbone", args.backbone)),
        resample_per_episode=False,
        morph_low=list(manifest["morphology_low"]),
        morph_high=list(manifest["morphology_high"]),
    )


def _safe_correlation(x, y):
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return None
    value = float(np.corrcoef(x, y)[0, 1])
    return value if np.isfinite(value) else None


def _summary(values):
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


def evaluate(
    env,
    agent_conf,
    agent_state,
    num_envs: int,
    horizon: int,
    seed: int,
    *,
    zero_action: bool = False,
):
    wrapped = VecEnv(LogWrapper(env))
    keys = jax.random.split(jax.random.PRNGKey(seed), num_envs)
    observation, state = jax.jit(wrapped.reset)(keys)
    morphology = state.env_state.additional_carry.morphology

    variables = {
        "params": agent_state.train_state.params,
        "run_stats": agent_state.train_state.run_stats,
    }

    def deterministic_action(obs):
        if zero_action:
            return jnp.zeros(
                (*obs.shape[:-1], env.info.action_space.shape[0]),
                dtype=obs.dtype,
            )
        (policy, _), _ = agent_conf.network.apply(variables, obs, mutable=["run_stats"])
        return policy.mean()

    def rollout_step(carry, _):
        obs, env_state, returns, lengths, completed = carry
        action = deterministic_action(obs)
        next_obs, reward, _, done, _, next_state = wrapped.step(env_state, action)
        active = ~completed
        returns = returns + reward * active
        lengths = lengths + active.astype(lengths.dtype)
        completed = completed | done.astype(jnp.bool_)
        return (next_obs, next_state, returns, lengths, completed), None

    initial = (
        observation,
        state,
        jnp.zeros((num_envs,), dtype=jnp.float32),
        jnp.zeros((num_envs,), dtype=jnp.int32),
        jnp.zeros((num_envs,), dtype=jnp.bool_),
    )
    rollout = jax.jit(lambda carry: jax.lax.scan(rollout_step, carry, None, horizon))
    started = time.perf_counter()
    (final_obs, final_state, returns, lengths, completed), _ = rollout(initial)
    del final_obs, final_state
    jax.block_until_ready(returns)
    rollout_seconds = time.perf_counter() - started
    return (
        np.asarray(morphology),
        np.asarray(returns),
        np.asarray(lengths),
        np.asarray(completed),
        rollout_seconds,
    )


def main():
    args = parse_args()
    manifest_path, training_manifest = _checkpoint_manifest(args.checkpoint)
    backbone = str(training_manifest.get("backbone", args.backbone)).lower()
    algorithm = PPOJax if backbone == "mlp" else URMAPPO
    agent_conf, agent_state = algorithm.load_agent(args.checkpoint)
    env, _ = build_online_env(_build_env_args(args, training_manifest))

    morphology, returns, lengths, completed, rollout_seconds = evaluate(
        env, agent_conf, agent_state, args.num_envs, args.horizon, args.seed
    )
    (
        baseline_morphology,
        baseline_returns,
        baseline_lengths,
        baseline_completed,
        baseline_seconds,
    ) = evaluate(
        env,
        agent_conf,
        agent_state,
        args.num_envs,
        args.horizon,
        args.seed,
        zero_action=True,
    )
    if not np.array_equal(morphology, baseline_morphology):
        raise AssertionError(
            "Policy and zero-action baseline sampled different bodies."
        )
    correlations = {
        name: _safe_correlation(morphology[:, i], returns)
        for i, name in enumerate(MORPHOLOGY_NAMES)
    }
    result = {
        "experiment": "heldout_online_morphology_evaluation",
        "checkpoint": str(args.checkpoint),
        "training_manifest": str(manifest_path),
        "backbone": backbone,
        "seed": args.seed,
        "num_heldout_morphologies": args.num_envs,
        "num_unique_at_1e_6": int(
            len(np.unique(np.round(morphology, decimals=6), axis=0))
        ),
        "fixed_morphology_during_rollout": True,
        "horizon": args.horizon,
        "completed_first_episodes": int(np.sum(completed)),
        "completion_rate": float(np.mean(completed)),
        "first_episode_return": _summary(returns),
        "first_episode_length": _summary(lengths),
        "zero_action_completed_first_episodes": int(np.sum(baseline_completed)),
        "zero_action_completion_rate": float(np.mean(baseline_completed)),
        "zero_action_first_episode_return": _summary(baseline_returns),
        "zero_action_first_episode_length": _summary(baseline_lengths),
        "mean_return_improvement_over_zero_action": float(
            np.mean(returns - baseline_returns)
        ),
        "mean_length_improvement_over_zero_action": float(
            np.mean(lengths - baseline_lengths)
        ),
        "fraction_beating_zero_action_return": float(
            np.mean(returns > baseline_returns)
        ),
        "return_correlation_by_morphology": correlations,
        "morphology_mean": morphology.mean(axis=0).tolist(),
        "morphology_min": morphology.min(axis=0).tolist(),
        "morphology_max": morphology.max(axis=0).tolist(),
        "first_morphologies": morphology[: min(16, len(morphology))].tolist(),
        "rollout_compile_and_run_seconds": rollout_seconds,
        "zero_action_rollout_compile_and_run_seconds": baseline_seconds,
        "transitions_per_second_including_compile": (
            args.num_envs * args.horizon / rollout_seconds
        ),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--backbone", choices=["mlp", "urma", "urmav2"], default="urmav2"
    )
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--use-mjwarp", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
