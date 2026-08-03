"""Evaluate an online-morphology AMP policy per body and per motion.

The registered AMP acceptance criteria need more than a mean: a non-fall rate, a
breakdown by body and by motion clip, and a held-out motion. This evaluator
assigns bodies deterministically by environment slot (so every arm sees the same
bodies) and records which trajectory each episode actually started on, so the
per-motion split is measured rather than assumed.

Reset and rollout run inside one jitted function; a standalone
``jit(vmap(mjx_reset_with_slot))`` is unstable on the measured ROCm stack.
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

from loco_mujoco.core.wrappers import LogWrapper  # noqa: E402
from loco_mujoco.core.wrappers.mjx import LogEnvState, Metrics  # noqa: E402

from scaling.embodiment_catalog import (  # noqa: E402
    EmbodimentCatalog,
    fixed_balanced_assignment,
)
from scaling.online_amp import OnlineMorphAMP  # noqa: E402
from scaling.online_amp_train import build_online_amp_env  # noqa: E402


def _summary(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p10": float(np.percentile(values, 10)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }


def rollout(env, agent_conf, agent_state, *, num_envs, horizon, seed, zero_action):
    logged = LogWrapper(env)
    keys = jax.random.split(jax.random.PRNGKey(seed), num_envs)
    slots = jnp.arange(num_envs, dtype=jnp.int32)

    variables = (
        None
        if zero_action
        else {
            "params": agent_state.train_state.params,
            "run_stats": agent_state.train_state.run_stats,
        }
    )

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
        next_obs, reward, absorbing, done, _, next_state = vstep(state, action_for(obs))
        active = ~completed
        returns = returns + reward * active
        lengths = lengths + active.astype(lengths.dtype)
        fell = fell | (absorbing.astype(jnp.bool_) & active)
        completed = completed | done.astype(jnp.bool_)
        return (next_obs, next_state, returns, lengths, completed, fell), None

    def reset_and_rollout(keys, slots):
        env_state = jax.vmap(env.mjx_reset_with_slot, in_axes=(0, 0))(keys, slots)
        zeros = jnp.zeros((num_envs,), dtype=jnp.float32)
        izeros = jnp.zeros((num_envs,), dtype=jnp.int32)
        log_state = LogEnvState(
            env_state,
            Metrics(zeros, izeros, zeros, izeros, izeros, jnp.zeros((num_envs,), bool)),
        )
        initial = (
            env_state.observation,
            log_state,
            zeros,
            izeros,
            jnp.zeros((num_envs,), dtype=jnp.bool_),
            jnp.zeros((num_envs,), dtype=jnp.bool_),
        )
        (_, _, returns, lengths, completed, fell), _ = jax.lax.scan(
            step_fn, initial, None, horizon
        )
        return (
            env_state.additional_carry.body_index,
            # Which clip this episode actually started on.
            env_state.additional_carry.traj_state.traj_no,
            returns,
            lengths,
            completed,
            fell,
        )

    started = time.perf_counter()
    body_index, traj_no, returns, lengths, completed, fell = jax.jit(reset_and_rollout)(
        keys, slots
    )
    jax.block_until_ready(returns)
    return {
        "body_index": np.asarray(body_index).reshape(-1),
        "traj_no": np.asarray(traj_no).reshape(-1),
        "returns": np.asarray(returns),
        "lengths": np.asarray(lengths),
        "terminated": np.asarray(completed),
        "fell": np.asarray(fell),
        "seconds": time.perf_counter() - started,
    }


def evaluate_clip_set(
    manifest: dict,
    checkpoint: Path,
    catalog_path: Path,
    clips: list[str],
    *,
    replicas: int,
    horizon: int,
    seed: int,
    label: str,
) -> dict:
    catalog = EmbodimentCatalog.load(catalog_path)
    num_envs = catalog.num_bodies * replicas
    args = SimpleNamespace(
        catalog=catalog_path,
        catalog_mode="fixed_balanced",
        catalog_stride=1,
        clips=list(clips),
        frames_per_clip=manifest.get("frames_per_clip_limit"),
        xml=Path(
            manifest.get(
                "xml",
                WORKSPACE / "generated_variants" / "h1_morphology_nominal" / "h1.xml",
            )
        ),
        num_envs=num_envs,
        use_mjwarp=False,
        run_tag=f"ampeval_{label}_{seed}",
        joint_space_reward=bool(manifest.get("joint_space_reward_only", True)),
    )
    env, catalog, _, _ = build_online_amp_env(args)
    agent_conf, agent_state = OnlineMorphAMP.load_agent(checkpoint)

    policy = rollout(
        env,
        agent_conf,
        agent_state,
        num_envs=num_envs,
        horizon=horizon,
        seed=seed,
        zero_action=False,
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

    def group(values, keys, n):
        sums = np.bincount(keys, weights=values, minlength=n)
        counts = np.bincount(keys, minlength=n)
        return sums / np.maximum(counts, 1), counts

    n_bodies = catalog.num_bodies
    n_clips = len(clips)
    body_len, _ = group(policy["lengths"].astype(float), policy["body_index"], n_bodies)
    body_ret, _ = group(policy["returns"], policy["body_index"], n_bodies)
    body_nonfall, _ = group(
        (~policy["fell"]).astype(float), policy["body_index"], n_bodies
    )
    body_zero_ret, _ = group(baseline["returns"], baseline["body_index"], n_bodies)
    clip_len, clip_counts = group(
        policy["lengths"].astype(float), policy["traj_no"], n_clips
    )
    clip_ret, _ = group(policy["returns"], policy["traj_no"], n_clips)
    clip_nonfall, _ = group((~policy["fell"]).astype(float), policy["traj_no"], n_clips)

    return {
        "label": label,
        "clips": list(clips),
        "catalog": str(catalog_path),
        "catalog_hash": catalog.content_hash,
        "num_bodies": n_bodies,
        "replicas_per_body": replicas,
        "num_envs": num_envs,
        "horizon": horizon,
        "env_horizon": int(env.info.horizon),
        "mean_episode_length": float(np.mean(policy["lengths"])),
        "episode_length": _summary(body_len),
        "episode_return": _summary(body_ret),
        "zero_action_episode_return": _summary(body_zero_ret),
        "non_fall_rate_mean": float(np.mean(~policy["fell"])),
        "zero_action_non_fall_rate_mean": float(np.mean(~baseline["fell"])),
        "reached_horizon_rate_mean": float(np.mean(policy["lengths"] >= horizon)),
        "fraction_bodies_beating_zero_action": float(np.mean(body_ret > body_zero_ret)),
        "per_motion": [
            {
                "clip": clips[i],
                "traj_no": i,
                "num_episodes": int(clip_counts[i]),
                "mean_episode_length": float(clip_len[i]),
                "mean_return": float(clip_ret[i]),
                "non_fall_rate": float(clip_nonfall[i]),
            }
            for i in range(n_clips)
        ],
        "per_body": [
            {
                "body_id": int(catalog.body_ids[i]),
                "descriptor": catalog.descriptors[i].tolist(),
                "mean_episode_length": float(body_len[i]),
                "mean_return": float(body_ret[i]),
                "zero_action_return": float(body_zero_ret[i]),
                "non_fall_rate": float(body_nonfall[i]),
                "beats_zero_action": bool(body_ret[i] > body_zero_ret[i]),
            }
            for i in range(n_bodies)
        ],
        "rollout_seconds": policy["seconds"],
    }


def main() -> None:
    args = parse_args()
    manifest_path = args.checkpoint.parent.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    catalog_path = args.catalog or Path(manifest["catalog_path"])
    train_clips = args.train_clips or list(manifest["clips"])

    results = {
        "experiment": "online_amp_heldout_evaluation",
        "checkpoint": str(args.checkpoint),
        "training_manifest": str(manifest_path),
        "training_seed": manifest.get("seed"),
        "training_total_timesteps": manifest.get("total_timesteps"),
        "discriminator_blind_to_descriptor": manifest.get(
            "discriminator_blind_to_descriptor"
        ),
        "joint_space_reward_only": manifest.get("joint_space_reward_only"),
        "jax_backend": jax.default_backend(),
        "evaluations": {},
    }
    plan = [("train_motions", train_clips)]
    if args.heldout_clips:
        plan.append(("heldout_motions", args.heldout_clips))
    for label, clips in plan:
        print(f"[amp-eval] {label}: {clips}", flush=True)
        summary = evaluate_clip_set(
            manifest,
            args.checkpoint,
            catalog_path,
            clips,
            replicas=args.replicas,
            horizon=args.horizon,
            seed=args.seed,
            label=label,
        )
        results["evaluations"][label] = summary
        print(
            f"[amp-eval] {label}: mean length {summary['mean_episode_length']:.1f}, "
            f"non-fall {summary['non_fall_rate_mean']:.1%}, "
            f"beats zero {summary['fraction_bodies_beating_zero_action']:.1%}",
            flush=True,
        )
        for row in summary["per_motion"]:
            print(
                f"    {row['clip']:<20} n={row['num_episodes']:<5} "
                f"len={row['mean_episode_length']:7.1f} "
                f"non-fall={row['non_fall_rate']:.1%}",
                flush=True,
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(results, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"[amp-eval] wrote {args.output}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=None)
    parser.add_argument("--train-clips", nargs="+", default=None)
    parser.add_argument("--heldout-clips", nargs="+", default=["dance2_subject5"])
    parser.add_argument("--replicas", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
