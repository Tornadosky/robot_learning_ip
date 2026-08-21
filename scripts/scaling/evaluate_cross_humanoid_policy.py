"""Evaluate a padded cross-humanoid checkpoint per robot against zero action."""

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

from scaling.parallel_cross_humanoid_train import (  # noqa: E402
    build_cross_humanoid_env,
    trainer_for,
)


def _resolve_checkpoint(checkpoint: Path) -> Path:
    """Accept either the saved pickle or the directory holding it.

    ``save_agent`` names the file after the trainer class, so the name differs
    between the masked-MLP and URMA arms; globbing keeps callers from having to
    know which backbone produced the run.
    """
    if checkpoint.is_file():
        return checkpoint
    candidates = sorted(checkpoint.glob("*_saved.pkl"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one *_saved.pkl in {checkpoint}; found {candidates}."
        )
    return candidates[0]


def _find_manifest(checkpoint: Path):
    checkpoint = checkpoint.resolve()
    for parent in checkpoint.parents:
        candidate = parent / "manifest.json"
        if candidate.is_file():
            return candidate, json.loads(candidate.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"No manifest.json found above {checkpoint}.")


def _summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
    }


def _env_args(args, manifest):
    """Rebuild the training environment, optionally on a held-out topology.

    The padded layout must match the checkpoint exactly, so any robot the run
    trained on but we are not evaluating is still *reserved*: its slots stay in
    the observation vector, zero-filled and masked off.
    """
    trained = list(manifest["robots"])
    robots = list(args.robots) if args.robots else trained
    reserved = list(manifest.get("reserved_robots", []))
    reserve = [r for r in (*trained, *reserved) if r not in robots]
    return SimpleNamespace(
        robots=robots,
        source=str(manifest["source_robot"]),
        clip=str(manifest["clip"]),
        start_frame=int(manifest["window_start_frame"]),
        frames=int(manifest["window_frames"]),
        reference_mode=str(manifest["reference_mode"]),
        reference_root=args.reference_root,
        use_mjwarp=bool(args.use_mjwarp),
        envs_per_robot=int(args.envs_per_robot),
        total_envs=int(args.envs_per_robot) * len(robots),
        robot_one_hot=bool(manifest.get("robot_one_hot", True)),
        append_joint_features=bool(manifest.get("append_joint_features", False)),
        reserve_robots=reserve,
        clip_windows=manifest.get("clip_windows"),
        morphology=manifest.get("morphology"),
        blank_goal=bool(manifest.get("blank_goal_observation", False)),
        goal_for_critic=bool(manifest.get("goal_for_critic", False)),
        actor_latent_dim=int(manifest.get("actor_latent_dim", 0)),
        latent_codes=manifest.get("latent_codes"),
        reward_type=str(manifest.get("reward_type", "MimicReward")),
        # The goal class sets the observation WIDTH (MorphGoalTrajMimicRootErr
        # appends 3 root-error dims), so replaying a checkpoint under the stock
        # goal would build a network of the wrong shape. The terminal handler
        # and deviation threshold do not change the width, but evaluating a
        # policy under a different termination rule than it trained on measures
        # something else, so both are carried over too.
        goal_type=str(manifest.get("goal_type", "GoalTrajMimic")),
        terminal_handler=manifest.get("terminal_handler"),
        max_root_deviation=manifest.get("max_root_pos_deviation"),
        # Control/action semantics and reference preprocessing MUST match
        # training or the eval measures a different controller entirely: a
        # PD-trained policy replayed under torque control scored 6.8 steps
        # against its own training-log 306 (found 2026-08-18). Every field
        # below was previously dropped and silently defaulted.
        pd_control=manifest.get("pd_control") or None,
        pd_gain_scale=float(manifest.get("pd_gain_scale", 1.0) or 1.0),
        pd_action_scale=manifest.get("pd_action_scale"),
        root_frame=str(manifest.get("root_frame", "absolute")),
        reference_grounding=str(manifest.get("reference_grounding", "none")),
        foot_model=str(manifest.get("foot_model", "stock")),
        n_substeps=manifest.get("n_substeps"),
        env_horizon=manifest.get("env_horizon"),
        root_rot_margin_degrees=manifest.get("root_rot_margin_degrees"),
        terminal_tilt_degrees=manifest.get("terminal_tilt_degrees"),
        morphology_catalog_file=manifest.get("morphology_catalog_file"),
        joint_target_obs=bool(manifest.get("joint_target_obs", False)),
        # Evaluation rebuilds envs repeatedly; the census belongs to training.
        admission_census_resets=0,
    )


def evaluate(env, agent_conf, agent_state, horizon: int, seed: int, zero_action=False):
    """Roll out one arm.  ``fell`` is tracked from ``absorbing``, never ``done``.

    ``done`` also fires at the environment horizon and when the reference clip
    runs out, so a fall rate read from it can report 0% while the true fall rate
    is 94%.  ``absorbing`` is the actual terminal-state flag.
    """
    keys = jax.random.split(jax.random.PRNGKey(seed), env.num_envs)
    observation, state = jax.jit(env.reset)(keys)
    variables = {
        "params": agent_state.train_state.params,
        "run_stats": agent_state.train_state.run_stats,
    }

    def choose_action(obs):
        if zero_action:
            return jnp.zeros((obs.shape[0], env.max_action_dim), dtype=obs.dtype)
        (policy, _), _ = agent_conf.network.apply(variables, obs, mutable=["run_stats"])
        return policy.mean()

    def rollout_step(carry, _):
        obs, env_state, returns, lengths, completed, fell = carry
        action = choose_action(obs)
        next_obs, reward, absorbing, done, _, next_state = env.step(env_state, action)
        active = ~completed
        returns = returns + reward * active
        lengths = lengths + active.astype(lengths.dtype)
        fell = fell | (absorbing.astype(jnp.bool_) & active)
        completed = completed | done.astype(jnp.bool_)
        return (next_obs, next_state, returns, lengths, completed, fell), None

    initial = (
        observation,
        state,
        jnp.zeros((env.num_envs,), dtype=jnp.float32),
        jnp.zeros((env.num_envs,), dtype=jnp.int32),
        jnp.zeros((env.num_envs,), dtype=jnp.bool_),
        jnp.zeros((env.num_envs,), dtype=jnp.bool_),
    )
    rollout = jax.jit(lambda carry: jax.lax.scan(rollout_step, carry, None, horizon))
    started = time.perf_counter()
    (_, _, returns, lengths, completed, fell), _ = rollout(initial)
    jax.block_until_ready(returns)
    return (
        np.asarray(returns),
        np.asarray(lengths),
        np.asarray(completed),
        np.asarray(fell),
        time.perf_counter() - started,
    )


def _welch(a, b):
    """Two-sample Welch t statistic for policy-minus-zero-action mean return."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    va, vb = np.var(a, ddof=1) / len(a), np.var(b, ddof=1) / len(b)
    denominator = np.sqrt(va + vb)
    if denominator == 0.0:
        return None
    return float((np.mean(a) - np.mean(b)) / denominator)


def main():
    args = parse_args()
    args.checkpoint = _resolve_checkpoint(args.checkpoint)
    manifest_path, manifest = _find_manifest(args.checkpoint)
    backbone = str(manifest.get("backbone", "masked_mlp"))
    trainer = trainer_for(backbone)
    env_args = _env_args(args, manifest)
    if env_args.robot_one_hot and env_args.robots != list(manifest["robots"]):
        raise ValueError(
            "This checkpoint uses a robot one-hot, so the evaluated robot set and "
            "order must match training exactly; the one-hot index is positional."
        )
    agent_conf, agent_state = trainer.load_agent(args.checkpoint)
    env, build_metadata = build_cross_humanoid_env(env_args)
    returns, lengths, completed, fell, policy_seconds = evaluate(
        env, agent_conf, agent_state, args.horizon, args.seed
    )
    (
        zero_returns,
        zero_lengths,
        zero_completed,
        zero_fell,
        zero_seconds,
    ) = evaluate(
        env,
        agent_conf,
        agent_state,
        args.horizon,
        args.seed,
        zero_action=True,
    )

    per_robot = {}
    for group in env.groups:
        slc = slice(group.start, group.stop)
        per_robot[group.name] = {
            "num_envs": group.size,
            "num_joints": int(build_metadata["joint_counts"].get(group.name, 0)),
            "policy_return": _summary(returns[slc]),
            "zero_action_return": _summary(zero_returns[slc]),
            "policy_length": _summary(lengths[slc]),
            "zero_action_length": _summary(zero_lengths[slc]),
            "mean_return_improvement": float(np.mean(returns[slc] - zero_returns[slc])),
            "mean_length_improvement": float(np.mean(lengths[slc] - zero_lengths[slc])),
            "return_improvement_welch_t": _welch(returns[slc], zero_returns[slc]),
            "beats_zero_action_on_mean_return": bool(
                np.mean(returns[slc]) > np.mean(zero_returns[slc])
            ),
            "fraction_beating_zero_action_return": float(
                np.mean(returns[slc] > zero_returns[slc])
            ),
            # Fall rate comes from `absorbing`; `done` also fires at the horizon
            # and at reference-clip exhaustion.
            "policy_fall_rate": float(np.mean(fell[slc])),
            "zero_action_fall_rate": float(np.mean(zero_fell[slc])),
            "policy_non_fall_rate": float(1.0 - np.mean(fell[slc])),
            "zero_action_non_fall_rate": float(1.0 - np.mean(zero_fell[slc])),
            "policy_completed": int(np.sum(completed[slc])),
            "zero_action_completed": int(np.sum(zero_completed[slc])),
        }

    result = {
        "experiment": "heldout_cross_humanoid_evaluation",
        "checkpoint": str(args.checkpoint),
        "training_manifest": str(manifest_path),
        "backbone": backbone,
        "trained_robots": list(manifest["robots"]),
        "robot_one_hot": bool(env_args.robot_one_hot),
        "topology_held_out": [r for r in env.names if r not in manifest["robots"]],
        "reserved_robots": list(env_args.reserve_robots),
        "seed": args.seed,
        "horizon": args.horizon,
        "envs_per_robot": args.envs_per_robot,
        "robots": list(env.names),
        "overall_policy_return": _summary(returns),
        "overall_zero_action_return": _summary(zero_returns),
        "overall_policy_length": _summary(lengths),
        "overall_zero_action_length": _summary(zero_lengths),
        "overall_mean_return_improvement": float(np.mean(returns - zero_returns)),
        "overall_mean_length_improvement": float(np.mean(lengths - zero_lengths)),
        "overall_fraction_beating_zero_action_return": float(
            np.mean(returns > zero_returns)
        ),
        "every_robot_beats_zero_action": bool(
            all(
                entry["beats_zero_action_on_mean_return"]
                for entry in per_robot.values()
            )
        ),
        "per_robot": per_robot,
        "policy_rollout_compile_and_run_seconds": policy_seconds,
        "zero_action_rollout_compile_and_run_seconds": zero_seconds,
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
        "--robots",
        nargs="*",
        default=None,
        help="Override the evaluated robots, e.g. a held-out topology. Robots the "
        "run trained on stay reserved so the padded layout still matches.",
    )
    parser.add_argument("--envs-per-robot", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--use-mjwarp", action="store_true")
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=WORKSPACE / "external_data" / "cross_humanoid",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
