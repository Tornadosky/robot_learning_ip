"""Train one padded shared PPO policy across several humanoid topologies.

This is a bridge between the cross-humanoid reference cache and the fully
embodiment-aware policy work.  Each robot remains a homogeneous MJX vmap group;
observations/actions are padded to the largest robot and a robot one-hot is
appended before all groups enter one PPO rollout/update.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

WORKSPACE = Path(__file__).resolve().parents[2]
SCRIPTS = WORKSPACE / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from loco_mujoco.trajectory import Trajectory  # noqa: E402

from morphology_deepmimic import make_mimic_env  # noqa: E402
from scaling.cross_humanoid_retarget import HUMANOIDS  # noqa: E402
from scaling.cross_topology_urma import CrossTopologyURMAPPO  # noqa: E402
from scaling.joint_descriptions import build_joint_block_spec  # noqa: E402
from scaling.masked_mlp import MaskedParallelPPO  # noqa: E402
from scaling.parallel_env import (  # noqa: E402
    ParallelMorphVecEnv,
    balanced_group_sizes,
    describe_layout,
)

BACKBONES = ("masked_mlp", "urma", "urmav2")


def trainer_for(backbone: str):
    return MaskedParallelPPO if backbone == "masked_mlp" else CrossTopologyURMAPPO


def _largest_divisor_at_most(value: int, limit: int) -> int:
    for candidate in range(min(value, limit), 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


def build_config(args, total_envs: int, actual_timesteps: int):
    batch_size = args.num_steps * total_envs
    minibatches = _largest_divisor_at_most(batch_size, args.num_minibatches)
    updates = max(1, actual_timesteps // batch_size)
    return OmegaConf.create(
        {
            "experiment": {
                "hidden_layers": list(args.hidden),
                # URMAPPO reads these; the masked MLP ignores them.
                "backbone": "urma" if args.backbone == "urma" else "urmav2",
                "urma_activation": "elu",
                "urma_latent_slots": args.urma_latent_slots,
                "urma_joint_value_dim": args.urma_joint_value_dim,
                "lr": args.lr,
                "num_envs": total_envs,
                "num_steps": args.num_steps,
                "total_timesteps": float(actual_timesteps),
                "update_epochs": args.update_epochs,
                "proportion_env_reward": 0.0,
                "num_minibatches": minibatches,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_eps": 0.2,
                "init_std": args.init_std,
                "learnable_std": bool(args.learnable_std),
                "ent_coef": 0.0,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
                "activation": "tanh",
                "anneal_lr": False,
                "weight_decay": 0.0,
                "normalize_env": not args.no_normalize_reward,
                "debug": False,
                "n_seeds": 1,
                "vmap_across_seeds": True,
                "validation": {
                    "active": False,
                    "num_steps": 100,
                    "num_envs": min(100, total_envs),
                    "num": min(10, updates),
                },
            }
        }
    )


def reference_path(args, robot: str) -> Path:
    tag = f"start{args.start_frame}_{args.frames}f_{args.reference_mode}.npz"
    return args.reference_root / f"{args.source}_source" / args.clip / robot / tag


def _build_robot_env(args, robot: str):
    spec = HUMANOIDS[robot]
    path = reference_path(args, robot)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {robot} reference {path}. Run "
            "scripts/scaling/cross_humanoid_retarget.py first."
        )
    trajectory = Trajectory.load(str(path))
    env = make_mimic_env(
        spec.mjx_env_name,
        trajectory,
        use_mjwarp=args.use_mjwarp,
        nconmax=7000,
        headless=True,
    )
    return env, trajectory


def _reserved_dims(args):
    """Observation/action widths to keep free for robots we do not train on.

    A fixed-width network can only be replayed on a held-out topology if that
    topology's slots existed at training time.  Reserving them costs a few
    always-zero, always-masked inputs and nothing else.
    """
    reserve = [robot for robot in args.reserve_robots if robot not in args.robots]
    if not reserve:
        return 0, 0, 0, {}
    observation_dim = 0
    action_dim = 0
    details = {}
    for robot in reserve:
        env, _ = _build_robot_env(args, robot)
        details[robot] = {
            "observation_dim": int(env.info.observation_space.shape[0]),
            "action_dim": int(env.info.action_space.shape[0]),
        }
        observation_dim = max(observation_dim, details[robot]["observation_dim"])
        action_dim = max(action_dim, details[robot]["action_dim"])
        print(
            f"[cross-train] reserved slots for {robot}: "
            f"obs={details[robot]['observation_dim']} "
            f"act={details[robot]['action_dim']}",
            flush=True,
        )
        del env
    return observation_dim, action_dim, len(args.robots) + len(reserve), details


def build_cross_humanoid_env(args):
    envs = []
    joint_block_specs = []
    references = {}
    build_started = time.perf_counter()
    for robot in args.robots:
        env, trajectory = _build_robot_env(args, robot)
        envs.append(env)
        joint_block_specs.append(build_joint_block_spec(env, robot))
        references[robot] = {
            "path": str(reference_path(args, robot)),
            "samples": int(trajectory.data.n_samples),
            "frequency_hz": float(trajectory.info.frequency),
            "observation_dim": int(env.info.observation_space.shape[0]),
            "action_dim": int(env.info.action_space.shape[0]),
        }
        print(
            f"[cross-train] built {robot:>12s}: "
            f"obs={env.info.observation_space.shape[0]} "
            f"act={env.info.action_space.shape[0]} "
            f"samples={trajectory.data.n_samples}",
            flush=True,
        )

    reserved_obs, reserved_act, reserved_slots, reserved_detail = _reserved_dims(args)

    group_sizes = (
        tuple([args.envs_per_robot] * len(envs))
        if args.envs_per_robot is not None
        else balanced_group_sizes(args.total_envs, len(envs))
    )
    parallel_env = ParallelMorphVecEnv(
        envs,
        group_sizes,
        names=args.robots,
        history_length=1,
        pad_to_max_shapes=True,
        append_group_one_hot=bool(args.robot_one_hot),
        append_action_mask=True,
        joint_block_specs=joint_block_specs if args.append_joint_features else None,
        reserved_observation_dim=reserved_obs,
        reserved_action_dim=reserved_act,
        reserved_group_slots=reserved_slots,
    )
    layout = parallel_env.urma_input_layout
    metadata = {
        "source_robot": args.source,
        "clip": args.clip,
        "reference_mode": args.reference_mode,
        "window_start_frame": args.start_frame,
        "window_frames": args.frames,
        "references": references,
        "environment_build_seconds": time.perf_counter() - build_started,
        "group_sizes": list(group_sizes),
        "group_observation_dims": list(parallel_env.group_observation_dims),
        "group_action_dims": list(parallel_env.group_action_dims),
        "padded_observation_dim": parallel_env.output_observation_dim,
        "padded_action_dim": parallel_env.max_action_dim,
        "robot_one_hot": bool(args.robot_one_hot),
        "robot_one_hot_dim": parallel_env.one_hot_dim,
        "action_mask_observation_start": parallel_env.action_mask_observation_start,
        "append_joint_features": bool(parallel_env.append_joint_features),
        "joint_feature_start": parallel_env.joint_feature_start,
        "num_joint_slots": parallel_env.num_joint_slots,
        "reserved_robots": list(reserved_detail),
        "reserved_robot_dims": reserved_detail,
        "joint_counts": {
            spec.name: spec.num_joints for spec in (joint_block_specs or ())
        },
        "urma_input_layout": None if layout is None else vars(layout),
    }
    return parallel_env, metadata


def run_preflight(env, seed: int):
    keys = jax.random.split(jax.random.PRNGKey(seed), env.num_envs)
    started = time.perf_counter()
    observation, state = jax.jit(env.reset)(keys)
    jax.block_until_ready(observation)
    reset_seconds = time.perf_counter() - started
    action = jnp.zeros((env.num_envs, env.max_action_dim), dtype=observation.dtype)
    started = time.perf_counter()
    next_observation, reward, _, done, _, _ = jax.jit(env.step)(state, action)
    jax.block_until_ready(next_observation)
    step_seconds = time.perf_counter() - started
    result = {
        "observation_shape": list(observation.shape),
        "action_shape": list(action.shape),
        "reward_shape": list(reward.shape),
        "done_shape": list(done.shape),
        "reset_compile_and_run_seconds": reset_seconds,
        "step_compile_and_run_seconds": step_seconds,
        "finite_observations": bool(np.isfinite(np.asarray(next_observation)).all()),
        "finite_rewards": bool(np.isfinite(np.asarray(reward)).all()),
    }
    print(json.dumps(result, indent=2), flush=True)
    return result


def _finite_or_none(value):
    value = float(value)
    return value if np.isfinite(value) else None


def _device_memory_stats():
    stats = jax.devices()[0].memory_stats()
    if not stats:
        return {}
    return {
        key: int(value)
        for key, value in stats.items()
        if isinstance(value, (int, np.integer))
    }


def main():
    args = parse_args()
    if len(set(args.robots)) != len(args.robots):
        raise ValueError("--robots must not contain duplicates.")
    env, build_metadata = build_cross_humanoid_env(args)
    print(
        f"[cross-train] backend={jax.default_backend()} robots={len(args.robots)} "
        f"total_envs={env.num_envs} "
        f"layout={describe_layout(env.names, env.group_sizes)} "
        f"padded_obs={env.output_observation_dim} "
        f"padded_act={env.max_action_dim}",
        flush=True,
    )
    if args.preflight_only:
        preflight = run_preflight(env, args.seed)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "experiment": "parallel_cross_humanoid_preflight",
            "implementation": "grouped_static_mjx_padded_observation_action_masked_shared_ppo",
            "robots": list(args.robots),
            "num_robot_topologies": len(args.robots),
            "total_envs": env.num_envs,
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in jax.devices()],
            "preflight": preflight,
            **build_metadata,
        }
        (args.output_dir / "preflight.json").write_text(
            json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
        )
        return

    steps_per_update = args.num_steps * env.num_envs
    num_updates = max(1, int(args.total_timesteps // steps_per_update))
    actual_timesteps = num_updates * steps_per_update
    config = build_config(args, env.num_envs, actual_timesteps)
    trainer = trainer_for(args.backbone)
    agent_conf = trainer.init_agent_conf(env, config)
    train_fn = jax.jit(trainer.build_train_fn(env, agent_conf, mh=None))
    key = jax.random.PRNGKey(args.seed)

    print(
        f"[cross-train] lowering/compiling updates={num_updates} "
        f"steps/update={steps_per_update:,} total={actual_timesteps:,}",
        flush=True,
    )
    started = time.perf_counter()
    executable = train_fn.lower(key).compile()
    compile_seconds = time.perf_counter() - started
    print(f"[cross-train] compile complete in {compile_seconds:.1f}s", flush=True)

    started = time.perf_counter()
    output = executable(key)
    jax.block_until_ready(output["agent_state"])
    training_seconds = time.perf_counter() - started
    throughput = actual_timesteps / training_seconds

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoint_final"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    agent_path = trainer.save_agent(
        str(checkpoint_dir), agent_conf, output["agent_state"]
    )
    returns = np.asarray(output["training_metrics"].mean_episode_return)
    lengths = np.asarray(output["training_metrics"].mean_episode_length)
    manifest = {
        "experiment": "parallel_cross_humanoid",
        "implementation": "grouped_static_mjx_padded_observation_action_masked_shared_ppo",
        "backbone": args.backbone,
        "robots": list(args.robots),
        "num_robot_topologies": len(args.robots),
        "total_envs": env.num_envs,
        "num_steps": args.num_steps,
        "num_updates": num_updates,
        "num_minibatches_requested": args.num_minibatches,
        "num_minibatches_actual": int(config.experiment.num_minibatches),
        "update_epochs": args.update_epochs,
        "total_timesteps": int(actual_timesteps),
        "seed": args.seed,
        "hidden": list(args.hidden),
        "lr": args.lr,
        "compile_seconds": compile_seconds,
        "training_seconds": training_seconds,
        "steps_per_second": throughput,
        "steps_per_minute": throughput * 60.0,
        "mean_episode_return_last": _finite_or_none(returns[-1]),
        "mean_episode_length_last": _finite_or_none(lengths[-1]),
        "return_curve": [_finite_or_none(value) for value in returns],
        "length_curve": [_finite_or_none(value) for value in lengths],
        "agent_path": str(agent_path),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "device_memory_stats": _device_memory_stats(),
        "action_mask_counts": [
            int(np.asarray(env.action_mask[group.start]).sum()) for group in env.groups
        ],
        **build_metadata,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(
        f"[cross-train] trained {actual_timesteps:,} steps in "
        f"{training_seconds:.1f}s ({throughput / 1e6 * 60:.2f}M steps/min) "
        f"-> {args.output_dir}",
        flush=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robots", nargs="+", choices=HUMANOIDS, default=["h1", "g1", "atlas"]
    )
    parser.add_argument("--source", choices=HUMANOIDS, default="h1")
    parser.add_argument("--clip", default="dance2_subject4")
    parser.add_argument("--start-frame", type=int, default=19482)
    parser.add_argument("--frames", type=int, default=800)
    parser.add_argument(
        "--reference-mode", choices=["direct", "robot2robot"], default="direct"
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=WORKSPACE / "external_data" / "cross_humanoid",
    )
    parser.add_argument("--backbone", choices=BACKBONES, default="masked_mlp")
    parser.add_argument(
        "--robot-one-hot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append the robot index. Disable to test structural generalisation.",
    )
    parser.add_argument(
        "--append-joint-features",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Emit the padded per-joint block. Defaults on for URMA backbones.",
    )
    parser.add_argument(
        "--reserve-robots",
        nargs="*",
        choices=list(HUMANOIDS),
        default=[],
        help="Robots to keep padded slots for without training on them.",
    )
    parser.add_argument("--urma-latent-slots", type=int, default=64)
    parser.add_argument("--urma-joint-value-dim", type=int, default=4)
    parser.add_argument("--total-envs", type=int, default=384)
    parser.add_argument("--envs-per-robot", type=int, default=None)
    parser.add_argument("--total-timesteps", type=float, default=1e6)
    parser.add_argument("--num-steps", type=int, default=32)
    parser.add_argument("--num-minibatches", type=int, default=12)
    parser.add_argument("--update-epochs", type=int, default=1)
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 128])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--init-std", type=float, default=0.2)
    parser.add_argument("--learnable-std", action="store_true")
    parser.add_argument("--no-normalize-reward", action="store_true")
    parser.add_argument("--use-mjwarp", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--run-tag", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.append_joint_features is None:
        args.append_joint_features = args.backbone != "masked_mlp"
    if args.backbone != "masked_mlp" and not args.append_joint_features:
        parser.error("URMA backbones require --append-joint-features.")
    if args.output_dir is None:
        args.output_dir = (
            WORKSPACE / "experiments" / "scaling_cross_topology" / args.run_tag
        )
    return args


if __name__ == "__main__":
    main()
