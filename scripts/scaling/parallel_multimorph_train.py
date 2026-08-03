"""Train one PPO policy on several H1 morphologies in every update.

Unlike the existing round-robin trainers, all morphology groups contribute to
each rollout/update.  Start with ``--preflight-only`` or a very small timestep
budget, then increase morphology count while keeping ``--total-envs`` fixed.
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

from loco_mujoco.algorithms import PPOJax  # noqa: E402
from loco_mujoco.task_factories import ImitationFactory, LAFAN1DatasetConf  # noqa: E402

from morphology_deepmimic import (  # noqa: E402
    crop_trajectory,
    get_robot,
    ground_trajectory_constant,
    make_mimic_env,
    register_variant_env,
    resolve_window,
    variant_env_names,
)
from scaling.parallel_env import (  # noqa: E402
    ParallelMorphPPO,
    ParallelMorphVecEnv,
    balanced_group_sizes,
    describe_layout,
)


DEFAULT_VARIANTS = [
    "nominal",
    "combined",
    "tall_legs",
    "extreme_tall_light",
]


def _largest_divisor_at_most(value: int, limit: int) -> int:
    for candidate in range(min(value, limit), 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


def build_config(args, total_envs: int, actual_timesteps: int):
    batch_size = args.num_steps * total_envs
    minibatches = _largest_divisor_at_most(batch_size, args.num_minibatches)
    updates = max(1, actual_timesteps // batch_size)
    validation_slots = min(10, updates)
    return OmegaConf.create(
        {
            "experiment": {
                "hidden_layers": list(args.hidden),
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
                    "num": validation_slots,
                },
            }
        }
    )


def build_parallel_env(args):
    robot = get_robot("h1")
    source_env = ImitationFactory.make(
        robot.cpu_env_name,
        lafan1_dataset_conf=LAFAN1DatasetConf([args.clip]),
    )
    full_traj = source_env.th.traj
    start_frame, n_frames = resolve_window(full_traj, args.duration, args.start_frame)
    base_traj = crop_trajectory(full_traj, start_frame, n_frames)

    envs = []
    ground_offsets = {}
    build_started = time.perf_counter()
    for index, name in enumerate(args.variants):
        xml_path = WORKSPACE / "generated_variants" / f"h1_morphology_{name}" / "h1.xml"
        if not xml_path.is_file():
            raise FileNotFoundError(
                f"Missing generated morphology {name!r}: {xml_path}. "
                "Generate/validate it before parallel training."
            )
        env_name = variant_env_names(robot, name, f"parallel_{args.run_tag}_{index}")[1]
        register_variant_env(env_name, xml_path, robot.base_mjx_cls)
        traj, offset = ground_trajectory_constant(xml_path, base_traj)
        env = make_mimic_env(
            env_name,
            traj,
            use_mjwarp=args.use_mjwarp,
            nconmax=7000,
            headless=True,
        )
        envs.append(env)
        ground_offsets[name] = float(offset)
        print(
            f"[parallel] built {name:>22s}: offset={offset:+.4f}m "
            f"obs={env.info.observation_space.shape} "
            f"act={env.info.action_space.shape}",
            flush=True,
        )

    if args.envs_per_morph is not None:
        group_sizes = tuple([args.envs_per_morph] * len(envs))
    else:
        group_sizes = balanced_group_sizes(args.total_envs, len(envs))
    parallel_env = ParallelMorphVecEnv(
        envs,
        group_sizes,
        names=args.variants,
        history_length=1,
    )
    metadata = {
        "clip": args.clip,
        "window_start_frame": int(start_frame),
        "window_frames": int(n_frames),
        "frequency_hz": float(base_traj.info.frequency),
        "ground_offsets": ground_offsets,
        "env_build_seconds": time.perf_counter() - build_started,
        "group_sizes": list(group_sizes),
    }
    return parallel_env, metadata


def run_preflight(env: ParallelMorphVecEnv, seed: int):
    key = jax.random.PRNGKey(seed)
    reset_keys = jax.random.split(key, env.num_envs)
    reset = jax.jit(env.reset)
    t0 = time.perf_counter()
    obs, state = reset(reset_keys)
    jax.block_until_ready(obs)
    reset_seconds = time.perf_counter() - t0
    actions = jnp.zeros((env.num_envs, env.info.action_space.shape[0]), dtype=obs.dtype)
    step = jax.jit(env.step)
    t0 = time.perf_counter()
    next_obs, reward, _, done, _, _ = step(state, actions)
    jax.block_until_ready(next_obs)
    step_seconds = time.perf_counter() - t0
    result = {
        "obs_shape": list(obs.shape),
        "reward_shape": list(reward.shape),
        "done_shape": list(done.shape),
        "reset_compile_and_run_seconds": reset_seconds,
        "step_compile_and_run_seconds": step_seconds,
        "finite_observations": bool(np.isfinite(np.asarray(next_obs)).all()),
        "finite_rewards": bool(np.isfinite(np.asarray(reward)).all()),
    }
    print(json.dumps(result, indent=2), flush=True)
    return result


def _device_memory_stats():
    """Return JSON-safe allocator counters when the backend exposes them."""
    stats = jax.devices()[0].memory_stats()
    if not stats:
        return {}
    return {
        key: int(value)
        for key, value in stats.items()
        if isinstance(value, (int, np.integer))
    }


def _finite_or_none(value):
    value = float(value)
    return value if np.isfinite(value) else None


def main():
    args = parse_args()
    if len(set(args.variants)) != len(args.variants):
        raise ValueError("--variants must not contain duplicates.")
    env, build_meta = build_parallel_env(args)
    print(
        f"[parallel] backend={jax.default_backend()} morphologies={env.num_morphologies} "
        f"total_envs={env.num_envs} layout={describe_layout(env.names, env.group_sizes)}",
        flush=True,
    )

    if args.preflight_only:
        run_preflight(env, args.seed)
        return

    steps_per_update = args.num_steps * env.num_envs
    num_updates = max(1, int(args.total_timesteps // steps_per_update))
    actual_timesteps = num_updates * steps_per_update
    config = build_config(args, env.num_envs, actual_timesteps)
    agent_conf = ParallelMorphPPO.init_agent_conf(env, config)
    train_fn = jax.jit(ParallelMorphPPO.build_train_fn(env, agent_conf, mh=None))
    key = jax.random.PRNGKey(args.seed)

    print(
        f"[parallel] lowering/compiling updates={num_updates} "
        f"steps/update={steps_per_update:,} total={actual_timesteps:,}",
        flush=True,
    )
    t0 = time.perf_counter()
    executable = train_fn.lower(key).compile()
    compile_seconds = time.perf_counter() - t0
    print(f"[parallel] compile complete in {compile_seconds:.1f}s", flush=True)

    t0 = time.perf_counter()
    output = executable(key)
    jax.block_until_ready(output["agent_state"])
    train_seconds = time.perf_counter() - t0
    throughput = actual_timesteps / train_seconds

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoint_final"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    agent_path = PPOJax.save_agent(
        str(checkpoint_dir), agent_conf, output["agent_state"]
    )

    returns = np.asarray(output["training_metrics"].mean_episode_return)
    lengths = np.asarray(output["training_metrics"].mean_episode_length)
    memory_stats = _device_memory_stats()
    manifest = {
        "experiment": "parallel_multi_morphology",
        "implementation": "grouped_static_mjx_single_compiled_rollout",
        "robot": "h1",
        "variants": list(args.variants),
        "num_morphologies": env.num_morphologies,
        "group_sizes": list(env.group_sizes),
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
        "training_seconds": train_seconds,
        "steps_per_second": throughput,
        "steps_per_minute": throughput * 60.0,
        "agent_path": str(agent_path),
        "mean_episode_return_last": _finite_or_none(returns[-1]),
        "mean_episode_length_last": _finite_or_none(lengths[-1]),
        "return_curve": [_finite_or_none(value) for value in returns],
        "length_curve": [_finite_or_none(value) for value in lengths],
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "device_memory_stats": memory_stats,
        **build_meta,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(
        f"[parallel] trained {actual_timesteps:,} steps in {train_seconds:.1f}s "
        f"({throughput / 1e6 * 60:.2f}M steps/min) -> {output_dir}",
        flush=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    parser.add_argument("--clip", default="walk1_subject1")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--total-envs", type=int, default=2048)
    parser.add_argument("--envs-per-morph", type=int, default=None)
    parser.add_argument("--total-timesteps", type=float, default=60e6)
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--num-minibatches", type=int, default=32)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--hidden", type=int, nargs="+", default=[512, 256])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--init-std", type=float, default=0.2)
    parser.add_argument("--learnable-std", action="store_true")
    parser.add_argument("--no-normalize-reward", action="store_true")
    parser.add_argument("--use-mjwarp", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--run-tag", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = WORKSPACE / "experiments" / "scaling_parallel" / args.run_tag
    return args


if __name__ == "__main__":
    main()
