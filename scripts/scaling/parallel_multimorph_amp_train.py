"""AMP training over several H1 bodies and several motions simultaneously."""

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

from loco_mujoco.task_factories import ImitationFactory, LAFAN1DatasetConf  # noqa: E402
from loco_mujoco.trajectory import Trajectory, TrajectoryData, TrajectoryTransitions  # noqa: E402

from morphology_deepmimic import (  # noqa: E402
    get_robot,
    ground_trajectory_constant,
    make_mimic_env,
    register_variant_env,
    variant_env_names,
)
from scaling.parallel_env import (  # noqa: E402
    ParallelMorphAMP,
    ParallelMorphVecEnv,
    balanced_group_sizes,
    describe_layout,
)


DEFAULT_VARIANTS = ["nominal", "combined", "tall_legs", "heavy_torso"]
DEFAULT_CLIPS = [
    "dance2_subject1",
    "dance2_subject2",
    "dance2_subject3",
    "dance2_subject4",
]


def _largest_divisor_at_most(value: int, limit: int) -> int:
    for candidate in range(min(value, limit), 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


def _limit_frames_per_clip(trajectory: Trajectory, max_frames: int | None):
    if max_frames is None:
        return trajectory
    data_slices = []
    for trajectory_index in range(int(trajectory.data.n_trajectories)):
        length = min(max_frames, int(trajectory.data.len_trajectory(trajectory_index)))
        if length < 2:
            raise ValueError("Every AMP reference clip needs at least two frames.")
        data_slices.append(
            TrajectoryData.dynamic_slice_in_dim(
                trajectory.data,
                trajectory_index,
                0,
                length,
                backend=jnp,
            )
        )
    data, info = TrajectoryData.concatenate(
        data_slices, [trajectory.info] * len(data_slices), backend=jnp
    )
    return Trajectory(info=info, data=data)


def _concat_expert_datasets(datasets):
    fields = {
        "observations": jnp.concatenate([data.observations for data in datasets]),
        "next_observations": jnp.concatenate(
            [data.next_observations for data in datasets]
        ),
        "absorbings": jnp.concatenate([data.absorbings for data in datasets]),
        "dones": jnp.concatenate([data.dones for data in datasets]),
    }
    return TrajectoryTransitions(**fields)


def build_parallel_amp_env(args):
    robot = get_robot("h1")
    source_env = ImitationFactory.make(
        robot.cpu_env_name,
        lafan1_dataset_conf=LAFAN1DatasetConf(list(args.clips)),
    )
    base_trajectory = _limit_frames_per_clip(source_env.th.traj, args.frames_per_clip)

    envs = []
    expert_datasets = []
    ground_offsets = {}
    build_started = time.perf_counter()
    expert_seconds = 0.0
    for index, name in enumerate(args.variants):
        xml_path = WORKSPACE / "generated_variants" / f"h1_morphology_{name}" / "h1.xml"
        if not xml_path.is_file():
            raise FileNotFoundError(
                f"Missing generated morphology {name!r}: {xml_path}"
            )
        env_name = variant_env_names(robot, name, f"amp_{args.run_tag}_{index}")[1]
        register_variant_env(env_name, xml_path, robot.base_mjx_cls)
        trajectory, offset = ground_trajectory_constant(xml_path, base_trajectory)
        env = make_mimic_env(
            env_name,
            trajectory,
            use_mjwarp=args.use_mjwarp,
            nconmax=7000,
            headless=True,
        )
        expert_started = time.perf_counter()
        expert_dataset = env.create_dataset().to_jnp()
        expert_seconds += time.perf_counter() - expert_started
        envs.append(env)
        expert_datasets.append(expert_dataset)
        ground_offsets[name] = float(offset)
        print(
            f"[amp] built {name:>22s}: trajectories={env.th.n_trajectories} "
            f"expert={expert_dataset.observations.shape[0]:,} "
            f"obs={env.info.observation_space.shape}",
            flush=True,
        )

    if args.envs_per_morph is not None:
        group_sizes = tuple([args.envs_per_morph] * len(envs))
    else:
        group_sizes = balanced_group_sizes(args.total_envs, len(envs))
    parallel_env = ParallelMorphVecEnv(
        envs, group_sizes, names=args.variants, history_length=1
    )
    expert_dataset = _concat_expert_datasets(expert_datasets)
    metadata = {
        "clips": list(args.clips),
        "num_motions": len(args.clips),
        "frames_per_clip_limit": args.frames_per_clip,
        "frames_per_clip_actual": [
            int(base_trajectory.data.len_trajectory(i))
            for i in range(int(base_trajectory.data.n_trajectories))
        ],
        "ground_offsets": ground_offsets,
        "environment_and_grounding_seconds": time.perf_counter() - build_started,
        "expert_dataset_seconds": expert_seconds,
        "expert_observations": int(expert_dataset.observations.shape[0]),
        "group_sizes": list(group_sizes),
    }
    return parallel_env, expert_dataset, metadata


def build_config(args, total_envs: int, actual_timesteps: int, expert_size: int):
    batch_size = args.num_steps * total_envs
    updates = max(1, actual_timesteps // batch_size)
    return OmegaConf.create(
        {
            "experiment": {
                "hidden_layers": list(args.hidden),
                "lr": args.lr,
                "disc_lr": args.disc_lr,
                "disc_max_grad_norm": args.disc_max_grad_norm,
                "num_envs": total_envs,
                "num_steps": args.num_steps,
                "total_timesteps": float(actual_timesteps),
                "update_epochs": args.update_epochs,
                "num_minibatches": _largest_divisor_at_most(
                    batch_size, args.num_minibatches
                ),
                "disc_minibatch_size": min(args.disc_minibatch_size, expert_size),
                "n_disc_epochs": args.n_disc_epochs,
                "proportion_env_reward": args.proportion_env_reward,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_eps": 0.2,
                "init_std": args.init_std,
                "learnable_std": bool(args.learnable_std),
                "ent_coef": 0.0,
                "disc_ent_coef": args.disc_ent_coef,
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


def _finite_or_none(value):
    value = float(value)
    return value if np.isfinite(value) else None


def main():
    args = parse_args()
    if len(set(args.variants)) != len(args.variants):
        raise ValueError("--variants must not contain duplicates.")
    env, expert_dataset, build_meta = build_parallel_amp_env(args)
    print(
        f"[amp] backend={jax.default_backend()} morphologies={env.num_morphologies} "
        f"motions={len(args.clips)} total_envs={env.num_envs} "
        f"layout={describe_layout(env.names, env.group_sizes)}",
        flush=True,
    )

    steps_per_update = args.num_steps * env.num_envs
    num_updates = max(1, int(args.total_timesteps // steps_per_update))
    actual_timesteps = num_updates * steps_per_update
    config = build_config(
        args, env.num_envs, actual_timesteps, expert_dataset.observations.shape[0]
    )
    agent_conf = ParallelMorphAMP.init_agent_conf(env, config)
    agent_conf = agent_conf.add_expert_dataset(expert_dataset)
    train_fn = jax.jit(ParallelMorphAMP.build_train_fn(env, agent_conf, mh=None))
    key = jax.random.PRNGKey(args.seed)

    print(
        f"[amp] lowering/compiling updates={num_updates} "
        f"steps/update={steps_per_update:,} total={actual_timesteps:,}",
        flush=True,
    )
    t0 = time.perf_counter()
    executable = train_fn.lower(key).compile()
    compile_seconds = time.perf_counter() - t0
    print(f"[amp] compile complete in {compile_seconds:.1f}s", flush=True)

    t0 = time.perf_counter()
    output = executable(key)
    jax.block_until_ready(output["agent_state"])
    training_seconds = time.perf_counter() - t0
    throughput = actual_timesteps / training_seconds

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoint_final"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    agent_path = ParallelMorphAMP.save_agent(
        str(checkpoint_dir), agent_conf, output["agent_state"]
    )
    metrics = output["training_metrics"]
    returns = np.asarray(metrics.mean_episode_return)
    lengths = np.asarray(metrics.mean_episode_length)
    disc_policy = np.asarray(metrics.discriminator_output_policy)
    disc_expert = np.asarray(metrics.discriminator_output_expert)
    manifest = {
        "experiment": "parallel_multibody_multimotion_amp",
        "implementation": "grouped_static_mjx_shared_amp_policy_and_discriminator",
        "robot": "h1",
        "variants": list(args.variants),
        "num_morphologies": env.num_morphologies,
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
        "disc_lr": args.disc_lr,
        "n_disc_epochs": args.n_disc_epochs,
        "proportion_env_reward": args.proportion_env_reward,
        "compile_seconds": compile_seconds,
        "training_seconds": training_seconds,
        "steps_per_second": throughput,
        "steps_per_minute": throughput * 60.0,
        "mean_episode_return_last": _finite_or_none(returns[-1]),
        "mean_episode_length_last": _finite_or_none(lengths[-1]),
        "discriminator_policy_last": _finite_or_none(disc_policy[-1]),
        "discriminator_expert_last": _finite_or_none(disc_expert[-1]),
        "return_curve": [_finite_or_none(value) for value in returns],
        "discriminator_policy_curve": [_finite_or_none(value) for value in disc_policy],
        "discriminator_expert_curve": [_finite_or_none(value) for value in disc_expert],
        "agent_path": str(agent_path),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        **build_meta,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(
        f"[amp] trained {actual_timesteps:,} steps in {training_seconds:.1f}s "
        f"({throughput / 1e6 * 60:.2f}M steps/min) -> {args.output_dir}",
        flush=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    parser.add_argument("--clips", nargs="+", default=DEFAULT_CLIPS)
    parser.add_argument(
        "--frames-per-clip",
        type=int,
        default=1000,
        help="Limit each expert motion for fast experiments; use 0 for full clips.",
    )
    parser.add_argument("--total-envs", type=int, default=2048)
    parser.add_argument("--envs-per-morph", type=int, default=None)
    parser.add_argument("--total-timesteps", type=float, default=60e6)
    parser.add_argument("--num-steps", type=int, default=64)
    parser.add_argument("--num-minibatches", type=int, default=32)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--hidden", type=int, nargs="+", default=[512, 256])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--disc-lr", type=float, default=5e-5)
    parser.add_argument("--disc-max-grad-norm", type=float, default=0.5)
    parser.add_argument("--disc-minibatch-size", type=int, default=2048)
    parser.add_argument("--n-disc-epochs", type=int, default=10)
    parser.add_argument("--disc-ent-coef", type=float, default=0.0)
    parser.add_argument("--proportion-env-reward", type=float, default=0.5)
    parser.add_argument("--init-std", type=float, default=0.2)
    parser.add_argument("--learnable-std", action="store_true")
    parser.add_argument("--no-normalize-reward", action="store_true")
    parser.add_argument("--use-mjwarp", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--run-tag", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.frames_per_clip == 0:
        args.frames_per_clip = None
    if args.output_dir is None:
        args.output_dir = WORKSPACE / "experiments" / "scaling_amp" / args.run_tag
    return args


if __name__ == "__main__":
    main()
