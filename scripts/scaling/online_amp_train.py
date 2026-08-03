"""Train one AMP policy over online H1 embodiments and several motions.

Body, motion clip and motion phase are all selected at reset inside one MJX
graph: the body by the catalog schedule, the clip and phase by LocoMuJoCo's
trajectory handler.  There is no static body x motion branch product, so the
scale ladder (2x2 -> 16x4 -> 1000x4) changes numbers, not graph shapes.
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

from loco_mujoco.task_factories import ImitationFactory, LAFAN1DatasetConf  # noqa: E402
from loco_mujoco.trajectory import (  # noqa: E402
    Trajectory,
    TrajectoryData,
    TrajectoryTransitions,
)

from morphology_deepmimic import (  # noqa: E402
    get_robot,
    ground_trajectory_constant,
    make_mimic_env,
)
from scaling.embodiment_catalog import (  # noqa: E402
    EmbodimentCatalog,
    exposure_summary,
    fixed_balanced_assignment,
)
from scaling.online_amp import OnlineMorphAMP, keep_indices_excluding  # noqa: E402
from scaling.online_h1 import MORPHOLOGY_NAMES, register_online_h1_env  # noqa: E402
from scaling.online_h1_train import ONLINE_REWARD_PARAMS, provenance  # noqa: E402

DEFAULT_CLIPS = [
    "dance2_subject1",
    "dance2_subject2",
    "dance2_subject3",
    "dance2_subject4",
]

NOMINAL_XML = WORKSPACE / "generated_variants" / "h1_morphology_nominal" / "h1.xml"


def _largest_divisor_at_most(value: int, limit: int) -> int:
    for candidate in range(min(value, limit), 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


def _limit_frames_per_clip(trajectory: Trajectory, max_frames: int | None):
    if max_frames is None:
        return trajectory
    slices = []
    for index in range(int(trajectory.data.n_trajectories)):
        length = min(max_frames, int(trajectory.data.len_trajectory(index)))
        if length < 2:
            raise ValueError("Every AMP reference clip needs at least two frames.")
        slices.append(
            TrajectoryData.dynamic_slice_in_dim(
                trajectory.data, index, 0, length, backend=jnp
            )
        )
    data, info = TrajectoryData.concatenate(
        slices, [trajectory.info] * len(slices), backend=jnp
    )
    return Trajectory(info=info, data=data)


def pad_expert_to_policy_width(expert_dataset, observation_dim: int):
    """Give expert transitions the same width as policy observations.

    ``create_dataset`` returns the base LocoMuJoCo observation; the online
    environment appends a morphology descriptor for the policy.  The padded
    columns are a *constant* - which is precisely the channel a descriptor-blind
    discriminator must ignore, and the negative control must exploit.
    """
    width = int(expert_dataset.observations.shape[-1])
    if width == observation_dim:
        return expert_dataset, 0
    if width > observation_dim:
        raise ValueError(
            f"Expert observations ({width}) are wider than policy "
            f"observations ({observation_dim})."
        )
    pad = observation_dim - width

    def _pad(values):
        return jnp.concatenate(
            [values, jnp.zeros((values.shape[0], pad), dtype=values.dtype)], axis=-1
        )

    return (
        TrajectoryTransitions(
            observations=_pad(expert_dataset.observations),
            next_observations=_pad(expert_dataset.next_observations),
            absorbings=expert_dataset.absorbings,
            dones=expert_dataset.dones,
        ),
        pad,
    )


def build_online_amp_env(args):
    catalog = EmbodimentCatalog.load(args.catalog)
    robot = get_robot("h1")
    source_env = ImitationFactory.make(
        robot.cpu_env_name,
        lafan1_dataset_conf=LAFAN1DatasetConf(list(args.clips)),
    )
    trajectory = _limit_frames_per_clip(source_env.th.traj, args.frames_per_clip)
    grounded, ground_offset = ground_trajectory_constant(args.xml, trajectory)

    env_name = f"MjxH1OnlineAMP_{args.run_tag}"
    register_online_h1_env(env_name, args.xml)
    env = make_mimic_env(
        env_name,
        grounded,
        use_mjwarp=args.use_mjwarp,
        nconmax=7000,
        headless=True,
        morphology_low=catalog.bounds_low,
        morphology_high=catalog.bounds_high,
        append_morphology_to_observation=True,
        catalog_descriptors=catalog.descriptors,
        catalog_mode=args.catalog_mode,
        catalog_stride=args.catalog_stride,
        # Site/root targets in the default mimic reward assume the reference
        # body's limb lengths, so across online morphologies they penalise a
        # body simply for being a different size.  The joint-space subset is the
        # only part of the canonical reference that stays valid, and it is what
        # the 1,000-body walk runs use.
        **({"reward_params": ONLINE_REWARD_PARAMS} if args.joint_space_reward else {}),
    )
    started = time.perf_counter()
    expert_dataset = env.create_dataset().to_jnp()
    expert_seconds = time.perf_counter() - started
    expert_dataset, padded_columns = pad_expert_to_policy_width(
        expert_dataset, int(env.info.observation_space.shape[0])
    )

    metadata = {
        "expert_padded_columns": int(padded_columns),
        "clips": list(args.clips),
        "num_motions": len(args.clips),
        "frames_per_clip_limit": args.frames_per_clip,
        "frames_per_clip_actual": [
            int(grounded.data.len_trajectory(i))
            for i in range(int(grounded.data.n_trajectories))
        ],
        "nominal_ground_offset": float(ground_offset),
        "expert_dataset_seconds": expert_seconds,
        "expert_observations": int(expert_dataset.observations.shape[0]),
        "catalog_path": str(args.catalog),
        "catalog_hash": catalog.content_hash,
        "catalog_split": catalog.split,
        "catalog_mode": args.catalog_mode,
        "num_catalog_bodies": catalog.num_bodies,
        "morphology_names": list(MORPHOLOGY_NAMES),
        "morphology_low": list(catalog.bounds_low),
        "morphology_high": list(catalog.bounds_high),
    }
    return env, catalog, expert_dataset, metadata


def build_config(args, actual_timesteps: int, expert_size: int, observation_dim: int):
    batch_size = args.num_steps * args.num_envs
    updates = max(1, actual_timesteps // batch_size)
    # The morphology descriptor is the observation suffix; blind the
    # discriminator to exactly that slice.
    exclude_start = observation_dim - len(MORPHOLOGY_NAMES)
    return OmegaConf.create(
        {
            "experiment": {
                "hidden_layers": list(args.hidden),
                "lr": args.lr,
                "disc_lr": args.disc_lr,
                "disc_max_grad_norm": args.disc_max_grad_norm,
                "num_envs": args.num_envs,
                "num_steps": args.num_steps,
                "total_timesteps": float(actual_timesteps),
                "update_epochs": args.update_epochs,
                "num_minibatches": _largest_divisor_at_most(
                    batch_size, args.num_minibatches
                ),
                "disc_minibatch_size": min(args.disc_minibatch_size, expert_size),
                "n_disc_epochs": args.n_disc_epochs,
                "proportion_env_reward": args.proportion_env_reward,
                "disc_exclude_start": exclude_start if args.blind_discriminator else 0,
                "disc_exclude_stop": (
                    observation_dim if args.blind_discriminator else 0
                ),
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
                    "num_envs": min(100, args.num_envs),
                    "num": min(10, updates),
                },
            }
        }
    )


def leakage_check(agent_conf, agent_state, observation_dim: int, seed: int) -> dict:
    """The discriminator must not respond to the descriptor channel alone.

    Two batches differ only in the morphology columns.  If the discriminator is
    correctly blind, its logits are bit-identical.
    """
    key = jax.random.PRNGKey(seed)
    base = jax.random.normal(key, (64, observation_dim))
    start = observation_dim - len(MORPHOLOGY_NAMES)
    perturbed = base.at[:, start:].set(
        jax.random.normal(jax.random.PRNGKey(seed + 1), (64, len(MORPHOLOGY_NAMES)))
    )
    variables = {
        "params": agent_state.disc_train_state.params,
        "run_stats": agent_state.disc_train_state.run_stats,
    }
    logits_base, _ = agent_conf.discriminator.apply(
        variables, base, mutable=["run_stats"]
    )
    logits_perturbed, _ = agent_conf.discriminator.apply(
        variables, perturbed, mutable=["run_stats"]
    )
    difference = float(np.max(np.abs(np.asarray(logits_base - logits_perturbed))))
    return {
        "max_logit_difference_under_descriptor_perturbation": difference,
        "discriminator_is_descriptor_blind": bool(difference == 0.0),
        "excluded_slice": [start, observation_dim],
    }


def _finite_or_none(value):
    value = float(value)
    return value if np.isfinite(value) else None


def main() -> None:
    args = parse_args()
    build_started = time.perf_counter()
    env, catalog, expert_dataset, metadata = build_online_amp_env(args)
    build_seconds = time.perf_counter() - build_started
    observation_dim = int(env.info.observation_space.shape[0])
    print(
        f"[online-amp] backend={jax.default_backend()} bodies={catalog.num_bodies} "
        f"motions={len(args.clips)} envs={args.num_envs} obs={observation_dim} "
        f"expert={expert_dataset.observations.shape[0]:,}",
        flush=True,
    )

    steps_per_update = args.num_steps * args.num_envs
    num_updates = max(1, int(args.total_timesteps // steps_per_update))
    actual_timesteps = num_updates * steps_per_update
    config = build_config(
        args, actual_timesteps, expert_dataset.observations.shape[0], observation_dim
    )
    agent_conf = OnlineMorphAMP.init_agent_conf(env, config)
    agent_conf = agent_conf.add_expert_dataset(expert_dataset)
    keep = keep_indices_excluding(
        observation_dim,
        int(config.experiment.disc_exclude_start),
        int(config.experiment.disc_exclude_stop),
    )
    print(
        f"[online-amp] discriminator sees {len(keep)}/{observation_dim} columns "
        f"(blind={bool(args.blind_discriminator)})",
        flush=True,
    )

    train_fn = jax.jit(OnlineMorphAMP.build_train_fn(env, agent_conf, mh=None))
    key = jax.random.PRNGKey(args.seed)
    print(
        f"[online-amp] lowering/compiling updates={num_updates} "
        f"steps/update={steps_per_update:,} total={actual_timesteps:,}",
        flush=True,
    )
    t0 = time.perf_counter()
    executable = train_fn.lower(key).compile()
    compile_seconds = time.perf_counter() - t0
    print(f"[online-amp] compile complete in {compile_seconds:.1f}s", flush=True)

    t0 = time.perf_counter()
    output = executable(key)
    jax.block_until_ready(output["agent_state"])
    training_seconds = time.perf_counter() - t0
    throughput = actual_timesteps / training_seconds

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoint_final"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    agent_path = OnlineMorphAMP.save_agent(
        str(checkpoint_dir), agent_conf, output["agent_state"]
    )
    metrics = output["training_metrics"]
    returns = np.asarray(metrics.mean_episode_return)
    lengths = np.asarray(metrics.mean_episode_length)
    disc_policy = np.asarray(metrics.discriminator_output_policy)
    disc_expert = np.asarray(metrics.discriminator_output_expert)

    manifest = {
        "experiment": "online_morphology_multimotion_amp",
        "implementation": "single_mjx_graph_dynamic_body_x_multi_trajectory_amp",
        "robot": "h1",
        "num_envs": args.num_envs,
        "replicas_per_body": args.num_envs / catalog.num_bodies,
        "catalog_exposure": exposure_summary(
            fixed_balanced_assignment(args.num_envs, catalog.num_bodies),
            catalog.num_bodies,
        ),
        "body_motion_cells": catalog.num_bodies * len(args.clips),
        "static_branches": 1,
        "discriminator_blind_to_descriptor": bool(args.blind_discriminator),
        "joint_space_reward_only": bool(args.joint_space_reward),
        "reward_params": (
            ONLINE_REWARD_PARAMS if args.joint_space_reward else "loco_mujoco_default"
        ),
        "discriminator_input_dim": len(keep),
        "observation_dim": observation_dim,
        "leakage_check": leakage_check(
            agent_conf, output["agent_state"], observation_dim, args.seed
        ),
        "num_steps": args.num_steps,
        "num_updates": num_updates,
        "num_minibatches_actual": int(config.experiment.num_minibatches),
        "update_epochs": args.update_epochs,
        "n_disc_epochs": args.n_disc_epochs,
        "proportion_env_reward": args.proportion_env_reward,
        "total_timesteps": int(actual_timesteps),
        "seed": args.seed,
        "hidden": list(args.hidden),
        "lr": args.lr,
        "disc_lr": args.disc_lr,
        "compile_seconds": compile_seconds,
        "training_seconds": training_seconds,
        "steps_per_minute": throughput * 60.0,
        "environment_build_seconds": build_seconds,
        "mean_episode_return_last": _finite_or_none(returns[-1]),
        "mean_episode_length_last": _finite_or_none(lengths[-1]),
        "discriminator_policy_last": _finite_or_none(disc_policy[-1]),
        "discriminator_expert_last": _finite_or_none(disc_expert[-1]),
        "return_curve": [_finite_or_none(v) for v in returns],
        "length_curve": [_finite_or_none(v) for v in lengths],
        "discriminator_policy_curve": [_finite_or_none(v) for v in disc_policy],
        "discriminator_expert_curve": [_finite_or_none(v) for v in disc_expert],
        "agent_path": str(agent_path),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(d) for d in jax.devices()],
        "provenance": provenance(),
        **metadata,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(
        f"[online-amp] trained {actual_timesteps:,} steps in {training_seconds:.1f}s "
        f"({throughput / 1e6 * 60:.2f}M steps/min) -> {args.output_dir}",
        flush=True,
    )
    print(json.dumps(manifest["leakage_check"], indent=2), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument(
        "--catalog-mode",
        choices=["fixed_balanced", "catalog_resample"],
        default="catalog_resample",
    )
    parser.add_argument("--catalog-stride", type=int, default=1)
    parser.add_argument("--clips", nargs="+", default=DEFAULT_CLIPS)
    parser.add_argument("--frames-per-clip", type=int, default=1000)
    parser.add_argument("--xml", type=Path, default=NOMINAL_XML)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--total-timesteps", type=float, default=60e6)
    parser.add_argument("--num-steps", type=int, default=64)
    parser.add_argument("--num-minibatches", type=int, default=32)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 128])
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
    parser.add_argument(
        "--full-mimic-reward",
        dest="joint_space_reward",
        action="store_false",
        help=(
            "Use the default mimic reward including site/root targets. Those "
            "assume the reference body's proportions and are not valid across "
            "online morphologies."
        ),
    )
    parser.add_argument(
        "--no-blind-discriminator",
        dest="blind_discriminator",
        action="store_false",
        help="Negative control only: let the discriminator read the descriptor.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--run-tag", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.frames_per_clip == 0:
        args.frames_per_clip = None
    if args.output_dir is None:
        args.output_dir = (
            WORKSPACE / "experiments" / "scaling_1000" / "online_amp" / args.run_tag
        )
    return args


if __name__ == "__main__":
    main()
