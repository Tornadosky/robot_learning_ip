"""Train one PPO policy on H1 embodiments sampled inside one MJX graph."""

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
from loco_mujoco.core.wrappers import LogWrapper, VecEnv  # noqa: E402
from loco_mujoco.task_factories import ImitationFactory, LAFAN1DatasetConf  # noqa: E402

from morphology_deepmimic import (  # noqa: E402
    crop_trajectory,
    get_robot,
    ground_trajectory_constant,
    make_mimic_env,
    resolve_window,
)
from scaling.catalog_vec_env import with_catalog_vec_env  # noqa: E402
from scaling.morphology_terminal import (  # noqa: E402,F401  (registers the handler)
    MorphologyAwareRootPoseTrajTerminalStateHandler,
)
from scaling.embodiment_catalog import (  # noqa: E402
    EmbodimentCatalog,
    exposure_summary,
    fixed_balanced_assignment,
)
from scaling.online_h1 import (  # noqa: E402
    MORPHOLOGY_NAMES,
    MorphologyBounds,
    register_online_h1_env,
)
from scaling.urma_networks import (  # noqa: E402
    URMA_IMPLEMENTATION_REVISION,
    URMAPPO,
)


ONLINE_REWARD_PARAMS = {
    # A single canonical joint trajectory remains valid across same-DOF bodies.
    # Site targets require morphology-aware retargeting and are deliberately not
    # scored in this first online scale layer.
    "qpos_w_sum": 0.6,
    "qvel_w_sum": 0.4,
    "rpos_w_sum": 0.0,
    "rquat_w_sum": 0.0,
    "rvel_w_sum": 0.0,
}


def _largest_divisor_at_most(value: int, limit: int) -> int:
    for candidate in range(min(value, limit), 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


def build_config(args, actual_timesteps: int):
    batch_size = args.num_steps * args.num_envs
    num_updates = actual_timesteps // batch_size
    validation_slots = min(10, num_updates)
    return OmegaConf.create(
        {
            "experiment": {
                "hidden_layers": list(args.hidden),
                "backbone": str(getattr(args, "backbone", "mlp")),
                "urma_activation": str(getattr(args, "urma_activation", "elu")),
                "urma_latent_slots": int(getattr(args, "urma_latent_slots", 64)),
                "urma_joint_value_dim": int(getattr(args, "urma_joint_value_dim", 4)),
                "lr": args.lr,
                "num_envs": args.num_envs,
                "num_steps": args.num_steps,
                "total_timesteps": float(actual_timesteps),
                "update_epochs": args.update_epochs,
                "proportion_env_reward": 0.0,
                "num_minibatches": _largest_divisor_at_most(
                    batch_size, args.num_minibatches
                ),
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
                    "num_envs": min(100, args.num_envs),
                    "num": validation_slots,
                },
            }
        }
    )


def load_training_catalog(args):
    """Return the catalog for this run, or None in continuous mode."""
    catalog_path = getattr(args, "catalog", None)
    mode = str(getattr(args, "catalog_mode", "continuous"))
    if catalog_path is None:
        if mode != "continuous":
            raise ValueError(f"catalog_mode={mode} requires --catalog.")
        return None
    if mode == "continuous":
        raise ValueError("--catalog was given but catalog_mode is continuous.")
    catalog = EmbodimentCatalog.load(Path(catalog_path))
    if catalog.split != "train" and not getattr(args, "allow_nontrain_catalog", False):
        raise ValueError(
            f"Training catalogs must carry split 'train'; got {catalog.split!r}."
        )
    return catalog


def build_online_env(args):
    catalog = load_training_catalog(args)
    if catalog is not None and not getattr(args, "keep_morph_bounds", False):
        # The descriptor the policy sees is normalised by these bounds, so the
        # catalog's own bounds must be the training bounds.
        args.morph_low = list(catalog.bounds_low)
        args.morph_high = list(catalog.bounds_high)
    bounds = MorphologyBounds(tuple(args.morph_low), tuple(args.morph_high))
    bounds.validate()
    robot = get_robot("h1")
    source_env = ImitationFactory.make(
        robot.cpu_env_name,
        lafan1_dataset_conf=LAFAN1DatasetConf([args.clip]),
    )
    full_traj = source_env.th.traj
    start_frame, n_frames = resolve_window(full_traj, args.duration, args.start_frame)
    base_traj = crop_trajectory(full_traj, start_frame, n_frames)

    xml_path = WORKSPACE / "generated_variants" / "h1_morphology_nominal" / "h1.xml"
    if not xml_path.is_file():
        raise FileNotFoundError(f"Missing nominal H1 XML: {xml_path}")
    trajectory, ground_offset = ground_trajectory_constant(xml_path, base_traj)
    env_name = f"MjxH1OnlineMorph_{args.run_tag}"
    register_online_h1_env(env_name, xml_path)
    env = make_mimic_env(
        env_name,
        trajectory,
        use_mjwarp=args.use_mjwarp,
        nconmax=7000,
        headless=True,
        morphology_low=bounds.low,
        morphology_high=bounds.high,
        append_morphology_to_observation=True,
        append_urma_joint_features=(
            str(getattr(args, "backbone", "mlp")).lower() != "mlp"
        ),
        resample_morphology_on_episode_reset=bool(
            getattr(args, "resample_per_episode", False)
        ),
        catalog_descriptors=(None if catalog is None else catalog.descriptors),
        catalog_mode=str(getattr(args, "catalog_mode", "continuous")),
        catalog_stride=int(getattr(args, "catalog_stride", 1)),
        reward_params=ONLINE_REWARD_PARAMS,
        **(
            {"terminal_state_type": args.terminal_handler}
            if getattr(args, "terminal_handler", None)
            else {}
        ),
    )
    metadata = {
        "clip": args.clip,
        "window_start_frame": int(start_frame),
        "window_frames": int(n_frames),
        "frequency_hz": float(base_traj.info.frequency),
        "nominal_ground_offset": float(ground_offset),
        "morphology_names": list(MORPHOLOGY_NAMES),
        "morphology_low": list(bounds.low),
        "morphology_high": list(bounds.high),
        "catalog_mode": str(getattr(args, "catalog_mode", "continuous")),
        "init_checkpoint": (
            None
            if getattr(args, "init_checkpoint", None) is None
            else str(args.init_checkpoint)
        ),
        "terminal_handler": getattr(args, "terminal_handler", None),
        "catalog_path": (None if catalog is None else str(args.catalog)),
        "catalog_hash": (None if catalog is None else catalog.content_hash),
        "catalog_split": (None if catalog is None else catalog.split),
        "catalog_seed": (None if catalog is None else int(catalog.seed)),
        "catalog_generator_revision": (
            None if catalog is None else catalog.generator_revision
        ),
        "catalog_sampling_method": (
            None if catalog is None else catalog.sampling_method
        ),
        "num_catalog_bodies": (0 if catalog is None else catalog.num_bodies),
    }
    return env, metadata


def catalog_assignment_check(
    env, catalog, num_envs: int, seed: int, check_envs: int = 0
) -> dict:
    """Verify that slot-assigned resets really deliver the catalog bodies.

    Exposure over the full environment count is exact by construction
    (``slot % num_bodies``) and is always reported analytically.

    The optional device-side check runs the same vmapped reset the trainer uses,
    on a bounded prefix of the slots.  It is **off by default** because a
    standalone ``jit(vmap(mjx_reset_with_slot))`` graph proved unstable on the
    measured ROCm stack: job 10803152 aborted inside it with
    ``HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION``, and jobs 10806775-77 hung in
    it at 99% CPU for 1h50m after compiling successfully. The identical reset
    runs correctly *inside* the training graph, which is where it matters, and
    the assignment itself is covered by ``tests/test_online_h1.py``.
    """
    full_assignment = fixed_balanced_assignment(num_envs, catalog.num_bodies)
    if check_envs <= 0:
        summary = exposure_summary(full_assignment, catalog.num_bodies)
        summary.update(
            {
                "exposure_source": "analytic_fixed_balanced_schedule",
                "num_slots_checked_on_device": 0,
                "device_check_skipped_reason": (
                    "standalone slot-reset graph is unstable on this ROCm stack; "
                    "the in-graph reset is exercised by training itself"
                ),
                "replicas_per_body": num_envs / catalog.num_bodies,
            }
        )
        return summary

    checked = int(min(num_envs, check_envs))
    keys = jax.random.split(jax.random.PRNGKey(seed), checked)
    slots = jnp.arange(checked, dtype=jnp.int32)
    reset_fn = jax.jit(jax.vmap(env.mjx_reset_with_slot, in_axes=(0, 0)))
    started = time.perf_counter()
    state = reset_fn(keys, slots)
    jax.block_until_ready(state.observation)
    reset_seconds = time.perf_counter() - started

    body_index = np.asarray(state.additional_carry.body_index)
    morphology = np.asarray(state.additional_carry.morphology)
    expected_index = full_assignment[:checked]
    expected_morphology = catalog.descriptors[expected_index].astype(np.float32)
    observation = np.asarray(state.observation)
    descriptor_channel = observation[:, -len(MORPHOLOGY_NAMES) :]
    expected_channel = np.asarray(
        jax.vmap(env._normalized_morphology)(jnp.asarray(morphology))
    )
    summary = exposure_summary(full_assignment, catalog.num_bodies)
    summary.update(
        {
            "exposure_source": "analytic_fixed_balanced_schedule",
            "num_slots_checked_on_device": checked,
            "assignment_matches_schedule": bool(
                np.array_equal(body_index, expected_index)
            ),
            "descriptors_match_catalog": bool(
                np.allclose(morphology, expected_morphology, atol=1e-6)
            ),
            "observation_descriptor_consistent": bool(
                np.allclose(descriptor_channel, expected_channel, atol=1e-5)
            ),
            "finite_observations": bool(np.isfinite(observation).all()),
            "slot_reset_compile_and_run_seconds": reset_seconds,
            "replicas_per_body": num_envs / catalog.num_bodies,
        }
    )
    if not summary["assignment_matches_schedule"]:
        raise AssertionError("Slot assignment did not follow the balanced schedule.")
    if not summary["descriptors_match_catalog"]:
        raise AssertionError("Assigned descriptors do not match the catalog.")
    if not summary["observation_descriptor_consistent"]:
        raise AssertionError("Reset observation does not describe the assigned body.")
    return summary


def sample_descriptors(env, num_envs: int, seed: int):
    keys = jax.random.split(jax.random.PRNGKey(seed), num_envs)
    sample_fn = jax.jit(
        jax.vmap(lambda key: env.mjx_reset(key).additional_carry.morphology)
    )
    started = time.perf_counter()
    morphology_device = sample_fn(keys)
    jax.block_until_ready(morphology_device)
    compile_and_run_seconds = time.perf_counter() - started
    morphology = np.asarray(morphology_device)

    # A second call separates the cheap steady-state online generation/reset
    # cost from XLA compilation.  This is the relevant number when bodies are
    # resampled over a long run or curriculum.
    steady_keys = jax.random.split(jax.random.PRNGKey(seed + 1), num_envs)
    started = time.perf_counter()
    steady_morphology = sample_fn(steady_keys)
    jax.block_until_ready(steady_morphology)
    steady_seconds = time.perf_counter() - started
    rounded_unique = np.unique(np.round(morphology, decimals=6), axis=0)
    return morphology, {
        "num_sampled": int(len(morphology)),
        "num_unique_at_1e_6": int(len(rounded_unique)),
        "sample_reset_compile_and_run_seconds": compile_and_run_seconds,
        "sample_reset_steady_seconds": steady_seconds,
        "steady_morphologies_per_second": num_envs / steady_seconds,
        "sample_mean": morphology.mean(axis=0).tolist(),
        "sample_min": morphology.min(axis=0).tolist(),
        "sample_max": morphology.max(axis=0).tolist(),
        "first_samples": morphology[: min(16, len(morphology))].tolist(),
    }


def run_preflight(env, args):
    wrapped = VecEnv(LogWrapper(env))
    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.num_envs)
    reset = jax.jit(wrapped.reset)
    t0 = time.perf_counter()
    obs, state = reset(keys)
    jax.block_until_ready(obs)
    reset_seconds = time.perf_counter() - t0

    action = jnp.zeros((args.num_envs, env.info.action_space.shape[0]))
    step = jax.jit(wrapped.step)
    t0 = time.perf_counter()
    next_obs, reward, _, _, _, state = step(state, action)
    jax.block_until_ready(next_obs)
    step_seconds = time.perf_counter() - t0
    morphology = np.asarray(state.env_state.additional_carry.morphology)
    expected_descriptor = np.asarray(
        jax.vmap(env._normalized_morphology)(jnp.asarray(morphology))
    )
    if env.urma_input_layout is None:
        descriptor_observation = np.asarray(obs)[:, -len(MORPHOLOGY_NAMES) :]
    else:
        start = env.urma_input_layout.morphology_start
        descriptor_observation = np.asarray(obs)[
            :, start : start + len(MORPHOLOGY_NAMES)
        ]
    result = {
        "obs_shape": list(obs.shape),
        "reset_compile_and_run_seconds": reset_seconds,
        "step_compile_and_run_seconds": step_seconds,
        "finite_observations": bool(np.isfinite(np.asarray(next_obs)).all()),
        "finite_rewards": bool(np.isfinite(np.asarray(reward)).all()),
        "num_unique_at_1e_6": int(len(np.unique(np.round(morphology, 6), axis=0))),
        "descriptor_observation_consistent": bool(
            np.allclose(descriptor_observation, expected_descriptor)
        ),
        "urma_joint_features": env.urma_input_layout is not None,
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


def provenance() -> dict:
    """Git/Slurm/command fingerprint required by the experiment artifact contract."""
    import os
    import subprocess

    def _git(*command):
        try:
            return subprocess.run(
                ["git", *command],
                cwd=WORKSPACE,
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            ).stdout.strip()
        except Exception:
            return None

    import hashlib

    # The Viper working copy is an rsync of the tree, not a git checkout, so a
    # source fingerprint is the only fingerprint available there.
    source_hashes = {}
    for name in (
        "online_h1.py",
        "online_h1_train.py",
        "embodiment_catalog.py",
        "catalog_vec_env.py",
    ):
        path = Path(__file__).with_name(name)
        if path.is_file():
            source_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()

    status = _git("status", "--porcelain")
    return {
        "source_sha256": source_hashes,
        "git_revision": _git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "git_dirty_file_count": None if status is None else len(status.splitlines()),
        "command": " ".join(sys.argv),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_node": os.environ.get("SLURMD_NODENAME"),
        "hostname": os.environ.get("HOSTNAME") or os.environ.get("COMPUTERNAME"),
    }


def _parameter_count(agent_state):
    leaves = jax.tree_util.tree_leaves(agent_state.train_state.params)
    return int(sum(np.size(np.asarray(leaf)) for leaf in leaves))


def main():
    args = parse_args()
    build_started = time.perf_counter()
    env, metadata = build_online_env(args)
    build_seconds = time.perf_counter() - build_started
    print(
        f"[online] backend={jax.default_backend()} envs={args.num_envs} "
        f"obs={env.info.observation_space.shape} act={env.info.action_space.shape}",
        flush=True,
    )
    catalog = load_training_catalog(args)
    if catalog is None:
        _, descriptor_stats = sample_descriptors(env, args.num_envs, args.seed)
        print(
            f"[online] sampled {descriptor_stats['num_unique_at_1e_6']:,}/"
            f"{args.num_envs:,} unique embodiments",
            flush=True,
        )
        catalog_stats = None
    else:
        descriptor_stats = None
        catalog_stats = catalog_assignment_check(
            env, catalog, args.num_envs, args.seed, check_envs=args.catalog_check_envs
        )
        print(
            f"[online] catalog {catalog.num_bodies:,} bodies "
            f"({catalog.content_hash[:12]}) exposure "
            f"{catalog_stats['min_exposure']}-{catalog_stats['max_exposure']} "
            f"per body across {args.num_envs:,} slots",
            flush=True,
        )

    if args.preflight_only:
        run_preflight(env, args)
        return

    steps_per_update = args.num_steps * args.num_envs
    num_updates = max(1, int(args.total_timesteps // steps_per_update))
    actual_timesteps = num_updates * steps_per_update
    config = build_config(args, actual_timesteps)
    base_algorithm = PPOJax if args.backbone == "mlp" else URMAPPO
    # Only the environment wrapper differs, so checkpoints stay loadable by the
    # unmodified class and keep the unmodified class's file name.
    algorithm = (
        base_algorithm if catalog is None else with_catalog_vec_env(base_algorithm)
    )
    agent_conf = algorithm.init_agent_conf(env, config)
    key = jax.random.PRNGKey(args.seed)
    init_state = None
    if getattr(args, "init_checkpoint", None) is not None:
        # Warm start, e.g. a morphology-bounds curriculum: each stage continues
        # the previous stage's weights on a wider catalog.  Only the parameters
        # carry over; the optimiser and environment are rebuilt for this stage.
        _, init_state = base_algorithm.load_agent(args.init_checkpoint)
        print(f"[online] warm start from {args.init_checkpoint}", flush=True)
        resume_fn = algorithm.build_resume_train_fn(env, agent_conf, mh=None)
        # Close over the loaded state so the traced function keeps the same
        # one-argument signature as the from-scratch path; jitting the two-arg
        # form instead produces an executable that main() cannot call.
        train_fn = jax.jit(lambda rng_key: resume_fn(rng_key, init_state))
    else:
        train_fn = jax.jit(algorithm.build_train_fn(env, agent_conf, mh=None))

    print(
        f"[online] lowering/compiling updates={num_updates} "
        f"steps/update={steps_per_update:,} total={actual_timesteps:,}",
        flush=True,
    )
    t0 = time.perf_counter()
    executable = train_fn.lower(key).compile()
    compile_seconds = time.perf_counter() - t0
    print(f"[online] compile complete in {compile_seconds:.1f}s", flush=True)

    t0 = time.perf_counter()
    output = executable(key)
    jax.block_until_ready(output["agent_state"])
    training_seconds = time.perf_counter() - t0
    throughput = actual_timesteps / training_seconds

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoint_final"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    agent_path = base_algorithm.save_agent(
        str(checkpoint_dir), agent_conf, output["agent_state"]
    )
    returns = np.asarray(output["training_metrics"].mean_episode_return)
    lengths = np.asarray(output["training_metrics"].mean_episode_length)
    valid_lengths = lengths[np.isfinite(lengths) & (lengths > 0)]
    body_changes_on_reset = (
        args.catalog_mode == "catalog_resample"
        if catalog is not None
        else bool(args.resample_per_episode)
    )
    if body_changes_on_reset and len(valid_lengths):
        estimated_morphology_exposures = int(
            args.num_envs + np.sum(steps_per_update / valid_lengths)
        )
    else:
        estimated_morphology_exposures = int(args.num_envs)
    manifest = {
        "experiment": "online_h1_morphology",
        "implementation": "single_mjx_graph_dynamic_per_environment_model_arrays",
        "backbone": args.backbone,
        "embodiment_backbone_revision": (
            None if args.backbone == "mlp" else URMA_IMPLEMENTATION_REVISION
        ),
        "policy_parameter_count": _parameter_count(output["agent_state"]),
        "urma_layout": (
            None
            if env.urma_input_layout is None
            else {
                "num_joints": env.urma_input_layout.num_joints,
                "joint_description_dim": env.urma_input_layout.joint_description_dim,
                "joint_state_dim": env.urma_input_layout.joint_state_dim,
                "joint_feature_dim": env.urma_input_layout.joint_feature_dim,
                "general_observation_dim": len(env.urma_input_layout.general_indices),
            }
        ),
        "robot": "h1",
        "num_envs": args.num_envs,
        "num_online_morphologies": (
            args.num_envs if catalog is None else catalog.num_bodies
        ),
        "resample_morphology_per_episode": bool(args.resample_per_episode),
        "capacity_only": bool(args.capacity_only),
        "replicas_per_body": (
            None if catalog is None else args.num_envs / catalog.num_bodies
        ),
        "catalog_exposure": catalog_stats,
        "estimated_total_morphology_exposures": estimated_morphology_exposures,
        "morphology_exposure_estimator": (
            "initial_envs_plus_sum_steps_per_update_over_mean_completed_episode_length"
            if body_changes_on_reset
            else "one_persistent_sample_per_environment"
        ),
        "num_steps": args.num_steps,
        "num_updates": num_updates,
        "num_minibatches_requested": args.num_minibatches,
        "num_minibatches_actual": int(config.experiment.num_minibatches),
        "update_epochs": args.update_epochs,
        "gradient_minibatch_updates": int(
            num_updates * args.update_epochs * config.experiment.num_minibatches
        ),
        "total_timesteps": int(actual_timesteps),
        "seed": args.seed,
        "hidden": list(args.hidden),
        "lr": args.lr,
        "init_std": args.init_std,
        "learnable_std": bool(args.learnable_std),
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
        "environment_build_seconds": build_seconds,
        "reference_mode": "canonical_same_dof_joint_space",
        "collision_mesh_mode": "nominal_shared_mesh",
        "descriptor_stats": descriptor_stats,
        "provenance": provenance(),
        **metadata,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(
        f"[online] trained {actual_timesteps:,} steps in {training_seconds:.1f}s "
        f"({throughput / 1e6 * 60:.2f}M steps/min) -> {args.output_dir}",
        flush=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", default="walk1_subject1")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--total-timesteps", type=float, default=60e6)
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--num-minibatches", type=int, default=32)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--hidden", type=int, nargs="+", default=[512, 256])
    parser.add_argument("--backbone", choices=["mlp", "urma", "urmav2"], default="mlp")
    parser.add_argument("--urma-activation", default="elu")
    parser.add_argument("--urma-latent-slots", type=int, default=64)
    parser.add_argument("--urma-joint-value-dim", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--init-std", type=float, default=0.2)
    parser.add_argument("--learnable-std", action="store_true")
    parser.add_argument(
        "--resample-per-episode",
        action="store_true",
        help="Sample a new embodiment on every asynchronous episode reset.",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=None,
        help="Path to a deterministic training catalog JSON.",
    )
    parser.add_argument(
        "--catalog-mode",
        choices=["continuous", "fixed_balanced", "catalog_resample"],
        default="continuous",
    )
    parser.add_argument("--catalog-stride", type=int, default=1)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Warm-start weights from a checkpoint (morphology curriculum).",
    )
    parser.add_argument(
        "--terminal-handler",
        default=None,
        help=(
            "Terminal-state handler override. Wide morphology bounds need "
            "MorphologyAwareRootPoseTrajTerminalStateHandler, or tall bodies are "
            "declared absorbing at reset for standing higher than the reference."
        ),
    )
    parser.add_argument(
        "--catalog-check-envs",
        type=int,
        default=0,
        help=(
            "Environment slots verified on device before training starts. "
            "0 skips the device check; exposure is still reported analytically."
        ),
    )
    parser.add_argument(
        "--keep-morph-bounds",
        action="store_true",
        help="Do not adopt the catalog's bounds for descriptor normalisation.",
    )
    parser.add_argument(
        "--capacity-only",
        action="store_true",
        help="Label the manifest as a throughput/memory result, not a learning result.",
    )
    parser.add_argument("--no-normalize-reward", action="store_true")
    parser.add_argument("--use-mjwarp", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--morph-low", type=float, nargs=4, default=list(MorphologyBounds().low)
    )
    parser.add_argument(
        "--morph-high", type=float, nargs=4, default=list(MorphologyBounds().high)
    )
    parser.add_argument("--run-tag", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = WORKSPACE / "experiments" / "scaling_online" / args.run_tag
    return args


if __name__ == "__main__":
    main()
