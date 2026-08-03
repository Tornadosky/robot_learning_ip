"""Train a DeepMimic PPO policy on ONE morphology cell, saving a checkpoint timeline.

This is the physics-based counterpart of the kinematic replays: the policy must
balance and actuate a *modified* body (e.g. legs 1.55x, torso 0.55x) to track the
retargeted dance reference produced by retarget_dance_to_variant.py.

Unlike the stock train_deepmimic_dance.py (one-shot, final agent only), this
trains in K equal segments and saves a checkpoint after each, so the learning
progression can be rendered later (early flailing -> balanced tracking). Segments
resume via PPOJax.build_resume_train_fn; the resume fn is jitted once and reused,
so checkpointing adds no extra compile cost. (Caveat: the Adam optimizer state is
re-initialised at each segment boundary -- a minor, self-correcting perturbation.)

GPU JAX only (e.g. WSL2). MjWarp backend is broken with mujoco 3.9.0; default MJX.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import numpy as np
from omegaconf import OmegaConf

from loco_mujoco.algorithms import PPOJax
from loco_mujoco.trajectory import Trajectory

from morphology_deepmimic import (
    apply_group_gain_scales,
    cell_dir,
    control_config,
    crop_trajectory,
    get_robot,
    ground_trajectory_constant,
    make_mimic_env,
    prepare_variant,
    reference_path,
    resolve_window,
)
from retarget_dance_to_variant import retarget_cell

NUM_STEPS_PER_UPDATE = 200  # rollout length per PPO update (matches the baseline)


def build_config(args: argparse.Namespace, total_timesteps: float) -> OmegaConf:
    return OmegaConf.create(
        {
            "experiment": {
                "hidden_layers": list(args.hidden_layers),
                "lr": args.lr,
                "num_envs": args.num_envs,
                "num_steps": NUM_STEPS_PER_UPDATE,
                "total_timesteps": float(total_timesteps),
                "update_epochs": 4,
                "proportion_env_reward": 0.0,
                "num_minibatches": 32,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_eps": 0.2,
                "init_std": args.init_std,
                "learnable_std": args.learnable_std,
                "ent_coef": 0.0,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
                "activation": "tanh",
                "anneal_lr": False,
                "weight_decay": 0.0,
                "normalize_env": True,
                "debug": False,
                "n_seeds": 1,
                "vmap_across_seeds": True,
                "validation": {"active": False, "num_steps": 100, "num_envs": 100, "num": 10},
            }
        }
    )


def resolve_reference(robot, args, variant) -> tuple[Trajectory, int, int, float, str]:
    """Return the training reference trajectory + window metadata.

    Two reference modes:
      raw  : the clean cropped LAFAN1 clip (baseline-identical), grounded onto the
             variant body by a single constant root-z offset (no jitter, no SMPL).
      else : the SMPL-retargeted reference for this variant (built on demand).
    """
    from loco_mujoco.task_factories import ImitationFactory, LAFAN1DatasetConf

    source_env = ImitationFactory.make(
        robot.cpu_env_name, lafan1_dataset_conf=LAFAN1DatasetConf([args.clip])
    )
    full_traj = source_env.th.traj
    frequency = float(full_traj.info.frequency)
    start, n_frames = resolve_window(full_traj, args.duration, args.start_frame)

    if args.raw_reference:
        traj = crop_trajectory(full_traj, start, n_frames)
        traj, offset = ground_trajectory_constant(variant["xml_path"], traj)
        return traj, start, n_frames, frequency, f"raw+ground({offset:+.3f}m)"

    ref_path = reference_path(robot.key, args.clip, args.preset, start, n_frames)
    if not ref_path.exists():
        if args.no_retarget:
            raise FileNotFoundError(
                f"Reference {ref_path} missing and --no-retarget set. Run retarget first."
            )
        print(f"[train] reference missing; retargeting {robot.key}/{args.clip}/{args.preset} ...")
        retarget_cell(robot.key, args.preset, args.clip, duration=args.duration,
                      start_frame=args.start_frame, cache_tag=args.cache_tag)
    return Trajectory.load(str(ref_path)), start, n_frames, frequency, ref_path.name


def main() -> None:
    args = parse_args()
    robot = get_robot(args.robot)

    variant = prepare_variant(robot, args.preset, args.cache_tag)
    traj, start_frame, n_frames, frequency, ref_desc = resolve_reference(robot, args, variant)
    print(f"[train] {robot.key}/{args.clip}/{args.preset}: ref={ref_desc} "
          f"({int(traj.data.n_samples)} samples @ {frequency:.0f}Hz)")

    ctrl_params = control_config(robot.key, args.control)
    group_scales = {"hip": args.hip_gain_scale, "knee": args.knee_gain_scale,
                    "ankle": args.ankle_gain_scale, "arm": args.arm_gain_scale}
    if args.control == "pd":
        cp = ctrl_params["control_params"]
        if args.pd_gain_scale != 1.0:
            cp["p_gain"] = [g * args.pd_gain_scale for g in cp["p_gain"]]
            cp["d_gain"] = [g * args.pd_gain_scale for g in cp["d_gain"]]
        # Then optional per-joint-group stiffening (e.g. stiffer ankles only).
        apply_group_gain_scales(cp, group_scales)
    env = make_mimic_env(
        variant["mjx_env_name"], traj,
        use_mjwarp=args.use_mjwarp, nconmax=7000, headless=True,
        **ctrl_params,
    )
    print(f"[train] control={args.control} pd_gain_scale={args.pd_gain_scale} "
          f"lr={args.lr} init_std={args.init_std}")

    steps_per_update = NUM_STEPS_PER_UPDATE * args.num_envs
    total_updates = max(args.num_checkpoints, int(args.total_timesteps // steps_per_update))
    updates_per_segment = max(1, total_updates // args.num_checkpoints)
    seg_timesteps = updates_per_segment * steps_per_update

    config = build_config(args, seg_timesteps)
    agent_conf = PPOJax.init_agent_conf(env, config)
    train_fn = jax.jit(PPOJax.build_train_fn(env, agent_conf, mh=None))
    resume_fn = jax.jit(PPOJax.build_resume_train_fn(env, agent_conf, mh=None))

    out_dir = cell_dir(robot.key, args.clip, args.preset)
    if args.out_suffix:
        # Route experiment runs to a sibling dir so the baseline cell is never clobbered.
        out_dir = out_dir.parent / f"{args.preset}__{args.out_suffix}"
    ckpt_root = out_dir / "checkpoints"
    ckpt_root.mkdir(parents=True, exist_ok=True)
    # Save the exact reference used so rendering rolls out against the same target.
    ref_save_path = out_dir / "reference.npz"
    traj.save(str(ref_save_path))

    print(f"[train] backend={jax.default_backend()} envs={args.num_envs} "
          f"checkpoints={args.num_checkpoints} x {updates_per_segment} updates "
          f"(~{seg_timesteps:,} steps/segment)")

    rng = jax.random.PRNGKey(args.seed)
    agent_state = None
    if args.resume_from:
        # Continue a known-good checkpoint deterministically (the balance transition
        # is bifurcation-sensitive, so a crossed policy is worth extending directly
        # rather than re-rolling a fresh seed that may not cross).
        _, agent_state = PPOJax.load_agent(args.resume_from)
        print(f"[train] resumed agent from {args.resume_from} (+{args.resume_steps:,} prior steps)")
    return_curve: list[float] = []
    checkpoints = []
    t_start = time.time()
    for k in range(args.num_checkpoints):
        rng, seg_rng = jax.random.split(rng)
        if agent_state is None:
            out = train_fn(seg_rng)
        else:
            out = resume_fn(seg_rng, agent_state)
        jax.block_until_ready(out["agent_state"])
        agent_state = out["agent_state"]

        returns = np.asarray(out["training_metrics"].mean_episode_return)
        lengths = np.asarray(out["training_metrics"].mean_episode_length)
        return_curve.extend(float(r) for r in returns)
        cumulative_steps = args.resume_steps + (k + 1) * seg_timesteps

        ckpt_dir = ckpt_root / f"ckpt_{k:02d}_{cumulative_steps}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        agent_path = PPOJax.save_agent(str(ckpt_dir), agent_conf, agent_state)
        checkpoints.append({
            "index": k,
            "cumulative_steps": int(cumulative_steps),
            "agent_path": str(agent_path),
            "mean_episode_return": float(returns[-1]),
            "mean_episode_length": float(lengths[-1]),
        })
        print(f"[train] checkpoint {k + 1}/{args.num_checkpoints} "
              f"@ {cumulative_steps:,} steps: return={returns[-1]:.1f} "
              f"length={lengths[-1]:.1f}")
        if (
            args.target_return is not None
            and k + 1 >= args.min_checkpoints
            and float(returns[-1]) >= args.target_return
        ):
            print(
                f"[train] target return {args.target_return:.1f} reached; "
                "stopping this cell early"
            )
            break

    elapsed = time.time() - t_start
    manifest = {
        "robot": robot.mjx_env_name,
        "robot_key": robot.key,
        "preset": args.preset,
        "clip": args.clip,
        "reference": ref_desc,
        "raw_reference": bool(args.raw_reference),
        "control": args.control,
        "pd_gain_scale": args.pd_gain_scale,
        "group_gain_scales": group_scales,
        # Fully-resolved PD gains actually used, so rendering can replay exactly
        # (covers uniform + per-group scaling without re-deriving the recipe).
        "control_params_resolved": (
            {k: list(v) for k, v in ctrl_params["control_params"].items()}
            if args.control == "pd" else None
        ),
        "lr": args.lr,
        "init_std": args.init_std,
        "hidden_layers": list(args.hidden_layers),
        "out_suffix": args.out_suffix,
        "reference_path": str(ref_save_path),
        "window_start_frame": int(start_frame),
        "window_frames": int(n_frames),
        "frequency_hz": frequency,
        "num_envs": args.num_envs,
        "num_checkpoints": len(checkpoints),
        "planned_num_checkpoints": args.num_checkpoints,
        "updates_per_segment": int(updates_per_segment),
        "training_timesteps": int(len(checkpoints) * seg_timesteps),
        "total_timesteps": int(checkpoints[-1]["cumulative_steps"]),
        "resume_steps": int(args.resume_steps),
        "target_return": args.target_return,
        "target_reached": bool(
            args.target_return is not None
            and max(c["mean_episode_return"] for c in checkpoints) >= args.target_return
        ),
        "training_minutes": elapsed / 60.0,
        "mean_episode_return_first": float(return_curve[0]),
        "mean_episode_return_last": float(return_curve[-1]),
        "mean_episode_length_last": float(checkpoints[-1]["mean_episode_length"]),
        "checkpoints": checkpoints,
        "return_curve_every_update": [
            float(r) for r in np.asarray(return_curve)[:: max(1, len(return_curve) // 200)]
        ],
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[train] done in {elapsed / 60:.1f} min -> {manifest_path}")
    print(json.dumps({k: v for k, v in manifest.items()
                      if k not in ("checkpoints", "return_curve_every_update")}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", required=True, choices=["h1", "g1"])
    parser.add_argument("--preset", required=True)
    parser.add_argument("--clip", default="dance2_subject4")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--cache-tag", default="dance")
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--total-timesteps", type=float, default=300e6)
    parser.add_argument("--num-checkpoints", type=int, default=6,
                        help="Number of equal training segments / saved checkpoints.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--control", choices=["torque", "pd"], default="torque",
                        help="torque = DefaultControl; pd = PD position control (G1 gains).")
    parser.add_argument("--lr", type=float, default=1e-4, help="PPO learning rate.")
    parser.add_argument("--hidden-layers", type=int, nargs="+", default=[512, 256],
                        help="Actor/critic MLP hidden layer sizes (default [512 256]).")
    parser.add_argument("--init-std", type=float, default=0.2,
                        help="Initial policy action std (exploration).")
    parser.add_argument("--learnable-std", action="store_true",
                        help="Let the policy learn its action std (default fixed).")
    parser.add_argument("--pd-gain-scale", type=float, default=1.0,
                        help="Uniformly scale the PD p/d gains (pd control only).")
    parser.add_argument("--hip-gain-scale", type=float, default=1.0,
                        help="Extra per-group PD scale for the hip actuators (pd only).")
    parser.add_argument("--knee-gain-scale", type=float, default=1.0,
                        help="Extra per-group PD scale for the knee actuators (pd only).")
    parser.add_argument("--ankle-gain-scale", type=float, default=1.0,
                        help="Extra per-group PD scale for the ankle actuators (pd only).")
    parser.add_argument("--arm-gain-scale", type=float, default=1.0,
                        help="Extra per-group PD scale for the arm actuators (pd only).")
    parser.add_argument("--out-suffix", default="",
                        help="Route outputs to <preset>__<suffix> instead of clobbering the baseline cell.")
    parser.add_argument("--resume-from", default="",
                        help="Path to a saved PPOJax_saved.pkl to continue training from.")
    parser.add_argument("--resume-steps", type=int, default=0,
                        help="Step count already trained before --resume-from (for labels/plots).")
    parser.add_argument("--target-return", type=float, default=None,
                        help="Stop after a checkpoint reaches this mean episode return.")
    parser.add_argument("--min-checkpoints", type=int, default=1,
                        help="Minimum checkpoints to train before target-return early stopping.")
    parser.add_argument("--use-mjwarp", action="store_true")
    parser.add_argument("--raw-reference", action="store_true",
                        help="Train on the clean cropped LAFAN1 clip (constant-grounded onto "
                             "the variant body) instead of the SMPL-retargeted reference.")
    parser.add_argument("--no-retarget", action="store_true",
                        help="Fail instead of retargeting if the reference is missing.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
