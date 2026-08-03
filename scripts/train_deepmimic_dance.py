"""Train a DeepMimic-style PPO policy (loco-mujoco PPOJax + MimicReward) to dance.

Trains on a highlight window of a LAFAN1 clip. This is the physics-based
counterpart of the kinematic replays: the policy has to actually balance and
actuate the robot to track the mocap reference (the lower level of Disney's
ReActor bilevel formulation; ReActor itself has no public code).

Meant to run on a CUDA-capable JAX install (e.g. WSL2). Example:
    python train_deepmimic_dance.py --output-dir /mnt/c/.../external_data/deepmimic_dance
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from omegaconf import OmegaConf

import loco_mujoco
from loco_mujoco.algorithms import PPOJax
from loco_mujoco.task_factories import CustomDatasetConf, ImitationFactory
from loco_mujoco.trajectory import Trajectory, TrajectoryData


def pick_highlight_window_frames(traj: Trajectory, duration_seconds: float) -> tuple[int, int]:
    """Return (start_frame, n_frames) of the most dynamic window (max joint-velocity energy)."""
    frequency = float(traj.info.frequency)
    qvel = np.asarray(traj.data.qvel)
    energy = np.linalg.norm(qvel[:, 6:], axis=1)
    window = int(round(duration_seconds * frequency))
    if window >= len(energy):
        return 0, len(energy)
    cumulative = np.concatenate(([0.0], np.cumsum(energy)))
    sums = cumulative[window:] - cumulative[:-window]
    return int(np.argmax(sums)), window


def crop_trajectory(traj: Trajectory, start_frame: int, n_frames: int) -> Trajectory:
    data = TrajectoryData.dynamic_slice_in_dim(traj.data, 0, start_frame, n_frames, backend=jnp)
    return Trajectory(info=traj.info, data=data)


def load_cached_lafan1(env_name: str, clip: str) -> Trajectory:
    with open(loco_mujoco.PATH_TO_VARIABLES, "r") as file:
        variables = yaml.load(file, Loader=yaml.FullLoader)
    cache_root = Path(variables["LOCOMUJOCO_CONVERTED_LAFAN1_PATH"])
    cache_file = cache_root / env_name.replace("Mjx", "") / f"{clip}.npz"
    if not cache_file.exists():
        raise FileNotFoundError(
            f"{cache_file} not found. Load the clip once via LAFAN1DatasetConf to build the cache."
        )
    return Trajectory.load(str(cache_file))


def build_config(args: argparse.Namespace) -> OmegaConf:
    return OmegaConf.create(
        {
            "experiment": {
                "hidden_layers": [512, 256],
                "lr": 1e-4,
                "num_envs": args.num_envs,
                "num_steps": 200,
                "total_timesteps": float(args.total_timesteps),
                "update_epochs": 4,
                "proportion_env_reward": 0.0,
                "num_minibatches": 32,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_eps": 0.2,
                "init_std": 0.2,
                "learnable_std": False,
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
                "validation": {
                    "active": False,
                    "num_steps": 100,
                    "num_envs": 100,
                    "num": 10,
                },
            }
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", default="MjxUnitreeH1")
    parser.add_argument("--clip", default="dance2_subject4")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--start-frame", type=int, default=None,
                        help="Window start frame at trajectory frequency (default: auto by energy).")
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--total-timesteps", type=float, default=300e6)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--use-mjwarp", action="store_true",
                        help="Use the MjWarp backend (broken with mujoco 3.9.0 + mujoco-warp 3.9.0.1; default MJX).")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    full_traj = load_cached_lafan1(args.robot, args.clip)
    frequency = float(full_traj.info.frequency)
    if args.start_frame is None:
        start_frame, n_frames = pick_highlight_window_frames(full_traj, args.duration)
    else:
        start_frame = args.start_frame
        n_frames = min(int(round(args.duration * frequency)),
                       int(full_traj.data.n_samples) - start_frame)
    print(f"Training window: frames [{start_frame}, {start_frame + n_frames}) @ {frequency:.0f} Hz")
    traj = crop_trajectory(full_traj, start_frame, n_frames)

    env = ImitationFactory.make(
        args.robot,
        custom_dataset_conf=CustomDatasetConf(traj),
        use_mjwarp=args.use_mjwarp,
        nconmax=7000,
        headless=True,
        horizon=1000,
        goal_type="GoalTrajMimic",
        goal_params=dict(visualize_goal=False),
        reward_type="MimicReward",
        reward_params=dict(
            qpos_w_sum=0.4,
            qvel_w_sum=0.2,
            rpos_w_sum=0.5,
            rquat_w_sum=0.3,
            rvel_w_sum=0.1,
            sites_for_mimic=[
                "upper_body_mimic",
                "left_hand_mimic",
                "left_foot_mimic",
                "right_hand_mimic",
                "right_foot_mimic",
            ],
        ),
    )

    config = build_config(args)
    agent_conf = PPOJax.init_agent_conf(env, config)
    train_fn = jax.jit(PPOJax.build_train_fn(env, agent_conf, mh=None))

    rng = jax.random.PRNGKey(args.seed)
    print(f"Starting training: {args.total_timesteps:.0f} steps, {args.num_envs} envs, "
          f"backend={jax.default_backend()}")
    t_start = time.time()
    out = train_fn(rng)
    jax.block_until_ready(out["agent_state"])
    elapsed = time.time() - t_start
    print(f"Training finished in {elapsed / 60:.1f} min")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_path = PPOJax.save_agent(str(args.output_dir), agent_conf, out["agent_state"])
    print(f"Agent saved to: {save_path}")

    metrics = out["training_metrics"]
    returns = np.asarray(metrics.mean_episode_return)
    lengths = np.asarray(metrics.mean_episode_length)
    summary = {
        "robot": args.robot,
        "clip": args.clip,
        "window_start_frame": int(start_frame),
        "window_frames": int(n_frames),
        "frequency_hz": frequency,
        "num_envs": args.num_envs,
        "total_timesteps": float(args.total_timesteps),
        "training_minutes": elapsed / 60,
        "agent_path": str(save_path),
        "mean_episode_return_first": float(returns[0]),
        "mean_episode_return_last": float(returns[-1]),
        "mean_episode_length_last": float(lengths[-1]),
        "return_curve_every_update": [float(r) for r in returns[:: max(1, len(returns) // 200)]],
    }
    summary_file = args.output_dir / "training_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "return_curve_every_update"}, indent=2))


if __name__ == "__main__":
    main()
