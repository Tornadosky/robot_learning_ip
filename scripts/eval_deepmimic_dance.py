"""Roll out a trained DeepMimic dance policy in plain MuJoCo and record a video.

Loads the agent pkl saved by train_deepmimic_dance.py, rebuilds the same
imitation env (CPU MuJoCo, not Mjx) and records the policy physically tracking
the dance reference. Works with CPU JAX, so it can run on the Windows venv.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import yaml

import loco_mujoco
from loco_mujoco.algorithms import PPOJax
from loco_mujoco.task_factories import CustomDatasetConf, ImitationFactory
from loco_mujoco.trajectory import Trajectory, TrajectoryData

WORKSPACE = Path(__file__).resolve().parents[1]


def load_cached_lafan1(env_name: str, clip: str) -> Trajectory:
    with open(loco_mujoco.PATH_TO_VARIABLES, "r") as file:
        variables = yaml.load(file, Loader=yaml.FullLoader)
    cache_root = Path(variables["LOCOMUJOCO_CONVERTED_LAFAN1_PATH"])
    cache_file = cache_root / env_name.replace("Mjx", "") / f"{clip}.npz"
    return Trajectory.load(str(cache_file))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-path", type=Path, required=True,
                        help="Path to the saved agent pkl (training_summary.json expected alongside).")
    parser.add_argument("--n-steps", type=int, default=1500)
    parser.add_argument("--stochastic", action="store_true",
                        help="Sample actions instead of using the deterministic mean.")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    summary = json.loads((args.agent_path.parent / "training_summary.json").read_text(encoding="utf-8"))
    robot = summary["robot"].replace("Mjx", "")
    clip = summary["clip"]
    start_frame = int(summary["window_start_frame"])
    n_frames = int(summary["window_frames"])

    full_traj = load_cached_lafan1(robot, clip)
    data = TrajectoryData.dynamic_slice_in_dim(full_traj.data, 0, start_frame, n_frames, backend=jnp)
    traj = Trajectory(info=full_traj.info, data=data)

    env = ImitationFactory.make(
        robot,
        custom_dataset_conf=CustomDatasetConf(traj),
        headless=True,
        horizon=1000,
        goal_type="GoalTrajMimic",
        # visualize_goal draws reference arrows via mjv_initGeom, which crashes with mujoco 3.9 bindings
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

    agent_conf, agent_state = PPOJax.load_agent(str(args.agent_path))
    PPOJax.play_policy_mujoco(
        env,
        agent_conf,
        agent_state,
        deterministic=not args.stochastic,
        n_steps=args.n_steps,
        record=True,
        train_state_seed=0,
    )

    video_file = Path(env.video_file_path)
    output = args.output or (WORKSPACE / "videos" / f"deepmimic_{clip}_{robot}.mp4")
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(video_file, output)
    print(f"Recorded policy rollout: {output}")


if __name__ == "__main__":
    main()
