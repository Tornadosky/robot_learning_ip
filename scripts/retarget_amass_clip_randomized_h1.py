from __future__ import annotations

import argparse
from pathlib import Path

import jax
import numpy as np

from loco_mujoco.smpl.retargeting import (
    OPTIMIZED_SHAPE_FILE_NAME,
    extend_motion,
    fit_smpl_motion,
    fit_smpl_shape,
    get_converted_amass_dataset_path,
    get_smpl_model_path,
    load_amass_data,
    load_robot_conf_file,
)
from loco_mujoco.task_factories import CustomDatasetConf, ImitationFactory
from loco_mujoco.utils import setup_logger
from retarget_randomized_h1 import (
    WORKSPACE,
    create_randomized_h1_xml,
    register_variant_env,
    render_trajectory_frame,
)


DEFAULT_DATASET = "DanceDB/20120911_TheodorosSourmelis/Capoeira_Theodoros_v2_C3D_poses"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retarget a short AMASS clip onto a randomized Unitree H1 morphology."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--scale-range", type=float, default=0.035)
    parser.add_argument("--shape-iterations", type=int, default=12)
    parser.add_argument("--motion-iterations", type=int, default=5)
    parser.add_argument("--init-motion-iterations", type=int, default=30)
    return parser.parse_args()


def crop_motion(motion: dict, frames: int) -> dict:
    available = len(motion["pose_aa"])
    if frames < 3 or frames > available:
        raise ValueError(f"Requested {frames} frames from a motion with {available} frames.")
    return {
        key: value[:frames] if key in {"pose_aa", "trans"} else value
        for key, value in motion.items()
    }


def main() -> None:
    args = parse_args()
    env_name = f"RandomizedUnitreeH1Seed{args.seed}"
    xml_path, scale = create_randomized_h1_xml(args.seed, args.scale_range)
    register_variant_env(env_name, xml_path)

    conf = load_robot_conf_file("UnitreeH1")
    conf.optimization_params.torch_device = "cpu"
    conf.optimization_params.shape_iterations = args.shape_iterations
    conf.optimization_params.motion_iterations = args.motion_iterations
    conf.optimization_params.init_motion_iterations = args.init_motion_iterations

    logger = setup_logger("amass-proof", identifier="[LocoMuJoCo AMASS clip proof]")
    shape_path = (
        Path(get_converted_amass_dataset_path()) / env_name / OPTIMIZED_SHAPE_FILE_NAME
    )
    if not shape_path.exists():
        fit_smpl_shape(env_name, conf, get_smpl_model_path(), str(shape_path), logger)

    motion = crop_motion(load_amass_data(args.dataset), args.frames)
    proof_dir = WORKSPACE / "external_data" / "retarget_proof"
    dataset_id = Path(args.dataset).name.removesuffix("_poses").lower()
    clip_id = f"{dataset_id}_{args.frames}_frames"
    traj_path = proof_dir / f"randomized_h1_seed_{args.seed}_{clip_id}.npz"
    render_path = WORKSPACE / "images" / f"randomized_h1_seed_{args.seed}_{clip_id}.png"

    traj = fit_smpl_motion(
        env_name,
        conf,
        get_smpl_model_path(),
        motion,
        str(shape_path),
        logger,
        skip_steps=False,
    )
    traj = extend_motion(env_name, conf.env_params, traj)
    traj.save(str(traj_path))
    render_trajectory_frame(xml_path, traj, render_path)

    env = ImitationFactory.make(env_name, custom_dataset_conf=CustomDatasetConf(traj))
    obs = env.reset(jax.random.key(9))
    next_obs, reward, absorbing, done, _ = env.step(np.zeros(env.info.action_space.shape))

    print("AMASS randomized-H1 proof complete")
    print(f"  dataset: {args.dataset}")
    print(f"  morphology_scale_xyz: {np.array2string(scale, precision=6)}")
    print(f"  source_frames: {args.frames}")
    print(f"  target_samples: {int(traj.data.n_samples)}")
    print(f"  target_qpos_shape: {tuple(traj.data.qpos.shape)}")
    print(f"  target_qvel_shape: {tuple(traj.data.qvel.shape)}")
    print(f"  trajectory_complete: {traj.data.is_complete}")
    print(f"  obs_shape: {tuple(obs.shape)}")
    print(f"  next_obs_shape: {tuple(next_obs.shape)}")
    print(f"  reward: {float(reward)}")
    print(f"  absorbing: {bool(absorbing)}")
    print(f"  done: {bool(done)}")
    print(f"  randomized_xml: {xml_path}")
    print(f"  retargeted_trajectory: {traj_path}")
    print(f"  rendered_frame: {render_path}")


if __name__ == "__main__":
    main()
