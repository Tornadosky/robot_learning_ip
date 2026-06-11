"""Retarget a cropped AMASS clip onto one explicit H1 morphology preset."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from h1_morphology_variants import PRESETS, create_h1_variant_xml
from retarget_amass_clip_randomized_h1 import DEFAULT_DATASET
from retarget_randomized_h1 import (
    WORKSPACE,
    lift_trajectory_above_floor,
    register_variant_env,
    render_trajectory_frame,
)
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
from loco_mujoco.utils import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--preset", choices=PRESETS, default="extreme_combined")
    parser.add_argument("--cache-tag", default="extreme_quality")
    parser.add_argument("--start-frame", type=int, default=1000)
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--shape-iterations", type=int, default=800)
    parser.add_argument("--motion-iterations", type=int, default=35)
    parser.add_argument("--init-motion-iterations", type=int, default=1000)
    return parser.parse_args()


def crop_motion(motion: dict, start_frame: int, frames: int) -> dict:
    available = len(motion["pose_aa"])
    if start_frame < 0 or frames < 3 or start_frame + frames > available:
        raise ValueError(
            f"Requested frames [{start_frame}, {start_frame + frames}) from "
            f"a motion with {available} samples."
        )
    return {
        key: value[start_frame : start_frame + frames] if key in {"pose_aa", "trans"} else value
        for key, value in motion.items()
    }


def configure_target(args: argparse.Namespace):
    conf = deepcopy(load_robot_conf_file("UnitreeH1"))
    conf.optimization_params.torch_device = "cpu"
    conf.optimization_params.shape_iterations = args.shape_iterations
    conf.optimization_params.motion_iterations = args.motion_iterations
    conf.optimization_params.init_motion_iterations = args.init_motion_iterations
    return conf


def main() -> None:
    args = parse_args()
    preset = PRESETS[args.preset]
    env_name = f"H1Morphology_{preset.name}_{args.cache_tag}"
    xml_path = create_h1_variant_xml(preset)
    register_variant_env(env_name, xml_path)
    conf = configure_target(args)
    logger = setup_logger("amass-extreme", identifier="[LocoMuJoCo AMASS extreme H1]")

    shape_path = (
        Path(get_converted_amass_dataset_path()) / env_name / OPTIMIZED_SHAPE_FILE_NAME
    )
    if not shape_path.exists():
        fit_smpl_shape(env_name, conf, get_smpl_model_path(), str(shape_path), logger)

    motion = crop_motion(load_amass_data(args.dataset), args.start_frame, args.frames)
    traj = fit_smpl_motion(
        env_name,
        conf,
        get_smpl_model_path(),
        motion,
        str(shape_path),
        logger,
        skip_steps=False,
    )
    traj, max_floor_lift = lift_trajectory_above_floor(xml_path, traj)
    traj = extend_motion(env_name, conf.env_params, traj)

    dataset_id = Path(args.dataset).name.removesuffix("_poses").lower()
    clip_id = f"{dataset_id}_{preset.name}_start_{args.start_frame}_{args.frames}_frames"
    output_dir = WORKSPACE / "external_data" / "amass_extreme"
    output_dir.mkdir(parents=True, exist_ok=True)
    traj_path = output_dir / f"{clip_id}.npz"
    render_path = WORKSPACE / "images" / f"{clip_id}.png"
    traj.save(str(traj_path))
    render_trajectory_frame(xml_path, traj, render_path)

    print("AMASS extreme-H1 retarget complete")
    print(f"  dataset: {args.dataset}")
    print(f"  preset: {preset.name} ({preset.details})")
    print(f"  source_crop: [{args.start_frame}, {args.start_frame + args.frames})")
    print(f"  target_samples: {int(traj.data.n_samples)}")
    print(f"  trajectory_complete: {bool(traj.data.is_complete)}")
    print(f"  max_floor_lift_meters: {max_floor_lift:.6f}")
    print(f"  target_trajectory: {traj_path}")
    print(f"  rendered_frame: {render_path}")


if __name__ == "__main__":
    main()
