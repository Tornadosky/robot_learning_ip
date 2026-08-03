"""Retarget a LAFAN1 dance clip onto explicit H1 morphology variants and render a grid."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from h1_morphology_variants import PRESETS, create_h1_variant_xml
from dance_lafan1_robots import pick_highlight_window
from retarget_h1_morphology_sweep import min_floor_distance, render_outputs
from retarget_randomized_h1 import (
    WORKSPACE,
    crop_first_trajectory,
    lift_trajectory_above_floor,
    register_variant_env,
)
from loco_mujoco.environments import LocoEnv, UnitreeH1
from loco_mujoco.smpl.retargeting import (
    extend_motion,
    load_robot_conf_file,
    retarget_traj_from_robot_to_robot,
)
from loco_mujoco.task_factories import ImitationFactory, LAFAN1DatasetConf
from loco_mujoco.trajectory import Trajectory


DEFAULT_PRESETS = ["nominal", "tall_legs", "short_legs", "long_arms", "big_feet", "combined"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", default="dance2_subject4")
    parser.add_argument("--cache-tag", default="dance")
    parser.add_argument("--presets", nargs="+", choices=PRESETS, default=DEFAULT_PRESETS)
    parser.add_argument("--duration", type=float, default=30.0,
                        help="Length (seconds) of the source highlight window.")
    parser.add_argument("--start-seconds", type=float, default=None,
                        help="Override the automatic highlight window start.")
    # NOTE: this only affects fit_smpl_shape, which is CPU-only code (crashes on cuda).
    # The heavy source-motion pose fit is hardcoded to cuda inside loco-mujoco anyway.
    parser.add_argument("--torch-device", default="cpu")
    parser.add_argument("--shape-iterations", type=int, default=1000)
    parser.add_argument("--pose-iterations", type=int, default=400)
    parser.add_argument("--motion-iterations", type=int, default=25)
    parser.add_argument("--init-motion-iterations", type=int, default=1000)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--loops", type=int, default=1)
    parser.add_argument("--panel-width", type=int, default=360)
    parser.add_argument("--panel-height", type=int, default=270)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def configure_conf(conf, args: argparse.Namespace):
    conf = deepcopy(conf)
    conf.optimization_params.torch_device = args.torch_device
    conf.optimization_params.shape_iterations = args.shape_iterations
    conf.optimization_params.pose_iterations = args.pose_iterations
    conf.optimization_params.motion_iterations = args.motion_iterations
    conf.optimization_params.init_motion_iterations = args.init_motion_iterations
    return conf


def main() -> None:
    args = parse_args()

    source_env = ImitationFactory.make(
        "UnitreeH1", lafan1_dataset_conf=LAFAN1DatasetConf([args.clip])
    )
    full_traj = source_env.th.traj
    frequency = float(full_traj.info.frequency)
    if args.start_seconds is None:
        start_seconds, duration = pick_highlight_window(full_traj, args.duration)
    else:
        start_seconds, duration = args.start_seconds, args.duration
    start_frame = int(round(start_seconds * frequency))
    n_frames = min(int(round(duration * frequency)),
                   int(full_traj.data.n_samples) - start_frame)
    print(f"Source window: start={start_seconds:.1f}s frames={n_frames} @ {frequency:.0f}Hz")
    source_traj = crop_first_trajectory(full_traj, start_frame, n_frames)

    source_env_name = f"DanceUnitreeH1Source_{args.clip}_{args.cache_tag}"
    LocoEnv.registered_envs[source_env_name] = UnitreeH1
    source_conf = configure_conf(load_robot_conf_file("UnitreeH1"), args)

    fitted_motion_path = (
        WORKSPACE / "external_data" / "retarget_proof"
        / f"h1_{args.clip}_{args.cache_tag}_start_{start_frame}_{n_frames}_frames_smpl.npz"
    )
    output_dir = WORKSPACE / "external_data" / "morphology_sweep" / args.clip / args.cache_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    previous_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}

    variants = []
    summary = {}
    for preset_name in args.presets:
        preset = PRESETS[preset_name]
        xml_path = create_h1_variant_xml(preset)
        target_env_name = f"H1Morphology_{preset.name}_{args.cache_tag}"
        register_variant_env(target_env_name, xml_path)
        trajectory_path = output_dir / f"{preset.name}_{args.clip}.npz"
        if args.force or not trajectory_path.exists():
            print(f"Retargeting {args.clip} -> {preset.name} ...")
            target_conf = configure_conf(load_robot_conf_file("UnitreeH1"), args)
            target_traj = retarget_traj_from_robot_to_robot(
                source_env_name,
                source_traj,
                target_env_name,
                robot_conf_source=source_conf,
                robot_conf_target=target_conf,
                path_to_fitted_motion_source=str(fitted_motion_path),
            )
            pre_correction_min_distance = min_floor_distance(xml_path, target_traj)
            target_traj, max_floor_lift = lift_trajectory_above_floor(xml_path, target_traj)
            target_traj = extend_motion(target_env_name, target_conf.env_params, target_traj)
            target_traj.save(str(trajectory_path))
        else:
            target_traj = Trajectory.load(trajectory_path)
            pre_correction_min_distance = previous_summary.get(preset.name, {}).get(
                "pre_correction_min_floor_distance_meters"
            )
            max_floor_lift = previous_summary.get(preset.name, {}).get("max_floor_lift_meters")
        min_distance = min_floor_distance(xml_path, target_traj)
        variants.append((preset, xml_path, target_traj))
        summary[preset.name] = {
            **asdict(preset),
            "clip": args.clip,
            "xml_path": str(xml_path),
            "trajectory_path": str(trajectory_path),
            "trajectory_samples": int(target_traj.data.n_samples),
            "trajectory_complete": bool(target_traj.data.is_complete),
            "pre_correction_min_floor_distance_meters": pre_correction_min_distance,
            "max_floor_lift_meters": max_floor_lift,
            "min_floor_distance_meters": min_distance,
        }
        print(f"{preset.name}: complete={target_traj.data.is_complete}, min_floor_distance={min_distance:.6f} m")

    # render_outputs derives output names from args.task / args.cache_tag
    args.task = args.clip
    args.source_start_frame = start_frame
    args.source_frames = n_frames
    video_path, snapshot_path = render_outputs(
        Path(UnitreeH1.get_default_xml_file_path()),
        full_traj,
        start_frame,
        n_frames,
        variants,
        args,
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"video": str(video_path), "snapshot": str(snapshot_path), "summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
