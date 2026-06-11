"""Retarget one built-in H1 activity onto explicit morphology variants and render a grid."""

from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import cv2
import mujoco
import numpy as np
from PIL import Image

from h1_morphology_variants import PRESETS, H1MorphologyPreset, create_h1_variant_xml, wrap_text
from render_retarget_comparison import RobotRenderer
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
from loco_mujoco.task_factories import DefaultDatasetConf, ImitationFactory
from loco_mujoco.trajectory import Trajectory


DEFAULT_PRESETS = ["nominal", "tall_legs", "long_arms", "big_feet", "combined"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="squat")
    parser.add_argument("--cache-tag", default="quality")
    parser.add_argument("--presets", nargs="+", choices=PRESETS, default=DEFAULT_PRESETS)
    parser.add_argument("--source-start-frame", type=int, default=240)
    parser.add_argument("--source-frames", type=int, default=240)
    parser.add_argument("--shape-iterations", type=int, default=1000)
    parser.add_argument("--pose-iterations", type=int, default=400)
    parser.add_argument("--motion-iterations", type=int, default=25)
    parser.add_argument("--init-motion-iterations", type=int, default=1000)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--loops", type=int, default=2)
    parser.add_argument("--panel-width", type=int, default=360)
    parser.add_argument("--panel-height", type=int, default=270)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def configure_target(conf, args: argparse.Namespace):
    conf = deepcopy(conf)
    conf.optimization_params.torch_device = "cpu"
    conf.optimization_params.shape_iterations = args.shape_iterations
    conf.optimization_params.pose_iterations = args.pose_iterations
    conf.optimization_params.motion_iterations = args.motion_iterations
    conf.optimization_params.init_motion_iterations = args.init_motion_iterations
    return conf


def min_floor_distance(xml_path: Path, traj: Trajectory) -> float:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    foot_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in ("left_foot", "right_foot")
    ]
    min_distance = np.inf
    for qpos, qvel in zip(np.asarray(traj.data.qpos), np.asarray(traj.data.qvel)):
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        mujoco.mj_forward(model, data)
        for foot_id in foot_ids:
            min_distance = min(
                min_distance,
                mujoco.mj_geomDistance(model, data, floor_id, foot_id, 10.0, None),
            )
    return float(min_distance)


def add_panel_header(image: np.ndarray, title: str, details: str) -> np.ndarray:
    header = np.full((64, image.shape[1], 3), 28, dtype=np.uint8)
    cv2.putText(header, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (245, 245, 245), 2, cv2.LINE_AA)
    for index, line in enumerate(wrap_text(details, image.shape[1] - 24, font_scale=0.35)):
        cv2.putText(header, line, (12, 45 + index * 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (188, 204, 199), 1, cv2.LINE_AA)
    return np.vstack((header, image))


def compose_grid(panels: list[np.ndarray], columns: int = 3) -> np.ndarray:
    blank = np.full_like(panels[0], 28)
    rows = []
    for start in range(0, len(panels), columns):
        row = panels[start : start + columns]
        rows.append(np.hstack(row + [blank] * (columns - len(row))))
    return np.vstack(rows)


def render_outputs(
    source_xml: Path,
    source_traj: Trajectory,
    source_start_frame: int,
    source_frames: int,
    variants: list[tuple[H1MorphologyPreset, Path, Trajectory]],
    args: argparse.Namespace,
) -> tuple[Path, Path]:
    source_renderer = RobotRenderer(source_xml, args.panel_width, args.panel_height)
    variant_renderers = [
        RobotRenderer(xml_path, args.panel_width, args.panel_height) for _, xml_path, _ in variants
    ]
    source_frequency = float(source_traj.info.frequency)
    duration = source_frames / source_frequency
    for _, _, traj in variants:
        duration = max(duration, int(traj.data.n_samples) / float(traj.info.frequency))
    frames_per_loop = max(1, round(duration * args.fps))
    grid_rows = math.ceil((len(variants) + 1) / 3)
    frame_size = (args.panel_width * 3, (args.panel_height + 64) * grid_rows)

    output_stem = f"h1_morphology_retarget_{args.task}_{args.cache_tag}"
    video_path = WORKSPACE / "videos" / f"{output_stem}_grid.mp4"
    snapshot_path = WORKSPACE / "images" / f"{output_stem}_gallery.png"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open the MP4 video writer.")

    snapshot = None
    try:
        for _ in range(args.loops):
            for output_index in range(frames_per_loop):
                time_seconds = output_index / args.fps
                source_index = source_start_frame + min(round(time_seconds * source_frequency), source_frames - 1)
                panels = [
                    add_panel_header(
                        source_renderer.render(
                            np.asarray(source_traj.data.qpos[source_index]),
                            np.asarray(source_traj.data.qvel[source_index]),
                        ),
                        "Reference: standard H1",
                        f"original {args.task} clip",
                    )
                ]
                for (preset, _, traj), renderer in zip(variants, variant_renderers):
                    target_index = min(
                        round(time_seconds * float(traj.info.frequency)),
                        int(traj.data.n_samples) - 1,
                    )
                    panels.append(
                        add_panel_header(
                            renderer.render(
                                np.asarray(traj.data.qpos[target_index]),
                                np.asarray(traj.data.qvel[target_index]),
                            ),
                            preset.label,
                            preset.details,
                        )
                    )
                frame = compose_grid(panels)
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                if snapshot is None or output_index == frames_per_loop - 1:
                    snapshot = frame
    finally:
        writer.release()
        source_renderer.close()
        for renderer in variant_renderers:
            renderer.close()

    Image.fromarray(snapshot).save(snapshot_path)
    return video_path, snapshot_path


def main() -> None:
    args = parse_args()
    source_env_name = f"ProofUnitreeH1Source_{args.task}_{args.cache_tag}"
    fitted_motion_path = (
        WORKSPACE
        / "external_data"
        / "retarget_proof"
        / f"h1_{args.task}_{args.cache_tag}_start_{args.source_start_frame}_{args.source_frames}_frames_smpl.npz"
    )

    LocoEnv.registered_envs[source_env_name] = UnitreeH1
    source_conf = configure_target(load_robot_conf_file("UnitreeH1"), args)
    source_env = ImitationFactory.make("UnitreeH1", default_dataset_conf=DefaultDatasetConf([args.task]))
    source_traj = crop_first_trajectory(source_env.th.traj, args.source_start_frame, args.source_frames)

    output_dir = WORKSPACE / "external_data" / "morphology_sweep" / args.task / args.cache_tag
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
        trajectory_path = output_dir / f"{preset.name}_{args.task}.npz"
        if args.force or not trajectory_path.exists():
            target_conf = configure_target(load_robot_conf_file("UnitreeH1"), args)
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
            "task": args.task,
            "xml_path": str(xml_path),
            "trajectory_path": str(trajectory_path),
            "trajectory_samples": int(target_traj.data.n_samples),
            "trajectory_complete": bool(target_traj.data.is_complete),
            "pre_correction_min_floor_distance_meters": pre_correction_min_distance,
            "max_floor_lift_meters": max_floor_lift,
            "min_floor_distance_meters": min_distance,
        }
        print(f"{preset.name}: complete={target_traj.data.is_complete}, min_floor_distance={min_distance:.6f} m")

    video_path, snapshot_path = render_outputs(
        Path(UnitreeH1.get_default_xml_file_path()),
        source_env.th.traj,
        args.source_start_frame,
        args.source_frames,
        variants,
        args,
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"video": str(video_path), "snapshot": str(snapshot_path), "summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
