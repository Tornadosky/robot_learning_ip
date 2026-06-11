from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import mujoco
import numpy as np
from PIL import Image

from loco_mujoco.environments import UnitreeH1
from loco_mujoco.task_factories import DefaultDatasetConf, ImitationFactory
from loco_mujoco.trajectory import Trajectory
from retarget_randomized_h1 import WORKSPACE, create_randomized_h1_xml


class RobotRenderer:
    def __init__(self, xml_path: Path, width: int, height: int):
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, width=width, height=height)
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        self.camera.lookat[:] = [0.0, 0.0, 0.9]
        self.camera.distance = 3.9
        self.camera.azimuth = 135
        self.camera.elevation = -12

    def render(self, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)
        self.camera.lookat[:] = qpos[:3]
        self.renderer.update_scene(self.data, camera=self.camera)
        return self.renderer.render()

    def close(self) -> None:
        self.renderer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the source H1 reference beside a randomized-H1 retarget."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-tag", default="proof")
    parser.add_argument("--scale-range", type=float, default=0.035)
    parser.add_argument("--source-start-frame", type=int, default=240)
    parser.add_argument("--source-frames", type=int, default=240)
    parser.add_argument("--target-trajectory", type=Path)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--loops", type=int, default=2)
    parser.add_argument("--panel-width", type=int, default=480)
    parser.add_argument("--panel-height", type=int, default=360)
    return parser.parse_args()


def add_labels(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    header_height = 48
    combined = np.full(
        (left.shape[0] + header_height, left.shape[1] + right.shape[1], 3),
        28,
        dtype=np.uint8,
    )
    combined[header_height:, : left.shape[1]] = left
    combined[header_height:, left.shape[1] :] = right
    cv2.putText(
        combined,
        "Reference: standard H1",
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        combined,
        "Retargeted: randomized H1",
        (left.shape[1] + 16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    return combined


def render_pair(
    source_renderer: RobotRenderer,
    target_renderer: RobotRenderer,
    source_traj: Trajectory,
    target_traj: Trajectory,
    source_index: int,
    target_index: int,
) -> np.ndarray:
    source = source_renderer.render(
        np.asarray(source_traj.data.qpos[source_index]),
        np.asarray(source_traj.data.qvel[source_index]),
    )
    target = target_renderer.render(
        np.asarray(target_traj.data.qpos[target_index]),
        np.asarray(target_traj.data.qvel[target_index]),
    )
    return add_labels(source, target)


def main() -> None:
    args = parse_args()
    clip_id = f"{args.cache_tag}_start_{args.source_start_frame}_{args.source_frames}_frames"
    target_path = args.target_trajectory or (
        WORKSPACE
        / "external_data"
        / "retarget_proof"
        / f"randomized_h1_seed_{args.seed}_squat_{clip_id}.npz"
    )
    if not target_path.exists():
        raise FileNotFoundError(
            f"Missing {target_path}. Run retarget_randomized_h1.py with matching "
            "--start-frame and --frames values first."
        )

    target_xml, scale = create_randomized_h1_xml(args.seed, args.scale_range)
    source_xml = Path(UnitreeH1.get_default_xml_file_path())
    source_env = ImitationFactory.make(
        "UnitreeH1", default_dataset_conf=DefaultDatasetConf(["squat"])
    )
    source_traj = source_env.th.traj
    target_traj = Trajectory.load(target_path)

    source_start = args.source_start_frame
    source_stop = source_start + args.source_frames
    if source_start < 0 or source_stop > int(source_traj.data.n_samples):
        raise ValueError(
            f"Requested source frames [{source_start}, {source_stop}) from a "
            f"trajectory with {int(source_traj.data.n_samples)} samples."
        )

    source_renderer = RobotRenderer(source_xml, args.panel_width, args.panel_height)
    target_renderer = RobotRenderer(target_xml, args.panel_width, args.panel_height)
    source_frequency = float(source_traj.info.frequency)
    target_frequency = float(target_traj.info.frequency)
    duration = max(args.source_frames / source_frequency, int(target_traj.data.n_samples) / target_frequency)
    frames_per_loop = max(1, round(duration * args.fps))

    video_path = WORKSPACE / "videos" / f"randomized_h1_seed_{args.seed}_squat_{args.cache_tag}_comparison.mp4"
    contact_sheet_path = WORKSPACE / "images" / f"randomized_h1_seed_{args.seed}_squat_{args.cache_tag}_comparison.png"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (args.panel_width * 2, args.panel_height + 48),
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open the MP4 video writer.")

    contact_frames = []
    contact_indices = {round(v * (frames_per_loop - 1)) for v in (0.0, 0.33, 0.66, 1.0)}
    try:
        for _ in range(args.loops):
            for output_index in range(frames_per_loop):
                time_seconds = output_index / args.fps
                source_offset = min(round(time_seconds * source_frequency), args.source_frames - 1)
                target_index = min(
                    round(time_seconds * target_frequency),
                    int(target_traj.data.n_samples) - 1,
                )
                frame = render_pair(
                    source_renderer,
                    target_renderer,
                    source_traj,
                    target_traj,
                    source_start + source_offset,
                    target_index,
                )
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                if output_index in contact_indices and len(contact_frames) < 4:
                    contact_frames.append(frame)
    finally:
        writer.release()
        source_renderer.close()
        target_renderer.close()

    while len(contact_frames) < 4:
        contact_frames.append(contact_frames[-1])
    contact_sheet = np.vstack(
        [np.hstack(contact_frames[:2]), np.hstack(contact_frames[2:4])]
    )
    Image.fromarray(contact_sheet).save(contact_sheet_path)

    print("Retarget comparison rendered")
    print(f"  morphology_scale_xyz: {np.array2string(scale, precision=6)}")
    print(f"  source_frames: {args.source_frames}")
    print(f"  target_samples: {int(target_traj.data.n_samples)}")
    print(f"  duration_seconds_per_loop: {duration:.3f}")
    print(f"  loops: {args.loops}")
    print(f"  video: {video_path}")
    print(f"  contact_sheet: {contact_sheet_path}")


if __name__ == "__main__":
    main()
