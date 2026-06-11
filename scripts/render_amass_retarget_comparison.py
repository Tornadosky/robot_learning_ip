from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from loco_mujoco.smpl import SMPLH_BONE_ORDER_NAMES, SMPLH_Parser
from loco_mujoco.smpl.retargeting import get_smpl_model_path, load_amass_data
from loco_mujoco.trajectory import Trajectory
from h1_morphology_variants import PRESETS, create_h1_variant_xml
from render_retarget_comparison import RobotRenderer
from retarget_amass_clip_randomized_h1 import DEFAULT_DATASET
from retarget_randomized_h1 import WORKSPACE, create_randomized_h1_xml


SKELETON_EDGES = [
    ("Pelvis", "L_Hip"),
    ("L_Hip", "L_Knee"),
    ("L_Knee", "L_Ankle"),
    ("L_Ankle", "L_Toe"),
    ("Pelvis", "R_Hip"),
    ("R_Hip", "R_Knee"),
    ("R_Knee", "R_Ankle"),
    ("R_Ankle", "R_Toe"),
    ("Pelvis", "Torso"),
    ("Torso", "Spine"),
    ("Spine", "Chest"),
    ("Chest", "Neck"),
    ("Neck", "Head"),
    ("Chest", "L_Thorax"),
    ("L_Thorax", "L_Shoulder"),
    ("L_Shoulder", "L_Elbow"),
    ("L_Elbow", "L_Wrist"),
    ("Chest", "R_Thorax"),
    ("R_Thorax", "R_Shoulder"),
    ("R_Shoulder", "R_Elbow"),
    ("R_Elbow", "R_Wrist"),
]
SKELETON_INDICES = {
    name: SMPLH_BONE_ORDER_NAMES.index(name)
    for name in {name for edge in SKELETON_EDGES for name in edge}
}


class SkeletonRenderer:
    def __init__(self, positions: np.ndarray, width: int, height: int):
        self.positions = positions
        self.width = width
        self.height = height
        root_xy = positions[:, SKELETON_INDICES["Pelvis"], :2]
        relative_xy = positions[:, :, :2] - root_xy[:, None]
        angle = np.deg2rad(135)
        self.horizontal = relative_xy[..., 0] * np.cos(angle) - relative_xy[..., 1] * np.sin(angle)
        self.vertical = positions[..., 2]

        used = np.array(list(SKELETON_INDICES.values()))
        horizontal = self.horizontal[:, used]
        vertical = self.vertical[:, used]
        horizontal_extent = max(float(np.ptp(horizontal)), 1e-6)
        vertical_extent = max(float(np.ptp(vertical)), 1e-6)
        self.floor_height = float(vertical.min())
        self.scale = min(width * 0.68 / horizontal_extent, height * 0.78 / vertical_extent)

    def render(self, frame_index: int) -> np.ndarray:
        canvas = np.full((self.height, self.width, 3), (221, 227, 222), dtype=np.uint8)
        floor_y = self.height - 28
        for x in range(0, self.width, 48):
            cv2.line(canvas, (x, floor_y), (x, self.height), (194, 202, 197), 1)
        cv2.line(canvas, (0, floor_y), (self.width, floor_y), (165, 174, 169), 2)

        def point(name: str) -> tuple[int, int]:
            index = SKELETON_INDICES[name]
            x = int(self.width / 2 + self.horizontal[frame_index, index] * self.scale)
            y = int(floor_y - (self.vertical[frame_index, index] - self.floor_height) * self.scale)
            return x, y

        for start, end in SKELETON_EDGES:
            cv2.line(canvas, point(start), point(end), (62, 84, 109), 4, cv2.LINE_AA)
        for name in SKELETON_INDICES:
            cv2.circle(canvas, point(name), 5, (225, 111, 65), -1, cv2.LINE_AA)
            cv2.circle(canvas, point(name), 5, (51, 67, 82), 1, cv2.LINE_AA)
        return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a raw AMASS SMPLH skeleton beside its randomized-H1 retarget."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--preset", choices=PRESETS)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--scale-range", type=float, default=0.035)
    parser.add_argument("--target-trajectory", type=Path)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--loops", type=int, default=2)
    parser.add_argument("--panel-width", type=int, default=480)
    parser.add_argument("--panel-height", type=int, default=360)
    return parser.parse_args()


def get_smplh_positions(motion: dict, start_frame: int, frames: int) -> np.ndarray:
    frame_slice = slice(start_frame, start_frame + frames)
    pose = torch.from_numpy(np.asarray(motion["pose_aa"][frame_slice])).float()
    pose = torch.cat([pose, torch.zeros((frames, 156 - pose.shape[1]))], dim=-1)
    trans = torch.from_numpy(np.asarray(motion["trans"][frame_slice])).float()
    betas = torch.from_numpy(np.asarray(motion["betas"])[:16]).float()[None].repeat(frames, 1)
    parser = SMPLH_Parser(model_path=get_smpl_model_path(), gender="neutral").to("cpu")
    with torch.no_grad():
        transforms = parser.get_joint_transformations(pose.reshape(frames, -1, 3), betas, trans)
    return transforms[..., :3, 3].cpu().numpy()


def add_labels(left: np.ndarray, right: np.ndarray, target_label: str) -> np.ndarray:
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
        "Reference: AMASS SMPLH skeleton",
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        combined,
        target_label,
        (left.shape[1] + 16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    return combined


def main() -> None:
    args = parse_args()
    dataset_id = Path(args.dataset).name.removesuffix("_poses").lower()
    if args.preset:
        clip_id = f"{dataset_id}_{args.preset}_start_{args.start_frame}_{args.frames}_frames"
        target_path = args.target_trajectory or (
            WORKSPACE / "external_data" / "amass_extreme" / f"{clip_id}.npz"
        )
        xml_path = create_h1_variant_xml(PRESETS[args.preset])
        morphology_description = PRESETS[args.preset].details
        target_label = f"Retargeted: {args.preset.replace('_', ' ')} H1"
        output_stem = f"amass_{clip_id}"
    else:
        clip_id = f"{dataset_id}_{args.frames}_frames"
        target_path = args.target_trajectory or (
            WORKSPACE
            / "external_data"
            / "retarget_proof"
            / f"randomized_h1_seed_{args.seed}_{clip_id}.npz"
        )
        xml_path, scale = create_randomized_h1_xml(args.seed, args.scale_range)
        morphology_description = np.array2string(scale, precision=6)
        target_label = "Retargeted: randomized H1"
        output_stem = f"randomized_h1_seed_{args.seed}_{clip_id}"
    if not target_path.exists():
        raise FileNotFoundError(
            f"Missing {target_path}. Run retarget_amass_clip_randomized_h1.py "
            "with matching --dataset and --frames values first."
        )

    motion = load_amass_data(args.dataset)
    if args.start_frame < 0 or args.start_frame + args.frames > len(motion["pose_aa"]):
        raise ValueError(
            f"Requested frames [{args.start_frame}, {args.start_frame + args.frames}) "
            f"from {len(motion['pose_aa'])}."
        )
    positions = get_smplh_positions(motion, args.start_frame, args.frames)
    target_traj = Trajectory.load(target_path)

    skeleton_renderer = SkeletonRenderer(positions, args.panel_width, args.panel_height)
    robot_renderer = RobotRenderer(xml_path, args.panel_width, args.panel_height)
    source_frequency = float(np.asarray(motion["fps"]).item())
    target_frequency = float(target_traj.info.frequency)
    duration = max(args.frames / source_frequency, int(target_traj.data.n_samples) / target_frequency)
    frames_per_loop = max(1, round(duration * args.fps))

    video_path = WORKSPACE / "videos" / f"{output_stem}_comparison.mp4"
    contact_sheet_path = WORKSPACE / "images" / f"{output_stem}_comparison.png"
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
                source_index = min(round(time_seconds * source_frequency), args.frames - 1)
                target_index = min(
                    round(time_seconds * target_frequency),
                    int(target_traj.data.n_samples) - 1,
                )
                skeleton_frame = skeleton_renderer.render(source_index)
                robot_frame = robot_renderer.render(
                    np.asarray(target_traj.data.qpos[target_index]),
                    np.asarray(target_traj.data.qvel[target_index]),
                )
                frame = add_labels(skeleton_frame, robot_frame, target_label)
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                if output_index in contact_indices and len(contact_frames) < 4:
                    contact_frames.append(frame)
    finally:
        writer.release()
        robot_renderer.close()

    while len(contact_frames) < 4:
        contact_frames.append(contact_frames[-1])
    contact_sheet = np.vstack(
        [np.hstack(contact_frames[:2]), np.hstack(contact_frames[2:4])]
    )
    Image.fromarray(contact_sheet).save(contact_sheet_path)

    print("AMASS retarget comparison rendered")
    print(f"  dataset: {args.dataset}")
    print(f"  morphology: {morphology_description}")
    print(f"  source_start_frame: {args.start_frame}")
    print(f"  source_frames: {args.frames}")
    print(f"  target_samples: {int(target_traj.data.n_samples)}")
    print(f"  duration_seconds_per_loop: {duration:.3f}")
    print(f"  loops: {args.loops}")
    print(f"  video: {video_path}")
    print(f"  contact_sheet: {contact_sheet_path}")


if __name__ == "__main__":
    main()
