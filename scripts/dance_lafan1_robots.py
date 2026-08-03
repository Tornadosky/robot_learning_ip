"""Render LAFAN1 dance clips on several stock humanoids (per-robot videos + grid)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np
from PIL import Image

from loco_mujoco.environments import LocoEnv
from loco_mujoco.task_factories import ImitationFactory, LAFAN1DatasetConf

WORKSPACE = Path(__file__).resolve().parents[1]

DEFAULT_ROBOTS = ["UnitreeH1", "UnitreeG1", "ToddlerBot", "Atlas"]


class FollowRenderer:
    """Offscreen renderer that follows the robot root with a size-aware camera."""

    def __init__(self, xml_path: Path, width: int, height: int, root_height: float):
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, width=width, height=height)
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        self.lookat_z = max(0.12, 0.85 * root_height)
        self.camera.distance = float(np.clip(3.6 * root_height, 1.0, 6.0))
        self.camera.azimuth = 135
        self.camera.elevation = -12

    def render(self, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)
        self.camera.lookat[:] = [qpos[0], qpos[1], self.lookat_z]
        self.renderer.update_scene(self.data, camera=self.camera)
        return self.renderer.render()

    def close(self) -> None:
        self.renderer.close()


def add_header(image: np.ndarray, title: str) -> np.ndarray:
    header = np.full((44, image.shape[1], 3), 28, dtype=np.uint8)
    cv2.putText(header, title, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 2, cv2.LINE_AA)
    return np.vstack((header, image))


def pick_highlight_window(traj, duration_seconds: float) -> tuple[float, float]:
    """Return (start_seconds, duration_seconds) of the most dynamic window."""
    frequency = float(traj.info.frequency)
    qvel = np.asarray(traj.data.qvel)
    energy = np.linalg.norm(qvel[:, 6:], axis=1)
    window = int(round(duration_seconds * frequency))
    if window >= len(energy):
        return 0.0, len(energy) / frequency
    cumulative = np.concatenate(([0.0], np.cumsum(energy)))
    sums = cumulative[window:] - cumulative[:-window]
    start_frame = int(np.argmax(sums))
    return start_frame / frequency, duration_seconds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robots", nargs="+", default=DEFAULT_ROBOTS)
    parser.add_argument("--clip", default="dance2_subject4")
    parser.add_argument("--duration", type=float, default=30.0, help="Highlight length in seconds.")
    parser.add_argument("--start-seconds", type=float, default=None,
                        help="Override automatic highlight selection.")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--panel-width", type=int, default=480)
    parser.add_argument("--panel-height", type=int, default=400)
    parser.add_argument("--output-tag", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tag = args.output_tag or args.clip

    videos_dir = WORKSPACE / "videos"
    images_dir = WORKSPACE / "images"
    videos_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    trajectories = {}
    renderers = {}
    summary = {}
    for name in args.robots:
        print(f"Loading {name} / {args.clip} ...")
        env = ImitationFactory.make(name, lafan1_dataset_conf=LAFAN1DatasetConf([args.clip]))
        traj = env.th.traj
        xml_path = Path(LocoEnv.registered_envs[name].get_default_xml_file_path())
        root_height = float(np.median(np.asarray(traj.data.qpos)[:, 2]))
        trajectories[name] = traj
        renderers[name] = FollowRenderer(xml_path, args.panel_width, args.panel_height, root_height)
        summary[name] = {
            "clip": args.clip,
            "samples": int(traj.data.n_samples),
            "frequency_hz": float(traj.info.frequency),
            "clip_seconds": int(traj.data.n_samples) / float(traj.info.frequency),
            "median_root_height_m": root_height,
        }
        del env

    reference = trajectories[args.robots[0]]
    if args.start_seconds is None:
        start_seconds, duration = pick_highlight_window(reference, args.duration)
    else:
        start_seconds, duration = args.start_seconds, args.duration
    print(f"Highlight window: start={start_seconds:.1f}s duration={duration:.1f}s")

    output_frames = max(1, int(round(duration * args.fps)))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    panel_size = (args.panel_width, args.panel_height + 44)

    writers = {}
    for name in args.robots:
        path = videos_dir / f"{tag}_{name}.mp4"
        writers[name] = (cv2.VideoWriter(str(path), fourcc, args.fps, panel_size), path)

    columns = 2 if len(args.robots) > 2 else len(args.robots)
    rows = -(-len(args.robots) // columns)
    grid_path = videos_dir / f"{tag}_grid.mp4"
    grid_writer = cv2.VideoWriter(
        str(grid_path), fourcc, args.fps, (panel_size[0] * columns, panel_size[1] * rows)
    )

    snapshot = None
    snapshot_index = output_frames // 2
    try:
        for output_index in range(output_frames):
            time_seconds = start_seconds + output_index / args.fps
            panels = []
            for name in args.robots:
                traj = trajectories[name]
                frame = min(
                    int(round(time_seconds * float(traj.info.frequency))),
                    int(traj.data.n_samples) - 1,
                )
                image = renderers[name].render(
                    np.asarray(traj.data.qpos[frame]), np.asarray(traj.data.qvel[frame])
                )
                panel = add_header(image, name)
                panels.append(panel)
                writers[name][0].write(cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

            blank = np.full_like(panels[0], 28)
            grid_rows = []
            for start in range(0, len(panels), columns):
                row = panels[start : start + columns]
                grid_rows.append(np.hstack(row + [blank] * (columns - len(row))))
            grid = np.vstack(grid_rows)
            grid_writer.write(cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
            if output_index == snapshot_index:
                snapshot = grid
            if output_index % (10 * args.fps) == 0:
                print(f"  rendered {output_index}/{output_frames} frames")
    finally:
        grid_writer.release()
        for writer, _ in writers.values():
            writer.release()
        for renderer in renderers.values():
            renderer.close()

    gallery_path = images_dir / f"{tag}_gallery.png"
    Image.fromarray(snapshot).save(gallery_path)

    summary["highlight"] = {"start_seconds": start_seconds, "duration_seconds": duration}
    summary["outputs"] = {
        "grid_video": str(grid_path),
        "gallery_image": str(gallery_path),
        **{name: str(path) for name, (_, path) in writers.items()},
    }
    summary_path = images_dir / f"{tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["outputs"], indent=2))


if __name__ == "__main__":
    main()
