"""Compare how G1 morphology variants perform the SAME recorded dance.

We load the standard Unitree G1 LAFAN1 dance trajectory and replay its joint
trajectory (qpos) on each morphology variant. Because every variant keeps the
same joint DOF layout (only link sizes/masses change), the recorded joint
angles drive every body directly -- so you can see, side by side, how longer
legs / heavier torso / etc. change the way the exact same choreography looks.

This is a *kinematic* comparison (no physics, no SMPL retargeting needed), which
makes it fast and completely robust. Each frame is dropped onto the floor with a
generic lowest-geom clamp so variants of different leg length still stand on the
ground.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np
from PIL import Image

from loco_mujoco.task_factories import ImitationFactory, LAFAN1DatasetConf

from dance_lafan1_robots import pick_highlight_window
from g1_morphology_variants import PRESETS, create_g1_variant_xml


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_PRESETS = [
    "nominal",
    "tall_legs",
    "short_legs",
    "heavy_torso",
    "extreme_tall_light",
    "extreme_short_heavy",
]


class VariantReplayRenderer:
    """Offscreen renderer that replays qpos on one variant with a floor clamp."""

    def __init__(self, xml_path: Path, width: int, height: int, root_height: float):
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, width=width, height=height)
        self.floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        self.lookat_z = max(0.12, 0.85 * root_height)
        self.camera.distance = float(np.clip(3.8 * root_height, 1.0, 6.0))
        self.camera.azimuth = 135
        self.camera.elevation = -12

    def _clamp_to_floor(self, clearance: float = 0.02) -> None:
        # Lift (or drop) the root so the lowest geom rests just above the floor.
        mujoco.mj_forward(self.model, self.data)
        zs = [
            float(self.data.geom_xpos[g][2])
            for g in range(self.model.ngeom)
            if g != self.floor_id
        ]
        self.data.qpos[2] += clearance - min(zs)
        mujoco.mj_forward(self.model, self.data)

    def render(self, qpos: np.ndarray) -> np.ndarray:
        self.data.qpos[:] = qpos
        self.data.qvel[:] = 0.0
        self._clamp_to_floor()
        self.camera.lookat[:] = [self.data.qpos[0], self.data.qpos[1], self.lookat_z]
        self.renderer.update_scene(self.data, camera=self.camera)
        return self.renderer.render()

    def close(self) -> None:
        self.renderer.close()


def add_header(image: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    header = np.full((52, image.shape[1], 3), 28, dtype=np.uint8)
    cv2.putText(header, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(header, subtitle, (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 200, 195), 1, cv2.LINE_AA)
    return np.vstack((header, image))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", default="dance2_subject4")
    parser.add_argument("--presets", nargs="+", choices=list(PRESETS), default=DEFAULT_PRESETS)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--start-seconds", type=float, default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--panel-width", type=int, default=400)
    parser.add_argument("--panel-height", type=int, default=360)
    parser.add_argument("--output-tag", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tag = args.output_tag or f"g1_morphology_dance_{args.clip}"

    videos_dir = WORKSPACE / "videos"
    images_dir = WORKSPACE / "images"
    videos_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading standard G1 / {args.clip} ...")
    env = ImitationFactory.make("UnitreeG1", lafan1_dataset_conf=LAFAN1DatasetConf([args.clip]))
    traj = env.th.traj
    frequency = float(traj.info.frequency)
    qpos_all = np.asarray(traj.data.qpos)
    root_height = float(np.median(qpos_all[:, 2]))
    del env

    if args.start_seconds is None:
        start_seconds, duration = pick_highlight_window(traj, args.duration)
    else:
        start_seconds, duration = args.start_seconds, args.duration
    print(f"Highlight window: start={start_seconds:.1f}s duration={duration:.1f}s")

    renderers = {}
    summary = {}
    for name in args.presets:
        preset = PRESETS[name]
        xml_path = create_g1_variant_xml(preset)
        renderers[name] = VariantReplayRenderer(
            xml_path, args.panel_width, args.panel_height, root_height
        )
        summary[name] = {"label": preset.label, "details": preset.details, "xml_path": str(xml_path)}

    output_frames = max(1, int(round(duration * args.fps)))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    panel_size = (args.panel_width, args.panel_height + 52)

    columns = 3 if len(args.presets) > 2 else len(args.presets)
    rows = -(-len(args.presets) // columns)
    grid_path = videos_dir / f"{tag}_grid.mp4"
    grid_writer = cv2.VideoWriter(
        str(grid_path), fourcc, args.fps, (panel_size[0] * columns, panel_size[1] * rows)
    )

    snapshot = None
    snapshot_index = output_frames // 2
    try:
        for output_index in range(output_frames):
            time_seconds = start_seconds + output_index / args.fps
            frame = min(int(round(time_seconds * frequency)), len(qpos_all) - 1)
            panels = []
            for name in args.presets:
                preset = PRESETS[name]
                image = renderers[name].render(qpos_all[frame])
                panels.append(add_header(image, preset.label, preset.details))

            blank = np.full_like(panels[0], 28)
            grid_rows = []
            for start in range(0, len(panels), columns):
                row = panels[start : start + columns]
                grid_rows.append(np.hstack(row + [blank] * (columns - len(row))))
            grid = np.vstack(grid_rows)
            grid_writer.write(cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
            if output_index == snapshot_index:
                snapshot = grid
            if output_index % (5 * args.fps) == 0:
                print(f"  rendered {output_index}/{output_frames} frames")
    finally:
        grid_writer.release()
        for renderer in renderers.values():
            renderer.close()

    gallery_path = images_dir / f"{tag}_gallery.png"
    Image.fromarray(snapshot).save(gallery_path)

    summary_out = {
        "clip": args.clip,
        "highlight": {"start_seconds": start_seconds, "duration_seconds": duration},
        "frequency_hz": frequency,
        "presets": summary,
        "outputs": {"grid_video": str(grid_path), "gallery_image": str(gallery_path)},
    }
    summary_path = images_dir / f"{tag}_summary.json"
    summary_path.write_text(json.dumps(summary_out, indent=2), encoding="utf-8")
    print(json.dumps(summary_out["outputs"], indent=2))


if __name__ == "__main__":
    main()
