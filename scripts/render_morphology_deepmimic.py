"""Render the DeepMimic morphology results: per-variant timelines + a final grid.

Two modes (CPU MuJoCo + CPU JAX, so this runs on the Windows venv after the GPU
matrix has produced the trained checkpoints):

  --mode timeline : for ONE cell, roll out every saved checkpoint and play them
                    side by side in one video -- the learning progression
                    (early flailing -> balanced tracking), captioned with
                    training steps + tracking return.

  --mode grid     : for a set of cells (e.g. all variants of a robot+clip), roll
                    out each cell's FINAL checkpoint and tile them into one grid
                    video, each panel labelled with its morphology modification.

Each policy rollout is produced via PPOJax.play_policy_mujoco (writes an mp4),
then the mp4s are composited with OpenCV -- decoupled from the rollout internals.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np

from loco_mujoco.algorithms import PPOJax
from loco_mujoco.trajectory import Trajectory

from morphology_deepmimic import (
    WORKSPACE,
    cell_dir,
    control_config,
    get_robot,
    make_mimic_env,
    prepare_variant,
)

VIDEOS_DIR = WORKSPACE / "videos"
IMAGES_DIR = WORKSPACE / "images"


def to_local_path(p: str) -> str:
    """Translate a WSL '/mnt/<drive>/...' path to 'DRIVE:\\...' when on Windows.

    Manifests are written in WSL (Linux paths); rendering may run on the Windows
    venv (native GL), so the stored reference/agent paths need translating.
    """
    if os.name == "nt" and p.startswith("/mnt/") and len(p) > 6 and p[6] == "/":
        return f"{p[5].upper()}:" + p[6:].replace("/", os.sep)
    return p


def load_manifest(robot_key: str, clip: str, preset: str, out_suffix: str = "") -> dict:
    base = cell_dir(robot_key, clip, preset)
    if out_suffix:
        base = base.parent / f"{preset}__{out_suffix}"
    path = base / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"No manifest for {robot_key}/{clip}/{preset}: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["reference_path"] = to_local_path(manifest["reference_path"])
    for ckpt in manifest.get("checkpoints", []):
        ckpt["agent_path"] = to_local_path(ckpt["agent_path"])
    return manifest


def rollout_to_mp4(manifest: dict, agent_path: Path, out_mp4: Path, n_steps: int,
                   stochastic: bool = False) -> Path:
    """Roll out one trained checkpoint in CPU MuJoCo and copy the recorded mp4."""
    robot = get_robot(manifest["robot_key"])
    variant = prepare_variant(robot, manifest["preset"], "render")
    traj = Trajectory.load(manifest["reference_path"])
    # Must roll out with the SAME control modality the policy was trained under:
    # a PD policy outputs target joint angles, which a torque env would misread.
    resolved = manifest.get("control_params_resolved")
    if manifest.get("control") == "pd" and resolved:
        # Replay the exact gains recorded at training time (covers uniform +
        # per-joint-group scaling without re-deriving the recipe).
        ctrl_params = dict(control_type="PDControl",
                           control_params={k: list(v) for k, v in resolved.items()})
    else:
        ctrl_params = control_config(robot.key, manifest.get("control", "torque"))
        scale = manifest.get("pd_gain_scale", 1.0)
        if manifest.get("control") == "pd" and scale != 1.0:
            cp = ctrl_params["control_params"]
            cp["p_gain"] = [g * scale for g in cp["p_gain"]]
            cp["d_gain"] = [g * scale for g in cp["d_gain"]]
    env = make_mimic_env(variant["cpu_env_name"], traj, headless=True, **ctrl_params)
    agent_conf, agent_state = PPOJax.load_agent(str(agent_path))
    PPOJax.play_policy_mujoco(
        env, agent_conf, agent_state,
        deterministic=not stochastic, n_steps=n_steps, record=True, train_state_seed=0,
    )
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(env.video_file_path, out_mp4)
    return out_mp4


def read_frames(mp4: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(mp4))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def header(width: int, title: str, subtitle: str = "", height: int = 52) -> np.ndarray:
    bar = np.full((height, width, 3), 28, dtype=np.uint8)
    cv2.putText(bar, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (245, 245, 245), 2, cv2.LINE_AA)
    if subtitle:
        cv2.putText(bar, subtitle, (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (185, 202, 197), 1, cv2.LINE_AA)
    return bar


def compose(panel_sources: list[tuple[Path, str, str]], out_mp4: Path, snapshot_png: Path,
            columns: int, fps: int, panel_w: int, panel_h: int) -> None:
    """Tile several rollout mp4s into one grid video (frames aligned by index)."""
    clips = [(read_frames(mp4), title, sub) for mp4, title, sub in panel_sources]
    clips = [(frames, title, sub) for frames, title, sub in clips if frames]
    if not clips:
        raise RuntimeError("No non-empty rollout videos to compose.")
    n_frames = max(len(frames) for frames, _, _ in clips)

    def panel_at(frames: list[np.ndarray], title: str, sub: str, index: int) -> np.ndarray:
        frame = frames[min(index, len(frames) - 1)]  # freeze on last frame when ended
        frame = cv2.resize(frame, (panel_w, panel_h))
        return np.vstack((header(panel_w, title, sub), frame))

    rows = -(-len(clips) // columns)
    frame_size = (panel_w * columns, (panel_h + 52) * rows)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size)
    blank = np.full((panel_h + 52, panel_w, 3), 28, dtype=np.uint8)

    snapshot = None
    for index in range(n_frames):
        panels = [panel_at(f, t, s, index) for f, t, s in clips]
        grid_rows = []
        for start in range(0, len(panels), columns):
            row = panels[start:start + columns]
            grid_rows.append(np.hstack(row + [blank] * (columns - len(row))))
        grid = np.vstack(grid_rows)
        writer.write(grid)
        if index == n_frames // 2:
            snapshot = grid
    writer.release()
    snapshot_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(snapshot_png), snapshot if snapshot is not None else grid)
    print(f"[render] {out_mp4}\n[render] {snapshot_png}")


def mode_timeline(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.robot, args.clip, args.preset, args.out_suffix)
    preset_label = _preset_label(args.robot, args.preset)
    work = Path(tempfile.mkdtemp(prefix="dm_timeline_"))
    sources = []
    for ckpt in manifest["checkpoints"]:
        mp4 = work / f"ckpt_{ckpt['index']:02d}.mp4"
        try:
            rollout_to_mp4(manifest, Path(ckpt["agent_path"]), mp4, args.n_steps, args.stochastic)
        except Exception as exc:
            # A transiently unstable checkpoint can blow the sim up (NaN QACC);
            # skip that panel rather than losing the whole timeline.
            print(f"[render] skip checkpoint {ckpt['index']} ({ckpt['cumulative_steps']/1e6:.0f}M): {exc}")
            continue
        sources.append((
            mp4,
            f"{ckpt['cumulative_steps'] / 1e6:.0f}M steps",
            f"R={ckpt['mean_episode_return']:.0f}",
        ))
    tag = f"timeline_{args.robot}_{args.clip}_{args.preset}"
    if args.out_suffix:
        tag += f"__{args.out_suffix}"
    compose(sources, VIDEOS_DIR / f"{tag}.mp4", IMAGES_DIR / f"{tag}.png",
            columns=len(sources), fps=args.fps, panel_w=args.panel_width, panel_h=args.panel_height)
    shutil.rmtree(work, ignore_errors=True)
    print(f"[render] timeline for {preset_label}: {len(sources)} checkpoints")


def mode_grid(args: argparse.Namespace) -> None:
    work = Path(tempfile.mkdtemp(prefix="dm_grid_"))
    sources = []
    for preset in args.presets:
        try:
            manifest = load_manifest(args.robot, args.clip, preset, args.out_suffix)
        except FileNotFoundError as exc:
            print(f"[render] skip {preset}: {exc}")
            continue
        # Use the best checkpoint, not the last: training can diverge in a late
        # segment (seen on G1), so "final result" = best policy by tracking return.
        final = max(manifest["checkpoints"], key=lambda c: c["mean_episode_return"])
        mp4 = work / f"{preset}.mp4"
        rollout_to_mp4(manifest, Path(final["agent_path"]), mp4, args.n_steps, args.stochastic)
        sources.append((
            mp4,
            _preset_label(args.robot, preset),
            f"{_preset_details(args.robot, preset)} | R={final['mean_episode_return']:.0f}",
        ))
    tag = f"grid_{args.robot}_{args.clip}"
    if args.out_suffix:
        tag += f"__{args.out_suffix}"
    compose(sources, VIDEOS_DIR / f"{tag}.mp4", IMAGES_DIR / f"{tag}.png",
            columns=args.columns, fps=args.fps, panel_w=args.panel_width, panel_h=args.panel_height)
    shutil.rmtree(work, ignore_errors=True)


def _preset(robot_key: str, preset_name: str):
    return get_robot(robot_key).presets()[preset_name]


def _preset_label(robot_key: str, preset_name: str) -> str:
    return _preset(robot_key, preset_name).label


def _preset_details(robot_key: str, preset_name: str) -> str:
    return _preset(robot_key, preset_name).details


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=["timeline", "grid"])
    parser.add_argument("--robot", required=True, choices=["h1", "g1"])
    parser.add_argument("--clip", default="dance2_subject4")
    parser.add_argument("--preset", help="Cell preset for --mode timeline.")
    parser.add_argument("--out-suffix", default="",
                        help="Render an experiment cell <preset>__<suffix> (timeline mode).")
    parser.add_argument("--presets", nargs="+",
                        default=["nominal", "extreme_tall_light", "extreme_short_heavy",
                                 "extreme_combined", "combined"],
                        help="Cells to tile for --mode grid.")
    parser.add_argument("--n-steps", type=int, default=900)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--panel-width", type=int, default=400)
    parser.add_argument("--panel-height", type=int, default=360)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "timeline":
        if not args.preset:
            raise SystemExit("--mode timeline requires --preset")
        mode_timeline(args)
    else:
        mode_grid(args)


if __name__ == "__main__":
    main()
