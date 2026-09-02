"""Emit raw reference WINDOW sidecars for SONIC-style co-training (wave 6).

For every <robot>/<clip>.npz under --clip-dir, writes <clip>_win.npz next to
the existing _zq sidecar, with the SAME layout the tokenizer was trained on
(khaendler_fsq_clip.build_windows: rows = [t, t, t+1, ..., t+N-1] clamped,
channels = qpos, qvel[, left_foot_h, right_foot_h]) flattened per joint:

    z_q  : (T, J, (N+1)*C) float32   -- named z_q so tracking_clip.py's sidecar
                                        loader (joint-name mapping, hold) is
                                        reused unchanged via
                                        tracking_clip_latent_sidecar_suffix=_win
    joint_names, rows, channels, lookahead, foot_channels

Usage:
  python emit_window_sidecars.py --tokenizer experiments/fsq_khaendler/tokenizer_3t_v2 \
      --clip-dir experiments/fsq_khaendler/clips_3t_v2 --robots UnitreeH1 UnitreeG1 BoosterT1 \
      --clip dance2_subject4.npz walk1_subject1.npz
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))  # scripts/scaling
from khaendler_fsq_clip import load_clip_joints, load_clip_feet, build_windows, resolve_clips  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True, help="dir with config.json + descriptions.npz (defines lookahead/foot channels/joint order)")
    ap.add_argument("--clip-dir", required=True)
    ap.add_argument("--robots", nargs="+", required=True)
    ap.add_argument("--clip", nargs="+", required=True)
    ap.add_argument("--out", default=None, help="default: in place next to the clips")
    a = ap.parse_args()

    tok = Path(a.tokenizer)
    cfg = json.loads((tok / "config.json").read_text())
    lookahead = int(cfg["lookahead"])
    foot = bool(cfg.get("foot_channels", False))
    desc_store = np.load(tok / "descriptions.npz", allow_pickle=True)
    src_dir = Path(a.clip_dir)
    dst_dir = Path(a.out) if a.out else src_dir
    C = 4 if foot else 2

    for robot in a.robots:
        joint_names = [str(n) for n in desc_store[f"{robot}_joints"]]
        for clip in resolve_clips(a.clip):
            src = src_dir / robot / clip
            if not src.exists():
                print(f"{robot}/{clip}: missing, skipped"); continue
            qpos, qvel, _ = load_clip_joints(src, joint_names)
            feet = load_clip_feet(src) if foot else None
            w = build_windows(qpos, qvel, lookahead, feet)  # (T, N+1, J, C)
            T, R, J, Cc = w.shape
            assert Cc == C
            flat = np.transpose(w, (0, 2, 1, 3)).reshape(T, J, R * C).astype(np.float32)
            out_robot = dst_dir / robot
            out_robot.mkdir(parents=True, exist_ok=True)
            out = out_robot / (Path(clip).stem + "_win.npz")
            np.savez(out, z_q=flat, joint_names=np.array(joint_names), rows=R, channels=C,
                     lookahead=lookahead, foot_channels=foot)
            print(f"{robot}/{clip}: window sidecar {flat.shape} -> {out}  "
                  f"|qpos|max={np.abs(flat[:, :, 0::C]).max():.2f} |qvel|max={np.abs(flat[:, :, 1::C]).max():.2f}")


if __name__ == "__main__":
    main()
