"""Time-scramble a token sidecar, keeping the clip and the token STATISTICS intact.

THE control for the auxiliary-channel result. Adding the token on top of the
explicit reference improved tracking by 2-3.5 %. Two explanations fit that sign:

  (a) INFORMATION -- the token summarises frames t..t+9, i.e. ~250 ms of future
      the reference channel never shows, so it is a lookahead channel;
  (b) NOTHING IN PARTICULAR -- a quantised, temporally-blurred signal alongside
      the reference acts as a regulariser or a low-pass filter, and any
      similar-looking input would do the same.

Scrambling settles it. The policy still receives a token of the right shape, the
right marginal distribution and the right per-joint structure -- but attached to
the wrong moment, so its lookahead is worthless. If the gain survives, it was
never information.

Two scramble modes:
  shuffle -- permute frames globally (destroys all temporal structure)
  shift   -- roll by a large offset (keeps local smoothness, destroys alignment)

`shift` is the stricter control: the token still looks like a plausible motion
code stream, it just describes a different moment of the dance.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clip-dir", required=True)
    ap.add_argument("--clip", required=True, help="e.g. superM9.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--robots", nargs="+", default=["UnitreeH1", "UnitreeG1"])
    ap.add_argument("--mode", default="shift", choices=("shift", "shuffle"))
    ap.add_argument("--shift", type=int, default=7919,
                    help="frames to roll; a prime well away from any clip boundary")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src_dir, out_dir = Path(args.clip_dir), Path(args.out)
    stem = Path(args.clip).stem
    rng = np.random.default_rng(args.seed)
    perm = None

    for robot in args.robots:
        src_clip = src_dir / robot / args.clip
        src_zq = src_dir / robot / f"{stem}_zq.npz"
        if not src_zq.exists():
            print(f"SKIP {robot}: no sidecar at {src_zq}")
            continue
        z = np.load(src_zq, allow_pickle=True)
        zq = np.asarray(z["z_q"])
        T = zq.shape[0]
        # One permutation for every robot, so a two-topology arm still receives a
        # consistent (if wrong) stream rather than two unrelated ones.
        if perm is None:
            perm = (rng.permutation(T) if args.mode == "shuffle"
                    else (np.arange(T) + args.shift) % T)
        dst = out_dir / robot
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_clip, dst / args.clip)          # clip UNCHANGED
        np.savez(dst / f"{stem}_zq.npz", z_q=zq[perm].astype(np.float32),
                 joint_names=z["joint_names"])
        moved = float(np.mean(np.abs(np.asarray(zq[perm]) - zq)))
        print(f"{robot}: {T} frames, mode={args.mode}, mean|z_scrambled - z| = {moved:.4f}")

    (out_dir / "scramble_report.txt").write_text(
        f"mode={args.mode} shift={args.shift} seed={args.seed} clip={args.clip}\n")
    print("SCRAMBLE_OK")


if __name__ == "__main__":
    main()
