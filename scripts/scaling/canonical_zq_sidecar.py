"""Tile the SHARED canonical code stream into per-robot `<clip>_zq.npz` sidecars.

`canonical_fsq_clip.py reconstruct` writes `z_canonical.npz` — one FSQ code per
canonical timestep, identical for every robot, which is the whole point of the
design.  `tracking_clip.py` however reads a PER-ROBOT sidecar shaped
(T, J, latent) and maps it onto actuators by joint NAME (line 241), so the same
code has to be tiled across each robot's own joints.  Tiling is what makes H1
(19 wide) and G1 (23 wide) receive byte-identical content.

The clip npz itself is copied through UNCHANGED: a z-only arm still tracks the
true motion, and only the observation channel is replaced.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", required=True, help="dir with dataset.npz (joint names per robot)")
    ap.add_argument("--z", required=True, help="z_canonical.npz from canonical reconstruct")
    ap.add_argument("--clip-dir", required=True, help="source clip dir (originals)")
    ap.add_argument("--clip", required=True, help="clip file name, e.g. super5dance.npz")
    ap.add_argument("--out", required=True, help="destination clip dir")
    ap.add_argument("--robots", nargs="+", default=["UnitreeH1", "UnitreeG1"])
    args = ap.parse_args()

    ds = np.load(Path(args.bundle) / "dataset.npz", allow_pickle=True)
    z = np.load(args.z, allow_pickle=True)["z_q"].astype(np.float32)
    if z.ndim != 2:
        raise ValueError(f"expected (T, latent) shared codes, got {z.shape}")
    T, latent = z.shape
    out = Path(args.out)

    for robot in args.robots:
        joints = [str(n) for n in ds[f"{robot}_joints"]]
        src = Path(args.clip_dir) / robot / args.clip
        clip = np.load(src, allow_pickle=True)
        if clip["qpos"].shape[0] != T:
            raise ValueError(f"{robot}: clip {clip['qpos'].shape[0]} frames != z {T}")
        dst = out / robot
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst / args.clip)
        zq = np.broadcast_to(z[:, None, :], (T, len(joints), latent)).astype(np.float32)
        np.savez(dst / (Path(args.clip).stem + "_zq.npz"),
                 z_q=zq, joint_names=np.array(joints))
        print(f"{robot}: sidecar {zq.shape} over {len(joints)} joints -> {dst}")
    print(f"SIDECARS_OK latent_dim={latent} frames={T}")


if __name__ == "__main__":
    main()
