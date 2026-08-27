"""Build a motion NOBODY RECORDED out of an existing FSQ code stream.

The one thing a discrete motion token can do that an explicit reference cannot is
be *generated*: you cannot sample a plausible 19-dimensional joint trajectory out
of nothing, but you can walk a graph over a code vocabulary. This is the cheapest
honest version of that -- motion matching in code space, no training at all.

Method. For every frame the tokenizer gives a code window z[t] (J, latent). Build
a transition graph: from frame t, the legal successors are t+1 (continue) and any
frame u whose code window is within `--tol` of z[t+1] but which is at least
`--min-gap` frames away in the original timeline (a genuine cut, not a
neighbour). Random-walk that graph, taking a cut every `--dwell` frames on
average. What comes out is a choreography that exists in no recording, assembled
only from moments the codes say are interchangeable.

The output is a drop-in clip directory: the ORIGINAL clip's frames in the new
order, plus the matching `_zq` sidecar, so a z-only policy can be driven by it
through the unmodified pipeline and an explicit-reference policy can be run on
exactly the same motion as a matched control.

Root position is re-integrated across cuts (per-frame deltas accumulated) so the
robot does not teleport; height and orientation come from the source frame, which
is what REFROOT_FLOOR reads anyway.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def build_sequence(z, length, dwell, tol, min_gap, rng, max_candidates=64):
    """Random-walk the code graph; returns the frame indices of the new motion."""
    T = z.shape[0]
    flat = z.reshape(T, -1)
    # Normalising by the global scale makes `tol` interpretable across tokenizers.
    scale = float(np.sqrt((flat ** 2).sum(axis=1).mean())) + 1e-9
    idx = [int(rng.integers(0, T - 1))]
    cuts = 0
    for _ in range(length - 1):
        t = idx[-1]
        nxt = min(t + 1, T - 1)
        if rng.random() < 1.0 / max(dwell, 1):
            # distance from the frame we WOULD have gone to, so the cut is
            # judged on continuation, not on the current pose
            d = np.linalg.norm(flat - flat[nxt][None, :], axis=1) / scale
            far = np.abs(np.arange(T) - t) >= min_gap
            cand = np.where(far & (d < tol))[0]
            if cand.size:
                if cand.size > max_candidates:
                    cand = rng.choice(cand, max_candidates, replace=False)
                nxt = int(cand[int(rng.integers(0, cand.size))])
                cuts += 1
        idx.append(nxt)
    return np.array(idx, dtype=np.int64), cuts


def reintegrate_root(qpos, idx):
    """Continuous root xy across cuts: accumulate the source clip's own deltas."""
    out = np.array(qpos[idx])
    src_next = np.minimum(idx + 1, qpos.shape[0] - 1)
    step = qpos[src_next, 0:2] - qpos[idx, 0:2]      # per-frame travel at each source frame
    xy = np.zeros((len(idx), 2), dtype=out.dtype)
    xy[0] = qpos[idx[0], 0:2]
    for i in range(1, len(idx)):
        xy[i] = xy[i - 1] + step[i - 1]
    out[:, 0:2] = xy
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clip-dir", required=True)
    ap.add_argument("--clip", default="superM9.npz")
    ap.add_argument("--robots", nargs="+", default=["UnitreeH1", "UnitreeG1"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-clip", default="novel.npz")
    ap.add_argument("--length", type=int, default=9000)
    ap.add_argument("--dwell", type=int, default=40, help="mean frames between cuts")
    ap.add_argument("--tol", type=float, default=0.25, help="code distance for a legal cut")
    ap.add_argument("--min-gap", type=int, default=200, help="minimum temporal distance of a cut")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src_dir = Path(args.clip_dir)
    out_dir = Path(args.out)
    rng = np.random.default_rng(args.seed)

    # The walk is planned ONCE, on the source robot's codes, and applied to every
    # robot -- so the two robots receive the same choreography, as they do for any
    # other clip in this pipeline.
    lead = args.robots[0]
    z_lead = np.load(src_dir / lead / (Path(args.clip).stem + "_zq.npz"),
                     allow_pickle=True)["z_q"]
    idx, cuts = build_sequence(z_lead, args.length, args.dwell, args.tol,
                               args.min_gap, rng)

    consecutive = int((np.diff(idx) == 1).sum())
    report = {
        "source_clip": args.clip, "source_frames": int(z_lead.shape[0]),
        "frames": int(len(idx)), "cuts": int(cuts),
        "consecutive_fraction": consecutive / max(len(idx) - 1, 1),
        "unique_source_frames": int(np.unique(idx).size),
        "tol": args.tol, "dwell": args.dwell, "min_gap": args.min_gap, "seed": args.seed,
    }

    for robot in args.robots:
        src = np.load(src_dir / robot / args.clip, allow_pickle=True)
        z = np.load(src_dir / robot / (Path(args.clip).stem + "_zq.npz"),
                    allow_pickle=True)
        d = dict(src)
        d["qpos"] = reintegrate_root(np.asarray(src["qpos"]), idx).astype(np.float32)
        d["qvel"] = np.asarray(src["qvel"])[idx].astype(np.float32)
        if "site_xpos" in d:
            d["site_xpos"] = np.asarray(src["site_xpos"])[idx].astype(np.float32)
        dst = out_dir / robot
        dst.mkdir(parents=True, exist_ok=True)
        np.savez(dst / args.out_clip, **d)
        np.savez(dst / (Path(args.out_clip).stem + "_zq.npz"),
                 z_q=np.asarray(z["z_q"])[idx].astype(np.float32),
                 joint_names=z["joint_names"])
        print(f"{robot}: {len(idx)} frames -> {dst}")

    (out_dir / "stitch_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("NOVEL_CLIP_OK")


if __name__ == "__main__":
    main()
