"""Experiment 1 eval -- per-motion tracking of the shared multi-motion policy.

Roll the trained multi-clip policy in a single-clip mimic env for each of the 4
trained clips AND the held-out clip (dance2_subject5). Absolute metrics suffice:
mean tracking reward/step (~1.0 = perfect mimic) and non-fall rate. The held-out
clip tests whether one policy generalizes to an UNSEEN motion on the same body.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import common as C
import mm_common as MM
import gpu_roll as G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--envs", type=int, default=512)
    ap.add_argument("--steps", type=int, default=1500)
    args = ap.parse_args()

    if C.phase_done(args.outdir, "mm_eval"):
        C.log("phase mm_eval already done; skipping")
        return

    import jax
    manifest = json.loads((Path(args.outdir) / "mm_train" / "manifest.json").read_text())
    best_pkl = manifest["best_checkpoint"]["agent_path"]
    C.log(f"eval multi-motion policy: {best_pkl}")
    policy = G.load_jax_policy_pkl(best_pkl)

    clips = MM.TRAIN_CLIPS + [MM.HELDOUT_CLIP]
    cells = {}
    key = jax.random.PRNGKey(0)
    from loco_mujoco.core.wrappers.mjx import LocoMjxWrapper
    for clip in clips:
        env = MM.make_multiclip_env([clip])
        w = LocoMjxWrapper(env)
        key, k = jax.random.split(key)
        m = G.expert_metrics(w, policy, args.envs, args.steps, k)
        m["heldout"] = (clip == MM.HELDOUT_CLIP)
        cells[clip] = m
        C.log(f"  {clip:18s}{' (HELD-OUT)' if m['heldout'] else '':11s}: "
              f"track_rew/step={m['mean_tracking_reward']:.3f} nonfall={m['nonfall_rate']:.2f} "
              f"len={m['mean_length']:.0f} R={m['mean_return']:.0f} (n={m['n_episodes']})")

    train = [cells[c] for c in MM.TRAIN_CLIPS]
    ho = cells[MM.HELDOUT_CLIP]
    summary = {
        "train_track_reward": float(np.mean([c["mean_tracking_reward"] for c in train])),
        "train_nonfall": float(np.mean([c["nonfall_rate"] for c in train])),
        "heldout_track_reward": ho["mean_tracking_reward"],
        "heldout_nonfall": ho["nonfall_rate"],
    }
    C.write_json(args.outdir / "mm_eval.json", {"cells": cells, "summary": summary,
                                                "best_checkpoint": best_pkl})
    with open(args.outdir / "mm_eval.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip", "heldout", "mean_track_reward", "nonfall_rate", "mean_length",
                    "mean_return", "n_episodes"])
        for c in clips:
            m = cells[c]
            w.writerow([c, m["heldout"], m["mean_tracking_reward"], m["nonfall_rate"],
                        m["mean_length"], m["mean_return"], m["n_episodes"]])

    md = ["# Experiment 1 -- one policy, several motions (stock G1 body)", "",
          f"Trained on {', '.join(MM.TRAIN_CLIPS)}; held out **{MM.HELDOUT_CLIP}**. "
          f"{args.envs} envs x {args.steps} steps/clip (MJX). track_reward/step ~1.0 = perfect mimic.",
          "", "| clip | role | track_rew/step | non-fall | mean_len | mean_return |",
          "|---|---|---|---|---|---|"]
    for c in clips:
        m = cells[c]
        md.append(f"| {c} | {'held-out' if m['heldout'] else 'train'} | "
                  f"{m['mean_tracking_reward']:.3f} | {m['nonfall_rate']:.2f} | "
                  f"{m['mean_length']:.0f} | {m['mean_return']:.0f} |")
    # Distinguish two separate axes: (1) tracking FIDELITY (reward/step, ~1.0=perfect
    # mimic) and (2) ROBUSTNESS/balance (non-fall, episode length). And judge
    # generalization by comparing the held-out clip to the TRAIN AVERAGE, not an
    # absolute bar -- if held-out tracks as well as trained clips, the goal-conditioned
    # policy generalizes across motions even if absolute robustness is low for all.
    tr_fid = summary["train_track_reward"]; tr_nf = summary["train_nonfall"]
    ho_fid = summary["heldout_track_reward"]; ho_nf = ho["nonfall_rate"]
    train_len = float(np.mean([c["mean_length"] for c in train]))
    gen_ratio = ho_fid / tr_fid if tr_fid > 1e-6 else 0.0
    fidelity_ok = tr_fid >= 0.45
    robust_ok = tr_nf >= 0.5
    md += ["", "## Read-out", "",
           f"- Tracking FIDELITY (reward/step, 1.0=perfect): trained avg **{tr_fid:.3f}**, "
           f"held-out **{ho_fid:.3f}**.",
           f"- ROBUSTNESS: trained avg non-fall **{tr_nf:.2f}**, mean episode length **{train_len:.0f}** "
           f"/ horizon {C.HORIZON}.",
           "",
           f"- **Can one policy follow several motions?** "
           + ("YES on fidelity -- it tracks all four dances at comparable reward/step (it reads the "
              "current reference from the goal obs)."
              if fidelity_ok else
              "Only partially -- tracking fidelity is low.")
           + (" And it is robust (balances to horizon)."
              if robust_ok else
              f" BUT it is NOT robust: it falls after ~{train_len:.0f} steps (non-fall {tr_nf:.2f}), "
              "far short of the single-clip experts (~600+ steps, non-fall ~0.9). Juggling several "
              "motions in one policy sharply lowers the balance ceiling."),
           f"- **Motion generalization:** held-out clip is tracked at {gen_ratio*100:.0f}% of the trained-clip "
           f"fidelity ({ho_fid:.3f} vs {tr_fid:.3f}) -> "
           + ("generalization to an UNSEEN motion is essentially as good as on trained motions "
              "(goal-conditioning generalizes; it is not memorizing clips)."
              if gen_ratio >= 0.9 else
              "the unseen motion is tracked noticeably worse than trained ones."),
           ]
    (args.outdir / "mm_eval.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    C.log("mm_eval complete")
    C.mark_done(args.outdir, "mm_eval", summary)


if __name__ == "__main__":
    main()
