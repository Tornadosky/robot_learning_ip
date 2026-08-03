"""Phase 4 -- closed-loop evaluation of the shared BC policies.

Runs every trained BC policy on every variant in the actual simulator and scores
it against the expert that owns that body. Highlights:
  * seen variants (all models)
  * held-out variant (extreme_tall_light): the descriptor policy trained on only 4
    variants vs the one trained on all 5 -> morphology generalization probe.

Metrics per (model, variant): mean return, success/non-fall rate, episode length,
normalized return vs own-expert, mean tracking reward, action smoothness.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import common as C
from models import TorchActor


def eval_cell_gpu(wenv, actor, n_envs, n_steps, key):
    """Closed-loop eval of a torch BC policy in N MJX envs (host loop, jitted step)."""
    import gpu_roll as G
    stats, _ = G.torch_rollout(wenv, actor.batch, n_envs, n_steps, key, collect=False)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--steps", type=int, default=1500)
    args = ap.parse_args()

    if C.phase_done(args.outdir, "eval"):
        C.log("phase eval already done; skipping")
        return

    import torch, jax
    import gpu_roll as G
    device = "cuda" if torch.cuda.is_available() else "cpu"
    selection = C.load_selection(args.outdir)
    order = selection["variants_order"]
    heldout = C.HELDOUT_VARIANT

    # own-expert reference returns (transfer-matrix diagonal)
    tm = json.loads((args.outdir / "transfer_matrix.json").read_text())
    expert_ref = {v: tm["cells"][f"{v}__ON__{v}"]["mean_return"] for v in order}

    summary = json.loads((args.outdir / "bc_offline_summary.json").read_text())
    models = {n: m["checkpoint"] for n, m in summary.items()}

    rows = []
    cells = {}
    key = jax.random.PRNGKey(0)
    for mname, ckpt in models.items():
        for vi, v in enumerate(order):
            wenv = G.build_mjx_env(selection["variants"][v])
            actor = TorchActor(ckpt, vi, device=device)
            key, k = jax.random.split(key)
            agg = eval_cell_gpu(wenv, actor, args.envs, args.steps, k)
            ref = expert_ref[v]
            agg["normalized_return"] = float(agg["mean_return"] / ref) if ref > 1e-6 else None
            agg["expert_ref_return"] = ref
            agg["is_heldout_variant"] = (v == heldout)
            # the held-out-trained descriptor model only "generalizes" on held-out variant
            agg["is_generalization_cell"] = (mname.endswith("_heldout") and v == heldout)
            cells[f"{mname}__{v}"] = agg
            rows.append([mname, v, agg["mean_return"], agg["normalized_return"],
                         agg["nonfall_rate"], agg["mean_length"],
                         agg["mean_tracking_reward"], agg["action_smoothness"],
                         agg["is_heldout_variant"]])
            C.log(f"{mname:38s} on {v:18s}: R={agg['mean_return']:6.1f} "
                  f"norm={agg['normalized_return'] if agg['normalized_return'] is None else round(agg['normalized_return'],2)} "
                  f"nonfall={agg['nonfall_rate']:.2f} len={agg['mean_length']:.0f}")

    C.write_json(args.outdir / "bc_closed_loop_eval.json",
                 {"cells": cells, "expert_ref": expert_ref,
                  "envs": args.envs, "steps": args.steps})
    with open(args.outdir / "bc_closed_loop_eval.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "variant", "mean_return", "normalized_return", "nonfall_rate",
                    "mean_length", "mean_tracking_reward", "action_smoothness", "is_heldout_variant"])
        w.writerows(rows)

    # markdown: normalized-return matrix (model x variant)
    mnames = list(models.keys())
    def cellval(m, v, key):
        c = cells[f"{m}__{v}"][key]
        return c
    md = ["# Closed-loop BC evaluation", "",
          f"{args.envs} envs x {args.steps} steps/cell (MJX). Normalized return = BC mean return / own-expert "
          "mean return on the same body (1.00 = matches the dedicated expert).", "",
          "## Normalized return (rows = shared policy, cols = variant)", "",
          "| model \\\\ variant | " + " | ".join(order) + " |",
          "|" + "---|" * (len(order) + 1)]
    for m in mnames:
        vals = []
        for v in order:
            nr = cellval(m, v, "normalized_return")
            vals.append("n/a" if nr is None else f"{nr:.2f}")
        md.append(f"| {m} | " + " | ".join(vals) + " |")
    md += ["", "## Non-fall rate", "",
           "| model \\\\ variant | " + " | ".join(order) + " |",
           "|" + "---|" * (len(order) + 1)]
    for m in mnames:
        md.append(f"| {m} | " + " | ".join(f"{cellval(m,v,'nonfall_rate'):.2f}" for v in order) + " |")

    # held-out generalization callout
    md += ["", "## Held-out variant generalization (" + heldout + ")", ""]
    for m in mnames:
        c = cells[f"{m}__{heldout}"]
        tag = " (trained WITHOUT this variant)" if m.endswith("_heldout") else ""
        nr = c["normalized_return"]
        md.append(f"- **{m}**{tag}: normalized return "
                  f"{'n/a' if nr is None else round(nr,2)}, non-fall {c['nonfall_rate']:.2f}")
    (args.outdir / "bc_closed_loop_eval.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    C.log("closed-loop eval complete")
    C.mark_done(args.outdir, "eval", {"n_models": len(models)})


if __name__ == "__main__":
    main()
