"""Phase 1 -- expert transfer matrix.

Evaluate each of the 5 expert policies on each of the 5 robot variants (drop
expert E's brain into robot V, which is well-defined because all variants share
G1's 23-DoF obs/action layout). The diagonal is "expert on its own body"; the
off-diagonal answers "are the variants actually different enough that an expert
does not just transfer for free?".

Writes transfer_matrix.{json,csv,md}; videos are produced best-effort and never
block the matrix.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

import common as C


def run_matrix(selection: dict, n_envs: int, n_steps: int) -> dict:
    """GPU MJX matrix: each cell rolls out N envs for T(>horizon) steps, auto-reset,
    and aggregates completed episodes. T>horizon so full-horizon survivors count."""
    import gpu_roll as G
    import jax
    order = selection["variants_order"]
    variants = selection["variants"]
    policies = {p: G.load_jax_policy(variants[p]) for p in order}
    C.log(f"loaded {len(policies)} experts (GPU MJX, {n_envs} envs x {n_steps} steps/cell)")

    cells = {}
    key = jax.random.PRNGKey(0)
    for ev in order:                       # evaluation robot (column)
        wenv = G.build_mjx_env(variants[ev])
        C.log(f"== eval variant {ev} ==")
        for ex in order:                   # expert policy (row)
            key, k = jax.random.split(key)
            agg = G.expert_metrics(wenv, policies[ex], n_envs, n_steps, k)
            cells[f"{ex}__ON__{ev}"] = agg
            C.log(f"  expert {ex:18s} -> {ev:18s}: "
                  f"R={agg['mean_return']:7.1f}+-{agg['std_return']:5.1f} "
                  f"len={agg['mean_length']:5.0f} nonfall={agg['nonfall_rate']:.2f} "
                  f"(n={agg['n_episodes']})")
    return {"variants_order": order, "n_envs": n_envs, "n_steps": n_steps,
            "n_episodes": "auto (MJX auto-reset)", "cells": cells}


def _matrix(values: dict, order: list, key: str) -> np.ndarray:
    m = np.zeros((len(order), len(order)))
    for i, ex in enumerate(order):
        for j, ev in enumerate(order):
            m[i, j] = values["cells"][f"{ex}__ON__{ev}"][key]
    return m


def write_outputs(values: dict, outdir: Path) -> dict:
    order = values["variants_order"]
    C.write_json(outdir / "transfer_matrix.json", values)

    # CSV: long form, every metric per cell.
    metrics = ["mean_return", "std_return", "mean_length", "nonfall_rate",
               "mean_tracking_reward", "survived_full", "n_episodes"]
    with open(outdir / "transfer_matrix.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["expert", "eval_variant"] + metrics)
        for ex in order:
            for ev in order:
                cell = values["cells"][f"{ex}__ON__{ev}"]
                w.writerow([ex, ev] + [cell.get(m) for m in metrics])

    ret = _matrix(values, order, "mean_return")
    nonfall = _matrix(values, order, "nonfall_rate")
    # normalized: each column divided by that variant's own expert (diagonal).
    diag = np.diag(ret).copy()
    diag_safe = np.where(diag <= 1e-6, 1.0, diag)
    norm = ret / diag_safe[None, :]

    def md_table(mat, fmt):
        lines = ["| expert \\\\ eval | " + " | ".join(order) + " |",
                 "|" + "---|" * (len(order) + 1)]
        for i, ex in enumerate(order):
            lines.append("| **" + ex + "** | " +
                         " | ".join(fmt(mat[i, j]) for j in range(len(order))) + " |")
        return "\n".join(lines)

    # off-diagonal analysis
    own = float(np.mean(np.diag(norm)))
    offdiag = norm.copy()
    np.fill_diagonal(offdiag, np.nan)
    cross_mean = float(np.nanmean(offdiag))
    # surprising cross success / worst cross failure
    best_cross = np.unravel_index(np.nanargmax(offdiag), offdiag.shape)
    worst_cross = np.unravel_index(np.nanargmin(offdiag), offdiag.shape)

    md = [
        "# Expert transfer matrix",
        "",
        f"5 experts x 5 evaluation variants, GPU MJX rollouts "
        f"({values.get('n_envs','?')} envs x {values.get('n_steps','?')} steps/cell, "
        f"auto-reset; deterministic policy, mimic tracking reward, horizon {C.HORIZON}).",
        "",
        "## Mean return (rows = expert policy, cols = evaluation robot)",
        "",
        md_table(ret, lambda x: f"{x:.0f}"),
        "",
        "## Non-fall rate",
        "",
        md_table(nonfall, lambda x: f"{x:.2f}"),
        "",
        "## Return normalized to each variant's own expert (diagonal = 1.00)",
        "",
        md_table(norm, lambda x: f"{x:.2f}"),
        "",
        "## Read-out",
        "",
        f"- Mean diagonal (own-body, normalized): **{own:.2f}** (1.00 by construction).",
        f"- Mean off-diagonal (cross-body, normalized): **{cross_mean:.2f}**.",
        f"- A low off-diagonal means the variants are genuinely different "
        "(experts do NOT transfer for free) -- which is the precondition for the "
        "shared-policy question to be interesting.",
        f"- Best surprising cross-transfer: expert **{order[best_cross[0]]}** on "
        f"**{order[best_cross[1]]}** = {offdiag[best_cross]:.2f} of own-expert.",
        f"- Worst cross-transfer: expert **{order[worst_cross[0]]}** on "
        f"**{order[worst_cross[1]]}** = {offdiag[worst_cross]:.2f} of own-expert.",
        "",
    ]
    (outdir / "transfer_matrix.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    return {
        "own_norm": own, "cross_norm": cross_mean,
        "best_cross": [order[best_cross[0]], order[best_cross[1]], float(offdiag[best_cross])],
        "worst_cross": [order[worst_cross[0]], order[worst_cross[1]], float(offdiag[worst_cross])],
    }


def make_videos(selection: dict, summary: dict, outdir: Path) -> None:
    """Best-effort diagnostic videos. Never raises into the phase."""
    vdir = outdir / "videos"
    vdir.mkdir(parents=True, exist_ok=True)
    order = selection["variants_order"]
    variants = selection["variants"]
    jobs = [
        ("own_success", order[0], order[0]),               # base expert on base body
        ("cross_fail", summary["worst_cross"][0], summary["worst_cross"][1]),
        ("cross_surprise", summary["best_cross"][0], summary["best_cross"][1]),
    ]
    try:
        import sys
        sys.path.insert(0, str(C.SCRIPTS))
        from render_morphology_deepmimic import rollout_to_mp4
        import json
        for tag, ex, ev in jobs:
            # roll out expert EX inside variant EV's body (use EV's cell manifest
            # for the body/reference, but EX's agent pkl for the brain).
            ev_cell = Path(variants[ev]["cell_dir"])
            man = json.loads((ev_cell / "manifest.json").read_text(encoding="utf-8"))
            man["reference_path"] = str(ev_cell / "reference.npz")
            out = vdir / f"transfer_{tag}__{ex}_on_{ev}.mp4"
            rollout_to_mp4(man, Path(variants[ex]["agent_pkl"]), out, n_steps=600)
            C.log(f"video: {out.name}")
    except Exception as exc:  # rendering/GL is fragile on headless Windows
        (vdir / "VIDEO_README.txt").write_text(
            f"Diagnostic video rendering failed/skipped: {exc!r}\n"
            "Metrics in transfer_matrix.* are unaffected.\n", encoding="utf-8")
        C.log(f"video rendering skipped: {exc!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--envs", type=int, default=512)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    if C.phase_done(args.outdir, "transfer"):
        C.log("phase transfer already done; skipping")
        return
    selection = C.load_selection(args.outdir)
    values = run_matrix(selection, args.envs, args.steps)
    summary = write_outputs(values, args.outdir)
    if not args.no_video:
        make_videos(selection, summary, args.outdir)
    C.mark_done(args.outdir, "transfer", summary)


if __name__ == "__main__":
    main()
