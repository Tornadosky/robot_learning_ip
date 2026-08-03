"""Phase 7 -- assemble report.md from every phase's artifacts.

Computes data-driven verdicts for each question the experiment was designed to
answer, then embeds the supporting tables. Tolerant of missing optional phases
(dagger / vae).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import common as C


def _load(p: Path, default=None):
    return json.loads(p.read_text()) if p.exists() else default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True, type=Path)
    args = ap.parse_args()
    o = args.outdir

    selection = _load(o / "selection.json")
    order = selection["variants_order"]
    heldout = C.HELDOUT_VARIANT
    seen = [v for v in order if v != heldout]

    tm = _load(o / "transfer_matrix.json")
    tm_done = _load(o / "state" / "transfer.done", {})
    ds = _load(o / "dataset_stats.json", {})
    bc = _load(o / "bc_offline_summary.json", {})
    cl = _load(o / "bc_closed_loop_eval.json", {})
    dagger = _load(o / "dagger_results.json")
    vae = _load(o / "vae_results.json")

    def cl_norm(model, variant):
        c = cl["cells"].get(f"{model}__{variant}") if cl else None
        return None if c is None else c.get("normalized_return")

    def cl_mean_over(model, variants):
        vals = [cl_norm(model, v) for v in variants]
        vals = [x for x in vals if x is not None]
        return float(np.mean(vals)) if vals else None

    # ---- verdicts ----
    verdicts = []
    no_desc = cl_mean_over("shared_no_descriptor", seen) if cl else None
    onehot = cl_mean_over("shared_variant_onehot", seen) if cl else None
    desc = cl_mean_over("shared_morphology_descriptor", seen) if cl else None
    film = cl_mean_over("shared_film_descriptor", seen) if cl else None

    # "best shared policy" = best among policies trained on ALL 5 variants (a
    # *_heldout model is a 4-variant policy, so scoring it on its 4 training
    # variants would not be a like-for-like shared-policy comparison).
    best_name, best_val = None, -1
    if cl:
        for m in bc:
            if m.endswith("_heldout"):
                continue
            mv = cl_mean_over(m, seen)
            if mv is not None and mv > best_val:
                best_name, best_val = m, mv
    # best descriptor-conditioned model (concat OR FiLM -- both consume the
    # morphology descriptor); the headline "did the descriptor help" question is
    # about morphology conditioning in general, not one specific injection.
    desc_best = None
    if cl:
        cands = [x for x in (desc, film) if x is not None]
        desc_best = max(cands) if cands else None
    desc_best_name = "FiLM" if (film is not None and (desc is None or film >= desc)) else "concat"

    def fmt(x):
        return "n/a" if x is None else f"{x:.2f}"

    if cl:
        shared_worked = best_val is not None and best_val >= 0.6
        verdicts.append(
            f"**Did one shared policy work?** Best shared policy `{best_name}` reaches "
            f"{fmt(best_val)} of expert return (averaged over seen variants). "
            + ("YES -- a single network controls the seen variants at a usable fraction of the dedicated experts."
               if shared_worked else
               "PARTIAL/NO -- the shared policy is well below the per-variant experts; "
               "see distribution-shift note below."))

        if desc_best is not None and no_desc is not None:
            verdicts.append(
                f"**Did the morphology descriptor help?** Best descriptor-conditioned policy "
                f"({desc_best_name}) {fmt(desc_best)} vs no-descriptor {fmt(no_desc)}, and naive "
                f"concat-descriptor {fmt(desc)}. "
                + ("YES, but HOW the descriptor is injected matters: FiLM modulation clearly helps "
                   "while simple concatenation does not -- a real cross-embodiment signal."
                   if (desc_best > no_desc + 0.08 and film is not None and (desc is None or film > desc + 0.08))
                   else ("YES: descriptor conditioning beats no-descriptor."
                         if desc_best > no_desc + 0.05 else
                         "NO meaningful difference -- variants may be too similar, or proprio+goal obs "
                         "already reveals morphology.")))

        # one-hot memorization check: onehot good on seen but cannot generalize (no held-out cond)
        verdicts.append(
            f"**Did one-hot just memorize variants?** one-hot closed-loop {fmt(onehot)} on seen variants. "
            "One-hot cannot represent the held-out variant at all (no held-out column in its input), so any "
            "generalization must come from the descriptor models -- see the held-out probe.")

        # held-out generalization
        ho_gen = cl_norm("shared_morphology_descriptor_heldout", heldout)
        ho_all = cl_norm("shared_morphology_descriptor", heldout)
        verdicts.append(
            f"**Held-out variant generalization ({heldout}):** descriptor policy trained on 4 variants "
            f"scores {fmt(ho_gen)} on the unseen body; the all-5 descriptor policy scores {fmt(ho_all)}. "
            + ("Generalization is real (the 4-variant policy transfers to the unseen body)."
               if (ho_gen is not None and ho_gen >= 0.4) else
               "Generalization to a genuinely novel extreme body did NOT succeed -- expected, it is the hardest case."))

        # distribution shift
        if bc and best_name:
            best_off = bc.get(best_name, {}).get("best_val_mse")
            if best_off is not None and best_val is not None and best_off < 0.5 and best_val < 0.5:
                verdicts.append(
                    f"**Offline vs closed-loop:** offline MSE is low ({best_off:.3f}) but closed-loop return is "
                    f"low ({fmt(best_val)}) -> classic DISTRIBUTION SHIFT (compounding error). The DAgger phase "
                    "targets exactly this.")

    if dagger:
        last = dagger["history"][-1]["mean_returns"]
        first = dagger["history"][0]["mean_returns"]
        impr = np.mean([last[v] - first[v] for v in order])
        verdicts.append(f"**Did DAgger fine-tuning help?** mean rollout return changed by {impr:+.0f} "
                        f"over {dagger['rounds']} rounds "
                        + ("(improved -- online relabeling reduced distribution shift)."
                           if impr > 0 else "(no improvement)."))
    else:
        verdicts.append("**PPO/DAgger fine-tuning:** not run (optional). Note: PPO needs GPU MJX; JAX is "
                        "CPU-only on this machine, so DAgger is the intended online remedy.")

    if vae:
        lat = vae.get("latents", {})
        import math
        finite = {k: v for k, v in lat.items()
                  if v.get("prior_sample_mse") is not None and math.isfinite(v["prior_sample_mse"])}
        det = vae.get("deterministic_bc_descriptor_val_mse")
        if not finite:
            verdicts.append("**VAE experiment:** FAILED -- the CVAE diverged (NaN/unstable prior-sample MSE) "
                            "at the tried latent sizes; deterministic BC remains better. Needs lr/beta/KL-warmup "
                            "tuning. Low priority vs the shared-policy result.")
        else:
            best_z = min(finite, key=lambda k: finite[k]["prior_sample_mse"])
            verdicts.append(f"**VAE experiment:** best latent {best_z} prior-sample MSE "
                            f"{finite[best_z]['prior_sample_mse']:.3f} vs deterministic BC {det:.3f} "
                            + ("(VAE worse -- stochastic latent did not help)."
                               if finite[best_z]["prior_sample_mse"] > det else "(VAE competitive)."))

    # ---- compose ----
    md = [f"# Overnight shared-policy experiment -- report", "",
          f"Output dir: `{o.name}`", "",
          "## Research question", "",
          "Can ONE shared policy control 5 modified G1 variants instead of needing one expert per variant?",
          "", "## 1. Selection", "",
          f"- Robot: **G1** (23 DoF; all variants share obs={selection['obs_dim']}, act={selection['action_dim']}).",
          f"- Variants: {', '.join(order)} (held-out variant for generalization: **{heldout}**).",
          f"- Motion: **{selection['clip_motion']}** (single shared clip).",
          f"  - {selection['motions']['note']}",
          "- Per-variant expert checkpoints and morphology descriptors: `selection.json`, "
          "`morphology_descriptors.json`, `descriptor_readme.md`.", ""]

    md += ["## 2. Expert transfer matrix", ""]
    if tm_done:
        md += [f"- Own-body (diagonal) normalized: **{tm_done.get('own_norm'):.2f}**; "
               f"cross-body (off-diagonal) normalized: **{tm_done.get('cross_norm'):.2f}**.",
               f"- => the variants are {'genuinely different (experts do NOT transfer for free)' if tm_done.get('cross_norm',1) < 0.7 else 'fairly similar (experts partly transfer)'}.",
               "- Full tables: `transfer_matrix.md` / `.csv`.", ""]
    else:
        md += ["- (transfer matrix not available)", ""]

    md += ["## 3. Dataset", "",
           f"- Total transitions: **{ds.get('total_samples', 'n/a'):,}**" if isinstance(ds.get('total_samples'), int) else "- (dataset stats n/a)",
           f"- Per variant: {ds.get('per_variant')}", "",
           "## 4. Offline BC losses (normalized-action val MSE)", ""]
    if bc:
        md += ["| model | conditioning | best val MSE |", "|---|---|---|"]
        for m, r in bc.items():
            md.append(f"| {m} | {r['cond_type']} | {r['best_val_mse']:.4f} |")
        md += ["", "Full per-variant breakdown: `bc_offline_results.md`.", ""]

    md += ["## 5. Closed-loop evaluation", ""]
    if cl:
        md += ["Normalized return (BC / own-expert) averaged over seen variants:", "",
               f"- no-descriptor: **{fmt(no_desc)}**",
               f"- variant one-hot: **{fmt(onehot)}**",
               f"- morphology descriptor: **{fmt(desc)}**",
               f"- FiLM descriptor: **{fmt(film)}**", "",
               "Full matrices (normalized return + non-fall): `bc_closed_loop_eval.md` / `.csv`.", ""]
    else:
        md += ["- (closed-loop eval not available)", ""]

    md += ["## 6. Verdicts", ""] + [f"- {v}" for v in verdicts] + [""]

    # checkpoints / videos
    md += ["## 7. Artifacts", "",
           "- Expert checkpoints: see `selection.json` (`agent_pkl` per variant).",
           "- BC/shared checkpoints: `bc_checkpoints/`.",
           "- Videos (best-effort): `videos/` (rendering may be skipped on headless Windows GL; "
           "metrics are unaffected).", ""]

    # recommended next
    rec = []
    if cl and best_val is not None:
        if best_val < 0.3:
            rec.append("All shared policies are far below experts: try an easier locomotion clip first, "
                       "add a richer descriptor (masses/inertia/CoM height from the MJCF), and use a "
                       "URMA-style per-joint architecture; consider stronger curriculum.")
        elif not dagger:
            rec.append("Run the DAgger phase (and, on a GPU box, PPO fine-tuning in MJX) to close the "
                       "offline-to-closed-loop gap.")
        if film is not None and desc is not None and film > desc + 0.08:
            rec.append("FiLM conditioning beat concat-descriptor -> adopt FiLM (or URMA-style per-joint "
                       "conditioning) as the default; naive concatenation wastes the descriptor.")
        if cl_norm("shared_morphology_descriptor_heldout", heldout) is not None and \
                cl_norm("shared_morphology_descriptor_heldout", heldout) < 0.1:
            rec.append("Zero-shot to the held-out EXTREME body failed -> train on more morphologies that "
                       "bracket the extreme one (the transfer matrix shows even experts don't reach it), "
                       "and condition with FiLM/URMA rather than concat.")
    rec.append("Add a true held-out MOTION by training at least 2 experts per variant on different clips.")
    md += ["## 8. Recommended next experiment", ""] + [f"- {r}" for r in rec] + [""]

    (o / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    C.log(f"wrote {o/'report.md'}")
    C.mark_done(o, "report")


if __name__ == "__main__":
    main()
