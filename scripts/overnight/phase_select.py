"""Phase 0 -- variant/motion selection + morphology descriptors.

Writes:
  selection.json              chosen variants (+ best expert ckpt), motions, splits
  morphology_descriptors.json raw + normalized descriptors per variant
  morphology_descriptors.npy  normalized matrix (n_variants x n_features)
  descriptor_readme.md        what the descriptor is and how it is normalized
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import common as C


def build_selection() -> dict:
    variants = {}
    for preset in C.SELECTED_VARIANTS:
        ck = C.best_checkpoint_for(preset)
        desc = C.descriptor_for(preset)
        ck["descriptor"] = desc
        variants[preset] = ck
        C.log(f"selected {preset:18s} cell={ck['cell']:26s} "
              f"return={ck['expert_mean_return']:.0f} steps={ck['checkpoint_steps']:,}")

    selection = {
        "robot": C.ROBOT_KEY,
        "clip_motion": C.CLIP,
        "variants_order": C.SELECTED_VARIANTS,
        "variants": variants,
        "heldout_variant": C.HELDOUT_VARIANT,
        "train_variants": [v for v in C.SELECTED_VARIANTS if v != C.HELDOUT_VARIANT],
        "motions": {
            "note": (
                "Every trained G1 expert is a single-clip DeepMimic tracker for "
                f"'{C.CLIP}' (an expressive dance / turning motion). There is no second "
                "motion that all five variants share an expert for (only the nominal "
                "variant additionally has a 'dance2_subject2' expert), so the motion axis "
                "is effectively single. The scientifically meaningful generalization probe "
                "is therefore a HELD-OUT VARIANT (extreme_tall_light), not a held-out motion. "
                "A held-out-phase split of the same clip is used as a within-motion "
                "distribution-shift check in closed-loop eval."
            ),
            "train_motion": C.CLIP,
            "motion_kind": "expressive_dance",
            "heldout_motion_available": False,
        },
        "obs_dim": 450,
        "action_dim": 23,
        "horizon": C.HORIZON,
    }
    return selection


def build_descriptors(selection: dict, outdir: Path) -> None:
    order = selection["variants_order"]
    keys = C.DESCRIPTOR_KEYS
    raw = np.array([[selection["variants"][v]["descriptor"][k] for k in keys] for v in order],
                   dtype=np.float64)
    # z-score normalize each feature across the 5 variants; constant features -> 0.
    mean = raw.mean(axis=0)
    std = raw.std(axis=0)
    std_safe = np.where(std < 1e-8, 1.0, std)
    norm = (raw - mean) / std_safe
    norm[:, std < 1e-8] = 0.0

    np.save(outdir / "morphology_descriptors.npy", norm.astype(np.float32))
    C.write_json(outdir / "morphology_descriptors.json", {
        "variants_order": order,
        "feature_keys": keys,
        "raw": {v: dict(zip(keys, raw[i].tolist())) for i, v in enumerate(order)},
        "normalized": {v: dict(zip(keys, norm[i].tolist())) for i, v in enumerate(order)},
        "feature_mean": dict(zip(keys, mean.tolist())),
        "feature_std": dict(zip(keys, std.tolist())),
    })

    readme = [
        "# Morphology descriptors",
        "",
        "No URMA/URMAv2 descriptor code exists in this repo, so the descriptor is built",
        "directly from the variant generator's exposed morphology knobs (the only",
        "parameters that actually differ between these five G1 bodies).",
        "",
        "## Features (7)",
        "",
        "| feature | meaning |",
        "|---|---|",
        "| leg_length_scale | thigh/shin length multiplier |",
        "| arm_length_scale | upper/lower arm length multiplier |",
        "| shoulder_width_scale | shoulder offset multiplier |",
        "| foot_scale_x/y/z | foot geom size multipliers |",
        "| torso_mass_scale | torso mass multiplier |",
        "",
        "## Normalization",
        "",
        "Each feature is z-scored across the 5 selected variants (mean 0, std 1).",
        "Features that are constant across the set collapse to 0 (carry no signal).",
        "The normalized matrix is saved row-aligned with `variants_order`.",
        "",
        "## Raw values",
        "",
        "| variant | " + " | ".join(keys) + " |",
        "|" + "---|" * (len(keys) + 1),
    ]
    for i, v in enumerate(order):
        readme.append("| " + v + " | " + " | ".join(f"{raw[i,j]:.3f}" for j in range(len(keys))) + " |")
    (outdir / "descriptor_readme.md").write_text("\n".join(readme) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True, type=Path)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    if C.phase_done(args.outdir, "select"):
        C.log("phase select already done; skipping")
        return

    selection = build_selection()
    C.write_json(args.outdir / "selection.json", selection)
    build_descriptors(selection, args.outdir)
    C.log(f"wrote selection.json + descriptors to {args.outdir}")
    C.mark_done(args.outdir, "select")


if __name__ == "__main__":
    main()
