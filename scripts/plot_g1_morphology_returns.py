"""Plot DeepMimic learning curves + final tracking score across G1 morphologies.

Reads per-cell ``manifest.json`` files written by ``train_deepmimic_morphology.py``
and overlays the mean-episode-return learning curves (x = training progress %, so
runs of different budgets line up) plus a bar chart of the best tracking score per
morphology -- making the impact of body shape on G1 trainability visible.

Two cell sets (``--mode``):
  best   : the best policy achieved for each body, across recipes. Only the
           compact bodies (nominal, short&heavy) reach "proper level"; the
           lengthened-leg bodies plateau regardless of PD-gain scale.
  stock  : the uniform baseline recipe (stock PD gains, 300M) for all five
           bodies -- an apples-to-apples morphology comparison.

G1 counterpart of the H1 morphology comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write a PNG, never open a window
import matplotlib.pyplot as plt
import numpy as np

WORKSPACE = Path(__file__).resolve().parents[1]
CELL_ROOT = WORKSPACE / "external_data" / "deepmimic_morphology" / "g1" / "dance2_subject4"

# Plot/display order + human labels.
PRESET_LABELS = {
    "nominal": "Nominal G1",
    "extreme_short_heavy": "Extreme: short & heavy",
    "combined": "Combined moderate",
    "extreme_tall_light": "Extreme: tall & light",
    "extreme_combined": "Extreme: combined",
}

# Best policy achieved per body (cell dir name under CELL_ROOT). Nominal needs the
# 3x-gain fix; every other body's best is its stock-gain 300M cell (stiffer gains
# destabilise the lengthened-leg bodies and underperform on short&heavy).
CELLS_BEST = {
    "nominal": "nominal__stiff3_lr3",
    "extreme_short_heavy": "extreme_short_heavy",
    "combined": "combined",
    "extreme_tall_light": "extreme_tall_light",
    "extreme_combined": "extreme_combined",
}

# Uniform baseline recipe (stock PD gains, 300M) for a fair morphology comparison.
CELLS_STOCK = {preset: preset for preset in PRESET_LABELS}

# Episode length (out of horizon 1000) above which the policy is "tracking the
# dance", not just surviving a moment -- used only to annotate the plot.
PROPER_LEVEL_LEN = 600.0


def best_checkpoint(manifest: dict) -> dict:
    return max(manifest["checkpoints"], key=lambda c: c["mean_episode_return"])


def recipe_tag(manifest: dict) -> str:
    scale = manifest.get("pd_gain_scale")
    steps = manifest.get("total_timesteps", 0) / 1e6
    gain = f"{scale:g}x gains" if scale else "stock gains"
    return f"{gain}, {steps:.0f}M"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["best", "stock"], default="best")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    cell_map = CELLS_BEST if args.mode == "best" else CELLS_STOCK
    output = args.output or WORKSPACE / "images" / f"g1_morphology_returns_{args.mode}.png"

    cells = []
    for preset in PRESET_LABELS:
        path = CELL_ROOT / cell_map[preset] / "manifest.json"
        if not path.exists():
            print(f"[plot] skip {preset}: no manifest at {path}")
            continue
        cells.append((preset, json.loads(path.read_text(encoding="utf-8"))))
    if not cells:
        raise SystemExit("No manifests found to plot.")

    fig, (ax_curve, ax_bar) = plt.subplots(
        1, 2, figsize=(13.5, 5.2), gridspec_kw={"width_ratios": [2.1, 1]}
    )
    colors = plt.cm.viridis(np.linspace(0.08, 0.85, len(cells)))

    title_mode = "best policy per body" if args.mode == "best" else "uniform stock-gain recipe"

    labels, best_returns, best_lengths = [], [], []
    for (preset, manifest), color in zip(cells, colors):
        curve = np.asarray(manifest["return_curve_every_update"], dtype=float)
        x = np.linspace(0.0, 100.0, len(curve))
        ck = best_checkpoint(manifest)
        label = PRESET_LABELS[preset]
        if args.mode == "best":
            label = f"{label} ({recipe_tag(manifest)})"
        ax_curve.plot(x, curve, color=color, linewidth=2.0, label=label)
        ax_curve.annotate(f"{curve[-1]:.0f}", xy=(100.0, curve[-1]),
                          xytext=(4, 0), textcoords="offset points",
                          va="center", fontsize=8, color=color)
        labels.append(PRESET_LABELS[preset])
        best_returns.append(ck["mean_episode_return"])
        best_lengths.append(ck["mean_episode_length"])

    ax_curve.set_title(f"G1 DeepMimic dance tracking by morphology — {title_mode}")
    ax_curve.set_xlabel("Training progress (%)")
    ax_curve.set_ylabel("Mean episode return")
    ax_curve.grid(alpha=0.25)
    ax_curve.legend(loc="upper left", fontsize=8)

    order = np.argsort(best_returns)[::-1]
    bar_colors = [colors[i] for i in order]
    bars = ax_bar.bar(range(len(order)), [best_returns[i] for i in order], color=bar_colors)
    ax_bar.set_xticks(range(len(order)))
    ax_bar.set_xticklabels([labels[i] for i in order], rotation=25, ha="right", fontsize=8)
    ax_bar.set_ylabel("Best mean episode return")
    ax_bar.set_title("Best tracking score per morphology")
    for rank, i in enumerate(order):
        ax_bar.text(rank, best_returns[i], f"{best_returns[i]:.0f}\nlen {best_lengths[i]:.0f}",
                    ha="center", va="bottom", fontsize=7)
    ax_bar.grid(alpha=0.25, axis="y")
    ax_bar.margins(y=0.18)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=130)
    print(f"[plot] saved {output}")

    summary = {}
    for preset, manifest in cells:
        ck = best_checkpoint(manifest)
        summary[preset] = {
            "label": PRESET_LABELS[preset],
            "cell": cell_map[preset],
            "recipe": recipe_tag(manifest),
            "pd_gain_scale": manifest.get("pd_gain_scale"),
            "best_return": ck["mean_episode_return"],
            "best_length": ck["mean_episode_length"],
            "proper_level": ck["mean_episode_length"] >= PROPER_LEVEL_LEN,
        }
    out_json = output.with_suffix(".json")
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
