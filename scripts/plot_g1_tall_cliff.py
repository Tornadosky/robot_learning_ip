"""Plot the G1 'tall & light' controllability cliff.

x-axis = leg-length scale (taller to the right); y-axis = best episode length
reached (out of horizon 1000). Points are green where G1 crossed the balance
transition and tracks the dance, red where it stays stuck. Anchored by the known
bodies (short&heavy 0.66x, nominal 1.0x, extreme_tall_light 1.55x) plus the swept
tall&light family (1.20 / 1.35 / 1.50x). The drop-off marks the cliff.

A second panel overlays the learning curves that have a saved manifest (the
crossers + anchors) so the qualitative difference -- sharp rise vs flat crawl --
is visible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WORKSPACE = Path(__file__).resolve().parents[1]
ROOT = WORKSPACE / "external_data" / "deepmimic_morphology" / "g1" / "dance2_subject4"
PROPER_LEVEL_LEN = 600.0

# Anchor bodies: (leg_scale, label, cell_dir). Best-achieved cell for each.
ANCHORS = [
    (0.66, "short&heavy", "extreme_short_heavy"),
    (1.00, "nominal", "nominal__stiff3_lr3"),
    (1.55, "extreme tall&light", "extreme_tall_light"),
]
# Swept tall&light family: (leg_scale, label, preset). Crossers finalize to
# {preset}__cliff; stuck ones live only in the summary json.
SWEEP = [
    (1.20, "tall&light 1.20x", "tall_light_leg120"),
    (1.35, "tall&light 1.35x", "tall_light_leg135"),
    (1.50, "tall&light 1.50x", "tall_light_leg150"),
]


def best_checkpoint(manifest: dict) -> dict:
    return max(manifest["checkpoints"], key=lambda c: c["mean_episode_return"])


def load_point(cell: str):
    path = ROOT / cell / "manifest.json"
    if not path.exists():
        return None
    m = json.loads(path.read_text(encoding="utf-8"))
    ck = best_checkpoint(m)
    return m, ck["mean_episode_length"], ck["mean_episode_return"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=WORKSPACE / "images" / "g1_tall_cliff.png")
    args = parser.parse_args()

    summary = {}
    spath = ROOT / "tall_cliff_summary.json"
    if spath.exists():
        summary = json.loads(spath.read_text(encoding="utf-8"))

    # Assemble (leg_scale, label, length, crossed, manifest|None).
    points = []
    for leg, label, cell in ANCHORS:
        got = load_point(cell)
        if got:
            m, length, _ = got
            points.append((leg, label, length, length >= PROPER_LEVEL_LEN, m))
    for leg, label, preset in SWEEP:
        got = load_point(f"{preset}__cliff")
        if got:
            m, length, _ = got
            points.append((leg, label, length, length >= PROPER_LEVEL_LEN, m))
        elif preset in summary:
            s = summary[preset]
            points.append((leg, label, s.get("best_length", 0.0),
                           s.get("status") == "crossed", None))
    points.sort(key=lambda p: p[0])

    fig, (ax_cliff, ax_curve) = plt.subplots(1, 2, figsize=(14, 5.4),
                                             gridspec_kw={"width_ratios": [1.15, 1]})

    legs = [p[0] for p in points]
    lengths = [p[2] for p in points]
    crossed = [p[3] for p in points]
    ax_cliff.plot(legs, lengths, color="0.6", linewidth=1.5, zorder=1)
    for leg, label, length, cross, _ in points:
        ax_cliff.scatter([leg], [length], s=120, zorder=3,
                         color="#2c7", edgecolor="k" if cross else "none") if cross else \
        ax_cliff.scatter([leg], [length], s=120, zorder=3, color="#d44", marker="X")
        ax_cliff.annotate(f"{label}\nlen {length:.0f}", xy=(leg, length),
                          xytext=(0, 10), textcoords="offset points",
                          ha="center", fontsize=7.5)
    ax_cliff.axhspan(PROPER_LEVEL_LEN, 1000, color="#2c7", alpha=0.08)
    ax_cliff.axhline(PROPER_LEVEL_LEN, color="#2c7", ls="--", lw=1, alpha=0.6)
    ax_cliff.text(min(legs), PROPER_LEVEL_LEN + 12, "proper tracking", fontsize=8, color="#2a7")
    ax_cliff.set_title("G1 tall&light controllability cliff")
    ax_cliff.set_xlabel("Leg-length scale (taller →)")
    ax_cliff.set_ylabel("Best episode length (of 1000)")
    ax_cliff.grid(alpha=0.25)
    ax_cliff.set_ylim(0, 1000)

    # Learning curves for points that have a manifest.
    curve_points = [p for p in points if p[4] is not None]
    colors = plt.cm.viridis(np.linspace(0.08, 0.85, len(curve_points)))
    for (leg, label, length, cross, m), color in zip(curve_points, colors):
        curve = np.asarray(m["return_curve_every_update"], dtype=float)
        x = np.linspace(0.0, 100.0, len(curve))
        ax_curve.plot(x, curve, color=color, lw=2.0,
                      label=f"{label} ({'cross' if cross else 'stuck'})")
    ax_curve.set_title("Learning curves (crossers rise, stuck stay flat)")
    ax_curve.set_xlabel("Training progress (%)")
    ax_curve.set_ylabel("Mean episode return")
    ax_curve.grid(alpha=0.25)
    ax_curve.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=130)
    print(f"[plot] saved {args.output}")

    out = {f"leg_{leg:.2f}": {"label": label, "length": length, "crossed": cross}
           for leg, label, length, cross, _ in points}
    args.output.with_suffix(".json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
