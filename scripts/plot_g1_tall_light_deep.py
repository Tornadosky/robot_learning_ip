"""Plot the G1 extreme_tall_light 'deep training' result: the slow crawl that
was mistaken for a controllability cliff actually crosses the balance transition
with ~3x the steps. Old 300M run plateaued at len 284; the 1B run climbs to 596.

No MuJoCo / GL needed -- reads the training manifests directly.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent / "external_data" / "deepmimic_morphology" / "g1" / "dance2_subject4"
IMAGES = Path(__file__).resolve().parent.parent / "images"


def curve(cell: str):
    m = json.loads((ROOT / cell / "manifest.json").read_text())
    cks = m["checkpoints"]
    x = [c["cumulative_steps"] / 1e6 for c in cks]
    ln = [c["mean_episode_length"] for c in cks]
    rt = [c["mean_episode_return"] for c in cks]
    return x, ln, rt


def main() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Old, under-trained 300M run (the "cliff" evidence).
    x0, l0, r0 = curve("extreme_tall_light")
    # Deep 1B run (crossed).
    x1, l1, r1 = curve("extreme_tall_light__deep1b")
    # Optional: the in-progress / finished 1.8B extension.
    deepbig = ROOT / "extreme_tall_light__deepbig" / "manifest.json"
    has_big = deepbig.exists()
    if has_big:
        x2, l2, r2 = curve("extreme_tall_light__deepbig")

    for ax, key, (yl0, yl1) in [(ax1, "length", (l0, l1)), (ax2, "return", (r0, r1))]:
        ax.plot(x0, yl0, "o-", color="#c0392b", label="old 300M run (under-trained)")
        ax.plot(x1, yl1, "o-", color="#27ae60", label="deep 1B run (crossed)")
        if has_big:
            yb = l2 if key == "length" else r2
            ax.plot(x2, yb, "s-", color="#2980b9", alpha=0.8, label="1.8B extension")
        ax.set_xlabel("environment steps (millions)")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=9)

    ax1.axhline(284, ls="--", color="#c0392b", alpha=0.6)
    ax1.text(20, 296, "old plateau (284)", color="#c0392b", fontsize=8)
    ax1.axhline(731, ls=":", color="gray", alpha=0.7)
    ax1.text(20, 742, "nominal G1 ceiling (731)", color="gray", fontsize=8)
    ax1.set_ylabel("mean episode length (of 1000)")
    ax1.set_title("G1 extreme tall&light: episode length")

    ax2.set_ylabel("mean episode return")
    ax2.set_title("G1 extreme tall&light: tracking return")

    fig.suptitle("Extreme tall&light G1 is trainable after all — it was under-trained, not at a control cliff",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = IMAGES / "g1_tall_light_deep_curve.png"
    fig.savefig(out, dpi=130)
    print(f"[plot] wrote {out}")


if __name__ == "__main__":
    main()
