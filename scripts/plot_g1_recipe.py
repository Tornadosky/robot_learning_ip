"""Overlay DeepMimic learning curves for several G1 training recipes (+ H1 ref).

Unlike plot_deepmimic_results.py (which labels runs by clip+window and so collides
when several runs share a clip), this takes explicit `path:label` pairs so different
hyperparameter recipes on the same clip can be compared directly. Plots both the
mean-episode-return and the mean-episode-length curves, since the G1 story is about
the episode-length "balance transition", not just return.
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


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True,
                        help="manifest.json:Label pairs (label optional).")
    parser.add_argument("--title", default="G1 nominal — DeepMimic dance tracking recipes")
    parser.add_argument("--output", type=Path, default=WORKSPACE / "images" / "g1_recipe_comparison.png")
    args = parser.parse_args()

    runs = []
    for spec in args.runs:
        # Split on the LAST colon so Windows drive letters (C:\) survive.
        path_str, _, label = spec.rpartition(":")
        if not path_str:  # no colon -> whole thing is a path
            path_str, label = spec, ""
        path = Path(path_str)
        summary = load(path)
        if not label:
            label = f"{summary.get('preset','?')} {summary.get('control','?')}"
        runs.append((label, summary))

    fig, (ax_r, ax_l) = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.turbo(np.linspace(0.08, 0.92, len(runs)))

    for (label, s), color in zip(runs, colors):
        curve = np.asarray(s["return_curve_every_update"], dtype=float)
        x = np.linspace(0.0, s.get("total_timesteps", len(curve)) / 1e6, len(curve))
        ax_r.plot(x, curve, color=color, lw=2.0, label=label)
        ax_r.annotate(f"{s['mean_episode_return_last']:.0f}", xy=(x[-1], curve[-1]),
                      xytext=(3, 0), textcoords="offset points", va="center",
                      fontsize=8, color=color)
        # Episode-length progression from the per-checkpoint records.
        ck = s.get("checkpoints", [])
        if ck:
            cx = [c["cumulative_steps"] / 1e6 for c in ck]
            cl = [c["mean_episode_length"] for c in ck]
            ax_l.plot(cx, cl, "-o", color=color, lw=2.0, ms=4, label=label)

    ax_r.set_title(args.title)
    ax_r.set_xlabel("Training steps (millions)")
    ax_r.set_ylabel("Mean episode return")
    ax_r.grid(alpha=0.25)
    ax_r.legend(loc="upper left", fontsize=8)

    ax_l.set_title("Episode length (balance) — higher = stays upright longer")
    ax_l.set_xlabel("Training steps (millions)")
    ax_l.set_ylabel("Mean episode length (of 1000)")
    ax_l.axhline(1000, color="gray", ls="--", lw=1, alpha=0.5)
    ax_l.grid(alpha=0.25)
    ax_l.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=130)
    print(f"Saved {args.output}")

    comparison = {
        label: {
            "control": s.get("control"), "lr": s.get("lr"), "init_std": s.get("init_std"),
            "pd_gain_scale": s.get("pd_gain_scale"), "total_timesteps": s.get("total_timesteps"),
            "return_last": s.get("mean_episode_return_last"),
            "length_last": s.get("mean_episode_length_last"),
            "training_minutes": s.get("training_minutes"),
        }
        for label, s in runs
    }
    args.output.with_suffix(".json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
