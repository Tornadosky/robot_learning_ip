"""Plot DeepMimic (PPOJax + MimicReward) training results.

Reads one or more `training_summary.json` files written by
`train_deepmimic_dance.py` and draws the mean-episode-return learning curves on a
single axis so multiple runs (e.g. different dance subjects) can be compared.
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


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summaries",
        nargs="+",
        type=Path,
        default=[
            WORKSPACE / "external_data" / "deepmimic_dance" / "training_summary.json",
            WORKSPACE / "external_data" / "deepmimic_dance_subject2" / "training_summary.json",
        ],
        help="training_summary.json files to overlay.",
    )
    parser.add_argument("--output", type=Path, default=WORKSPACE / "images" / "deepmimic_training_results.png")
    args = parser.parse_args()

    summaries = [(path, load_summary(path)) for path in args.summaries if path.exists()]
    if not summaries:
        raise SystemExit("No training_summary.json files found among: " + ", ".join(map(str, args.summaries)))

    fig, (ax_curve, ax_bar) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [2.1, 1]})
    colors = plt.cm.viridis(np.linspace(0.1, 0.8, len(summaries)))

    def run_label(summary: dict) -> str:
        # Two runs can share a clip name but differ in window length, so encode
        # the reference-window duration (seconds) to keep labels unambiguous.
        window_seconds = summary.get("window_frames", 0) / max(1.0, summary.get("frequency_hz", 1.0))
        return f"{summary['clip']} [{window_seconds:.0f}s win]"

    labels = []
    finals = []
    for (path, summary), color in zip(summaries, colors):
        curve = np.asarray(summary["return_curve_every_update"], dtype=float)
        # x axis is the training-progress fraction (0..1), so runs with a
        # different number of logged points still line up for comparison.
        x = np.linspace(0.0, 1.0, len(curve))
        label = run_label(summary)
        labels.append(run_label(summary))
        finals.append(summary["mean_episode_return_last"])
        ax_curve.plot(x * 100.0, curve, color=color, linewidth=2.0, label=label)
        ax_curve.annotate(
            f"{summary['mean_episode_return_last']:.0f}",
            xy=(100.0, curve[-1]),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=color,
        )

    ax_curve.set_title("DeepMimic dance tracking — learning curves")
    ax_curve.set_xlabel("Training progress (%)")
    ax_curve.set_ylabel("Mean episode return")
    ax_curve.grid(alpha=0.25)
    ax_curve.legend(loc="lower right", fontsize=9)

    # Right panel: final return per run as a quick at-a-glance bar chart.
    bar_colors = colors
    ax_bar.bar(range(len(finals)), finals, color=bar_colors)
    ax_bar.set_xticks(range(len(finals)))
    ax_bar.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax_bar.set_ylabel("Final mean episode return")
    ax_bar.set_title("Final tracking score")
    for i, value in enumerate(finals):
        ax_bar.text(i, value, f"{value:.0f}", ha="center", va="bottom", fontsize=9)
    ax_bar.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=130)
    print(f"Saved {args.output}")

    # Also dump a tiny machine-readable comparison next to the figure.
    comparison = {
        run_label(summary): {
            "robot": summary["robot"],
            "training_minutes": summary.get("training_minutes"),
            "mean_episode_return_first": summary.get("mean_episode_return_first"),
            "mean_episode_return_last": summary.get("mean_episode_return_last"),
            "mean_episode_length_last": summary.get("mean_episode_length_last"),
            "window_start_frame": summary.get("window_start_frame"),
            "window_frames": summary.get("window_frames"),
        }
        for _, summary in summaries
    }
    comparison_path = args.output.with_suffix(".json")
    comparison_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
