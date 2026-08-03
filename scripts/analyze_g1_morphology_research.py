"""Build the G1 morphology research table and summary plots.

For each main G1 morphology preset, scan every available training cell, select
the checkpoint with the highest mean episode return, and separately find the
fastest observed checkpoint that reached episode length 500. This keeps the
"best final policy" and "sample efficiency" questions distinct.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from g1_morphology_variants import PRESETS

WORKSPACE = Path(__file__).resolve().parents[1]
ROOT = WORKSPACE / "external_data" / "deepmimic_morphology" / "g1" / "dance2_subject4"
IMAGES = WORKSPACE / "images"

ORDER = [
    "nominal",
    "tall_legs",
    "short_legs",
    "long_arms",
    "broad_shoulders",
    "big_feet",
    "heavy_torso",
    "combined",
    "extreme_tall_light",
    "extreme_short_heavy",
    "extreme_combined",
]

SHORT_LABELS = {
    "nominal": "Nominal",
    "tall_legs": "Tall legs",
    "short_legs": "Short legs",
    "long_arms": "Long arms",
    "broad_shoulders": "Broad shoulders",
    "big_feet": "Big feet",
    "heavy_torso": "Heavy torso",
    "combined": "Combined",
    "extreme_tall_light": "Extreme tall/light",
    "extreme_short_heavy": "Extreme short/heavy",
    "extreme_combined": "Extreme combined",
}

CROSS_LEN = 500.0


def load_manifests() -> dict[str, list[tuple[str, dict]]]:
    grouped = {preset: [] for preset in ORDER}
    for path in ROOT.glob("*/manifest.json"):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        preset = manifest.get("preset")
        if preset in grouped:
            grouped[preset].append((path.parent.name, manifest))
    return grouped


def best_checkpoint(manifest: dict) -> dict:
    return max(manifest["checkpoints"], key=lambda c: c["mean_episode_return"])


def recipe_name(manifest: dict) -> str:
    gain = manifest.get("pd_gain_scale")
    lr = manifest.get("lr")
    steps_m = manifest.get("total_timesteps", 0) / 1e6
    suffix = manifest.get("out_suffix") or "stock"
    gain_text = f"{gain:g}x PD" if gain is not None else manifest.get("control", "legacy")
    lr_text = f"{lr:g}" if lr is not None else "legacy"
    return f"{suffix}: {gain_text}, lr={lr_text}, {steps_m:.0f}M"


def first_cross(manifest: dict) -> dict | None:
    return next(
        (ck for ck in manifest["checkpoints"] if ck["mean_episode_length"] >= CROSS_LEN),
        None,
    )


def build_rows(grouped: dict[str, list[tuple[str, dict]]]) -> tuple[list[dict], dict[str, dict]]:
    rows = []
    selected: dict[str, dict] = {}
    for preset in ORDER:
        candidates = grouped[preset]
        if not candidates:
            rows.append({
                "preset": preset,
                "label": PRESETS[preset].label,
                "status": "missing",
            })
            continue

        best_cell, best_manifest = max(
            candidates,
            key=lambda item: best_checkpoint(item[1])["mean_episode_return"],
        )
        best = best_checkpoint(best_manifest)
        crossings = [
            (cross["cumulative_steps"], cell, cross)
            for cell, manifest in candidates
            if (cross := first_cross(manifest)) is not None
        ]
        fastest = min(crossings, default=None, key=lambda item: item[0])
        max_budget = max(manifest["total_timesteps"] for _, manifest in candidates)
        preset_conf = PRESETS[preset]

        row = {
            "preset": preset,
            "label": preset_conf.label,
            "status": "crossed" if fastest else "not_reached",
            "best_cell": best_cell,
            "best_recipe": recipe_name(best_manifest),
            "best_checkpoint_steps": best["cumulative_steps"],
            "best_return": best["mean_episode_return"],
            "best_length": best["mean_episode_length"],
            "final_return_of_best_cell": best_manifest["mean_episode_return_last"],
            "final_length_of_best_cell": best_manifest["mean_episode_length_last"],
            "return_drop_after_best": (
                best["mean_episode_return"] - best_manifest["mean_episode_return_last"]
            ),
            "fastest_steps_to_len500": fastest[0] if fastest else None,
            "fastest_cross_cell": fastest[1] if fastest else None,
            "max_tested_steps": max_budget,
            "leg_length_scale": preset_conf.leg_length_scale,
            "arm_length_scale": preset_conf.arm_length_scale,
            "shoulder_width_scale": preset_conf.shoulder_width_scale,
            "foot_scale_x": preset_conf.foot_scale_xyz[0],
            "foot_scale_y": preset_conf.foot_scale_xyz[1],
            "foot_scale_z": preset_conf.foot_scale_xyz[2],
            "torso_mass_scale": preset_conf.torso_mass_scale,
        }
        rows.append(row)
        selected[preset] = {
            "cell": best_cell,
            "manifest": best_manifest,
            "checkpoint": best,
        }
    return rows, selected


def write_table(rows: list[dict]) -> tuple[Path, Path]:
    IMAGES.mkdir(parents=True, exist_ok=True)
    csv_path = IMAGES / "g1_morphology_research_summary.csv"
    json_path = IMAGES / "g1_morphology_research_summary.json"
    fields = list(rows[0])
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return csv_path, json_path


def plot_summary(rows: list[dict]) -> Path:
    valid = [row for row in rows if row["status"] != "missing"]
    labels = [SHORT_LABELS[row["preset"]] for row in valid]
    x = np.arange(len(valid))
    colors = plt.cm.tab20(np.linspace(0.0, 0.9, len(valid)))

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    returns = [row["best_return"] for row in valid]
    axes[0].bar(x, returns, color=colors)
    axes[0].set_ylabel("Best mean return")
    axes[0].set_title("G1 morphology: best trained policy")
    axes[0].grid(axis="y", alpha=0.25)
    for i, value in enumerate(returns):
        axes[0].text(i, value + 10, f"{value:.0f}", ha="center", fontsize=8)

    lengths = [row["best_length"] for row in valid]
    axes[1].bar(x, lengths, color=colors)
    axes[1].axhline(CROSS_LEN, color="#c43d3d", linestyle="--", linewidth=1.5,
                    label="tracking threshold (len 500)")
    axes[1].set_ylabel("Best episode length")
    axes[1].set_ylim(0, 900)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(loc="upper right")
    for i, value in enumerate(lengths):
        axes[1].text(i, value + 15, f"{value:.0f}", ha="center", fontsize=8)

    crossing_m = []
    bar_colors = []
    for row, color in zip(valid, colors):
        if row["fastest_steps_to_len500"] is not None:
            crossing_m.append(row["fastest_steps_to_len500"] / 1e6)
            bar_colors.append(color)
        else:
            crossing_m.append(row["max_tested_steps"] / 1e6)
            bar_colors.append("#b8b8b8")
    axes[2].bar(x, crossing_m, color=bar_colors)
    axes[2].set_ylabel("Steps to len 500 (millions)")
    axes[2].set_title("Sample efficiency; gray bars did not reach len 500")
    axes[2].grid(axis="y", alpha=0.25)
    for i, (row, value) in enumerate(zip(valid, crossing_m)):
        text = f"{value:.0f}M" if row["fastest_steps_to_len500"] is not None else f">{value:.0f}M"
        axes[2].text(i, value + 15, text, ha="center", fontsize=8)

    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=28, ha="right")
    fig.tight_layout()
    out = IMAGES / "g1_morphology_research_metrics.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_learning_curves(selected: dict[str, dict]) -> Path:
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = plt.cm.tab20(np.linspace(0.0, 0.9, len(selected)))
    for (preset, item), color in zip(selected.items(), colors):
        manifest = item["manifest"]
        curve = np.asarray(manifest["return_curve_every_update"], dtype=float)
        start_steps = manifest.get("resume_steps", 0)
        steps = np.linspace(start_steps / 1e6, manifest["total_timesteps"] / 1e6, len(curve))
        ax.plot(steps, curve, linewidth=1.8, color=color, label=SHORT_LABELS[preset])
    ax.set_xlabel("Environment steps (millions)")
    ax.set_ylabel("Mean episode return")
    ax.set_title("Best available G1 training cell per morphology")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    out = IMAGES / "g1_morphology_research_learning_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    grouped = load_manifests()
    rows, selected = build_rows(grouped)
    csv_path, json_path = write_table(rows)
    metrics_plot = plot_summary(rows)
    curves_plot = plot_learning_curves(selected)

    print(json.dumps({
        "csv": str(csv_path),
        "json": str(json_path),
        "metrics_plot": str(metrics_plot),
        "learning_curves_plot": str(curves_plot),
        "selected_cells": {
            preset: item["cell"] for preset, item in selected.items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
