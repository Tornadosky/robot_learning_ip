#!/usr/bin/env python3
"""Create metric-focused training plots from parse_training_log.py output."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

FAMILIES = ("h1", "g1", "t1")


def load_rows(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, float] = {}
            for key, value in raw.items():
                if key == "timestamp" or not value:
                    continue
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(number):
                    row[key] = number
            if "steps/nr_env_steps" in row:
                rows.append(row)
    return rows


def series(rows: list[dict[str, float]], key: str) -> tuple[list[float], list[float]]:
    x, y = [], []
    for row in rows:
        if key in row:
            x.append(row["steps/nr_env_steps"])
            y.append(row[key])
    return x, y


def plot_family_metric(rows, key_template: str, title: str, ylabel: str, out: Path) -> bool:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.2))
    drawn = False
    for family in FAMILIES:
        key = key_template.format(family=family)
        x, y = series(rows, key)
        if x:
            ax.plot(x, y, label=family.upper())
            drawn = True
    if not drawn:
        plt.close(fig)
        return False
    ax.set_title(title)
    ax.set_xlabel("aggregate environment steps")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return True


def plot_multi(rows, specs: list[tuple[str, str]], title: str, ylabel: str, out: Path) -> bool:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.2))
    drawn = False
    for key, label in specs:
        x, y = series(rows, key)
        if x:
            ax.plot(x, y, label=label)
            drawn = True
    if not drawn:
        plt.close(fig)
        return False
    ax.set_title(title)
    ax.set_xlabel("aggregate environment steps")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return True


def write_markdown(rows: list[dict[str, float]], created: list[str], out: Path) -> None:
    last = rows[-1]
    lines = [
        "# Training metric summary",
        "",
        f"Last parsed aggregate step: `{int(last['steps/nr_env_steps']):,}`",
        "",
        "| Family | Joint RMSE (rad) | Normalized qvel RMSE | Heading error (deg) | rpos error | rquat error |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for family in FAMILIES:
        def val(key: str) -> str:
            number = last.get(key)
            return "n/a" if number is None else f"{number:.5f}"
        lines.append(
            "| {name} | {joint} | {qvel} | {heading} | {rpos} | {rquat} |".format(
                name=family.upper(),
                joint=val(f"derived/joint_rmse/{family}"),
                qvel=val(f"derived/normalized_qvel_rmse/{family}"),
                heading=val(f"derived/root_heading_error_deg/{family}"),
                rpos=val(f"env_info/rpos_tracking_error/{family}"),
                rquat=val(f"env_info/rquat_tracking_error/{family}"),
            )
        )
    lines.extend(["", "Generated plots:"])
    lines.extend(f"- `{name}`" for name in created)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics_csv", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = load_rows(args.metrics_csv)
    if not rows:
        raise SystemExit("no metric rows found")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    created: list[str] = []

    plots = [
        (plot_family_metric, (rows, "derived/joint_rmse/{family}", "Joint tracking RMSE", "RMSE (rad)", args.out_dir / "joint_rmse.png")),
        (plot_family_metric, (rows, "derived/normalized_qvel_rmse/{family}", "Normalized joint-velocity RMSE", "normalized RMSE", args.out_dir / "qvel_rmse.png")),
        (plot_family_metric, (rows, "derived/root_heading_error_deg/{family}", "Absolute root-heading error", "mean error (degrees)", args.out_dir / "root_heading_error_deg.png")),
        (plot_family_metric, (rows, "env_info/rpos_tracking_error/{family}", "DeepMimic relative body-position error", "mean normalized squared error", args.out_dir / "body_rpos_error.png")),
        (plot_family_metric, (rows, "env_info/rquat_tracking_error/{family}", "DeepMimic body-orientation error", "mean quaternion distance", args.out_dir / "body_rquat_error.png")),
        (plot_family_metric, (rows, "env_info/foot_height_error/{family}", "Foot-height tracking error", "mean normalized squared error", args.out_dir / "foot_height_error.png")),
        (plot_family_metric, (rows, "env_info/foot_penetration_m/{family}", "Foot penetration", "metres", args.out_dir / "foot_penetration.png")),
        (plot_family_metric, (rows, "env_info/foot_slip_speed_sq/{family}", "Foot slip", "squared speed", args.out_dir / "foot_slip.png")),
        (plot_family_metric, (rows, "env_curriculum/morphology_coeff/{family}", "Morphology-randomization coefficient", "coefficient", args.out_dir / "morphology_coeff.png")),
        (plot_family_metric, (rows, "rollout/episode_return/{family}", "Episode return (context only)", "return", args.out_dir / "episode_return.png")),
        (plot_family_metric, (rows, "rollout/episode_length/{family}", "Episode length (context only)", "steps", args.out_dir / "episode_length.png")),
    ]
    for fn, fn_args in plots:
        if fn(*fn_args):
            created.append(Path(fn_args[-1]).name)

    health_specs = [
        ("policy_ratio/approx_kl", "approx KL"),
        ("policy_ratio/clip_fraction", "clip fraction"),
        ("policy/std_dev", "policy std dev"),
        ("v_value/explained_variance", "explained variance"),
    ]
    path = args.out_dir / "ppo_health.png"
    if plot_multi(rows, health_specs, "PPO health indicators", "value", path):
        created.append(path.name)

    grad_specs = [
        ("gradients/policy_grad_norm", "policy grad norm"),
        ("gradients/critic_grad_norm", "critic grad norm"),
        ("loss/critic_loss", "critic loss"),
        ("loss/policy_gradient_loss", "policy gradient loss"),
    ]
    path = args.out_dir / "optimization_health.png"
    if plot_multi(rows, grad_specs, "Optimization health", "value", path):
        created.append(path.name)

    write_markdown(rows, created, args.out_dir / "training_summary.md")
    (args.out_dir / "plot_manifest.json").write_text(
        json.dumps({"plots": created, "rows": len(rows)}, indent=2), encoding="utf-8"
    )
    print(json.dumps({"plots": created, "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
