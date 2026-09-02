#!/usr/bin/env python3
"""Aggregate crossevaluation JSON files into compact tables."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def mean(values):
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(vals) / len(vals) if vals else None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("eval_dir", type=Path)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    files = sorted(args.eval_dir.glob("*.json"))
    records = []
    worst = defaultdict(list)
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        cond = data.get("eval_condition", {})
        for robot, metrics in data.get("robots", {}).items():
            record = {
                "file": path.name,
                "robot": robot,
                "seed": cond.get("seed", data.get("seed")),
                "morphology_coeff": cond.get("morphology_coeff", 0.0),
                "zero_action": bool(cond.get("zero_action", False)),
                "joint_rmse_absolute_rad": metrics.get("raw_rmse_rad_absolute"),
                "joint_rmse_shape_rad": metrics.get("raw_rmse_rad"),
                "reference_floor_absolute_rad": metrics.get("reference_vs_raw_rmse_rad_absolute"),
                "heading_error_deg_mean": metrics.get("heading_error_deg_mean"),
                "heading_error_deg_p95": metrics.get("heading_error_deg_p95"),
                "alive_fraction": metrics.get("alive_fraction"),
            }
            for key, value in metrics.get("environment_tracking_metrics", {}).items():
                record[f"env_{key}"] = value
            for key, value in metrics.get("foot_metrics", {}).items():
                record[f"foot_{key}"] = value
            records.append(record)
            for name, value in metrics.get("per_joint_rmse_rad_absolute", {}).items():
                worst[(robot, cond.get("morphology_coeff", 0.0), bool(cond.get("zero_action", False)))].append((float(value), name, path.name))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    columns = sorted({k for r in records for k in r})
    preferred = ["file", "robot", "seed", "morphology_coeff", "zero_action", "joint_rmse_absolute_rad", "joint_rmse_shape_rad", "reference_floor_absolute_rad", "heading_error_deg_mean", "heading_error_deg_p95", "alive_fraction"]
    columns = preferred + [c for c in columns if c not in preferred]
    with (args.out_dir / "evaluation_records.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader(); w.writerows(records)

    grouped = defaultdict(list)
    for r in records:
        grouped[(r["robot"], r["morphology_coeff"], r["zero_action"])].append(r)
    summary = []
    metric_keys = [k for k in columns if k not in ("file", "robot", "seed", "morphology_coeff", "zero_action")]
    for (robot, morph, zero), rows in sorted(grouped.items()):
        item = {"robot": robot, "morphology_coeff": morph, "zero_action": zero, "runs": len(rows)}
        for key in metric_keys:
            m = mean([r.get(key) for r in rows])
            if m is not None:
                item[f"mean_{key}"] = m
        summary.append(item)
    (args.out_dir / "evaluation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    worst_out = {}
    for key, vals in worst.items():
        vals.sort(reverse=True)
        worst_out[f"{key[0]}|morph={key[1]}|zero={key[2]}"] = [
            {"joint": name, "rmse_rad": value, "source": source}
            for value, name, source in vals[:12]
        ]
    (args.out_dir / "worst_joints.json").write_text(json.dumps(worst_out, indent=2, sort_keys=True), encoding="utf-8")

    lines = ["# Evaluation summary", ""]
    for item in summary:
        lines.append(
            "- {robot}, morphology={morphology_coeff}, zero_action={zero_action}, runs={runs}: "
            "joint RMSE={rmse}, heading={heading}, alive={alive}".format(
                **item,
                rmse=("n/a" if "mean_joint_rmse_absolute_rad" not in item else f"{item['mean_joint_rmse_absolute_rad']:.4f} rad"),
                heading=("n/a" if "mean_heading_error_deg_mean" not in item else f"{item['mean_heading_error_deg_mean']:.1f} deg"),
                alive=("n/a" if "mean_alive_fraction" not in item else f"{item['mean_alive_fraction']:.1%}"),
            )
        )
    (args.out_dir / "evaluation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"files": len(files), "records": len(records), "groups": len(summary)}, indent=2))


if __name__ == "__main__":
    main()
