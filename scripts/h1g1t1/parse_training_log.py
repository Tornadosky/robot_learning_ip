#!/usr/bin/env python3
"""Parse RL-X/URMA2 console tables into machine-readable training evidence."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

TABLE_RE = re.compile(r"^\[(?P<stamp>\d{2}-\d{2} \d{2}:\d{2}:\d{2})\].*?INFO - │(?P<key>.*?)│\s*(?P<value>.*?)\s*│\s*$")
FATAL_RE = re.compile(
    r"(?:Traceback \(most recent call last\)|Uncaught exception|CUDA_ERROR|ROCM_ERROR|"
    r"RESOURCE_EXHAUSTED|out of memory|nan detected|FloatingPointError|Segmentation fault)",
    re.IGNORECASE,
)
FAMILIES = ("h1", "g1", "t1")


def _number(text: str) -> float | None:
    text = text.strip().replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _derived(row: dict[str, float | str]) -> None:
    for family in FAMILIES:
        joint = row.get(f"env_info/joint_tracking_error/{family}")
        if isinstance(joint, (int, float)) and math.isfinite(float(joint)):
            row[f"derived/joint_rmse/{family}"] = math.sqrt(max(float(joint), 0.0))
        qvel = row.get(f"env_info/qvel_tracking_error/{family}")
        if isinstance(qvel, (int, float)) and math.isfinite(float(qvel)):
            row[f"derived/normalized_qvel_rmse/{family}"] = math.sqrt(max(float(qvel), 0.0))
        heading = row.get(f"env_info/root_heading_error/{family}")
        if isinstance(heading, (int, float)) and math.isfinite(float(heading)):
            row[f"derived/root_heading_error_deg/{family}"] = math.degrees(float(heading))


def parse_lines(lines: Iterable[str]) -> list[dict[str, float | str]]:
    """Return one row per timestamped metric table containing nr_env_steps."""
    groups: "OrderedDict[str, dict[str, float | str]]" = OrderedDict()
    for line in lines:
        match = TABLE_RE.match(line.rstrip("\n"))
        if not match:
            continue
        value = _number(match.group("value"))
        if value is None:
            continue
        stamp = match.group("stamp")
        row = groups.setdefault(stamp, {"timestamp": stamp})
        row[match.group("key").strip()] = value

    rows: list[dict[str, float | str]] = []
    for row in groups.values():
        if "steps/nr_env_steps" not in row:
            continue
        _derived(row)
        rows.append(row)
    rows.sort(key=lambda item: float(item["steps/nr_env_steps"]))
    return rows


def find_fatals(lines: Iterable[str]) -> list[str]:
    found: list[str] = []
    for line in lines:
        if FATAL_RE.search(line):
            found.append(line.rstrip())
    return found


def write_wide(rows: list[dict[str, float | str]], path: Path) -> None:
    keys = ["timestamp", "steps/nr_env_steps"]
    remaining = sorted({key for row in rows for key in row} - set(keys))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys + remaining, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_long(rows: list[dict[str, float | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "nr_env_steps", "metric", "value"])
        for row in rows:
            for key, value in sorted(row.items()):
                if key in ("timestamp", "steps/nr_env_steps"):
                    continue
                writer.writerow([row["timestamp"], row["steps/nr_env_steps"], key, value])


def build_summary(rows: list[dict[str, float | str]], fatals: list[str]) -> dict[str, object]:
    summary: dict[str, object] = {
        "metric_rows": len(rows),
        "fatal_line_count": len(fatals),
        "fatal_lines": fatals[-50:],
        "first_step": int(rows[0]["steps/nr_env_steps"]) if rows else None,
        "last_step": int(rows[-1]["steps/nr_env_steps"]) if rows else None,
        "last": {},
        "best": {},
    }
    if not rows:
        return summary
    last = rows[-1]
    watched = []
    for family in FAMILIES:
        watched.extend(
            [
                f"derived/joint_rmse/{family}",
                f"derived/normalized_qvel_rmse/{family}",
                f"derived/root_heading_error_deg/{family}",
                f"env_info/rpos_tracking_error/{family}",
                f"env_info/rquat_tracking_error/{family}",
                f"env_info/foot_height_error/{family}",
                f"env_info/foot_penetration_m/{family}",
                f"env_info/foot_slip_speed_sq/{family}",
                f"env_curriculum/morphology_coeff/{family}",
                f"rollout/episode_return/{family}",
                f"rollout/episode_length/{family}",
            ]
        )
    watched.extend(
        [
            "policy_ratio/approx_kl",
            "policy_ratio/clip_fraction",
            "policy/std_dev",
            "gradients/policy_grad_norm",
            "gradients/critic_grad_norm",
            "loss/policy_gradient_loss",
            "loss/critic_loss",
            "throughput/env_steps_per_second",
        ]
    )
    summary["last"] = {key: last[key] for key in watched if key in last}
    lower_is_better = [
        key
        for key in watched
        if key.startswith("derived/")
        or "tracking_error" in key
        or "penetration" in key
        or "slip" in key
    ]
    best: dict[str, float] = {}
    for key in lower_is_better:
        values = [float(row[key]) for row in rows if key in row and math.isfinite(float(row[key]))]
        if values:
            best[key] = min(values)
    summary["best"] = best
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    text = args.log.read_text(encoding="utf-8", errors="replace").splitlines()
    rows = parse_lines(text)
    fatals = find_fatals(text)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_wide(rows, args.out_dir / "training_metrics_wide.csv")
    write_long(rows, args.out_dir / "training_metrics_long.csv")
    summary = build_summary(rows, fatals)
    (args.out_dir / "training_parse_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
