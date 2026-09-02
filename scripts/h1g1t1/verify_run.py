#!/usr/bin/env python3
"""Verify that RL-X really reached the requested step and produced a model."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from parse_training_log import find_fatals, parse_lines  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    lines = args.log.read_text(encoding="utf-8", errors="replace").splitlines() if args.log.is_file() else []
    rows = parse_lines(lines)
    fatals = find_fatals(lines)
    last_step = int(rows[-1]["steps/nr_env_steps"]) if rows else None
    checks = {
        "log_exists": args.log.is_file() and args.log.stat().st_size > 0,
        "metric_rows_present": bool(rows),
        "no_fatal_lines": not fatals,
        "requested_final_step_reached": last_step == args.expected_step,
        "latest_model_exists": args.model.is_file() and args.model.stat().st_size > 0,
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "last_step": last_step,
        "expected_step": args.expected_step,
        "model": str(args.model),
        "model_bytes": args.model.stat().st_size if args.model.is_file() else 0,
        "fatal_lines": fatals[-50:],
        "metric_rows": len(rows),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
