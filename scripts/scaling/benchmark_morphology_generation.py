"""Benchmark safe, throw-away static H1 XML morphology generation."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[2]
SCRIPTS = WORKSPACE / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from h1_morphology_variants import PRESETS, create_h1_variant_xml  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variants", nargs="+", choices=PRESETS, default=list(PRESETS)[:10]
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be positive.")

    records = []
    total_started = time.perf_counter()
    # The generator copies assets and compiles each XML for validation.  A
    # private temporary root guarantees that existing research assets cannot
    # be overwritten by the benchmark.
    with tempfile.TemporaryDirectory(prefix="h1_generation_benchmark_") as temp:
        root = Path(temp)
        for repeat in range(args.repeats):
            for name in args.variants:
                started = time.perf_counter()
                path = create_h1_variant_xml(PRESETS[name], root / f"repeat_{repeat}")
                records.append(
                    {
                        "repeat": repeat,
                        "variant": name,
                        "seconds": time.perf_counter() - started,
                        "validated_xml_bytes": path.stat().st_size,
                    }
                )
    total_seconds = time.perf_counter() - total_started
    timings = [record["seconds"] for record in records]
    result = {
        "experiment": "static_h1_morphology_generation_benchmark",
        "implementation": "copy_assets_modify_mjspec_write_and_compile_validate",
        "num_generated": len(records),
        "num_variants": len(args.variants),
        "repeats": args.repeats,
        "total_seconds": total_seconds,
        "mean_seconds_per_morphology": statistics.mean(timings),
        "median_seconds_per_morphology": statistics.median(timings),
        "morphologies_per_second": len(records) / total_seconds,
        "records": records,
    }
    payload = json.dumps(result, indent=2, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
