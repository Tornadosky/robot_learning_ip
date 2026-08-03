"""Train moderate G1 morphology variants with the stock-gain 300M recipe.

This complements ``auto_g1_moderate_variants.py``. The fast 3x-PD recipe is good
for nominal G1 and short legs but brittle on several moderate morphology edits.
The stock-gain 300M pass gives an apples-to-apples baseline and often behaves
better for morphology stress cases.

Run inside WSL conda env ``locodm``.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent / "external_data" / "deepmimic_morphology" / "g1" / "dance2_subject4"

PRESETS = [
    "tall_legs",
    "short_legs",
    "long_arms",
    "broad_shoulders",
    "big_feet",
    "heavy_torso",
]

SUFFIX = "stock300m"
TOTAL_TIMESTEPS = "300e6"
NUM_CHECKPOINTS = 6
CROSS_LEN = 500.0
START = time.time()


def elapsed_min() -> float:
    return (time.time() - START) / 60.0


def manifest_path(preset: str) -> Path:
    return ROOT / f"{preset}__{SUFFIX}" / "manifest.json"


def run_train(preset: str) -> bool:
    out = manifest_path(preset)
    if out.exists():
        print(f"[stock300] skip {preset}: {out} exists", flush=True)
        return True

    cmd = [
        sys.executable, "train_deepmimic_morphology.py",
        "--robot", "g1",
        "--preset", preset,
        "--clip", "dance2_subject4",
        "--duration", "30",
        "--num-envs", "2048",
        "--total-timesteps", TOTAL_TIMESTEPS,
        "--num-checkpoints", str(NUM_CHECKPOINTS),
        "--raw-reference",
        "--control", "pd",
        "--seed", "1",
        "--out-suffix", SUFFIX,
    ]
    print(f"\n[stock300] t={elapsed_min():.1f}min $ {' '.join(cmd)}", flush=True)
    rc = subprocess.run(cmd, cwd=str(SCRIPTS)).returncode
    ok = rc == 0 and out.exists()
    print(f"[stock300] {preset} rc={rc} ok={ok}", flush=True)
    return ok


def summarize_manifest(preset: str) -> dict:
    path = manifest_path(preset)
    if not path.exists():
        return {"preset": preset, "status": "missing", "cell": f"{preset}__{SUFFIX}"}

    manifest = json.loads(path.read_text(encoding="utf-8"))
    checkpoints = manifest["checkpoints"]
    best_return = max(checkpoints, key=lambda c: c["mean_episode_return"])
    best_length = max(checkpoints, key=lambda c: c["mean_episode_length"])
    first_cross = next(
        (c for c in checkpoints if c["mean_episode_length"] >= CROSS_LEN),
        None,
    )
    return {
        "preset": preset,
        "status": "crossed" if first_cross else "needs_extension",
        "cell": f"{preset}__{SUFFIX}",
        "recipe": "1x PD, lr=1e-4, 300M",
        "total_timesteps": manifest["total_timesteps"],
        "best_return": best_return["mean_episode_return"],
        "best_return_length": best_return["mean_episode_length"],
        "best_length": best_length["mean_episode_length"],
        "best_length_return": best_length["mean_episode_return"],
        "steps_to_len500": first_cross["cumulative_steps"] if first_cross else None,
        "final_return": manifest["mean_episode_return_last"],
        "final_length": manifest["mean_episode_length_last"],
    }


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    for preset in PRESETS:
        ok = run_train(preset)
        results[preset] = summarize_manifest(preset)
        if not ok:
            results[preset]["status"] = "train_failed"
        summary = {
            "recipe": "1x PD, lr=1e-4, init_std=0.2, 300M, raw reference",
            "cross_len": CROSS_LEN,
            "elapsed_min": elapsed_min(),
            "results": results,
        }
        (ROOT / "auto_moderate_stock300_summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(results[preset], indent=2), flush=True)

    summary = {
        "recipe": "1x PD, lr=1e-4, init_std=0.2, 300M, raw reference",
        "cross_len": CROSS_LEN,
        "elapsed_min": elapsed_min(),
        "results": results,
    }
    out = ROOT / "auto_moderate_stock300_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[stock300] done elapsed={elapsed_min():.1f}min -> {out}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
