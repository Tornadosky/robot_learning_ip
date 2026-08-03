"""Gain-scale probe for the TALL G1 bodies that get stuck under the 3x recipe.

extreme_tall_light (legs 1.55x) and extreme_combined (legs 1.45x) have a high CoM
and collapse under 3x PD gains (episode length ~130, vs ~650-731 for the bodies
that cross). Stock gains (1x) put them in a better basin (~280) but still short of
"proper level" (~700). This probes intermediate gain scales to find one that lets
the tall bodies actually balance and track.

Each cell is the same 120M recipe as the main run, only --pd-gain-scale varies.
Results land under {preset}__stiff{scale}_lr3; the best scale per body is reported
so the assembly step can pick the winning cell.

Run inside WSL conda env locodm (GPU JAX):
    python run_g1_tall_gainsweep.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent / "external_data" / "deepmimic_morphology" / "g1" / "dance2_subject4"

TALL_PRESETS = ["extreme_tall_light", "extreme_combined", "combined"]
# 3x overshoots the tall bodies (len ~130); stock(1x) plateaus (~280). Probe the
# gap, stiffest-first so a crossing body early-stops fast.
GAIN_SCALES = [2.5, 2.0, 1.5]
CROSS_LENGTH = 320.0


def best_checkpoint(manifest: dict) -> dict:
    return max(manifest["checkpoints"], key=lambda c: c["mean_episode_return"])


def scale_tag(scale: float) -> str:
    # 2.0 -> "stiff2_lr3", 1.5 -> "stiff15_lr3"
    s = ("%g" % scale).replace(".", "")
    return f"stiff{s}_lr3"


def train_one(preset: str, scale: float) -> Path | None:
    suffix = scale_tag(scale)
    cmd = [
        sys.executable, "train_deepmimic_morphology.py",
        "--robot", "g1", "--preset", preset, "--clip", "dance2_subject4",
        "--duration", "30", "--num-envs", "2048",
        "--total-timesteps", "120e6", "--num-checkpoints", "4",
        "--raw-reference", "--control", "pd",
        "--lr", "3e-4", "--init-std", "0.2", "--pd-gain-scale", str(scale),
        "--seed", "1", "--out-suffix", suffix,
    ]
    print(f"\n$ {' '.join(cmd)}", flush=True)
    rc = subprocess.run(cmd, cwd=str(SCRIPTS)).returncode
    out_dir = ROOT / f"{preset}__{suffix}"
    if rc != 0 or not (out_dir / "manifest.json").exists():
        print(f"[sweep] {preset} scale={scale} FAILED rc={rc}", flush=True)
        return None
    return out_dir


def main() -> None:
    results = {}
    for preset in TALL_PRESETS:
        results[preset] = {}
        for scale in GAIN_SCALES:
            out_dir = train_one(preset, scale)
            if out_dir is None:
                results[preset][scale_tag(scale)] = {"status": "failed"}
                continue
            m = json.loads((out_dir / "manifest.json").read_text())
            ck = best_checkpoint(m)
            crossed = ck["mean_episode_length"] >= CROSS_LENGTH
            results[preset][scale_tag(scale)] = {
                "status": "crossed" if crossed else "stuck",
                "best_return": ck["mean_episode_return"],
                "best_length": ck["mean_episode_length"],
            }
            print(f"[sweep] {preset} scale={scale}: best R={ck['mean_episode_return']:.0f} "
                  f"len={ck['mean_episode_length']:.0f} ({'crossed' if crossed else 'stuck'})",
                  flush=True)
            if crossed:
                break  # found a working scale for this body

    (ROOT / "tall_gainsweep_summary.json").write_text(json.dumps(results, indent=2))
    print("\n[sweep] SWEEP_DONE")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
