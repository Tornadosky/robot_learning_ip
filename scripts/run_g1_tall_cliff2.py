"""Refined tall&light cliff run: the soft-gain candidate at full budget.

Stage-1 cliff probes showed the tall&light bodies only respond to SOFT gains: at
legs 1.20x, 3x and 2x stay flat (~104) but 1.5x climbs and was still accelerating
at the 60M cut (76/100/119/152). So the crossing window for these bodies -- if it
exists -- is at ~1.5x with more steps, not at stiffer gains.

This trains each graduated tall&light body at 1.5x PD gains to the FULL 120M
budget (8 checkpoints for a clean curve) and finalizes every cell to
{preset}__cliff, so the cliff plot has full learning curves at all severities.
Each run is ~22 min; the three together give the cliff floor + edge.

Run inside WSL conda env locodm (GPU JAX):
    python run_g1_tall_cliff2.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent / "external_data" / "deepmimic_morphology" / "g1" / "dance2_subject4"
PRESETS = ["tall_light_leg120", "tall_light_leg135", "tall_light_leg150"]
SCALE = "1.5"
STEPS, CKPTS = "120e6", "8"
PROPER_LEN = 600.0


def best_checkpoint(manifest: dict) -> dict:
    return max(manifest["checkpoints"], key=lambda c: c["mean_episode_return"])


def train(preset: str) -> Path | None:
    suffix = "cliff"
    cmd = [
        sys.executable, "train_deepmimic_morphology.py",
        "--robot", "g1", "--preset", preset, "--clip", "dance2_subject4",
        "--duration", "30", "--num-envs", "2048",
        "--total-timesteps", STEPS, "--num-checkpoints", CKPTS,
        "--raw-reference", "--control", "pd", "--lr", "3e-4", "--init-std", "0.2",
        "--pd-gain-scale", SCALE, "--seed", "1", "--out-suffix", suffix,
    ]
    print(f"\n$ {' '.join(cmd)}", flush=True)
    rc = subprocess.run(cmd, cwd=str(SCRIPTS)).returncode
    out_dir = ROOT / f"{preset}__{suffix}"
    if rc != 0 or not (out_dir / "manifest.json").exists():
        print(f"[cliff2] {preset} FAILED rc={rc}", flush=True)
        return None
    return out_dir


def main() -> None:
    results = {}
    for preset in PRESETS:
        out_dir = train(preset)
        if out_dir is None:
            results[preset] = {"status": "failed"}
            continue
        m = json.loads((out_dir / "manifest.json").read_text())
        ck = best_checkpoint(m)
        lens = [c["mean_episode_length"] for c in m["checkpoints"]]
        bl = ck["mean_episode_length"]
        crossed = bl >= PROPER_LEN
        results[preset] = {"status": "crossed" if crossed else "stuck",
                           "scale": float(SCALE),
                           "best_return": ck["mean_episode_return"], "best_length": bl}
        print(f"[cliff2] {preset}: best R={ck['mean_episode_return']:.0f} len={bl:.0f} "
              f"({'CROSSED' if crossed else 'stuck'}) curve={'/'.join('%.0f' % l for l in lens)}",
              flush=True)

    (ROOT / "tall_cliff_summary.json").write_text(json.dumps(results, indent=2))
    print("\n[cliff2] CLIFF_DONE")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
