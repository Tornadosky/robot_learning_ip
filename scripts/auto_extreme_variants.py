"""Deep-train the remaining failed G1 morphology variants, now that the
'controllability cliff' was shown to be under-training (extreme_tall_light
crossed to len ~690 with stock gains + ~1.8B steps, see auto_tall_light.py).

Targets (both lengthened-leg, both stuck ~260-286 at 300M = same under-trained
crawl signature tall&light had):
  - extreme_combined (legs 1.45x, ~50 kg)
  - combined         (legs 1.10x)

Per variant: one continuous from-scratch 1.8B run, stock gains, seed 1, raw ref,
18 x 100M checkpoints (resume is broken cross-process, so a single long run is
the honest test). Finalize the BEST checkpoint by episode length. Run inside WSL
conda env locodm (GPU JAX).
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent / "external_data" / "deepmimic_morphology" / "g1" / "dance2_subject4"

PRESETS = ["extreme_combined", "combined"]
STEPS = "1.8e9"
CKPTS = 18
SUFFIX = "deepbig"
CROSS = 500.0  # episode length (of 1000) that counts as "tracking / crossed"
START = time.time()


def elapsed_min() -> float:
    return (time.time() - START) / 60.0


def train(preset: str) -> Path | None:
    cmd = [
        sys.executable, "train_deepmimic_morphology.py",
        "--robot", "g1", "--preset", preset, "--clip", "dance2_subject4",
        "--duration", "30", "--num-envs", "2048",
        "--total-timesteps", STEPS, "--num-checkpoints", str(CKPTS),
        "--raw-reference", "--control", "pd", "--seed", "1", "--out-suffix", SUFFIX,
    ]
    print(f"\n[auto] t={elapsed_min():.0f}min  $ {' '.join(cmd)}", flush=True)
    rc = subprocess.run(cmd, cwd=str(SCRIPTS)).returncode
    out = ROOT / f"{preset}__{SUFFIX}"
    if rc != 0 or not (out / "manifest.json").exists():
        print(f"[auto] {preset} FAILED rc={rc}", flush=True)
        return None
    return out


def main() -> None:
    results = {}
    for preset in PRESETS:
        out = train(preset)
        if out is None:
            results[preset] = {"status": "FAILED"}
            continue
        m = json.loads((out / "manifest.json").read_text())
        cks = m["checkpoints"]
        best = max(cks, key=lambda c: c["mean_episode_length"])
        lens = [round(c["mean_episode_length"], 1) for c in cks]
        rets = [round(c["mean_episode_return"], 1) for c in cks]
        crossed = best["mean_episode_length"] >= CROSS
        results[preset] = {
            "status": "CROSSED" if crossed else "best_effort",
            "best_len": round(best["mean_episode_length"], 1),
            "best_ret": round(best["mean_episode_return"], 1),
            "best_ckpt": best["agent_path"],
            "lens": lens, "rets": rets,
            "steps_M": [round(c["cumulative_steps"] / 1e6) for c in cks],
        }
        print(f"[auto] {preset}: {results[preset]['status']} "
              f"best len={best['mean_episode_length']:.0f} ret={best['mean_episode_return']:.0f}",
              flush=True)
        (ROOT / "auto_extreme_variants_summary.json").write_text(json.dumps(results, indent=2))

    results["elapsed_min"] = elapsed_min()
    (ROOT / "auto_extreme_variants_summary.json").write_text(json.dumps(results, indent=2))
    print(f"\n[auto] AUTO_VARIANTS_DONE elapsed={elapsed_min():.0f}min", flush=True)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
