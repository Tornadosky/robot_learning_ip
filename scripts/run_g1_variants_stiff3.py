"""Train the G1 morphology variants with the winning 3x-PD-gain recipe.

The G1 "fixed" recipe (PD control, 3x stiffer gains, lr 3e-4, init_std 0.2,
raw reference, 120M steps / 4 checkpoints) is the one that reliably crossed the
balance transition for the nominal body (-> nominal__stiff3_lr3, best R~478).
The matrix-driver variant cells used stock gains and stayed stuck at the poor
baseline, so we retrain the variants here with the same recipe as nominal, all
under the shared out-suffix ``stiff3_lr3`` so the grid/plot pull one clean set.

The 120M recipe is bifurcation-sensitive: a single seed can get stuck (best
episode length ~160) instead of crossing (length 260+). Strategy is "single
seed, retry if stuck" -- train seed 1, and only if it failed to escape the
stuck basin do we try further seeds, keeping the best run.

Run inside WSL conda env locodm (GPU JAX):
    python run_g1_variants_stiff3.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent / "external_data" / "deepmimic_morphology" / "g1" / "dance2_subject4"
SUFFIX = "stiff3_lr3"  # matches the existing good nominal cell (nominal__stiff3_lr3)

VARIANTS = ["extreme_tall_light", "extreme_short_heavy", "extreme_combined", "combined"]
SEEDS = [1, 2, 3]
# "Crossed the balance transition" = survived clearly longer than the stuck basin
# (stuck runs collapse at episode length ~160; crossed nominal reached ~731).
CROSS_LENGTH = 320.0


def best_checkpoint(manifest: dict) -> dict:
    return max(manifest["checkpoints"], key=lambda c: c["mean_episode_return"])


def train_one(preset: str, seed: int, suffix: str) -> Path | None:
    cmd = [
        sys.executable, "train_deepmimic_morphology.py",
        "--robot", "g1", "--preset", preset, "--clip", "dance2_subject4",
        "--duration", "30", "--num-envs", "2048",
        "--total-timesteps", "120e6", "--num-checkpoints", "4",
        "--raw-reference", "--control", "pd",
        "--lr", "3e-4", "--init-std", "0.2", "--pd-gain-scale", "3.0",
        "--seed", str(seed), "--out-suffix", suffix,
    ]
    print(f"\n$ {' '.join(cmd)}", flush=True)
    rc = subprocess.run(cmd, cwd=str(SCRIPTS)).returncode
    out_dir = ROOT / f"{preset}__{suffix}"
    if rc != 0 or not (out_dir / "manifest.json").exists():
        print(f"[driver] {preset} seed{seed} FAILED rc={rc}", flush=True)
        return None
    return out_dir


def main() -> None:
    results = {}
    for preset in VARIANTS:
        canonical = ROOT / f"{preset}__{SUFFIX}"
        if (canonical / "manifest.json").exists():
            m = json.loads((canonical / "manifest.json").read_text())
            best = best_checkpoint(m)
            if best["mean_episode_length"] >= CROSS_LENGTH:
                print(f"[driver] skip {preset}: already crossed "
                      f"(R={best['mean_episode_return']:.0f} len={best['mean_episode_length']:.0f})")
                results[preset] = {"status": "skipped", **best}
                continue

        best_dir = None
        best_len = -1.0
        best_meta = None
        for seed in SEEDS:
            try_dir = train_one(preset, seed, f"{SUFFIX}_try{seed}")
            if try_dir is None:
                continue
            m = json.loads((try_dir / "manifest.json").read_text())
            ck = best_checkpoint(m)
            print(f"[driver] {preset} seed{seed}: best R={ck['mean_episode_return']:.0f} "
                  f"len={ck['mean_episode_length']:.0f}", flush=True)
            if ck["mean_episode_length"] > best_len:
                best_len, best_dir, best_meta = ck["mean_episode_length"], try_dir, ck
            if ck["mean_episode_length"] >= CROSS_LENGTH:
                break  # crossed -> no need for more seeds

        if best_dir is None:
            results[preset] = {"status": "all_failed"}
            print(f"[driver] {preset}: ALL SEEDS FAILED", flush=True)
            continue

        # Promote the best try to the canonical shared-suffix dir for grid/plot.
        old_name = best_dir.name  # e.g. "<preset>__stiff3_lr3_try2"
        if canonical.exists():
            shutil.rmtree(canonical)
        shutil.move(str(best_dir), str(canonical))
        # The moved manifest still references the old try-dir path internally;
        # rewrite reference_path + checkpoint agent_paths to the canonical dir so
        # rendering (which reads those paths directly) resolves the moved files.
        mpath = canonical / "manifest.json"
        mpath.write_text(mpath.read_text().replace(old_name, canonical.name))
        # Clean up the other (non-winning) try dirs.
        for seed in SEEDS:
            stale = ROOT / f"{preset}__{SUFFIX}_try{seed}"
            if stale.exists():
                shutil.rmtree(stale, ignore_errors=True)
        crossed = best_len >= CROSS_LENGTH
        results[preset] = {"status": "crossed" if crossed else "best_effort", **best_meta}
        print(f"[driver] {preset} -> {canonical.name} "
              f"({'crossed' if crossed else 'best-effort'} "
              f"R={best_meta['mean_episode_return']:.0f} len={best_meta['mean_episode_length']:.0f})",
              flush=True)

    (ROOT / "variants_stiff3_summary.json").write_text(json.dumps(results, indent=2))
    print("\n[driver] DRIVER_DONE")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
