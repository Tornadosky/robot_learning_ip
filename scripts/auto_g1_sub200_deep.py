"""Deep-extend every G1 morphology whose best return is below acceptance.

Each run resumes the final stock-300M checkpoint, evaluates every additional
100M steps, and stops at R=250. The 1.5B-step extension cap matches the 1.8B
cumulative budget that solved the existing extreme morphology experiments.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent / "external_data" / "deepmimic_morphology" / "g1" / "dance2_subject4"

PRESETS = ["tall_legs", "long_arms", "broad_shoulders", "heavy_torso"]
SOURCE_SUFFIX = "stock300m"
OUT_SUFFIX = "deep250"
TARGET_RETURN = 250.0
ACCEPT_RETURN = 200.0
EXTENSION_TIMESTEPS = "1500e6"
NUM_CHECKPOINTS = 15
START = time.time()


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def best_checkpoint(manifest: dict) -> dict:
    return max(manifest["checkpoints"], key=lambda ck: ck["mean_episode_return"])


def run_preset(preset: str) -> dict:
    output_path = ROOT / f"{preset}__{OUT_SUFFIX}" / "manifest.json"
    if output_path.exists():
        manifest = load_manifest(output_path)
        best = best_checkpoint(manifest)
        if best["mean_episode_return"] >= ACCEPT_RETURN:
            print(
                f"[deep250] skip {preset}: accepted checkpoint already exists "
                f"(R={best['mean_episode_return']:.1f})",
                flush=True,
            )
            return summarize(preset, manifest)
        raise RuntimeError(
            f"Existing {output_path} is below R={ACCEPT_RETURN:.0f}; "
            "use a new suffix or resume it explicitly."
        )

    source_path = ROOT / f"{preset}__{SOURCE_SUFFIX}" / "manifest.json"
    source = load_manifest(source_path)
    resume = max(source["checkpoints"], key=lambda ck: ck["cumulative_steps"])
    agent_path = Path(resume["agent_path"])
    if not agent_path.exists():
        raise FileNotFoundError(agent_path)

    cmd = [
        sys.executable,
        "train_deepmimic_morphology.py",
        "--robot", "g1",
        "--preset", preset,
        "--clip", "dance2_subject4",
        "--duration", "30",
        "--num-envs", "2048",
        "--total-timesteps", EXTENSION_TIMESTEPS,
        "--num-checkpoints", str(NUM_CHECKPOINTS),
        "--raw-reference",
        "--control", "pd",
        "--seed", "1",
        "--resume-from", str(agent_path),
        "--resume-steps", str(resume["cumulative_steps"]),
        "--target-return", str(TARGET_RETURN),
        "--out-suffix", OUT_SUFFIX,
    ]
    print(f"\n[deep250] $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=SCRIPTS, check=True)
    manifest = load_manifest(output_path)
    return summarize(preset, manifest)


def summarize(preset: str, manifest: dict) -> dict:
    best = best_checkpoint(manifest)
    return {
        "preset": preset,
        "accepted": best["mean_episode_return"] >= ACCEPT_RETURN,
        "target_reached": best["mean_episode_return"] >= TARGET_RETURN,
        "best_return": best["mean_episode_return"],
        "best_length": best["mean_episode_length"],
        "best_steps": best["cumulative_steps"],
        "final_steps": manifest["checkpoints"][-1]["cumulative_steps"],
        "training_minutes": manifest["training_minutes"],
    }


def main() -> None:
    results = {}
    summary_path = ROOT / "auto_sub200_deep_summary.json"
    for preset in PRESETS:
        results[preset] = run_preset(preset)
        summary_path.write_text(
            json.dumps({"elapsed_hours": (time.time() - START) / 3600, "results": results}, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(results[preset], indent=2), flush=True)

    rejected = [name for name, result in results.items() if not result["accepted"]]
    print(f"\n[deep250] complete in {(time.time() - START) / 3600:.2f}h", flush=True)
    if rejected:
        raise RuntimeError(f"Below R={ACCEPT_RETURN:.0f} after deep extension: {rejected}")


if __name__ == "__main__":
    main()
