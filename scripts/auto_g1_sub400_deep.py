"""Continue every selected G1 policy at or below R=400 until it reaches R=410.

The source cell and its best checkpoint come from the current research summary.
Each continuation inherits the source control and optimizer recipe, evaluates at
roughly 100M-step intervals, and has up to 1.5B additional environment steps.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
WORKSPACE = SCRIPTS.parent
ROOT = WORKSPACE / "external_data" / "deepmimic_morphology" / "g1" / "dance2_subject4"
SUMMARY = WORKSPACE / "images" / "g1_morphology_research_summary.json"

ORDER = [
    "extreme_tall_light",
    "extreme_combined",
    "tall_legs",
    "long_arms",
    "broad_shoulders",
    "big_feet",
    "heavy_torso",
]
OUT_SUFFIX = "deep410"
ACCEPT_RETURN = 400.0
TARGET_RETURN = 410.0
EXTENSION_TIMESTEPS = "1500e6"
NUM_CHECKPOINTS = 15
START = time.time()


def load_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def best_checkpoint(manifest: dict) -> dict:
    return max(manifest["checkpoints"], key=lambda ck: ck["mean_episode_return"])


def append_option(cmd: list[str], flag: str, value: object) -> None:
    cmd.extend((flag, str(value)))


def source_rows() -> dict[str, dict]:
    rows = {row["preset"]: row for row in load_json(SUMMARY)}
    return {
        preset: rows[preset]
        for preset in ORDER
        if float(rows[preset]["best_return"]) <= ACCEPT_RETURN
    }


def build_command(preset: str, source: dict, resume: dict) -> list[str]:
    cmd = [
        sys.executable,
        "train_deepmimic_morphology.py",
        "--robot", "g1",
        "--preset", preset,
        "--clip", "dance2_subject4",
        "--duration", "30",
        "--num-envs", str(source.get("num_envs", 2048)),
        "--total-timesteps", EXTENSION_TIMESTEPS,
        "--num-checkpoints", str(NUM_CHECKPOINTS),
        "--control", source.get("control", "pd"),
        "--seed", "1",
        "--resume-from", resume["agent_path"],
        "--resume-steps", str(resume["cumulative_steps"]),
        "--target-return", str(TARGET_RETURN),
        "--out-suffix", OUT_SUFFIX,
    ]
    if source.get("raw_reference"):
        cmd.append("--raw-reference")
    append_option(cmd, "--pd-gain-scale", source.get("pd_gain_scale", 1.0))
    append_option(cmd, "--lr", source.get("lr", 1e-4))
    append_option(cmd, "--init-std", source.get("init_std", 0.2))
    group_scales = source.get("group_gain_scales", {})
    for group in ("hip", "knee", "ankle", "arm"):
        append_option(cmd, f"--{group}-gain-scale", group_scales.get(group, 1.0))
    cmd.append("--hidden-layers")
    cmd.extend(str(width) for width in source.get("hidden_layers", [512, 256]))
    return cmd


def summarize(preset: str, source_cell: str, manifest: dict) -> dict:
    best = best_checkpoint(manifest)
    return {
        "preset": preset,
        "source_cell": source_cell,
        "accepted": best["mean_episode_return"] > ACCEPT_RETURN,
        "target_reached": best["mean_episode_return"] >= TARGET_RETURN,
        "best_return": best["mean_episode_return"],
        "best_length": best["mean_episode_length"],
        "best_steps": best["cumulative_steps"],
        "additional_steps": best["cumulative_steps"] - manifest.get("resume_steps", 0),
        "training_minutes": manifest["training_minutes"],
    }


def run_preset(preset: str, row: dict) -> dict:
    source_cell = row["best_cell"]
    source = load_json(ROOT / source_cell / "manifest.json")
    resume = best_checkpoint(source)
    if not Path(resume["agent_path"]).exists():
        raise FileNotFoundError(resume["agent_path"])

    output_path = ROOT / f"{preset}__{OUT_SUFFIX}" / "manifest.json"
    if output_path.exists():
        existing = load_json(output_path)
        result = summarize(preset, source_cell, existing)
        if result["accepted"]:
            print(
                f"[deep410] skip {preset}: R={result['best_return']:.1f} already accepted",
                flush=True,
            )
            return result
        raise RuntimeError(f"Existing {output_path} is still at or below R={ACCEPT_RETURN:.0f}")

    cmd = build_command(preset, source, resume)
    print(
        f"\n[deep410] {preset}: source={source_cell} "
        f"R={resume['mean_episode_return']:.1f} steps={resume['cumulative_steps']:,}",
        flush=True,
    )
    print(f"[deep410] $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=SCRIPTS, check=True)
    return summarize(preset, source_cell, load_json(output_path))


def main() -> None:
    rows = source_rows()
    results = {}
    summary_path = ROOT / "auto_sub400_deep_summary.json"
    for preset in ORDER:
        if preset not in rows:
            continue
        results[preset] = run_preset(preset, rows[preset])
        summary_path.write_text(
            json.dumps({"elapsed_hours": (time.time() - START) / 3600, "results": results}, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(results[preset], indent=2), flush=True)

    rejected = [name for name, result in results.items() if not result["accepted"]]
    print(f"\n[deep410] complete in {(time.time() - START) / 3600:.2f}h", flush=True)
    if rejected:
        raise RuntimeError(f"At or below R={ACCEPT_RETURN:.0f}: {rejected}")


if __name__ == "__main__":
    main()
