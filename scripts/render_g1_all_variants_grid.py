"""Render all main G1 morphology variants at their best observed checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from g1_morphology_variants import PRESETS
from render_morphology_deepmimic import (
    IMAGES_DIR,
    VIDEOS_DIR,
    compose,
    load_manifest,
    rollout_to_mp4,
)

WORKSPACE = Path(__file__).resolve().parents[1]
ROOT = WORKSPACE / "external_data" / "deepmimic_morphology" / "g1" / "dance2_subject4"
SUMMARY = WORKSPACE / "images" / "g1_morphology_research_summary.json"

ORDER = [
    "nominal",
    "tall_legs",
    "short_legs",
    "long_arms",
    "broad_shoulders",
    "big_feet",
    "heavy_torso",
    "combined",
    "extreme_tall_light",
    "extreme_short_heavy",
    "extreme_combined",
]

# Headless OpenCV builds throw from GUI cleanup even though recording succeeded.
for _fn in ("destroyAllWindows", "destroyWindow", "imshow", "namedWindow",
            "waitKey", "startWindowThread", "setWindowProperty"):
    setattr(cv2, _fn, lambda *args, **kwargs: 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-steps", type=int, default=900)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--panel-width", type=int, default=400)
    parser.add_argument("--panel-height", type=int, default=360)
    parser.add_argument("--output-stem", default="grid_g1_all_variants_best_rewards")
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()

    rows = {
        row["preset"]: row
        for row in json.loads(SUMMARY.read_text(encoding="utf-8"))
        if row.get("best_cell")
    }

    work = VIDEOS_DIR / ".grid_cache" / args.output_stem
    work.mkdir(parents=True, exist_ok=True)
    sources = []
    for preset in ORDER:
        row = rows.get(preset)
        if row is None:
            print(f"[render] skip {preset}: no selected cell")
            continue

        cell = row["best_cell"]
        prefix = f"{preset}__"
        suffix = cell[len(prefix):] if cell.startswith(prefix) else ""
        manifest = load_manifest("g1", "dance2_subject4", preset, suffix)
        best = max(manifest["checkpoints"], key=lambda c: c["mean_episode_return"])
        mp4 = work / f"{preset}.mp4"
        print(
            f"[render] {preset}: {row['best_cell']} "
            f"R={best['mean_episode_return']:.1f} "
            f"len={best['mean_episode_length']:.0f}",
            flush=True,
        )
        if mp4.exists() and mp4.stat().st_size > 0:
            print(f"[render] reuse cached panel: {mp4.name}", flush=True)
        else:
            rollout_to_mp4(
                manifest,
                Path(best["agent_path"]),
                mp4,
                args.n_steps,
                args.stochastic,
            )
        sources.append((
            mp4,
            PRESETS[preset].label,
            (
                f"R={best['mean_episode_return']:.1f} | "
                f"len={best['mean_episode_length']:.0f} | "
                f"{best['cumulative_steps'] / 1e6:.0f}M steps"
            ),
        ))

    compose(
        sources,
        VIDEOS_DIR / f"{args.output_stem}.mp4",
        IMAGES_DIR / f"{args.output_stem}.png",
        columns=args.columns,
        fps=args.fps,
        panel_w=args.panel_width,
        panel_h=args.panel_height,
    )
    print(f"[render] done: {len(sources)} panels", flush=True)


if __name__ == "__main__":
    main()
