"""Render a final grid video of the best trained G1 policy per morphology.

The best policy for each body spans different recipes: nominal uses the 3x-gain
fix, short-heavy works best in the stock 300M cell, and the tall/combined bodies
need the long stock-gain ``deepbig`` runs. The shared
``render_morphology_deepmimic --mode grid`` takes one out-suffix for all presets,
so this driver reuses that module's rollout + compositing helpers but picks each
panel's cell explicitly.

CPU MuJoCo + CPU JAX -> runs on the Windows .venv after the GPU training.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from render_morphology_deepmimic import (
    IMAGES_DIR, VIDEOS_DIR, _preset_details, _preset_label,
    compose, load_manifest, rollout_to_mp4,
)

ROBOT = "g1"
CLIP = "dance2_subject4"

# preset -> out-suffix of its best trained cell ("" = stock-gain no-suffix cell).
BEST_CELLS = {
    "nominal": "stiff3_lr3",
    "extreme_short_heavy": "",
    "combined": "deepbig",
    "extreme_tall_light": "deepbig",
    "extreme_combined": "deepbig",
}
ORDER = ["nominal", "extreme_tall_light", "extreme_short_heavy", "combined", "extreme_combined"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-steps", type=int, default=900)
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--panel-width", type=int, default=400)
    parser.add_argument("--panel-height", type=int, default=360)
    parser.add_argument("--output-stem", default="grid_g1_best_rewards")
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()

    work = Path(tempfile.mkdtemp(prefix="g1_best_grid_"))
    sources = []
    for preset in ORDER:
        suffix = BEST_CELLS[preset]
        try:
            manifest = load_manifest(ROBOT, CLIP, preset, suffix)
        except FileNotFoundError as exc:
            print(f"[render] skip {preset}: {exc}")
            continue
        # Best checkpoint by tracking return (training can diverge late).
        best = max(manifest["checkpoints"], key=lambda c: c["mean_episode_return"])
        mp4 = work / f"{preset}.mp4"
        print(f"[render] {preset}: cell suffix='{suffix or '(stock)'}' "
              f"R={best['mean_episode_return']:.0f} len={best['mean_episode_length']:.0f}")
        rollout_to_mp4(manifest, Path(best["agent_path"]), mp4, args.n_steps, args.stochastic)
        scale = manifest.get("pd_gain_scale")
        recipe = f"{scale:g}x PD" if scale else (suffix or "stock")
        sources.append((
            mp4,
            _preset_label(ROBOT, preset),
            f"{recipe} | R={best['mean_episode_return']:.1f} | len={best['mean_episode_length']:.0f}",
        ))

    compose(sources, VIDEOS_DIR / f"{args.output_stem}.mp4", IMAGES_DIR / f"{args.output_stem}.png",
            columns=args.columns, fps=args.fps,
            panel_w=args.panel_width, panel_h=args.panel_height)
    shutil.rmtree(work, ignore_errors=True)
    print(f"[render] done: {len(sources)} panels")


if __name__ == "__main__":
    main()
