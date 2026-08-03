"""Drive the full DeepMimic morphology matrix: robots x variants x clips.

Approved scope: 2 robots (H1, G1) x 5 variants (nominal, extreme_tall_light,
extreme_short_heavy, extreme_combined, combined) x 2 clips (dance2_subject4,
dance2_subject2) = 20 cells.

Each cell runs as two isolated subprocesses -- retarget (torch/CUDA) then train
(JAX/CUDA) -- so the two CUDA stacks never share a process, a single failed cell
doesn't abort the matrix, and JIT caches don't accumulate across 20 trainings.
The driver is resumable: a cell whose manifest.json already exists is skipped
unless --force is given.

Run this on the GPU box (WSL2):
    python scripts/run_morphology_matrix.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from morphology_deepmimic import DEEPMIMIC_ROOT, cell_dir

SCRIPTS = Path(__file__).resolve().parent
ROBOTS_DEFAULT = ["h1", "g1"]
PRESETS_DEFAULT = [
    "nominal",
    "extreme_tall_light",
    "extreme_short_heavy",
    "extreme_combined",
    "combined",
]
CLIPS_DEFAULT = ["dance2_subject4", "dance2_subject2"]


def run(cmd: list[str]) -> int:
    print("\n$ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(SCRIPTS)).returncode


def run_cell(robot: str, preset: str, clip: str, args: argparse.Namespace) -> str:
    manifest = cell_dir(robot, clip, preset) / "manifest.json"
    if manifest.exists() and not args.force:
        print(f"[matrix] skip {robot}/{clip}/{preset} (manifest exists)")
        return "skipped"

    # The SMPL retarget step is only needed for the retargeted-reference mode; with
    # --raw-reference the trainer grounds the clean LAFAN1 clip onto the body itself.
    if not args.raw_reference:
        retarget_cmd = [
            sys.executable, "retarget_dance_to_variant.py",
            "--robot", robot, "--preset", preset, "--clip", clip,
            "--duration", str(args.duration),
        ]
        if args.start_frame is not None:
            retarget_cmd += ["--start-frame", str(args.start_frame)]
        if run(retarget_cmd) != 0:
            return "retarget_failed"

    train_cmd = [
        sys.executable, "train_deepmimic_morphology.py",
        "--robot", robot, "--preset", preset, "--clip", clip,
        "--duration", str(args.duration),
        "--num-envs", str(args.num_envs),
        "--total-timesteps", str(args.total_timesteps),
        "--num-checkpoints", str(args.num_checkpoints),
    ]
    train_cmd += ["--raw-reference"] if args.raw_reference else ["--no-retarget"]
    train_cmd += ["--control", args.control]
    if args.start_frame is not None:
        train_cmd += ["--start-frame", str(args.start_frame)]
    if args.use_mjwarp:
        train_cmd += ["--use-mjwarp"]
    return "trained" if run(train_cmd) == 0 else "train_failed"


def main() -> None:
    args = parse_args()
    cells = [(r, p, c) for r in args.robots for c in args.clips for p in args.presets]
    print(f"[matrix] {len(cells)} cells: robots={args.robots} presets={args.presets} clips={args.clips}")

    results = {}
    t_start = time.time()
    for index, (robot, preset, clip) in enumerate(cells, 1):
        key = f"{robot}/{clip}/{preset}"
        print(f"\n===== cell {index}/{len(cells)}: {key} =====")
        try:
            status = run_cell(robot, preset, clip, args)
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # keep the matrix going; record the failure
            status = f"error: {exc}"
        results[key] = status
        if status not in ("trained", "skipped") and not args.keep_going:
            print(f"[matrix] cell {key} -> {status}; stopping (use --keep-going to continue).")
            break

    DEEPMIMIC_ROOT.mkdir(parents=True, exist_ok=True)
    summary = {
        "elapsed_minutes": (time.time() - t_start) / 60.0,
        "num_envs": args.num_envs,
        "total_timesteps": args.total_timesteps,
        "num_checkpoints": args.num_checkpoints,
        "results": results,
    }
    summary_path = DEEPMIMIC_ROOT / "matrix_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[matrix] done in {summary['elapsed_minutes']:.1f} min -> {summary_path}")
    print(json.dumps(results, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robots", nargs="+", default=ROBOTS_DEFAULT, choices=["h1", "g1"])
    parser.add_argument("--presets", nargs="+", default=PRESETS_DEFAULT)
    parser.add_argument("--clips", nargs="+", default=CLIPS_DEFAULT)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--total-timesteps", type=float, default=300e6)
    parser.add_argument("--num-checkpoints", type=int, default=6)
    parser.add_argument("--use-mjwarp", action="store_true")
    parser.add_argument("--control", choices=["torque", "pd"], default="torque",
                        help="Control modality passed to the trainer (pd recommended for G1).")
    parser.add_argument("--raw-reference", action="store_true",
                        help="Train on the clean grounded LAFAN1 clip (skips SMPL retarget).")
    parser.add_argument("--force", action="store_true", help="Re-run cells even if a manifest exists.")
    parser.add_argument("--keep-going", action="store_true", help="Continue past a failed cell.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
