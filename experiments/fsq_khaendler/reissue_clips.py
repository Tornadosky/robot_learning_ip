#!/usr/bin/env python3
"""Re-issue LAFAN1 clips for every family, with the joint limits actually enforced.

WHAT I GOT WRONG FIRST. I reported that only `dance2_subject4` could be
regenerated locally, because the retargeter's default source env is UnitreeH1v2
and only that one H1v2 clip was on this machine. That was wrong: the source is a
PARAMETER, not a constant, and this repo has **11 full-length UnitreeH1 clips and
10 UnitreeG1 clips** locally -- both families verified feasible by
`reference_feasibility_matrix.py`, and H1 is already what produced the cached
SMPL fits in `deepmimic_morphology/`. So the full re-issue needs no downloads.

THE TWO-STAGE COST. `motion_transfer_robot_to_robot` fits the SMPL pose to the
SOURCE robot (torch, the expensive half) and then runs a per-target physics fit
(cheap). The source fit is cached via `path_to_fitted_motion_source` and reused
for EVERY target robot, so N motions x M robots costs N source fits, not N*M.

WHY THE OUTPUT IS A NEW DIRECTORY. `LAFAN1/` is what every trained checkpoint and
every crosseval baseline was scored against. Overwriting it in place would
silently invalidate comparisons to work already done, so the re-issue lands in
`LAFAN1_fixed/` and a run opts in by pointing `--clip_dir` / `--raw_clip_dir` at it.

Paths are passed EXPLICITLY rather than through loco_mujoco's variables file:
that file is WSL-pathed (`/mnt/c/...`) and does not resolve on Windows.

    .venv/Scripts/python.exe experiments/fsq_khaendler/reissue_clips.py --dry
    .venv/Scripts/python.exe experiments/fsq_khaendler/reissue_clips.py --clips dance2_subject1 --targets BoosterT1
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SMPL_MODEL = ROOT / "external_data/smpl"
SHAPE_ROOT = ROOT / "external_data/amass_converted/AMASS"
SRC_CLIPS = ROOT / "external_data/amass_converted/LAFAN1"
OUT_ROOT = ROOT / "external_data/amass_converted/LAFAN1_fixed"
FIT_CACHE = ROOT / "external_data/amass_converted/smpl_source_fits"

SOURCE_ENV = "UnitreeH1"
SOURCE_SUBDIR = "UnitreeH1"
# Only the families the fix is for. H1 and G1 are already clean and are
# deliberately not re-issued -- there is no reason to perturb known-good
# references that trained checkpoints were scored against.
TARGETS = ["BoosterT1", "Atlas", "Talos", "ToddlerBot"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="*", default=None,
                    help="clip stems; default = every clip the source robot has")
    ap.add_argument("--targets", nargs="*", default=TARGETS)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--src-dir", default=None, help="source clip root (default external_data/amass_converted/LAFAN1)")
    ap.add_argument("--out-root", default=None, help="output root (default external_data/amass_converted/LAFAN1_fixed)")
    ap.add_argument("--no-extend", action="store_true", help="skip extend_motion (FK sites/xpos); default extends so foot channels exist")
    args = ap.parse_args()
    global SRC_CLIPS, OUT_ROOT
    if args.src_dir: SRC_CLIPS = Path(args.src_dir)
    if args.out_root: OUT_ROOT = Path(args.out_root)

    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger("reissue")

    clips = args.clips or sorted(p.stem for p in (SRC_CLIPS / SOURCE_SUBDIR).glob("*.npz"))
    print(f"source {SOURCE_ENV}: {len(clips)} clips -> {len(args.targets)} targets "
          f"= {len(clips)} source fits + {len(clips) * len(args.targets)} target fits")
    for c in clips:
        print(f"  {c}")
    if args.dry:
        return

    from loco_mujoco.core.trajectory import Trajectory
    from loco_mujoco.smpl import retargeting as R

    FIT_CACHE.mkdir(parents=True, exist_ok=True)
    conf_source = R.load_robot_conf_file(SOURCE_ENV)
    src_shape_dir = SHAPE_ROOT / SOURCE_ENV
    src_shape_dir.mkdir(parents=True, exist_ok=True)

    for clip in clips:
        src_path = SRC_CLIPS / SOURCE_SUBDIR / f"{clip}.npz"
        if not src_path.exists():
            print(f"!! {clip}: no source trajectory, skipped")
            continue
        traj_source = Trajectory.load(str(src_path))
        fit_path = FIT_CACHE / f"{SOURCE_ENV}_{clip}_smpl.npz"

        for target in args.targets:
            out_dir = OUT_ROOT / target
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{clip}.npz"
            if out_path.exists():
                print(f"== {target}/{clip}: exists, skipped")
                continue
            conf_target = R.load_robot_conf_file(target)
            tgt_shape_dir = SHAPE_ROOT / target
            tgt_shape_dir.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            try:
                traj = R.motion_transfer_robot_to_robot(
                    SOURCE_ENV, conf_source, traj_source, str(src_shape_dir),
                    target, conf_target, str(tgt_shape_dir), str(SMPL_MODEL),
                    logger, path_to_fitted_motion_source=str(fit_path))
            except Exception as exc:                       # noqa: BLE001
                print(f"!! {target}/{clip}: FAILED after {time.time() - t0:.0f}s "
                      f"-- {type(exc).__name__}: {exc}")
                continue
            if not args.no_extend:
                # restore xpos/site_xpos/body_names via FK so the tokenizer foot channels and
                # derive_clip_signs work on the re-issued clip (LAFAN1_fixed lacked them)
                from loco_mujoco.datasets.humanoids.LAFAN1.load import extend_motion
                traj = extend_motion(target, conf_target, traj, replace_qvel_with_finite_diff=False)
            traj.save(str(out_path))
            err = R.LAST_FIT_DIAGNOSTICS.get("site_error_m")
            note = f", site err {1000 * err.mean():.1f} mm" if err is not None else ""
            print(f"OK {target}/{clip}: {time.time() - t0:.0f}s{note} -> {out_path}")


if __name__ == "__main__":
    main()
