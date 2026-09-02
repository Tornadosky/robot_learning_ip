"""Fetch every LAFAN1 motion HuggingFace holds for the given robots, in OUR clip
format (loco-mujoco Trajectory.save after extend_motion: qpos/qvel/xpos/
site_xpos/... 28 keys, 40 Hz), into external_data/amass_converted/LAFAN1_all/.

Runs in the Windows .venv (CPU; ~27 s per clip incl. download + FK):
  .venv/Scripts/python.exe scripts/scaling/wave7/fetch_lafan1_all.py --robots UnitreeH1 UnitreeG1 BoosterT1
Idempotent: existing outputs are skipped. Errors are logged and the loop continues.
"""
import argparse
import os
import time
import traceback
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
R = Path(__file__).resolve().parents[3]
OUT = R / "external_data/amass_converted/LAFAN1_all"

ALL = ["dance1_subject1", "dance1_subject2", "dance1_subject3",
       "dance2_subject1", "dance2_subject2", "dance2_subject3", "dance2_subject4", "dance2_subject5",
       "fallAndGetUp1_subject1", "fallAndGetUp1_subject4", "fallAndGetUp1_subject5",
       "fallAndGetUp2_subject2", "fallAndGetUp2_subject3", "fallAndGetUp3_subject1",
       "fight1_subject2", "fight1_subject3", "fight1_subject5", "fightAndSports1_subject1", "fightAndSports1_subject4",
       "jumps1_subject1", "jumps1_subject2", "jumps1_subject5",
       "run1_subject2", "run1_subject5", "run2_subject1", "run2_subject4",
       "sprint1_subject2", "sprint1_subject4",
       "walk1_subject1", "walk1_subject2", "walk1_subject5",
       "walk2_subject1", "walk2_subject3", "walk2_subject4",
       "walk3_subject1", "walk3_subject2", "walk3_subject3", "walk3_subject4", "walk3_subject5",
       "walk4_subject1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robots", nargs="+", required=True)
    ap.add_argument("--motions", nargs="*", default=ALL)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    from loco_mujoco.datasets.humanoids.LAFAN1.load import load_lafan1_trajectory
    for robot in a.robots:
        d = Path(a.out) / robot
        d.mkdir(parents=True, exist_ok=True)
        for m in a.motions:
            out = d / f"{m}.npz"
            if out.exists():
                print(f"{robot}/{m}: exists", flush=True)
                continue
            t = time.time()
            try:
                traj = load_lafan1_trajectory(robot, [m])
                traj.save(str(out))
                import numpy as np
                z = np.load(out, allow_pickle=True)
                print(f"{robot}/{m}: OK qpos {z['qpos'].shape} sites {z['site_xpos'].shape} {time.time() - t:.0f}s", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"{robot}/{m}: FAIL {type(e).__name__}: {str(e)[:160]}", flush=True)
                traceback.print_exc()
    print("FETCH DONE", flush=True)


if __name__ == "__main__":
    main()
