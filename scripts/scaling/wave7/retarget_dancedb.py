"""Batch-retarget the local AMASS DanceDB corpus (77 dances) to UnitreeH1 through
loco-mujoco's SMPL pipeline, writing clips in our LAFAN1 format (extend_motion:
qpos/qvel/xpos/site_xpos, 28 keys). Other robots then come from
experiments/fsq_khaendler/reissue_clips.py --src-dir <out> --targets ...

  .venv/Scripts/python.exe scripts/scaling/wave7/retarget_dancedb.py [--limit N] [--subjects 20140526_StephanosKoullapis ...]
Resumable (skips existing outputs). torch CUDA. Log: prints one line per clip.
"""
import argparse
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
R = Path(__file__).resolve().parents[3]
AMASS = R / "external_data/amass/DanceDB"
OUT = R / "external_data/amass_converted/DanceDB/UnitreeH1"


def load_amass_npz(path):
    """Same dict as loco_mujoco.smpl.retargeting.load_amass_data, from an explicit file."""
    import numpy as np
    e = dict(np.load(open(path, "rb"), allow_pickle=True))
    fr = e.get("mocap_framerate", e.get("mocap_frame_rate"))
    if fr is None:
        raise ValueError("Framerate not found in the data file.")
    trans = e["trans"]
    pose_aa = np.concatenate([e["poses"][:, :66], np.zeros((trans.shape[0], 6))], axis=-1)
    return {"pose_aa": pose_aa, "gender": e["gender"], "trans": trans, "betas": e["betas"], "fps": fr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--env", default="UnitreeH1")
    ap.add_argument("--motion-iterations", type=int, default=None, help="override conf (fewer = faster)")
    a = ap.parse_args()
    sys.path.insert(0, str(R / "loco-mujoco"))
    # LOCOMUJOCO_VARIABLES.yaml carries WSL paths; the Windows venv needs native ones
    import loco_mujoco
    vars_path = R / "tmp" / "locomujoco_vars_native.yaml"
    vars_path.parent.mkdir(parents=True, exist_ok=True)
    ext = R / "external_data"
    vars_path.write_text("".join(f"{k}: {v}\n" for k, v in {
        "LOCOMUJOCO_AMASS_PATH": ext / "amass", "LOCOMUJOCO_CONVERTED_AMASS_PATH": ext / "amass_converted/AMASS",
        "LOCOMUJOCO_CONVERTED_DEFAULT_PATH": ext / "amass_converted/DEFAULT", "LOCOMUJOCO_CONVERTED_LAFAN1_PATH": ext / "amass_converted/LAFAN1",
        "LOCOMUJOCO_MODELS_PATH": R / "tmp/lm_models", "LOCOMUJOCO_SMPL_MODEL_PATH": ext / "smpl"}.items()))
    loco_mujoco.PATH_TO_VARIABLES = str(vars_path)
    from loco_mujoco.smpl.retargeting import (load_robot_conf_file, load_amass_data, fit_smpl_shape, fit_smpl_motion,
                                              extend_motion, get_smpl_model_path)
    from loco_mujoco.smpl.retargeting import get_converted_amass_dataset_path
    from loco_mujoco.utils import setup_logger
    conf = load_robot_conf_file(a.env)
    if a.motion_iterations:
        conf.optimization_params.motion_iterations = a.motion_iterations
    logger = setup_logger("dancedb", identifier="[DanceDB retarget]")
    shape_path = Path(get_converted_amass_dataset_path()) / a.env / "shape_optimized.pkl"
    if not shape_path.exists():
        shape_path.parent.mkdir(parents=True, exist_ok=True)
        fit_smpl_shape(a.env, conf, get_smpl_model_path(), str(shape_path), logger)
    OUT.mkdir(parents=True, exist_ok=True)
    subjects = a.subjects or sorted(p.name for p in AMASS.iterdir() if p.is_dir())
    done = 0
    for subj in subjects:
        for clip in sorted((AMASS / subj).glob("*_poses.npz")):
            stem = f"{subj.split('_', 1)[-1]}__{clip.stem.removesuffix('_poses')}"
            out = OUT / f"{stem}.npz"
            if out.exists():
                print(f"{stem}: exists", flush=True); continue
            t = time.time()
            try:
                motion = load_amass_npz(clip)  # loco-mujoco's registry lookup breaks on Windows paths; read the file directly
                traj = fit_smpl_motion(a.env, conf, get_smpl_model_path(), motion, str(shape_path), logger, skip_steps=False)
                traj = extend_motion(a.env, conf.env_params, traj)
                traj.save(str(out))
                import numpy as np
                z = np.load(out, allow_pickle=True)
                print(f"{stem}: OK qpos {z['qpos'].shape} freq {float(z['frequency'])} {time.time() - t:.0f}s", flush=True)
                done += 1
            except Exception as e:  # noqa: BLE001
                print(f"{stem}: FAIL {type(e).__name__}: {str(e)[:160]} ({time.time() - t:.0f}s)", flush=True)
                traceback.print_exc()
            if a.limit and done >= a.limit:
                print("limit reached", flush=True); return
    print("DANCEDB DONE", flush=True)


if __name__ == "__main__":
    main()
