#!/usr/bin/env python3
"""Do the clip's joints and the TRAINING model's joints describe the same robot?

`derive_clip_signs.py` reproduces H1/G1/BoosterT1's recorded world poses from
their clip angles to a residual of ~1.5e-4 by choosing a sign per joint. On
Atlas, Talos and ToddlerBot it cannot: residuals are 0.043 / 0.830 / 2.395, i.e.
21x / 415x / 1198x the validated families. So for those three the clip angles do
NOT map onto the loco_mjx model by a sign table, and the reference a policy would
chase is not the motion the clip records.

Feasibility (`reference_feasibility_matrix.py`) compares the clip's column for a
joint NAME against that name's range in the loco_mjx model. That is only
meaningful if the two are the same joint. This checks the precondition:

  * joint-name overlap between clip and training model
  * qpos width vs the model's nq
  * link geometry: body-to-parent offsets, which a sign flip cannot change

If names and counts line up but geometry does not, the two stacks ship different
BODIES under the same name and no sign table will ever fix it.

    .venv/Scripts/python.exe experiments/fsq_khaendler/clip_model_agreement.py
"""
from pathlib import Path

import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[2]
import os as _os
CLIPS = Path(_os.environ.get("CLIPS_DIR", str(ROOT / "external_data/amass_converted/LAFAN1")))
MJX = ROOT / "loco_mjx/loco_mjx/environments/robots/{}/data/plane.xml"
LMJ = ROOT / ".venv/Lib/site-packages/loco_mujoco_models/{}/{}.xml"

FAMILIES = {
    "UnitreeH1": "unitree_h1", "UnitreeG1": "unitree_g1",
    "BoosterT1": "booster_t1", "Atlas": "atlas",
    "Talos": "talos", "ToddlerBot": "toddlerbot",
    "Apollo": "apptronik_apollo", "FourierGR1T2": "fourier_gr1t2",
}


def joints(model):
    out = {}
    for j in range(model.njnt):
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if n and model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE:
            out[n] = j
    return out


def bodies(model):
    return {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b): b
            for b in range(model.nbody)}


def main():
    print(f"{'robot':<12}{'clip jnts':>10}{'mjx jnts':>10}{'matched':>9}"
          f"{'clip nq':>9}{'mjx nq':>8}   {'geometry (body_pos vs the fit model)':<40}")
    for sub, mdir in FAMILIES.items():
        cp = CLIPS / sub / "dance2_subject4.npz"
        mx = Path(str(MJX).format(mdir))
        lm = Path(str(LMJ).format(mdir, mdir))
        if not cp.exists() or not mx.exists():
            print(f"{sub:<12} missing clip or model")
            continue
        d = np.load(cp, allow_pickle=True)
        cj = [str(n) for n in d["joint_names"]][1:]
        nq = int(np.asarray(d["qpos"]).shape[1])
        m_mjx = mujoco.MjModel.from_xml_path(str(mx))
        jm = joints(m_mjx)
        matched = sum(1 for n in cj if n in jm)

        geo = "loco_mujoco model not found"
        if lm.exists():
            m_src = mujoco.MjModel.from_xml_path(str(lm))
            bs, bm = bodies(m_src), bodies(m_mjx)
            shared = [b for b in bs if b in bm and b not in (None, "world")]
            if shared:
                diff = np.array([
                    np.linalg.norm(m_src.body_pos[bs[b]] - m_mjx.body_pos[bm[b]])
                    for b in shared])
                geo = (f"{len(shared)}/{len(bs)} bodies shared, "
                       f"max |dpos| {1000 * diff.max():7.1f} mm, "
                       f"mean {1000 * diff.mean():6.1f} mm")
            else:
                geo = f"NO shared body names ({len(bs)} vs {len(bm)})"
        print(f"{sub:<12}{len(cj):>10}{len(jm):>10}{matched:>9}"
              f"{nq:>9}{m_mjx.nq:>8}   {geo:<40}")


if __name__ == "__main__":
    main()
