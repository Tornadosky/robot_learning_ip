"""Where does a family's FK residual come from? Per-body position error after
applying a sign table, plus a joint-axis comparison between the loco_mjx model
and the loco-mujoco model the clip was retargeted on.

  .venv/Scripts/python.exe scripts/scaling/wave7/fk_residual_breakdown.py --robot apollo \
      --clip-dir external_data/amass_converted/LAFAN1_all --signs experiments/fsq_khaendler/clip_signs_apollo.json
"""
import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

R = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(R / "scripts/scaling"))
import derive_clip_signs as D  # noqa: E402

LM_XML = {  # loco-mujoco model xml (the retarget target) per family key
    "apollo": "apptronik_apollo/apptronik_apollo.xml", "gr1t2": "fourier_gr1t2/gr1t2.xml", "atlas": "atlas/atlas.xml",
    "talos": "talos/talos.xml", "h1": "unitree_h1/h1.xml", "g1": "unitree_g1/g1.xml", "booster_t1": "booster_t1/t1.xml",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", required=True)
    ap.add_argument("--clip", default="dance2_subject4.npz")
    ap.add_argument("--clip-dir", required=True)
    ap.add_argument("--signs", default=None, help="json from derive_clip_signs (key = robot or top-level)")
    ap.add_argument("--frames", type=int, default=40)
    a = ap.parse_args()
    model_dir, clip_sub = D.FAMILIES[a.robot]
    model = mujoco.MjModel.from_xml_path(str(D.ROBOT_DIR / model_dir / "data" / "plane.xml"))
    data = mujoco.MjData(model)
    clip = D.load_clip(Path(a.clip_dir) / clip_sub / a.clip)
    free_nq = 7 if int(clip["jnt_type"][0]) == mujoco.mjtJoint.mjJNT_FREE else 0
    names, col_of, adr_of = [], [], []
    col = free_nq
    for jn, jt in zip(clip["joint_names"], clip["jnt_type"]):
        if int(jt) == mujoco.mjtJoint.mjJNT_FREE:
            continue
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid >= 0:
            names.append(jn); col_of.append(col); adr_of.append(model.jnt_qposadr[jid])
        col += 1
    unmatched = [jn for jn in clip["joint_names"] if jn not in names and jn != clip["joint_names"][0]]
    print(f"{a.robot}: {len(names)} matched joints, unmatched clip joints: {unmatched}")
    signs = np.ones(len(names))
    if a.signs:
        js = json.load(open(a.signs)); js = js.get(a.robot, js).get("signs", js.get(a.robot, js))
        signs = np.array([js.get(n, 1.0) for n in names])
    shared, body_ids, clip_body_idx = [], [], []
    for bi, bn in enumerate(clip["body_names"]):
        if bn == "world":
            continue
        mid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bn)
        if mid >= 0:
            shared.append(bn); body_ids.append(mid); clip_body_idx.append(bi)
    body_ids = np.asarray(body_ids); clip_body_idx = np.asarray(clip_body_idx)
    frames = D.pick_frames(clip["qpos"][:, np.asarray(col_of)], a.frames)
    ref_pos = clip["xpos"][np.ix_(frames, clip_body_idx)]; ref_pos = ref_pos - ref_pos[:, :1, :]
    pos, quat = D.fk_poses(model, data, clip, frames, col_of, adr_of, signs, body_ids)
    err = np.linalg.norm(pos - ref_pos, axis=-1).mean(axis=0)
    print("per-body mean |dpos| (m), worst first:")
    for bn, e in sorted(zip(shared, err), key=lambda x: -x[1])[:14]:
        print(f"  {bn:32s} {e:.4f}")
    # joint-axis comparison with the loco-mujoco model
    import importlib.util
    spec = importlib.util.find_spec("loco_mujoco_models")
    if spec and spec.submodule_search_locations and a.robot in LM_XML:
        lm = Path(list(spec.submodule_search_locations)[0]) / LM_XML[a.robot]
        if lm.exists():
            m2 = mujoco.MjModel.from_xml_path(str(lm))
            print(f"joint axes loco_mjx vs loco-mujoco ({lm.name}); listing differences:")
            nd = 0
            for jn in names:
                j1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn); j2 = mujoco.mj_name2id(m2, mujoco.mjtObj.mjOBJ_JOINT, jn)
                if j2 < 0:
                    print(f"  {jn}: absent in loco-mujoco model"); continue
                a1, a2 = model.jnt_axis[j1], m2.jnt_axis[j2]
                b1 = model.body_quat[model.jnt_bodyid[j1]]; b2 = m2.body_quat[m2.jnt_bodyid[j2]]
                r1 = model.jnt_range[j1]; r2 = m2.jnt_range[j2]
                if not np.allclose(a1, a2, atol=1e-3) or not np.allclose(b1, b2, atol=1e-3):
                    nd += 1; print(f"  {jn:22s} axis {a1} vs {a2} | body_quat {np.round(b1,3)} vs {np.round(b2,3)} | range {np.round(r1,2)} vs {np.round(r2,2)}")
            print(f"  {nd} joints differ in axis/body frame out of {len(names)}")
        else:
            print("loco-mujoco xml not found:", lm)


if __name__ == "__main__":
    main()
