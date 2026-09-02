"""Screen a robot family for tracking: name aliases (clip -> loco_mjx model),
joint signs read from the joint-axis comparison against the loco-mujoco model
the clip was retargeted on (falls back to the greedy sweep for joints absent
there), and the FK residual that results. Writes a JSON table usable by
clip_reference.py (see EXTRA tables).

  .venv/Scripts/python.exe scripts/scaling/wave7/screen_family.py --robot gr1t2 \
      --clip-dir external_data/amass_converted/LAFAN1_all --out experiments/fsq_khaendler/clip_tables_gr1t2.json
"""
import argparse
import json
import re
import sys
from pathlib import Path

import mujoco
import numpy as np

R = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(R / "scripts/scaling"))
import derive_clip_signs as D  # noqa: E402

LM_XML = {"apollo": "apptronik_apollo/apptronik_apollo.xml", "gr1t2": "fourier_gr1t2/gr1t2.xml", "atlas": "atlas/atlas.xml",
          "talos": "talos/talos.xml", "h1": "unitree_h1/h1.xml", "g1": "unitree_g1/g1.xml", "booster_t1": "booster_t1/t1.xml"}

# clip name -> loco_mjx model name. Functions return None when no rule applies.
def alias_gr1t2(name, kind):
    if kind == "joint" and name.startswith("joint_"):
        return name[len("joint_"):] + "_joint"
    if kind == "body":
        if name == "base":
            return "trunk"
        if name.startswith("link_"):
            return name[len("link_"):] + "_link"
    return None

ATLAS_JOINTS = {"ankle_angle_l": "ankle_angle_y_l", "l_leg_akx": "ankle_angle_x_l",
                "ankle_angle_r": "ankle_angle_y_r", "r_leg_akx": "ankle_angle_x_r"}

def alias_atlas(name, kind):
    return ATLAS_JOINTS.get(name) if kind == "joint" else None

ALIASES = {"gr1t2": alias_gr1t2, "atlas": alias_atlas}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", required=True)
    ap.add_argument("--clip", default="dance2_subject4.npz")
    ap.add_argument("--clip-dir", required=True)
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    model_dir, clip_sub = D.FAMILIES[a.robot]
    model = mujoco.MjModel.from_xml_path(str(D.ROBOT_DIR / model_dir / "data" / "plane.xml"))
    data = mujoco.MjData(model)
    clip = D.load_clip(Path(a.clip_dir) / clip_sub / a.clip)
    rule = ALIASES.get(a.robot, lambda n, k: None)

    # --- rename clip joints/bodies to model names
    jalias, balias = {}, {}
    model_joints = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(model.njnt)}
    model_bodies = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(model.nbody)}
    for jn in clip["joint_names"]:
        if jn in model_joints:
            continue
        cand = rule(jn, "joint")
        if cand and cand in model_joints:
            jalias[jn] = cand
    for bn in clip["body_names"]:
        if bn in model_bodies:
            continue
        cand = rule(bn, "body")
        if cand and cand in model_bodies:
            balias[bn] = cand
    clip["joint_names"] = [jalias.get(n, n) for n in clip["joint_names"]]
    clip["body_names"] = [balias.get(n, n) for n in clip["body_names"]]

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
    unmatched = [jn for jn, jt in zip(clip["joint_names"], clip["jnt_type"]) if jn not in names and int(jt) != mujoco.mjtJoint.mjJNT_FREE]
    shared, body_ids, clip_body_idx = [], [], []
    for bi, bn in enumerate(clip["body_names"]):
        if bn == "world":
            continue
        mid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bn)
        if mid >= 0:
            shared.append(bn); body_ids.append(mid); clip_body_idx.append(bi)
    body_ids = np.asarray(body_ids); clip_body_idx = np.asarray(clip_body_idx)
    print(f"{a.robot}: joints matched {len(names)} (aliased {len(jalias)}), unmatched clip joints {unmatched}; bodies shared {len(shared)} (aliased {len(balias)})")

    # --- signs from the axis comparison with the loco-mujoco model
    signs = np.ones(len(names)); source = {}
    import importlib.util
    spec = importlib.util.find_spec("loco_mujoco_models")
    lm = Path(list(spec.submodule_search_locations)[0]) / LM_XML[a.robot] if spec else None
    m2 = mujoco.MjModel.from_xml_path(str(lm)) if lm and lm.exists() else None
    inv = {v: k for k, v in jalias.items()}
    for i, jn in enumerate(names):
        if m2 is None:
            break
        j1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        j2 = mujoco.mj_name2id(m2, mujoco.mjtObj.mjOBJ_JOINT, inv.get(jn, jn))
        if j2 < 0:
            source[jn] = "unknown"; continue
        a1, a2 = model.jnt_axis[j1], m2.jnt_axis[j2]
        b1 = model.body_quat[model.jnt_bodyid[j1]]; b2 = m2.body_quat[m2.jnt_bodyid[j2]]
        if np.allclose(a1, -a2, atol=1e-3) and np.allclose(b1, b2, atol=1e-3):
            signs[i] = -1.0; source[jn] = "axis-flip"
        elif np.allclose(a1, a2, atol=1e-3) and np.allclose(b1, b2, atol=1e-3):
            source[jn] = "same"
        else:
            source[jn] = "frame-differs"
    frames = D.pick_frames(clip["qpos"][:, np.asarray(col_of)], a.frames)
    ref_pos = clip["xpos"][np.ix_(frames, clip_body_idx)]; ref_pos = ref_pos - ref_pos[:, :1, :]
    ref_quat = clip["xquat"][np.ix_(frames, clip_body_idx)]

    def cost(sg):
        pos, quat = D.fk_poses(model, data, clip, frames, col_of, adr_of, sg, body_ids)
        e_pos = float(np.linalg.norm(pos - ref_pos, axis=-1).mean())
        dot = np.abs((quat * ref_quat).sum(axis=-1)).clip(0.0, 1.0)
        return e_pos + float((2.0 * np.arccos(dot)).mean()), pos

    c0, _ = cost(np.ones(len(names)))
    c1, pos = cost(signs)
    # greedy refinement for joints the axis rule could not decide
    best = c1
    for _ in range(4):
        improved = False
        for i, jn in enumerate(names):
            if source.get(jn) in ("axis-flip", "same"):
                continue
            t = signs.copy(); t[i] = -t[i]
            c, _ = cost(t)
            if c < best - 1e-9:
                signs, best, improved = t, c, True
        if not improved:
            break
    c2, pos = cost(signs)
    err = np.linalg.norm(pos - ref_pos, axis=-1).mean(axis=0)
    print(f"residual: all+1 {c0:.5f} -> axis-signs {c1:.5f} -> refined {c2:.5f}   ({int((signs < 0).sum())} negated; validated families ~1.5e-4)")
    print("worst bodies:", ", ".join(f"{bn} {e:.3f}" for bn, e in sorted(zip(shared, err), key=lambda x: -x[1])[:6]))
    print("undecided joints:", [jn for jn in names if source.get(jn) not in ("axis-flip", "same")])
    if a.out:
        json.dump({"robot": a.robot, "clip_subdir": clip_sub, "model_dir": model_dir, "joint_aliases": jalias, "body_aliases": balias,
                   "signs": {jn: float(s) for jn, s in zip(names, signs)}, "sign_source": source,
                   "residual": c2, "unmatched_clip_joints": unmatched}, open(a.out, "w"), indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
