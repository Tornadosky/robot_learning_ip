"""C2 -- do the two H1 body-randomization paths produce the *same body*?

The pipeline this goal needs has two halves that must agree on what a body is:

  * **offline retargeting / rendering** works on a real ``MjModel``, so it uses
    the XML variant generator (``scripts/h1_morphology_variants.py``), which
    edits an ``MjSpec`` and writes a new ``h1.xml``;
  * **training** must not pay one XLA compile per body, so it uses the dynamic
    same-topology array path (``scripts/scaling/online_h1.py::_apply_morphology``),
    which patches ``body_pos / body_ipos / body_mass / body_inertia / site_pos``
    on a single graph.

If those two disagree, every offline reference is retargeted to a body the
policy never actually controls. This script measures the disagreement instead
of assuming it away:

1. rebuild each morphology through both paths;
2. diff every model array, not just the five the dynamic path touches;
3. run forward kinematics on shared poses and report site-position divergence
   (the mimic sites are exactly what ``MimicReward`` scores);
4. run the physical-validity checks Phase 2 of the goal document requires.

Pure MuJoCo — runs on Windows or WSL, no dataset, no GPU, seconds per body.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE / "scripts"))

from h1_morphology_variants import H1MorphologyPreset, create_h1_variant_xml  # noqa: E402

MIMIC_SITES = (
    "upper_body_mimic",
    "left_hand_mimic",
    "left_foot_mimic",
    "right_hand_mimic",
    "right_foot_mimic",
)

# The dimensions the dynamic path supports (online_h1.MORPHOLOGY_NAMES).
DYNAMIC_DIMS = (
    "leg_length_scale", "arm_length_scale", "shoulder_width_scale",
    "torso_mass_scale", "torso_length_scale", "total_mass_scale",
    "damping_scale", "armature_scale", "strength_scale", "friction_scale",
    "torso_com_x_offset",
)

# Arrays worth diffing. Anything not in the dynamic path's write set that still
# differs between the two models is a genuine divergence.
COMPARED_ARRAYS = (
    "body_pos", "body_ipos", "body_mass", "body_inertia", "body_quat",
    "site_pos", "site_quat", "jnt_pos", "jnt_axis", "jnt_range",
    "geom_pos", "geom_size", "geom_quat", "mesh_vert",
    "actuator_gainprm", "actuator_biasprm", "actuator_ctrlrange",
    "dof_damping", "dof_armature", "geom_friction",
)
DYNAMIC_WRITE_SET = {
    "body_pos", "body_ipos", "body_mass", "body_inertia", "site_pos",
    "dof_damping", "dof_armature", "actuator_gainprm", "geom_friction",
}


def body_id(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def site_id(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)


def apply_dynamic_morphology(model: mujoco.MjModel, morph: dict) -> mujoco.MjModel:
    """Replicate ``online_h1.OnlineMorphMjxUnitreeH1._apply_morphology`` on an MjModel.

    Kept deliberately array-for-array identical to the JAX version so the
    comparison below is about the *method*, not about a reimplementation.
    """
    leg, arm, shoulder, torso_mass = (
        morph["leg_length_scale"], morph["arm_length_scale"],
        morph["shoulder_width_scale"], morph["torso_mass_scale"],
    )
    torso_len = morph.get("torso_length_scale", 1.0)
    total_mass = morph.get("total_mass_scale", 1.0)
    damping = morph.get("damping_scale", 1.0)
    armature = morph.get("armature_scale", 1.0)
    strength = morph.get("strength_scale", 1.0)
    friction = morph.get("friction_scale", 1.0)
    com_x = morph.get("torso_com_x_offset", 0.0)
    out = mujoco.MjModel.from_xml_path(model_xml_of(model))  # fresh nominal copy

    leg_inertial_ids = np.array([body_id(out, f"{s}_{l}_link")
                                 for s in ("left", "right") for l in ("hip_pitch", "knee")])
    leg_position_ids = np.array([body_id(out, f"{s}_{l}_link")
                                 for s in ("left", "right") for l in ("knee", "ankle")])
    upper_arm_ids = np.array([body_id(out, f"{s}_shoulder_yaw_link") for s in ("left", "right")])
    forearm_ids = np.array([body_id(out, f"{s}_elbow_link") for s in ("left", "right")])
    shoulder_ids = np.array([body_id(out, f"{s}_shoulder_pitch_link") for s in ("left", "right")])
    hand_site_ids = np.array([site_id(out, f"{s}_hand_mimic") for s in ("left", "right")])
    upper_body_site_id = site_id(out, "upper_body_mimic")
    torso_id = body_id(out, "torso_link")

    out.body_pos[leg_position_ids] *= np.array([1.0, 1.0, leg])
    out.body_pos[upper_arm_ids] *= np.array([1.0, 1.0, arm])
    out.body_pos[forearm_ids] *= arm
    out.body_pos[shoulder_ids] *= np.array([1.0, shoulder, torso_len])

    body_scale = np.ones_like(out.body_ipos)
    body_scale[leg_inertial_ids, 2] = leg
    body_scale[upper_arm_ids, 2] = arm
    body_scale[forearm_ids, 0] = arm
    body_scale[torso_id, 2] = torso_len
    volume_scale = np.prod(body_scale, axis=1)
    inertia_scale = volume_scale * np.square(np.mean(body_scale, axis=1))

    out.body_ipos[:] = out.body_ipos * body_scale
    out.body_mass[:] = out.body_mass * volume_scale
    out.body_inertia[:] = out.body_inertia * inertia_scale[:, None]
    out.body_mass[torso_id] *= torso_mass
    out.body_inertia[torso_id] *= torso_mass
    out.body_mass[:] *= total_mass
    out.body_inertia[:] *= total_mass
    out.body_ipos[torso_id, 0] += com_x

    out.site_pos[hand_site_ids] *= np.array([arm, 1.0, 1.0])
    out.site_pos[upper_body_site_id] *= np.array([1.0, 1.0, torso_len])

    out.dof_damping[:] *= damping
    out.dof_armature[:] *= armature
    out.actuator_gainprm[:, 0] *= strength
    out.geom_friction[:, 0] *= friction
    return out


_XML_OF: dict[int, str] = {}


def model_xml_of(model: mujoco.MjModel) -> str:
    return _XML_OF[id(model)]


def load(xml: Path) -> mujoco.MjModel:
    m = mujoco.MjModel.from_xml_path(str(xml))
    _XML_OF[id(m)] = str(xml)
    return m


def array_diff(a: np.ndarray, b: np.ndarray) -> dict:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return {"shape_mismatch": [list(a.shape), list(b.shape)]}
    d = np.abs(a - b)
    denom = np.maximum(np.abs(a), np.abs(b))
    rel = np.where(denom > 1e-12, d / np.maximum(denom, 1e-12), 0.0)
    return {
        "max_abs": float(d.max()) if d.size else 0.0,
        "max_rel": float(rel.max()) if rel.size else 0.0,
        "n_changed": int((d > 1e-9).sum()),
        "n_total": int(d.size),
    }


def fk_sites(model: mujoco.MjModel, qpos_batch: np.ndarray) -> np.ndarray:
    data = mujoco.MjData(model)
    ids = [site_id(model, n) for n in MIMIC_SITES]
    out = np.zeros((len(qpos_batch), len(ids), 3))
    for i, q in enumerate(qpos_batch):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        out[i] = data.site_xpos[ids]
    return out


def validity(model: mujoco.MjModel) -> dict:
    data = mujoco.MjData(model)
    if model.nkey:
        data.qpos[:] = model.key_qpos[0]
    mujoco.mj_forward(model, data)
    ok_step = True
    try:
        for _ in range(10):
            mujoco.mj_step(model, data)
        ok_step = bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
    except Exception:
        ok_step = False
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    feet = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
            for n in ("left_foot", "right_foot")]
    data2 = mujoco.MjData(model)
    if model.nkey:
        data2.qpos[:] = model.key_qpos[0]
    mujoco.mj_forward(model, data2)
    foot_dist = [float(mujoco.mj_geomDistance(model, data2, floor, f, 10.0, None)) for f in feet]
    return {
        "total_mass_kg": float(model.body_mass.sum()),
        "min_body_mass": float(model.body_mass[1:].min()),
        "min_inertia": float(model.body_inertia[1:].min()),
        "all_finite": bool(np.isfinite(model.body_pos).all()
                           and np.isfinite(model.body_inertia).all()),
        "positive_mass": bool((model.body_mass[1:] > 0).all()),
        "positive_inertia": bool((model.body_inertia[1:] > 0).all()),
        "valid_jnt_ranges": bool((model.jnt_range[:, 1] >= model.jnt_range[:, 0]).all()),
        "keyframe_foot_floor_distance_m": foot_dist,
        "ten_steps_finite": ok_step,
        "standing_height_m": float(data2.qpos[2]),
    }


def model_hash(model: mujoco.MjModel) -> str:
    h = hashlib.sha256()
    for name in COMPARED_ARRAYS:
        arr = getattr(model, name, None)
        if arr is None:
            continue
        h.update(name.encode())
        h.update(np.ascontiguousarray(np.asarray(arr, dtype=np.float64)).tobytes())
    return h.hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--xml-root", type=Path,
                    default=WORKSPACE / "experiments" / "h1_morphology_deepmimic_20260808" / "body_catalog" / "xml")
    ap.add_argument("--n-poses", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # Deterministic catalog: nominal + 4 seeded bodies inside the online_h1
    # default bounds (all dims). Foot scale is pinned to 1.0 because the dynamic
    # path has no foot dimension -- including it would guarantee a mismatch for
    # a reason that has nothing to do with the question being asked.
    from scaling.online_h1 import MORPHOLOGY_SPEC

    assert tuple(n for n, _, _ in MORPHOLOGY_SPEC) == DYNAMIC_DIMS, (
        "c2's DYNAMIC_DIMS is out of sync with online_h1.MORPHOLOGY_SPEC"
    )
    low = np.array([lo for _, lo, _ in MORPHOLOGY_SPEC])
    high = np.array([hi for _, _, hi in MORPHOLOGY_SPEC])
    nominal_vals = [1.0 if n.endswith("_scale") else 0.0 for n in DYNAMIC_DIMS]
    rng = np.random.default_rng(args.seed)
    catalog = [("body00_nominal", dict(zip(DYNAMIC_DIMS, nominal_vals)), args.seed)]
    for i in range(1, 5):
        sub = np.random.default_rng(1000 + i)
        vals = sub.uniform(low, high)
        catalog.append((f"body{i:02d}_seed{1000 + i}", dict(zip(DYNAMIC_DIMS, vals.tolist())), 1000 + i))

    nominal_xml = Path(
        __import__("loco_mujoco").environments.UnitreeH1.get_default_xml_file_path()
    )
    nominal = load(nominal_xml)

    # Shared random poses within joint limits for the FK comparison.
    lo, hi = nominal.jnt_range[1:, 0], nominal.jnt_range[1:, 1]
    poses = np.tile(np.zeros(nominal.nq), (args.n_poses, 1))
    poses[:, 2] = 1.0
    poses[:, 3] = 1.0
    poses[:, 7:] = rng.uniform(lo, hi, size=(args.n_poses, nominal.nq - 7))

    args.xml_root.mkdir(parents=True, exist_ok=True)
    records = []
    for name, morph, seed in catalog:
        preset = H1MorphologyPreset(name=f"c2_{name}", label=name, **morph)
        xml_path = create_h1_variant_xml(preset, output_root=args.xml_root)
        xml_model = load(xml_path)
        dyn_model = apply_dynamic_morphology(nominal, morph)
        _XML_OF[id(dyn_model)] = str(nominal_xml)

        diffs = {}
        for arr in COMPARED_ARRAYS:
            a, b = getattr(xml_model, arr, None), getattr(dyn_model, arr, None)
            if a is None or b is None:
                continue
            d = array_diff(a, b)
            if d.get("shape_mismatch") or d.get("n_changed", 0) > 0:
                diffs[arr] = d

        s_xml, s_dyn = fk_sites(xml_model, poses), fk_sites(dyn_model, poses)
        err = np.linalg.norm(s_xml - s_dyn, axis=-1)

        # How different is this body from nominal at all? (guards "descriptor
        # changed but the physics did not")
        s_nom = fk_sites(nominal, poses)
        vs_nom = np.linalg.norm(s_xml - s_nom, axis=-1)

        records.append({
            "body_id": name,
            "seed": seed,
            "morphology": morph,
            "xml_path": str(xml_path),
            "xml_model_hash": model_hash(xml_model),
            "dynamic_model_hash": model_hash(dyn_model),
            "arrays_differing_xml_vs_dynamic": diffs,
            "arrays_differing_outside_dynamic_write_set": sorted(
                set(diffs) - DYNAMIC_WRITE_SET
            ),
            "fk_site_error_xml_vs_dynamic_m": {
                "mean": float(err.mean()), "max": float(err.max()),
                "per_site_mean": {s: float(err[:, i].mean()) for i, s in enumerate(MIMIC_SITES)},
            },
            "fk_site_distance_from_nominal_m": {
                "mean": float(vs_nom.mean()), "max": float(vs_nom.max()),
                "per_site_mean": {s: float(vs_nom[:, i].mean()) for i, s in enumerate(MIMIC_SITES)},
            },
            "validity_xml": validity(xml_model),
            "validity_dynamic": validity(dyn_model),
        })
        print(f"{name}: fk err xml-vs-dyn mean={err.mean() * 1000:.3f} mm "
              f"max={err.max() * 1000:.3f} mm | dist from nominal mean={vs_nom.mean() * 1000:.1f} mm "
              f"| extra arrays: {sorted(set(diffs) - DYNAMIC_WRITE_SET)}")

    out = {
        "component": "C2_body_equivalence",
        "question": "does the XML variant path build the same body as the dynamic array path?",
        "bounds": {"low": low.tolist(), "high": high.tolist(), "dims": list(DYNAMIC_DIMS)},
        "n_poses": args.n_poses,
        "nominal_xml": str(nominal_xml),
        "mujoco": mujoco.__version__,
        "bodies": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
