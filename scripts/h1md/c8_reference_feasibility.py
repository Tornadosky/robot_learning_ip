"""C8 -- can a body physically follow its reference, before any RL is involved?

Phase 4 of the goal document trains single-body specialists to separate
"reference or actuator infeasible" from "shared training failed". A specialist
costs a training run each. A PD controller tracking the reference joint angles
answers the same question in seconds and cannot be confounded by an RL
hyperparameter, so it runs first and the specialists only need to run where it
disagrees.

Per body x reference method this reports:

* **pairing assertion** -- the reference handed to the environment is the one
  built for that body (the goal document's "assert against mismatches");
* **zero-action rollout** -- the matched control every claim must beat;
* **PD-tracking rollout** -- torque = kp*(q_ref - q) - kd*qd, over a small gain
  sweep, giving survival time and tracking error under real physics/contact.

A reference no PD controller can hold is a reference PPO should not be asked to
hold either. Runs on CPU MuJoCo in seconds; no compile, no GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE / "scripts"))
sys.path.insert(0, str(WORKSPACE / "scripts" / "h1md"))

from c6_reward_discrimination import (  # noqa: E402
    E, W_DANCE, build_model, catalog, reground, site_quantities, terms, total,
)


def build_references(qpos, qvel, bodies, xml_root, kern_cls):
    """fk and ik_scaled references for every catalog body (see C3)."""
    import jax
    import jax.numpy as jnp
    from c3_reference_methods import LIMB_SCALE_INDEX

    kern = kern_cls(xml_root / "h1_morphology_c2_body00_nominal" / "h1.xml")
    Q = jnp.asarray(qpos.astype(np.float32))
    morphs = jnp.asarray(np.array(
        [[b[1]["leg_length_scale"], b[1]["arm_length_scale"], b[1]["shoulder_width_scale"]]
         for b in bodies], dtype=np.float32))
    nom_sites = np.asarray(kern.fk_bt(jnp.ones((1, 3), jnp.float32), Q))[0]
    nom_targets = jnp.asarray(nom_sites[:, kern.TARGET_SITES, :])
    root_nom = jnp.asarray(qpos[:, 0:3].astype(np.float32))

    def scaled_targets(morph):
        rel = nom_targets - root_nom[:, None, :]
        s = morph[LIMB_SCALE_INDEX][None, :, None]
        root = root_nom.at[:, 2].multiply(morph[0])
        return root[:, None, :] + rel * s

    xs, _ = kern.ik(morphs, Q, jax.vmap(scaled_targets)(morphs))
    xs = np.asarray(xs)

    refs = {}
    for bi, (name, morph) in enumerate(bodies):
        model = build_model(name, morph, xml_root)
        q_fk, _ = reground(model, qpos)
        q_ik = qpos.copy()
        q_ik[:, kern.ACT_QADR] = xs[bi, :, : kern.NU]
        q_ik[:, 0:3] += xs[bi, :, kern.NU: kern.NU + 3]
        # clamp to joint limits -- the GN solve is unconstrained (C3 caveat)
        q_ik[:, kern.ACT_QADR] = np.clip(q_ik[:, kern.ACT_QADR], kern.jnt_low, kern.jnt_high)
        refs[name] = {"fk": q_fk, "ik_scaled": q_ik}
    return refs, kern


def reference_qvel(source_qvel: np.ndarray, ref_qpos: np.ndarray, dt: float) -> np.ndarray:
    """Reference velocities: keep the mocap root velocities (incl. angular, which a
    finite difference of quaternions would get wrong) and finite-difference the
    actuated joints, which the IK solve may have changed."""
    qvel = source_qvel.copy()
    qvel[:, 6:] = np.gradient(ref_qpos[:, 7:], dt, axis=0)
    qvel[:, 0:3] = np.gradient(ref_qpos[:, 0:3], dt, axis=0)
    return qvel


def reference_site_quantities(model, ref_qpos, ref_qvel, site_ids, body_ids):
    """Precomputed once per (body, method) -- doing this inside the rollout loop
    dominated the runtime and bought nothing."""
    from loco_mujoco.core.utils.math import calculate_relative_site_quatities
    d = mujoco.MjData(model)
    out = []
    for q, v in zip(ref_qpos, ref_qvel):
        d.qpos[:] = q
        d.qvel[:] = v
        mujoco.mj_forward(model, d)
        p, _, _ = calculate_relative_site_quatities(d, site_ids, body_ids, model.body_rootid, np)
        out.append(p)
    return np.asarray(out)


def rollout(model, ref_qpos, ref_qvel, ref_rpos, act_qadr, dt, mode, kp=0.0, kd=0.0,
            fall_height_frac=0.6):
    """Roll the physics forward from the reference start; return survival + tracking."""
    from loco_mujoco.core.utils.math import calculate_relative_site_quatities

    data = mujoco.MjData(model)
    data.qpos[:] = ref_qpos[0]
    data.qvel[:] = ref_qvel[0]
    mujoco.mj_forward(model, data)
    start_z = float(data.qpos[2])
    n_sub = max(1, int(round(dt / model.opt.timestep)))

    site_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n)
                         for n in ("upper_body_mimic", "left_hand_mimic", "left_foot_mimic",
                                   "right_hand_mimic", "right_foot_mimic")])
    body_ids = np.array([model.site_bodyid[s] for s in site_ids])

    joint_err, site_err, rewards = [], [], []
    survived = len(ref_qpos)
    for i in range(len(ref_qpos)):
        if mode == "pd":
            tau = kp * (ref_qpos[i, act_qadr] - data.qpos[act_qadr]) - kd * data.qvel[act_qadr - 1]
            data.ctrl[:] = np.clip(tau, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
        else:
            data.ctrl[:] = 0.0
        for _ in range(n_sub):
            mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all() or data.qpos[2] < fall_height_frac * start_z:
            survived = i + 1
            break

        joint_err.append(np.abs(data.qpos[act_qadr] - ref_qpos[i, act_qadr]))
        p, _, _ = calculate_relative_site_quatities(data, site_ids, body_ids, model.body_rootid, np)
        site_err.append(np.linalg.norm(p - ref_rpos[i], axis=-1))
        qpos_dist = float(np.mean((data.qpos[act_qadr] - ref_qpos[i, act_qadr]) ** 2))
        rewards.append(total(terms(qpos_dist, 0.0, float(np.mean((p - ref_rpos[i]) ** 2)),
                                   0.0, 0.0, 0.0), W_DANCE))

    return {
        "survived_frames": int(survived),
        "survived_frac": float(survived / len(ref_qpos)),
        "mean_joint_error_deg": float(np.degrees(np.mean(joint_err))) if joint_err else float("nan"),
        "mean_site_error_cm": float(np.mean(site_err) * 100) if site_err else float("nan"),
        "mean_reward_dance_weights": float(np.mean(rewards)) if rewards else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--start", type=int, default=19482)
    ap.add_argument("--frames", type=int, default=800)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--kp", type=float, nargs="+", default=[100.0, 300.0, 800.0])
    ap.add_argument("--kd-frac", type=float, default=0.05)
    ap.add_argument("--xml-root", type=Path,
                    default=WORKSPACE / "experiments" / "h1_morphology_deepmimic_20260808" / "body_catalog" / "xml")
    args = ap.parse_args()

    from c3_reference_methods import IKKernel
    from loco_mujoco.task_factories import ImitationFactory, LAFAN1DatasetConf

    env = ImitationFactory.make("UnitreeH1", lafan1_dataset_conf=LAFAN1DatasetConf(["dance2_subject4"]))
    th = env.th.traj
    freq = float(th.info.frequency)
    qpos = np.asarray(th.data.qpos)[args.start:args.start + args.frames][:: args.stride].astype(np.float64)
    qvel = np.asarray(th.data.qvel)[args.start:args.start + args.frames][:: args.stride].astype(np.float64)
    dt = args.stride / freq
    print(f"{len(qpos)} frames @ {1 / dt:.1f} Hz control")

    bodies = catalog()
    refs, kern = build_references(qpos, qvel, bodies, args.xml_root, IKKernel)

    records = []
    for name, morph in bodies:
        model = build_model(name, morph, args.xml_root)
        entry = {"body_id": name, "morphology": morph, "methods": {}}
        for method, q_ref in refs[name].items():
            # pairing assertion: the reference must be the one built for THIS body
            other = [n for n, _ in bodies if n != name]
            same_as_other = any(np.allclose(q_ref, refs[o][method]) for o in other) if other else False
            v_ref = reference_qvel(qvel, q_ref, dt)
            site_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n)
                                 for n in ("upper_body_mimic", "left_hand_mimic", "left_foot_mimic",
                                           "right_hand_mimic", "right_foot_mimic")])
            body_ids = np.array([model.site_bodyid[s] for s in site_ids])
            r_ref = reference_site_quantities(model, q_ref, v_ref, site_ids, body_ids)

            res = {
                "pairing": {
                    "reference_frames": int(len(q_ref)),
                    "identical_to_another_bodys_reference": bool(same_as_other),
                    "distinct_from_nominal_cm": float(
                        np.linalg.norm(q_ref[:, 0:3] - refs[bodies[0][0]][method][:, 0:3], axis=-1).mean() * 100),
                },
                "zero_action": rollout(model, q_ref, v_ref, r_ref, kern.ACT_QADR, dt, "zero"),
                "pd": {},
            }
            for kp in args.kp:
                res["pd"][f"kp{int(kp)}"] = rollout(
                    model, q_ref, v_ref, r_ref, kern.ACT_QADR, dt, "pd", kp=kp, kd=args.kd_frac * kp)
            best = max(res["pd"].values(), key=lambda r: (r["survived_frac"], -r["mean_joint_error_deg"]))
            res["best_pd"] = best
            entry["methods"][method] = res
            print(f"{name:22s} {method:10s} zero {res['zero_action']['survived_frac']:.2f} | "
                  f"pd best {best['survived_frac']:.2f} surv, "
                  f"{best['mean_joint_error_deg']:.2f} deg, {best['mean_site_error_cm']:.2f} cm, "
                  f"R {best['mean_reward_dance_weights']:.3f}")
        records.append(entry)

    out = {
        "component": "C8_reference_feasibility",
        "question": "can a body physically follow its reference, independent of RL?",
        "control_hz": 1 / dt,
        "pd_note": "torque = kp*(q_ref - q) - kd*qd, clipped to actuator ctrlrange; kd = "
                   f"{args.kd_frac} * kp",
        "reward_note": "MimicReward formulas with the dance weights, qpos+rpos terms only",
        "ik_note": "ik_scaled references clamped to joint limits after the unconstrained GN solve",
        "bodies": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
