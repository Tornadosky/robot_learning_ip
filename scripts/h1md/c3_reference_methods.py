"""C3 -- three ways to build a per-body reference, gated on validity not speed.

Phase 3 of the goal document says a reference that is quantitatively invalid
must be fixed or excluded, and that PPO must not be asked to compensate for a
bad teacher. Throughput was already measured elsewhere
(`experiments/retarget_scaling_probe/`); what was not measured is whether the
references those methods produce are *trackable*. This script builds all three
on the frozen C1 window and the C2 catalog, then gates them:

  fk         joint angles preserved, root re-grounded per frame from the body's
             own FK. No IK. Feasible by construction -- it is a pose the body
             can hold -- but it does not preserve the motion in space.
  ik_world   Gauss-Newton onto the *nominal* world site positions. Preserves
             the choreography literally; likely unreachable off-nominal.
  ik_scaled  Gauss-Newton onto root-relative site targets scaled by the limb
             that carries them. "A scaled body wants a scaled motion."

Gates applied to each:
  * joint-limit violations (the GN kernel is unconstrained, so this is real);
  * achieved site residual;
  * temporal smoothness -- per-frame independent IK can produce a reference no
    actuator could follow;
  * foot penetration and foot sliding against the real contact geometry;
  * distance to the other methods expressed in MimicReward units.

Run under WSL dance_env. GPU is used if visible.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE / "scripts"))
sys.path.insert(0, str(WORKSPACE / "scripts" / "h1md"))

from c6_reward_discrimination import (  # noqa: E402
    E, W_DANCE, build_model, catalog, reground, site_quantities, terms, total,
)

TARGET_SITE_NAMES = ("left_foot_mimic", "right_foot_mimic", "left_hand_mimic", "right_hand_mimic")
# which morphology scale carries each target site: feet -> leg, hands -> arm
LIMB_SCALE_INDEX = np.array([0, 0, 1, 1])


class IKKernel:
    """Batched Gauss-Newton over (bodies x frames) on mjx.kinematics.

    Adapted from experiments/retarget_scaling_probe/retarget_ik_v2.py. Contacts
    are disabled for the kinematic solve only; every validity gate below is then
    evaluated on the untouched contact-enabled model.
    """

    def __init__(self, xml_path: Path):
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        model.geom_contype[:] = 0
        model.geom_conaffinity[:] = 0
        self.model = model
        self.mx = mjx.put_model(model)
        bid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
        sid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n)
        self.LEG_POS = np.array([bid(f"{s}_{l}_link") for s in ("left", "right") for l in ("knee", "ankle")])
        self.UPPER_ARM = np.array([bid(f"{s}_shoulder_yaw_link") for s in ("left", "right")])
        self.FOREARM = np.array([bid(f"{s}_elbow_link") for s in ("left", "right")])
        self.SHOULDER = np.array([bid(f"{s}_shoulder_pitch_link") for s in ("left", "right")])
        self.HAND_SITES = np.array([sid("left_hand_mimic"), sid("right_hand_mimic")])
        self.TARGET_SITES = np.array([sid(n) for n in TARGET_SITE_NAMES])
        self.BASE_BODY_POS = jnp.asarray(self.mx.body_pos)
        self.BASE_SITE_POS = jnp.asarray(self.mx.site_pos)
        self.ACT_QADR = np.array([model.jnt_qposadr[model.actuator_trnid[i, 0]] for i in range(model.nu)])
        self.NU = model.nu
        self._d0 = mjx.make_data(self.mx)
        self.jnt_low = model.jnt_range[model.actuator_trnid[:, 0], 0]
        self.jnt_high = model.jnt_range[model.actuator_trnid[:, 0], 1]

        self.fk_bt = jax.jit(jax.vmap(jax.vmap(self._fk_sites, in_axes=(None, 0)), in_axes=(0, None)))
        self.ik = jax.jit(jax.vmap(jax.vmap(self._gauss_newton, in_axes=(None, 0, 0)), in_axes=(0, None, 0)))

    def _apply(self, morph):
        leg, arm, shoulder = morph[0], morph[1], morph[2]
        p = self.BASE_BODY_POS
        p = p.at[self.LEG_POS].set(self.BASE_BODY_POS[self.LEG_POS] * jnp.array([1.0, 1.0, leg]))
        p = p.at[self.UPPER_ARM].set(self.BASE_BODY_POS[self.UPPER_ARM] * jnp.array([1.0, 1.0, arm]))
        p = p.at[self.FOREARM].set(self.BASE_BODY_POS[self.FOREARM] * arm)
        p = p.at[self.SHOULDER].set(self.BASE_BODY_POS[self.SHOULDER] * jnp.array([1.0, shoulder, 1.0]))
        s = self.BASE_SITE_POS.at[self.HAND_SITES].set(
            self.BASE_SITE_POS[self.HAND_SITES] * jnp.array([arm, 1.0, 1.0]))
        return self.mx.replace(body_pos=p, site_pos=s)

    def _fk_sites(self, morph, qpos):
        return mjx.kinematics(self._apply(morph), self._d0.replace(qpos=qpos)).site_xpos

    def _residual(self, x, morph, qpos, tgt):
        q = qpos.at[self.ACT_QADR].set(x[: self.NU])
        q = q.at[0:3].add(x[self.NU: self.NU + 3])
        return (self._fk_sites(morph, q)[self.TARGET_SITES] - tgt).reshape(-1)

    def _gauss_newton(self, morph, qpos, tgt, iters=8, damping=1e-3):
        x0 = jnp.concatenate([qpos[self.ACT_QADR], jnp.zeros(3)])

        def step(x, _):
            r = self._residual(x, morph, qpos, tgt)
            J = jax.jacfwd(self._residual)(x, morph, qpos, tgt)
            H = J.T @ J + damping * jnp.eye(x.shape[0])
            return x - jnp.linalg.solve(H, J.T @ r), None

        x, _ = jax.lax.scan(step, x0, None, length=iters)
        return x, self._residual(x, morph, qpos, tgt)


def foot_metrics(model, qpos_batch, dt: float, contact_thresh: float = 0.01) -> dict:
    """Penetration and sliding measured on the real contact-enabled model."""
    data = mujoco.MjData(model)
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    feet = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("left_foot", "right_foot")]
    fsites = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n)
              for n in ("left_foot_mimic", "right_foot_mimic")]
    dist = np.zeros((len(qpos_batch), 2))
    xy = np.zeros((len(qpos_batch), 2, 2))
    for i, q in enumerate(qpos_batch):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        for j, f in enumerate(feet):
            dist[i, j] = mujoco.mj_geomDistance(model, data, floor, f, 10.0, None)
        xy[i] = data.site_xpos[fsites][:, :2]
    in_contact = dist < contact_thresh
    step_xy = np.linalg.norm(np.diff(xy, axis=0), axis=-1)          # (T-1, 2)
    slide = step_xy[in_contact[:-1]]
    return {
        "min_foot_floor_distance_m": float(dist.min()),
        "penetration_frames": int((dist < 0).sum()),
        "penetration_frac": float((dist < 0).mean()),
        "contact_frac": float(in_contact.mean()),
        "foot_slide_cm_per_s_mean": float(slide.mean() / dt * 100) if slide.size else 0.0,
        "foot_slide_cm_per_s_p95": float(np.percentile(slide, 95) / dt * 100) if slide.size else 0.0,
    }


def smoothness(qpos_batch: np.ndarray, dt: float) -> dict:
    """Second/third derivative of the actuated joint angles of the reference."""
    q = qpos_batch[:, 7:]
    acc = np.diff(q, n=2, axis=0) / dt**2
    jerk = np.diff(q, n=3, axis=0) / dt**3
    return {
        "joint_acc_rms_rad_s2": float(np.sqrt((acc**2).mean())),
        "joint_acc_max_rad_s2": float(np.abs(acc).max()),
        "joint_jerk_rms_rad_s3": float(np.sqrt((jerk**2).mean())),
    }


def limit_violations(qpos_batch, act_qadr, low, high) -> dict:
    q = qpos_batch[:, act_qadr]
    over = np.maximum(q - high, 0.0)
    under = np.maximum(low - q, 0.0)
    excess = over + under
    return {
        "violating_frames_frac": float((excess > 1e-6).any(axis=1).mean()),
        "violating_entries_frac": float((excess > 1e-6).mean()),
        "max_excess_deg": float(np.degrees(excess.max())),
        "mean_excess_deg_where_violating": (
            float(np.degrees(excess[excess > 1e-6].mean())) if (excess > 1e-6).any() else 0.0
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--start", type=int, default=19482)
    ap.add_argument("--frames", type=int, default=800)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--xml-root", type=Path,
                    default=WORKSPACE / "experiments" / "h1_morphology_deepmimic_20260808" / "body_catalog" / "xml")
    args = ap.parse_args()

    from loco_mujoco.task_factories import ImitationFactory, LAFAN1DatasetConf

    env = ImitationFactory.make("UnitreeH1", lafan1_dataset_conf=LAFAN1DatasetConf(["dance2_subject4"]))
    th = env.th.traj
    freq = float(th.info.frequency)
    qpos = np.asarray(th.data.qpos)[args.start:args.start + args.frames][:: args.stride].astype(np.float64)
    qvel = np.asarray(th.data.qvel)[args.start:args.start + args.frames][:: args.stride].astype(np.float64)
    dt = args.stride / freq
    print(f"{len(qpos)} frames @ effective {1 / dt:.1f} Hz, backend={jax.default_backend()}")

    bodies = catalog()
    nominal_model = build_model(*bodies[0], args.xml_root)
    nominal_xml = args.xml_root / "h1_morphology_c2_body00_nominal" / "h1.xml"
    kern = IKKernel(nominal_xml)

    morphs = np.array([[b[1]["leg_length_scale"], b[1]["arm_length_scale"],
                        b[1]["shoulder_width_scale"]] for b in bodies], dtype=np.float32)
    Q = jnp.asarray(qpos.astype(np.float32))
    M = jnp.asarray(morphs)

    # nominal world site targets, and the scale-normalized variant
    nom_sites = np.asarray(kern.fk_bt(jnp.ones((1, 3), jnp.float32), Q))[0]
    nom_targets = jnp.asarray(nom_sites[:, kern.TARGET_SITES, :])
    root_nom = jnp.asarray(np.asarray(qpos)[:, 0:3].astype(np.float32))

    def scaled_targets(morph):
        rel = nom_targets - root_nom[:, None, :]
        s = morph[LIMB_SCALE_INDEX][None, :, None]
        root = root_nom.at[:, 2].multiply(morph[0])
        return root[:, None, :] + rel * s

    B, T = len(bodies), len(qpos)
    tgt_world = jnp.broadcast_to(nom_targets, (B, T, 4, 3))
    tgt_scaled = jax.vmap(scaled_targets)(M)

    solutions = {}
    timings = {}
    for label, tgt in (("ik_world", tgt_world), ("ik_scaled", tgt_scaled)):
        t0 = time.perf_counter(); xs, rs = kern.ik(M, Q, tgt); xs.block_until_ready()
        timings[label] = {"compile_plus_run_s": time.perf_counter() - t0}
        t0 = time.perf_counter(); xs, rs = kern.ik(M, Q, tgt); xs.block_until_ready()
        warm = time.perf_counter() - t0
        timings[label].update({"warm_run_s": warm, "cells": B * T, "cells_per_second": B * T / warm})
        solutions[label] = (np.asarray(xs), np.asarray(rs))
        print(f"{label}: warm {warm:.3f} s, {B * T / warm:,.0f} cells/s")

    records = []
    for bi, (name, morph) in enumerate(bodies):
        model = build_model(name, morph, args.xml_root)
        entry = {"body_id": name, "morphology": morph, "methods": {}}

        refs = {}
        # --- fk: joint angles preserved, root re-grounded from this body's FK
        q_fk, ground = reground(model, qpos)
        refs["fk"] = q_fk
        entry["methods"]["fk"] = {"regrounding": ground}

        # --- ik variants: joints + root translation from the solver
        for label in ("ik_world", "ik_scaled"):
            xs, rs = solutions[label]
            q = qpos.copy()
            q[:, kern.ACT_QADR] = xs[bi, :, : kern.NU]
            q[:, 0:3] += xs[bi, :, kern.NU: kern.NU + 3]
            # Re-ground exactly as `fk` is. Without this the IK output inherits
            # the raw clip's ground offset and the feet float ~1.2 cm for the
            # whole window, so the reference never establishes contact and never
            # tells the policy where to plant its feet. The IK site targets say
            # nothing about the floor, so this is not optional.
            q, ground_ik = reground(model, q)
            entry.setdefault("regrounding_ik", {})[label] = ground_ik
            refs[label] = q
            err = np.linalg.norm(rs[bi].reshape(T, 4, 3), axis=-1)
            entry["methods"][label] = {
                "site_residual_cm_mean": float(err.mean() * 100),
                "site_residual_cm_p99": float(np.percentile(err, 99) * 100),
                "joint_change_vs_source_deg_mean": float(
                    np.degrees(np.abs(xs[bi, :, : kern.NU] - qpos[:, kern.ACT_QADR]).mean())),
                "joint_change_vs_source_deg_max": float(
                    np.degrees(np.abs(xs[bi, :, : kern.NU] - qpos[:, kern.ACT_QADR]).max())),
                "root_shift_cm_mean": float(np.linalg.norm(xs[bi, :, kern.NU:], axis=-1).mean() * 100),
            }

        for label, q in refs.items():
            entry["methods"][label].update({
                "limits": limit_violations(q, kern.ACT_QADR, kern.jnt_low, kern.jnt_high),
                "smoothness": smoothness(q, dt),
                "feet": foot_metrics(model, q, dt),
            })

        # pairwise distance in MimicReward units: robot plays reference X, is
        # scored against reference Y
        rq = {k: site_quantities(model, v, qvel)[0] for k, v in refs.items()}
        pair = {}
        for a in refs:
            for b in refs:
                if a == b:
                    continue
                d = float(np.mean((rq[a] - rq[b]) ** 2))
                t = terms(0.0, 0.0, d, 0.0, 0.0, 0.0)
                pair[f"{a}_scored_against_{b}"] = {
                    "rpos_reward": t["rpos_reward"],
                    "total_dance": total(t, W_DANCE),
                    "site_rms_cm": float(np.sqrt((np.linalg.norm(rq[a] - rq[b], axis=-1) ** 2).mean()) * 100),
                }
        entry["reward_space_pairwise"] = pair
        records.append(entry)

        m = entry["methods"]
        print(f"{name}: "
              f"fk[lim {m['fk']['limits']['violating_frames_frac']:.2f} pen {m['fk']['feet']['penetration_frac']:.2f} "
              f"acc {m['fk']['smoothness']['joint_acc_rms_rad_s2']:.0f}] "
              f"ikw[res {m['ik_world']['site_residual_cm_mean']:.2f}cm lim {m['ik_world']['limits']['violating_frames_frac']:.2f} "
              f"pen {m['ik_world']['feet']['penetration_frac']:.2f} acc {m['ik_world']['smoothness']['joint_acc_rms_rad_s2']:.0f}] "
              f"iks[res {m['ik_scaled']['site_residual_cm_mean']:.2f}cm lim {m['ik_scaled']['limits']['violating_frames_frac']:.2f} "
              f"pen {m['ik_scaled']['feet']['penetration_frac']:.2f} acc {m['ik_scaled']['smoothness']['joint_acc_rms_rad_s2']:.0f}]")

    out = {
        "component": "C3_reference_methods",
        "question": "which per-body reference construction produces a *trackable* reference?",
        "window": {"start_frame_100hz": args.start, "frames": args.frames,
                   "stride": args.stride, "effective_hz": 1 / dt},
        "ik": {"iters": args.iters, "solves_for": "19 actuated joints + root xyz",
               "targets": list(TARGET_SITE_NAMES),
               "caveats": "unconstrained (no joint limits, no self-collision); contacts disabled for the solve only",
               "timings": timings, "backend": jax.default_backend()},
        "bodies": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
