"""What score could a PERFECT controller get on this clip?  The tracking ceiling.

"The policy tracks to 13 cm and dies after 31 steps" is not interpretable until
somebody measures what is achievable.  This script measures it, by replacing the
policy with a controller that already knows the whole future: a stable-PD servo
driven straight at the reference joint angles, at the training control rate, on
the training body, judged by the training terminal criteria and scored with the
training reward's own five-site metric.  Any torque policy is bounded above by
this, because a policy has to *infer* the target this controller is handed.

Four axes are swept, because each one is a different candidate explanation:

* **body** -- nominal and both morphology corners
  (``family_morphology.FAMILY_MORPHOLOGY_LOW/HIGH``).
* **reference variant** -- ``raw`` (what training uses), ``reground_all_geoms``
  (``reference_grounding``'s per-frame regrounding, which measures clearance
  over every robot geom) and ``reground_collidable`` (the same, restricted to
  the geoms that can actually contact the floor -- H1 and G1 declare contact
  only through four foot-sphere ``<pair>`` elements each, so every other geom
  in the all-geom measurement is a visual mesh that can never touch anything).
  Root-z shifts do not move *relative* site positions, so the three variants
  share a site-error target and differ only in where the feet meet the floor.
* **gain** -- 1x / 3x / 10x / 30x / 100x of the shipped (G1) or
  torque-limit-derived (H1) PD gains; the top of the sweep is the near-rigid
  setting.  Torque is clipped to the actuator's real authority either way, so
  the sweep also shows where clipping, not gain, becomes binding.
* **root** -- free, versus welded to the reference root pose.  The welded arm
  removes balance from the problem entirely, so the gap between the two is
  exactly how much of the failure is balance rather than limb tracking.

Everything is CPU MuJoCo; no MJX, no GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
for _p in (str(WORKSPACE / "scripts"), str(WORKSPACE / "scripts" / "h1md")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jax.numpy as jnp  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from loco_mujoco.core.utils.math import (  # noqa: E402
    calculate_relative_site_quatities,
)

from scaling.body_correct_reference import clamp_reference_qpos  # noqa: E402
from scaling.reference_grounding import (  # noqa: E402
    DEFAULT_CLEARANCE,
    _floor_geom,
    _robot_geoms,
    floor_clearance,
)
from scaling.stand_test import (  # noqa: E402
    BODIES,
    DEFAULT_CLIP,
    DEFAULT_REFERENCE_ROOT,
    TerminalCriteria,
    actuated_index,
    body_model,
    build_env,
    collidable_floor_geoms,
    joint_ranges,
    pd_gains,
    root_pitch_roll_deg,
)


# --------------------------------------------------------------------------- #
# reference variants
# --------------------------------------------------------------------------- #
def reground_root_z(model, qpos: np.ndarray, geoms: np.ndarray,
                    clearance: float = DEFAULT_CLEARANCE) -> np.ndarray:
    """Per-frame root height that puts the lowest of ``geoms`` at ``clearance``.

    Same computation as ``reference_grounding.ground_trajectory_per_frame``
    (``mj_geomDistance`` over the chosen geoms, root z only), returned as a bare
    height series because the oracle steps the model itself and never needs the
    rebuilt ``Trajectory``.
    """
    data = mujoco.MjData(model)
    floor = _floor_geom(model)
    out = np.array(qpos[:, 2], dtype=np.float64, copy=True)
    for i in range(qpos.shape[0]):
        data.qpos[:] = qpos[i]
        mujoco.mj_forward(model, data)
        out[i] += clearance - floor_clearance(model, data, floor, geoms)
    return out


def reference_variants(model, qpos: np.ndarray, height_offset: float) -> dict:
    """``{name: (root z series, stats)}`` for the three reference treatments."""
    floor = _floor_geom(model)
    variants = {
        "raw": (qpos[:, 2] + height_offset, {
            "description": "reference root z + the reset's morphology offset",
        }),
    }
    for name, geoms in (
        ("reground_all_geoms", _robot_geoms(model, floor)),
        ("reground_collidable", collidable_floor_geoms(model)),
    ):
        z = reground_root_z(model, qpos, geoms)
        variants[name] = (z, {
            "description": f"per-frame grounding over {geoms.size} geoms",
            "n_geoms": int(geoms.size),
            "shift_mean_m": float(np.mean(z - qpos[:, 2])),
            "shift_max_abs_m": float(np.max(np.abs(z - qpos[:, 2]))),
        })
    # How far the raw reference's feet float, as the simulator would see them:
    # the collidable regrounding had to move each frame by exactly the negative
    # of its clearance error, so the shift series reports the float directly.
    float_m = -(variants["reground_collidable"][0] - qpos[:, 2] - height_offset)
    float_m = float_m + DEFAULT_CLEARANCE
    variants["raw"][1]["raw_collidable_clearance_m"] = {
        "median": float(np.median(float_m)),
        "min": float(np.min(float_m)),
        "max": float(np.max(float_m)),
        "pct_frames_floating_over_2cm": float(100.0 * np.mean(float_m > 0.02)),
    }
    return variants


# --------------------------------------------------------------------------- #
# targets: exactly what the reward scores
# --------------------------------------------------------------------------- #
class SiteTargets:
    """The reward's five-site relative-position block, on the sampled body.

    Built with the same ``calculate_relative_site_quatities`` call and the same
    per-body joint clamp that ``body_correct_reference`` uses, on the same site
    and body ids the reward holds, so "site error" here is the reward's own
    quantity and not a look-alike.
    """

    def __init__(self, env, model):
        reward = env._reward_function
        self.site_ids = np.asarray(reward._rel_site_ids)
        self.body_ids = np.asarray(reward._rel_body_ids)
        self.body_rootid = np.asarray(model.body_rootid)
        self.sys = env.sys
        self.model = model
        self.data = mujoco.MjData(model)
        # calculate_relative_site_quatities drops row 0 (the main mimic site) and
        # expresses the rest relative to it, so these are the scored rows in order.
        self.relative_site_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, int(s))
            for s in self.site_ids[1:]
        ]

    def clamp(self, qpos: np.ndarray) -> np.ndarray:
        return clamp_reference_qpos(qpos, np.asarray(self.model.jnt_range),
                                    self.sys, np)

    def relative(self, qpos: np.ndarray) -> np.ndarray:
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)
        rpos, _, _ = calculate_relative_site_quatities(
            self.data, self.site_ids, self.body_ids, self.body_rootid, np)
        return np.asarray(rpos)


def verify_against_provider(env, model, targets: SiteTargets, frame: int) -> float:
    """Max |our target - ``body_correct_reference``'s| at one frame, in metres."""
    from scaling.body_correct_reference import cpu_reference_bundle

    reference = np.asarray(cpu_reference_bundle(
        env, model, 0, frame,
        rel_site_ids=targets.site_ids, rel_body_ids=targets.body_ids,
        include_site_velocity=False).relative_site_position)
    ours = targets.relative(targets.clamp(
        np.asarray(env.th.traj.data.qpos[frame], dtype=np.float64)))
    return float(np.max(np.abs(ours - reference)))


# --------------------------------------------------------------------------- #
# the oracle rollout
# --------------------------------------------------------------------------- #
def nlerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    if float(np.dot(q0, q1)) < 0.0:
        q1 = -q1
    q = (1.0 - alpha) * q0 + alpha * q1
    return q / np.linalg.norm(q)


def prepare_reference(model, targets: SiteTargets, qpos_ref: np.ndarray,
                      root_z: np.ndarray) -> tuple:
    """``(reference qpos, target joint angles, target site block)`` for one cell.

    Shared by every gain and both root modes of a (body, variant) pair, so the
    FK pass that builds the site targets runs once rather than ten times.
    """
    qpos_ids, _ = actuated_index(model)
    lo, hi = joint_ranges(model, qpos_ids)
    n = qpos_ref.shape[0]
    reference = np.stack([targets.clamp(qpos_ref[t]) for t in range(n)])
    reference[:, 2] = root_z
    # ``LocoEnv.set_sim_state_from_traj_data`` subtracts the episode's own start
    # XY at reset, so the robot begins at world (0, 0) and the terminal handler
    # compares against the reference's DISPLACEMENT. Rebasing here is what makes
    # the deviation check mean the same thing it means in training -- without it
    # every free-root rollout is terminal on step 0 for the clip's absolute
    # position (0.71 m on H1, 0.30 m on G1, against a 0.5 m threshold).
    reference[:, :2] -= reference[0, :2]
    target_angles = np.clip(reference[:, qpos_ids], lo, hi)
    target_sites = np.stack([targets.relative(reference[t]) for t in range(n)])
    return reference, target_angles, target_sites


def rollout(model, targets: SiteTargets, reference: np.ndarray,
            target_angles: np.ndarray, target_sites: np.ndarray,
            qvel_ref: np.ndarray, criteria: TerminalCriteria,
            p_gain: np.ndarray, d_gain: np.ndarray, weld_root: bool,
            n_substeps: int) -> dict:
    """One 800-step oracle rollout; returns the metric block for one cell."""
    qpos_ids, dof_ids = actuated_index(model)
    ctrl_lo = model.actuator_ctrlrange[:, 0].copy()
    ctrl_hi = model.actuator_ctrlrange[:, 1].copy()
    unlimited = ~model.actuator_ctrllimited.astype(bool)
    ctrl_lo[unlimited] = -np.inf
    ctrl_hi[unlimited] = np.inf
    limit = np.maximum(np.abs(ctrl_lo), np.abs(ctrl_hi))
    n = reference.shape[0]

    data = mujoco.MjData(model)
    data.qpos[:] = reference[0]
    data.qvel[:] = qvel_ref[0]
    mujoco.mj_forward(model, data)

    dt = float(model.opt.timestep)
    site_errors, heights, tilts, saturation = [], [], [], []
    per_site, per_joint, joint_errors = [], [], []
    steps_survived = n
    fall_reason = None
    diverged = False

    for t in range(n):
        actual = targets.relative(data.qpos)
        error = np.linalg.norm(actual - target_sites[t], axis=-1)
        site_errors.append(float(error.mean()))
        per_site.append(error)
        per_joint.append(np.abs(data.qpos[qpos_ids] - target_angles[t]))
        heights.append(float(data.qpos[2]))
        pitch, roll = root_pitch_roll_deg(data.qpos)
        tilts.append(max(abs(pitch), abs(roll)))
        joint_errors.append(float(np.sqrt(np.mean(
            (data.qpos[qpos_ids] - target_angles[t]) ** 2))))

        if not weld_root:
            reference_xy = reference[t, :2]
            if criteria.height_violation(data.qpos):
                fall_reason, steps_survived = "root_height", t
                break
            if criteria.rotation_violation(data.qpos):
                fall_reason, steps_survived = "root_rotation", t
                break
            if criteria.deviation_violation(data.qpos, reference_xy):
                fall_reason, steps_survived = "root_deviation", t
                break

        goal = target_angles[t]
        for k in range(n_substeps):
            gap = goal - data.qpos[qpos_ids] - data.qvel[dof_ids] * dt
            ctrl = p_gain * gap - d_gain * data.qvel[dof_ids]
            ctrl = np.clip(ctrl, ctrl_lo, ctrl_hi)
            data.ctrl[:] = ctrl
            saturation.append(float(np.mean(np.abs(ctrl) >= limit - 1e-9)))
            mujoco.mj_step(model, data)
            if weld_root and t + 1 < n:
                alpha = (k + 1) / n_substeps
                data.qpos[:3] = (1 - alpha) * reference[t, :3] + \
                    alpha * reference[t + 1, :3]
                data.qpos[3:7] = nlerp(reference[t, 3:7], reference[t + 1, 3:7],
                                       alpha)
                data.qvel[:6] = qvel_ref[t, :6]
        if not np.all(np.isfinite(data.qpos)):
            diverged = True
            steps_survived = t
            fall_reason = "integrator_diverged"
            break

    alive = slice(0, max(steps_survived, 1))
    errors = np.asarray(site_errors[alive])
    return {
        "steps_survived": int(steps_survived),
        "steps_total": int(n),
        "fall_reason": fall_reason,
        "integrator_diverged": diverged,
        "site_err_mean_m": float(errors.mean()),
        "site_err_median_m": float(np.median(errors)),
        "site_err_p90_m": float(np.percentile(errors, 90)),
        "site_err_first_step_m": float(site_errors[0]),
        "site_err_full_clip_mean_m": (float(np.mean(site_errors))
                                      if steps_survived == n else None),
        "site_err_per_site_m": {
            name: float(np.mean(np.stack(per_site[alive])[:, i]))
            for i, name in enumerate(targets.relative_site_names)
        },
        "worst_joints_rad": sorted(
            ((mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT,
                                int(model.actuator_trnid[a, 0])),
              float(np.mean(np.stack(per_joint[alive])[:, a])))
             for a in range(model.nu)),
            key=lambda kv: -kv[1])[:5],
        "joint_rms_err_rad_mean": float(np.mean(joint_errors[alive])),
        "torque_saturation_frac": float(np.mean(saturation)) if saturation else None,
        "root_height_m": {"init": float(heights[0]),
                          "min": float(np.min(heights[alive])),
                          "final": float(heights[max(steps_survived - 1, 0)])},
        "max_root_tilt_deg": float(np.max(tilts[alive])),
        "root_height_series_m": [round(h, 4) for h in heights[::10]],
        "site_err_series_m": [round(e, 4) for e in site_errors[::10]],
    }


def run_robot(env, robot: str, args) -> list[dict]:
    traj = env.th.traj
    qpos_ref = np.asarray(traj.data.qpos, dtype=np.float64)
    qvel_ref = np.asarray(traj.data.qvel, dtype=np.float64)
    n = min(args.steps, qpos_ref.shape[0])
    qpos_ref, qvel_ref = qpos_ref[:n], qvel_ref[:n]

    rows = []
    for label, morphology in BODIES.items():
        model = body_model(env, robot, morphology)
        offset = float(np.asarray(env.root_height_offset(jnp.asarray(morphology))))
        criteria = TerminalCriteria(env, offset)
        targets = SiteTargets(env, model)
        provider_gap = verify_against_provider(env, model, targets, 0)
        p_base, d_base, gain_source = pd_gains(robot, model)
        variants = reference_variants(model, qpos_ref, offset)

        for variant, (root_z, stats) in variants.items():
            reference, target_angles, target_sites = prepare_reference(
                model, targets, qpos_ref, root_z)
            for scale in args.gain_scales:
                for weld in (False, True):
                    record = rollout(
                        model, targets, reference, target_angles, target_sites,
                        qvel_ref, criteria, p_base * scale, d_base * scale,
                        weld, args.n_substeps)
                    record.update({
                        "robot": robot, "body": label,
                        "morphology": [float(v) for v in np.asarray(morphology)],
                        "reference_variant": variant,
                        "reference_variant_stats": stats,
                        "gain_scale": float(scale), "gain_source": gain_source,
                        "root": "welded" if weld else "free",
                        "root_height_offset_m": offset,
                        "target_vs_body_correct_reference_max_m": provider_gap,
                        "terminal_height_band_m": list(criteria.height_range),
                    })
                    rows.append(record)
                    print(f"[oracle] {robot:<3} {label:<11} {variant:<19} "
                          f"{scale:>5g}x {record['root']:<6} "
                          f"site {record['site_err_mean_m']*100:6.2f} cm "
                          f"(p90 {record['site_err_p90_m']*100:6.2f}) "
                          f"steps {record['steps_survived']:>4}/{n} "
                          f"sat {record['torque_saturation_frac'] or 0.0:.2f} "
                          f"{record['fall_reason'] or ''}", flush=True)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robots", nargs="+", default=["h1", "g1"])
    parser.add_argument("--clip-window", default=DEFAULT_CLIP)
    parser.add_argument("--reference-root", type=Path,
                        default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--n-substeps", type=int, default=5,
                        help="the env's own n_substeps (control rate = 1/(dt*n))")
    parser.add_argument("--gain-scales", type=float, nargs="+",
                        default=[1.0, 3.0, 10.0, 30.0, 100.0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for robot in args.robots:
        env = build_env(robot, args.clip_window, args.reference_root)
        rows.extend(run_robot(env, robot, args))
        del env

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "clip_window": args.clip_window,
        "steps": args.steps,
        "n_substeps": args.n_substeps,
        "gain_scales": list(args.gain_scales),
        "controller": "stable-PD (one-step position prediction) at simulation "
                      "frequency, torque clipped to actuator_ctrlrange; the "
                      "target is the reference joint angle after the per-body "
                      "joint-range clamp",
        "metric": "mean over the reward's 4 relative mimic sites of "
                  "||achieved - reference|| (metres), the same "
                  "calculate_relative_site_quatities block MorphMimicReward "
                  "scores; averaged over the steps the episode survived",
        "cells": rows,
    }, indent=2), encoding="utf-8")
    print(f"[oracle] -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
