"""Independent verification of morphology-conditioned FK reward targets.

For every (family, morphology, motion, phase) cell this script compares the
in-graph JAX/MJX target site quantities produced by ``MorphMimicReward``
(the exact production method, eager AND jitted) against an INDEPENDENT
calculation: a fresh CPU ``mujoco.MjModel`` whose knee/ankle attachment
offsets are re-scaled in numpy (the only morphology coordinate that affects
kinematics), evaluated with ``mujoco.mj_forward`` — the C engine, not MJX.

Regression properties asserted per family:
  A. jitted == eager (JIT stability of the FK pass).
  B. jax FK == independent CPU FK at every probed cell (tolerance --tol).
  C. corner-morphology targets differ from nominal targets (> --min-morph-diff)
     — FAILS if the implementation silently caches nominal FK.
  D. phase p targets match CPU phase p far better than CPU phase p±1
     — FAILS on an off-by-one command/cursor.
  E. nominal FK matches the trajectory's own stored site data
     — ties the new FK path to the stock MimicReward targets on the
     nominal body (they must agree there, and only there).
  F. everything finite.

Writes a JSON report and exits non-zero on any failure.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

WORKSPACE = Path(__file__).resolve().parents[2]
SCRIPTS = WORKSPACE / "scripts"
for p in (str(SCRIPTS), str(SCRIPTS / "h1md")):
    if p not in sys.path:
        sys.path.insert(0, p)

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from loco_mujoco.core.utils.math import calculate_relative_site_quatities

from scaling.family_morphology import (
    FAMILY_BODIES,
    FAMILY_MORPHOLOGY_HIGH,
    FAMILY_MORPHOLOGY_LOW,
)
from scaling.parallel_cross_humanoid_train import (
    _build_robot_env,
    _ensure_latent_defaults,
)


def _cpu_reference_quantities(env, reward, robot, morphology, qpos):
    """Independent CPU FK: numpy morphology mutation + mujoco.mj_forward."""
    model = copy.deepcopy(env._model)
    bodies = FAMILY_BODIES[robot]
    leg_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in (*bodies.knee_bodies, *bodies.ankle_bodies)
    ]
    if any(i < 0 for i in leg_ids):
        raise ValueError(f"{robot}: missing knee/ankle body")
    leg_scale = float(morphology[0])
    model.body_pos[leg_ids, 2] *= leg_scale  # the only kinematic coordinate

    data = mujoco.MjData(model)
    data.qpos[:] = np.asarray(qpos)
    mujoco.mj_forward(model, data)
    rpos, rangles, _ = calculate_relative_site_quatities(
        data, reward._rel_site_ids, reward._rel_body_ids, model.body_rootid, np
    )
    return np.asarray(rpos), np.asarray(rangles)


def _jax_target_quantities(env, reward, morphology, qpos, jitted):
    """The production FK pass (MorphMimicReward._traj_site_quantities)."""
    data0 = mjx.make_data(env.sys)
    carry = SimpleNamespace(morphology=jnp.asarray(morphology))

    def compute(m, q):
        body_model = env._apply_morphology(env.sys, m)
        rpos, rangles, _ = reward._traj_site_quantities(
            env, env._model, data0, q, carry, jnp, body_model=body_model
        )
        return rpos, rangles

    fn = jax.jit(compute) if jitted else compute
    rpos, rangles = fn(jnp.asarray(morphology), jnp.asarray(qpos))
    return np.asarray(rpos), np.asarray(rangles)


def _traj_stored_quantities(env, reward, traj_no, step):
    sample = env.th.traj.data.get(traj_no, step, jnp)
    rpos, rangles, _ = calculate_relative_site_quatities(
        sample, reward._rel_site_ids, reward._rel_body_ids,
        env.sys.body_rootid, jnp,
    )
    return np.asarray(rpos), np.asarray(rangles)


def verify_family(robot, args):
    env_args = _ensure_latent_defaults(
        SimpleNamespace(
            source=args.source,
            reference_mode="direct",
            reference_root=args.reference_root,
            clip=None,
            start_frame=None,
            frames=None,
            clip_windows=args.clip_windows,
            morphology="continuous",
            use_mjwarp=False,
            reward_type="MorphMimicReward",
        )
    )
    env, _ = _build_robot_env(env_args, robot)
    reward = env._reward_function
    assert type(reward).__name__ == "MorphMimicReward", type(reward).__name__

    nominal = np.ones(4, dtype=np.float32)
    low = FAMILY_MORPHOLOGY_LOW.copy()
    high = FAMILY_MORPHOLOGY_HIGH.copy()
    morphs = {"nominal": nominal, "low_corner": low, "high_corner": high}

    n_motions = int(np.asarray(env.th.traj.data.split_points).shape[0]) - 1
    cells = []
    failures = []
    nominal_rpos = {}

    for traj_no in range(n_motions):
        for step in args.phases:
            sample = env.th.traj.data.get(int(traj_no), int(step), jnp)
            qpos = np.asarray(sample.qpos)
            for mname, m in morphs.items():
                rpos_e, rang_e = _jax_target_quantities(env, reward, m, qpos, False)
                rpos_j, rang_j = _jax_target_quantities(env, reward, m, qpos, True)
                rpos_c, rang_c = _cpu_reference_quantities(env, reward, robot, m, qpos)

                jit_diff = float(np.abs(rpos_j - rpos_e).max())
                pos_err = float(np.abs(rpos_j - rpos_c).max())
                ang_err = float(np.abs(rang_j - rang_c).max())
                finite = bool(
                    np.isfinite(rpos_j).all() and np.isfinite(rang_j).all()
                )
                cell = {
                    "robot": robot, "traj_no": traj_no, "step": int(step),
                    "morph": mname, "jit_vs_eager": jit_diff,
                    "rpos_err_m": pos_err, "rangles_err": ang_err,
                    "finite": finite,
                }
                if jit_diff > 1e-6:
                    failures.append(f"{cell}: JIT != eager")
                if pos_err > args.tol or ang_err > args.tol_angle:
                    failures.append(f"{cell}: JAX FK != independent CPU FK")
                if not finite:
                    failures.append(f"{cell}: non-finite")

                if mname == "nominal":
                    nominal_rpos[(traj_no, int(step))] = rpos_j
                    # E: nominal FK == trajectory-stored site data
                    rpos_t, rang_t = _traj_stored_quantities(env, reward, traj_no, int(step))
                    stored_err = float(np.abs(rpos_j - rpos_t).max())
                    cell["nominal_vs_stored_m"] = stored_err
                    if stored_err > args.stored_tol:
                        failures.append(
                            f"{cell}: nominal FK differs from trajectory site data"
                        )
                else:
                    # C: morphology must MOVE the targets
                    morph_diff = float(
                        np.abs(rpos_j - nominal_rpos[(traj_no, int(step))]).max()
                    )
                    cell["vs_nominal_m"] = morph_diff
                    if morph_diff < args.min_morph_diff:
                        failures.append(
                            f"{cell}: corner morphology target ~= nominal target "
                            "(nominal-cached FK?)"
                        )
                cells.append(cell)

        # D: off-by-one detection at one dynamic phase per motion
        step = int(args.phases[len(args.phases) // 2])
        q_here = np.asarray(env.th.traj.data.get(int(traj_no), step, jnp).qpos)
        rpos_here, _ = _jax_target_quantities(env, reward, nominal, q_here, True)
        q_next = np.asarray(env.th.traj.data.get(int(traj_no), step + 1, jnp).qpos)
        rpos_next, _ = _cpu_reference_quantities(env, reward, robot, nominal, q_next)
        match_err = min(
            c["rpos_err_m"] for c in cells
            if c["robot"] == robot and c["traj_no"] == traj_no
            and c["step"] == step and c["morph"] == "nominal"
        )
        offbyone_gap = float(np.abs(rpos_here - rpos_next).max())
        cells.append({
            "robot": robot, "traj_no": traj_no, "check": "off_by_one",
            "step": step, "gap_to_next_phase_m": offbyone_gap,
            "match_err_m": match_err,
        })
        if offbyone_gap < max(10.0 * match_err, 1e-3):
            failures.append(
                f"{robot} traj{traj_no} step{step}: phase p vs p+1 gap "
                f"{offbyone_gap:.2e} not distinguishable from match error "
                f"{match_err:.2e}"
            )

    return cells, failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robots", nargs="+", default=["h1", "g1", "atlas"])
    parser.add_argument("--source", default="h1")
    parser.add_argument(
        "--clip-windows", nargs="+",
        default=["dance2_subject4:19482:800", "dance2_subject1:2000:800"],
    )
    parser.add_argument(
        "--reference-root", type=Path,
        default=WORKSPACE / "external_data" / "cross_humanoid",
    )
    parser.add_argument(
        "--phases", type=int, nargs="+", default=[3, 111, 400, 731]
    )
    parser.add_argument("--tol", type=float, default=5e-4)
    parser.add_argument("--tol-angle", type=float, default=2e-3)
    parser.add_argument("--stored-tol", type=float, default=5e-3)
    parser.add_argument("--min-morph-diff", type=float, default=5e-3)
    parser.add_argument(
        "--output", type=Path,
        default=WORKSPACE / "experiments" / "urma_fk_targets_20260815"
        / "metrics" / "fk_verification.json",
    )
    args = parser.parse_args()

    all_cells, all_failures = [], []
    for robot in args.robots:
        cells, failures = verify_family(robot, args)
        all_cells.extend(cells)
        all_failures.extend(failures)
        worst_pos = max(c.get("rpos_err_m", 0.0) for c in cells)
        n_checked = sum(1 for c in cells if "rpos_err_m" in c)
        print(
            f"[verify-fk] {robot:>6s}: {n_checked} cells, "
            f"worst |JAX-CPU| = {worst_pos:.2e} m, "
            f"failures so far = {len(all_failures)}",
            flush=True,
        )

    report = {
        "robots": args.robots,
        "clip_windows": args.clip_windows,
        "phases": args.phases,
        "tolerance_m": args.tol,
        "cells": all_cells,
        "failures": all_failures,
        "passed": not all_failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[verify-fk] report -> {args.output}", flush=True)
    if all_failures:
        print("[verify-fk] FAILURES:", flush=True)
        for failure in all_failures:
            print("  " + failure, flush=True)
        sys.exit(1)
    print("[verify-fk] ALL CHECKS PASSED", flush=True)


if __name__ == "__main__":
    main()
