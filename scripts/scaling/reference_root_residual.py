"""Root residual wrench of a reference clip: is the motion realizable at all?

``reference_feasibility.py`` inspects ``qfrc_inverse`` on the ACTUATED dofs only
and concluded the reference is within actuator limits.  The 6 dofs of the root
free joint were never looked at -- and they are the ones that decide whether the
clip is physically realizable.  A humanoid has no actuator on its root: the only
way generalized force can enter dofs 0..5 is through contact with the world.

So for every frame:

    qfrc_inverse[0:6] = (inertial + Coriolis + gravity terms of the root)
                        - (contact reaction projected on the root)

If the reference never touches the floor, ``mj_forward`` finds no contacts, the
whole gravity load lands in the root residual, and the required root force is
about one body weight upwards -- a force nothing on the robot can produce.  That
is a *hovering* reference, and no policy can track it while obeying physics.

The control arm is the per-frame regrounded reference from
``reference_grounding.ground_trajectory_per_frame``: if putting the feet on the
floor drives the vertical root residual toward zero, the residual was caused by
the missing contact and the grounding fix is the right one.

Conventions: for a MuJoCo free joint, dofs 0..2 are the linear force in the
GLOBAL frame and dofs 3..5 the torque about the body frame origin in the BODY
local frame.  ``vertical`` below is therefore ``qfrc_inverse[2]`` (global +z).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

WORKSPACE = Path(__file__).resolve().parents[2]
for _p in (str(WORKSPACE / "scripts"), str(WORKSPACE / "scripts" / "h1md")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jax.numpy as jnp  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from scaling.body_correct_reference import cpu_morphology_model  # noqa: E402
from scaling.family_morphology import (  # noqa: E402
    FAMILY_MORPHOLOGY_HIGH,
    FAMILY_MORPHOLOGY_LOW,
)
from scaling.parallel_cross_humanoid_train import (  # noqa: E402
    _build_robot_env,
    _ensure_latent_defaults,
)
from scaling.foot_contact_audit import floor_collidable_geoms  # noqa: E402
from scaling.reference_grounding import (  # noqa: E402
    DISTANCE_CUTOFF,
    _floor_geom,
    ground_trajectory_per_frame,
)

GRAVITY_TOL = 0.1  # |Fz| above this fraction of body weight = unsupported frame


def ground_collidable(model, qpos: np.ndarray, qvel: np.ndarray,
                      frequency: float, clearance: float):
    """Per-frame grounding restricted to geoms that CAN collide with the floor.

    ``reference_grounding._robot_geoms`` returns every non-world geom, and
    ``mj_geomDistance`` ignores contype/conaffinity, so the shipped grounder
    grounds the clip against whatever geom happens to hang lowest -- a *visual*
    mesh, typically -- leaving the geoms that can actually collide above the
    floor and ncon at 0. In the MJX-prepared model the only contacts that exist
    at all are the 8 explicit floor<->foot pairs. This is the same shift,
    computed on that set only, and is the fair control for "does grounding
    remove the root residual?".
    """
    data = mujoco.MjData(model)
    floor = _floor_geom(model)
    geoms = floor_collidable_geoms(model, floor)
    if not geoms:
        raise ValueError("model exposes no floor-collidable robot geoms")

    qpos = np.array(qpos, dtype=np.float64, copy=True)
    qvel = np.array(qvel, dtype=np.float64, copy=True)
    shifts = np.zeros(qpos.shape[0])
    before = np.zeros(qpos.shape[0])
    for i in range(qpos.shape[0]):
        data.qpos[:] = qpos[i]
        mujoco.mj_forward(model, data)
        before[i] = min(
            mujoco.mj_geomDistance(model, data, floor, g, DISTANCE_CUTOFF, None)
            for g in geoms)
        shifts[i] = clearance - before[i]
        qpos[i, 2] += shifts[i]
    qvel[:, 2] += np.gradient(shifts) * frequency
    stats = {
        "collidable_geoms": len(geoms),
        "clearance_target_m": float(clearance),
        "shift_mean_m": float(shifts.mean()),
        "shift_max_abs_m": float(np.abs(shifts).max()),
        "shift_rms_m": float(np.sqrt((shifts ** 2).mean())),
        "clearance_before_m": {"median": float(np.median(before)),
                               "min": float(before.min()),
                               "max": float(before.max())},
    }
    return qpos, qvel, stats


def _env(robot: str, reference_root: Path, clip_window: str):
    args = _ensure_latent_defaults(SimpleNamespace(
        source="h1", reference_mode="direct", reference_root=reference_root,
        clip=None, start_frame=None, frames=None, clip_windows=[clip_window],
        morphology="continuous", use_mjwarp=False,
        reward_type="MorphMimicReward", goal_type="GoalTrajMimic",
    ))
    return _build_robot_env(args, robot)[0]


def _stats(values: np.ndarray) -> dict:
    absolute = np.abs(values)
    return {
        "median": float(np.median(absolute)),
        "p95": float(np.percentile(absolute, 95)),
        "max": float(absolute.max()),
    }


def _total_normal_force(model, data) -> float:
    total = 0.0
    result = np.zeros(6)
    for c in range(data.ncon):
        mujoco.mj_contactForce(model, data, c, result)
        total += float(result[0])
    return total


def root_residual(model, qpos: np.ndarray, qvel: np.ndarray, frequency: float,
                  stride: int = 1) -> dict:
    """Per-frame root free-joint residual of an inverse-dynamics pass."""
    data = mujoco.MjData(model)
    mass = float(mujoco.mj_getTotalmass(model))
    weight = mass * float(abs(model.opt.gravity[2]))

    frames = list(range(1, qpos.shape[0] - 1, stride))
    residual = np.zeros((len(frames), 6))
    static = np.zeros((len(frames), 6))
    ncon = np.zeros(len(frames), dtype=np.int64)
    normal = np.zeros(len(frames))
    for i, t in enumerate(frames):
        data.qpos[:] = qpos[t]
        data.qvel[:] = qvel[t]
        mujoco.mj_forward(model, data)
        ncon[i] = int(data.ncon)
        normal[i] = _total_normal_force(model, data)
        data.qacc[:] = (qvel[t + 1] - qvel[t - 1]) * frequency / 2.0
        mujoco.mj_inverse(model, data)
        residual[i] = data.qfrc_inverse[:6]
        # QUASI-STATIC arm: the same pose held still. This removes the
        # acceleration term entirely, so what is left is purely "can the
        # contacts under this pose carry the robot's weight?". Regrounding
        # injects vertical acceleration through its per-frame shift, which
        # inflates the dynamic number; the static one is immune to that.
        # mj_forward OVERWRITES data.qacc with the forward-dynamics solution, so
        # qacc must be zeroed AFTER it -- zeroing first measures free fall and
        # reports a residual of exactly 0 for every frame.
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        data.qacc[:] = 0.0
        mujoco.mj_inverse(model, data)
        static[i] = data.qfrc_inverse[:6]

    vertical = residual[:, 2]
    horizontal = np.linalg.norm(residual[:, :2], axis=1)
    torque = residual[:, 3:6]
    return {
        "frames_analysed": len(frames),
        "total_mass_kg": mass,
        "body_weight_N": weight,
        "vertical_force_N": _stats(vertical),
        "vertical_force_signed": {
            "median": float(np.median(vertical)),
            "mean": float(vertical.mean()),
            "min": float(vertical.min()),
            "max": float(vertical.max()),
        },
        "vertical_force_fraction_of_weight": _stats(vertical / weight),
        "horizontal_force_N": {
            "median": float(np.median(horizontal)),
            "p95": float(np.percentile(horizontal, 95)),
            "max": float(horizontal.max()),
        },
        "horizontal_force_fraction_of_weight": {
            "median": float(np.median(horizontal) / weight),
            "p95": float(np.percentile(horizontal, 95) / weight),
            "max": float(horizontal.max() / weight),
        },
        "torque_Nm": {
            axis: _stats(torque[:, k]) for k, axis in enumerate("xyz")
        },
        "torque_norm_Nm": {
            "median": float(np.median(np.linalg.norm(torque, axis=1))),
            "p95": float(np.percentile(np.linalg.norm(torque, axis=1), 95)),
            "max": float(np.linalg.norm(torque, axis=1).max()),
        },
        "static_vertical_force_N": _stats(static[:, 2]),
        "static_vertical_force_fraction_of_weight": _stats(static[:, 2] / weight),
        "static_horizontal_force_fraction_of_weight": {
            "median": float(
                np.median(np.linalg.norm(static[:, :2], axis=1)) / weight),
            "p95": float(np.percentile(
                np.linalg.norm(static[:, :2], axis=1), 95) / weight),
        },
        "static_torque_norm_Nm": {
            "median": float(np.median(np.linalg.norm(static[:, 3:6], axis=1))),
            "p95": float(np.percentile(
                np.linalg.norm(static[:, 3:6], axis=1), 95)),
        },
        "static_pct_frames_vertical_over_10pct_weight": float(
            100.0 * np.mean(np.abs(static[:, 2]) > GRAVITY_TOL * weight)),
        "pct_frames_vertical_over_10pct_weight": float(
            100.0 * np.mean(np.abs(vertical) > GRAVITY_TOL * weight)),
        "pct_frames_vertical_over_50pct_weight": float(
            100.0 * np.mean(np.abs(vertical) > 0.5 * weight)),
        "contacts": {
            "pct_frames_no_contact": float(100.0 * np.mean(ncon == 0)),
            "ncon_median": float(np.median(ncon)),
            "ncon_max": int(ncon.max()),
            "normal_force_N_median": float(np.median(normal)),
            "normal_force_N_p95": float(np.percentile(normal, 95)),
            "normal_force_fraction_of_weight_median": float(
                np.median(normal) / weight),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robots", nargs="+", default=["h1", "g1"])
    parser.add_argument("--clip-window", default="dance2_subject4:19482:800")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--bodies", nargs="+",
                        default=["nominal", "low_corner", "high_corner"])
    parser.add_argument("--clearance", type=float, default=0.002,
                        help="reference_grounding's own default: a 2 mm GAP")
    parser.add_argument("--touch-clearance", type=float, default=-0.002,
                        help="grounding target that actually forms contacts")
    parser.add_argument(
        "--reference-root", type=Path,
        default=WORKSPACE / "external_data" / "cross_humanoid")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    morphologies = {
        "nominal": np.ones(4, dtype=np.float32),
        "low_corner": FAMILY_MORPHOLOGY_LOW.astype(np.float32),
        "high_corner": FAMILY_MORPHOLOGY_HIGH.astype(np.float32),
    }

    out: dict = {"clip_window": args.clip_window, "stride": args.stride,
                 "robots": {}}
    for robot in args.robots:
        env = _env(robot, args.reference_root, args.clip_window)
        traj = env.th.traj
        frequency = float(traj.info.frequency)
        n = int(np.asarray(traj.data.split_points)[1])
        qpos_raw = np.asarray(traj.data.qpos, dtype=np.float64)[:n]
        qvel_raw = np.asarray(traj.data.qvel, dtype=np.float64)[:n]

        out["robots"][robot] = {}
        for label in args.bodies:
            morphology = morphologies[label]
            model = (env._model if label == "nominal"
                     else cpu_morphology_model(env._model, robot, morphology))
            offset = float(np.asarray(
                env.root_height_offset(jnp.asarray(morphology))))

            # RAW: reference as trained on, plus the same root lift the env's
            # reset applies to a longer/shorter-legged body.
            qpos = qpos_raw.copy()
            qpos[:, 2] += offset
            raw = root_residual(model, qpos, qvel_raw, frequency, args.stride)
            raw["root_height_offset_m"] = offset

            # CONTROL: per-frame regrounded on THIS body's own geometry.
            # Two clearances, because the default one is a 2 mm GAP: MuJoCo's
            # contact margin is 0 in these models, so grounding to +2 mm still
            # yields ncon == 0 and cannot absorb any weight. The second arm
            # closes the gap so contacts actually exist.
            cell = {"raw": raw}
            for key, clearance in (("regrounded", args.clearance),
                                   ("regrounded_touch", args.touch_clearance)):
                grounded_traj, grounding_stats = ground_trajectory_per_frame(
                    model, traj, clearance)
                qpos_g = np.asarray(
                    grounded_traj.data.qpos, dtype=np.float64)[:n]
                qvel_g = np.asarray(
                    grounded_traj.data.qvel, dtype=np.float64)[:n]
                arm = root_residual(
                    model, qpos_g, qvel_g, frequency, args.stride)
                arm["grounding"] = grounding_stats
                cell[key] = arm

            qpos_c, qvel_c, collidable_stats = ground_collidable(
                model, qpos_raw, qvel_raw, frequency, args.touch_clearance)
            arm = root_residual(
                model, qpos_c, qvel_c, frequency, args.stride)
            arm["grounding"] = collidable_stats
            cell["regrounded_collidable"] = arm
            regrounded = cell["regrounded_collidable"]

            out["robots"][robot][label] = cell
            print(f"[root] {robot:<3} {label:<11} raw Fz "
                  f"{raw['vertical_force_fraction_of_weight']['median']:.2f} bw "
                  f"({raw['vertical_force_N']['median']:.0f} N) static "
                  f"{raw['static_vertical_force_fraction_of_weight']['median']:.2f}"
                  f" bw, "
                  f"{raw['pct_frames_vertical_over_10pct_weight']:.0f}% frames "
                  f">0.1bw, ncon med {raw['contacts']['ncon_median']:.0f} "
                  f"| regrounded(gap) Fz "
                  f"{cell['regrounded']['vertical_force_fraction_of_weight']['median']:.2f}"
                  f" bw ncon med "
                  f"{cell['regrounded']['contacts']['ncon_median']:.0f} "
                  f"| touch Fz "
                  f"{cell['regrounded_touch']['vertical_force_fraction_of_weight']['median']:.2f}"
                  f" bw ncon med "
                  f"{cell['regrounded_touch']['contacts']['ncon_median']:.0f} "
                  f"| collidable Fz "
                  f"{regrounded['vertical_force_fraction_of_weight']['median']:.2f}"
                  f" bw static "
                  f"{regrounded['static_vertical_force_fraction_of_weight']['median']:.2f}"
                  f" bw, ncon med "
                  f"{regrounded['contacts']['ncon_median']:.0f} "
                  f"(shift rms "
                  f"{regrounded['grounding']['shift_rms_m'] * 100:.1f} cm)",
                  flush=True)
        del env

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[root] -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
