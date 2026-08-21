"""Can the robot stand at all, before anyone asks it to dance?

Every cross-humanoid tracking result to date is reported as "the policy tracks
badly and the episode ends after N steps".  That number is uninterpretable until
the trivial control question is answered: put the robot at a standing pose, ask
it to do *nothing*, and see whether it is still standing five seconds later.  If
it is not, the tracking numbers are measuring a broken configuration and no
amount of RL tuning will move them.

Three control cases per body, all run at the simulation frequency exactly as
``loco_mujoco.core.control_functions.pd.PDControl`` does:

* ``zero``      -- ``ctrl = 0`` (what ``DefaultControl`` receives from a
                   zero action; H1 and G1 both train on direct torque).
* ``pd_1x``     -- ``ctrl = clip(p (q0 - q) - d qdot, ctrlrange)`` at the
                   shipped gains, holding the *initial* joint configuration.
* ``pd_3x``     -- the same at 3x gains, the setting
                   ``morphology_deepmimic`` records as necessary for G1 dance.

Gain provenance
---------------
``morphology_deepmimic.PD_GAINS`` ships gains for **g1 only**; H1 has always
trained on raw torque and has no shipped PD gains.  Rather than silently
substituting a proxy, H1's gains are *derived* here from the model's own
``ctrlrange`` (p = torque limit per rad, d = 0.03 p, the ratio G1's shipped
gains use) and every H1 record carries ``"gain_source": "derived_from_ctrlrange"``.

Termination is judged by the same criteria the training run uses
(``MorphologyAwareRootPoseTrajTerminalStateHandler``): the root height, with the
body's own grounding offset subtracted, against the reference clip's height band
+/- 0.3 m, and the root rotation against the clip's quaternion centroid + 30 deg.
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
from scaling.reference_grounding import (  # noqa: E402
    _floor_geom,
    _robot_geoms,
    floor_clearance,
)

DEFAULT_CLIP = "dance2_subject4:19482:800"
DEFAULT_REFERENCE_ROOT = WORKSPACE / "external_data" / "cross_humanoid"

BODIES = {
    "nominal": np.ones(4, dtype=np.float32),
    "low_corner": FAMILY_MORPHOLOGY_LOW.astype(np.float32),
    "high_corner": FAMILY_MORPHOLOGY_HIGH.astype(np.float32),
}


# --------------------------------------------------------------------------- #
# environment / model plumbing (shared with oracle_tracking.py)
# --------------------------------------------------------------------------- #
def build_env(robot: str, clip_window: str = DEFAULT_CLIP,
              reference_root: Path = DEFAULT_REFERENCE_ROOT):
    """The training environment for one family, built on CPU.

    Same construction as ``reference_feasibility``: the MJX class is what the
    trainer builds, but ``env._model`` is a plain ``mujoco.MjModel`` and
    ``env.th.traj`` the reference, which is all the offline analyses need.
    The terminal handler is the production one so its thresholds are read from
    the same object training uses, not re-derived here.
    """
    args = _ensure_latent_defaults(SimpleNamespace(
        source="h1", reference_mode="direct", reference_root=reference_root,
        clip=None, start_frame=None, frames=None, clip_windows=[clip_window],
        morphology="continuous", use_mjwarp=False,
        reward_type="MorphMimicReward", goal_type="GoalTrajMimic",
        terminal_handler="MorphologyAwareRootPoseTrajTerminalStateHandler",
        max_root_deviation=0.5,
    ))
    return _build_robot_env(args, robot)[0]


def body_model(env, robot: str, morphology: np.ndarray):
    """CPU model for one morphology corner (``nominal`` returns the base model)."""
    if np.allclose(morphology, 1.0):
        return env._model
    return cpu_morphology_model(env._model, robot, morphology)


def actuated_index(model):
    """``(qpos ids, dof ids)`` of the joints the actuators drive, in ctrl order."""
    qpos_ids, dof_ids = [], []
    for a in range(model.nu):
        joint = int(model.actuator_trnid[a, 0])
        qpos_ids.append(int(model.jnt_qposadr[joint]))
        dof_ids.append(int(model.jnt_dofadr[joint]))
    return np.asarray(qpos_ids), np.asarray(dof_ids)


def joint_ranges(model, qpos_ids):
    """``(lo, hi)`` per actuated joint; unlimited joints get +/- inf."""
    lo = np.full(len(qpos_ids), -np.inf)
    hi = np.full(len(qpos_ids), np.inf)
    for a in range(model.nu):
        joint = int(model.actuator_trnid[a, 0])
        if model.jnt_limited[joint]:
            lo[a], hi[a] = model.jnt_range[joint]
    return lo, hi


def pd_gains(robot: str, model) -> tuple[np.ndarray, np.ndarray, str]:
    """Shipped PD gains, or gains derived from the model's own torque limits.

    ``morphology_deepmimic.PD_GAINS`` only defines g1.  Returning a derived set
    for the others is fine as long as nobody can mistake it for a shipped
    number, hence the third return value.
    """
    from morphology_deepmimic import PD_GAINS

    if robot in PD_GAINS:
        gains = PD_GAINS[robot]
        p = np.asarray(gains["p_gain"], dtype=np.float64)
        d = np.asarray(gains["d_gain"], dtype=np.float64)
        if p.size != model.nu:
            raise ValueError(
                f"{robot}: shipped PD gains have {p.size} entries but the model "
                f"has {model.nu} actuators.")
        return p, d, "shipped(morphology_deepmimic.PD_GAINS)"
    limit = np.max(np.abs(model.actuator_ctrlrange), axis=1)
    return limit, 0.03 * limit, "derived_from_ctrlrange"


def collidable_floor_geoms(model) -> np.ndarray:
    """Robot geoms that can actually make contact with the floor.

    H1 and G1 as shipped set ``contype = conaffinity = 0`` on **every** geom and
    declare contact through explicit ``<pair>`` elements instead -- four foot
    spheres per robot against the floor plane, and nothing else.  A grounding
    routine that measures clearance over all robot geoms (which
    ``reference_grounding._robot_geoms`` does) is therefore measuring visual
    meshes that can never touch anything, so both scripts here report the
    collidable clearance alongside the all-geom one.
    """
    floor = _floor_geom(model)
    contype = np.asarray(model.geom_contype)
    conaff = np.asarray(model.geom_conaffinity)
    hits = set()
    for p in range(model.npair):
        g1, g2 = int(model.pair_geom1[p]), int(model.pair_geom2[p])
        if g1 == floor and int(model.geom_bodyid[g2]) != 0:
            hits.add(g2)
        if g2 == floor and int(model.geom_bodyid[g1]) != 0:
            hits.add(g1)
    for g in range(model.ngeom):
        if g == floor or int(model.geom_bodyid[g]) == 0:
            continue
        if (contype[g] & conaff[floor]) or (contype[floor] & conaff[g]):
            hits.add(g)
    return np.asarray(sorted(hits), dtype=np.int64)


def ground_qpos(model, qpos, clearance: float = 0.002, collidable_only: bool = True):
    """Raise/lower the root so the lowest geom sits at ``clearance``.

    ``collidable_only`` grounds against the geoms that can touch the floor.
    ``False`` reproduces ``reference_grounding``'s all-geom behaviour.
    """
    data = mujoco.MjData(model)
    floor = _floor_geom(model)
    geoms = (collidable_floor_geoms(model) if collidable_only
             else _robot_geoms(model, floor))
    if geoms.size == 0:
        raise ValueError("No geom can collide with the floor in this model.")
    out = np.array(qpos, dtype=np.float64, copy=True)
    data.qpos[:] = out
    mujoco.mj_forward(model, data)
    before = floor_clearance(model, data, floor, geoms)
    out[2] += clearance - before
    return out, float(clearance - before)


def geom_clearances(model, data, qpos, geoms) -> np.ndarray:
    """Per-geom signed distance to the floor at ``qpos``."""
    floor = _floor_geom(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    return np.asarray([
        mujoco.mj_geomDistance(model, data, floor, int(g), 10.0, None)
        for g in geoms
    ])


def ankle_pitch_joints(model) -> list[int]:
    """Joint ids of the ankle pitch dofs (the ones that level the sole)."""
    out = []
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        lowered = name.lower()
        if "ankle" in lowered and "roll" not in lowered and "yaw" not in lowered:
            out.append(j)
    return out


def flat_foot_pose(model, qpos, span: float = 0.4, samples: int = 161):
    """``qpos`` with the ankle pitch chosen so the whole sole reaches the floor.

    H1's shipped ``qpos0`` puts the ankle at 0 rad, at which the heel sphere sits
    9 mm above the toe sphere: the robot balances on a line through its toes and
    pitches backward at any gain.  That is a property of the *pose*, not of the
    robot, so the stand test needs a pose in which every collidable foot geom
    can actually reach the ground before it can say anything about the robot.

    Scans a single ankle-pitch offset (applied to both ankles) and keeps the one
    that minimises the spread of collidable-geom floor clearances, which is
    invariant to the subsequent root-height grounding.
    """
    geoms = collidable_floor_geoms(model)
    joints = ankle_pitch_joints(model)
    data = mujoco.MjData(model)
    base = np.array(qpos, dtype=np.float64, copy=True)
    if not joints or geoms.size < 2:
        return base, 0.0, float(np.ptp(geom_clearances(model, data, base, geoms)))
    adr = [int(model.jnt_qposadr[j]) for j in joints]
    best = (np.inf, 0.0)
    for offset in np.linspace(-span, span, samples):
        trial = base.copy()
        for j, a in zip(joints, adr):
            value = base[a] + offset
            if model.jnt_limited[j]:
                value = float(np.clip(value, *model.jnt_range[j]))
            trial[a] = value
        spread = float(np.ptp(geom_clearances(model, data, trial, geoms)))
        if spread < best[0]:
            best = (spread, float(offset))
    spread, offset = best
    out = base.copy()
    for j, a in zip(joints, adr):
        value = base[a] + offset
        if model.jnt_limited[j]:
            value = float(np.clip(value, *model.jnt_range[j]))
        out[a] = value
    return out, offset, spread


# --------------------------------------------------------------------------- #
# the training terminal criteria, evaluated offline
# --------------------------------------------------------------------------- #
class TerminalCriteria:
    """The production root-pose terminal test, callable on a plain ``MjData``.

    Thresholds are read off the environment's own handler
    (``root_height_range``, the quaternion centroid and its margin,
    ``max_root_pos_deviation``) so this cannot drift from what training used.
    """

    def __init__(self, env, height_offset: float):
        handler = env._terminal_state_handler
        if not handler.initialized:
            handler.init_from_traj(env.th)
        self.height_range = (float(handler.root_height_range[0]),
                             float(handler.root_height_range[1]))
        self.rot_threshold = float(handler._valid_threshold)
        self.centroid_quat = np.asarray(handler._centroid_quat, dtype=np.float64)
        self.max_root_deviation = float(handler.max_root_pos_deviation)
        self.height_offset = float(height_offset)

    def height_violation(self, qpos) -> bool:
        height = float(qpos[2]) - self.height_offset
        return height < self.height_range[0] or height > self.height_range[1]

    def rotation_violation(self, qpos) -> bool:
        quat = np.asarray(qpos[3:7], dtype=np.float64)
        quat = np.array([quat[1], quat[2], quat[3], quat[0]])  # scalar last
        quat = quat / np.linalg.norm(quat)
        dot = float(np.clip(np.dot(self.centroid_quat, quat), -1.0, 1.0))
        return 2.0 * np.arccos(dot) > self.rot_threshold

    def deviation_violation(self, qpos, reference_xy) -> bool:
        return bool(np.linalg.norm(np.asarray(qpos[:2]) - reference_xy)
                    > self.max_root_deviation)


def root_pitch_roll_deg(qpos) -> tuple[float, float]:
    """Pitch and roll of the root, in degrees, from the free-joint quaternion."""
    from scipy.spatial.transform import Rotation

    quat = np.asarray(qpos[3:7], dtype=np.float64)
    rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
    roll, pitch, _ = rot.as_euler("xyz", degrees=True)
    return float(pitch), float(roll)


# --------------------------------------------------------------------------- #
# the test
# --------------------------------------------------------------------------- #
def run_case(model, qpos_init, criteria, mode: str, p_gain, d_gain,
             seconds: float, control_decimation: int) -> dict:
    """One 5 s rollout under one control law; per-control-step diagnostics."""
    qpos_ids, dof_ids = actuated_index(model)
    lo, hi = joint_ranges(model, qpos_ids)
    ctrl_lo = model.actuator_ctrlrange[:, 0].copy()
    ctrl_hi = model.actuator_ctrlrange[:, 1].copy()
    unlimited = ~model.actuator_ctrllimited.astype(bool)
    ctrl_lo[unlimited] = -np.inf
    ctrl_hi[unlimited] = np.inf

    target = np.clip(np.asarray(qpos_init)[qpos_ids], lo, hi)

    data = mujoco.MjData(model)
    data.qpos[:] = qpos_init
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    sim_steps = int(round(seconds / model.opt.timestep))
    heights, pitches, rolls, ncons = [], [], [], []
    fall_step = None
    height_step = None
    rotation_step = None
    topple_step = None
    unstable = False
    #: a physical topple, independent of any criterion: the root has dropped
    #: 30 cm below where it started, or the trunk has tipped past 45 degrees.
    topple_height = float(qpos_init[2]) - 0.30

    for step in range(sim_steps):
        if mode == "zero":
            data.ctrl[:] = 0.0
        else:
            error = target - data.qpos[qpos_ids]
            ctrl = p_gain * error - d_gain * data.qvel[dof_ids]
            data.ctrl[:] = np.clip(ctrl, ctrl_lo, ctrl_hi)
        mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qpos)):
            unstable = True
            break
        if step % control_decimation == 0:
            control_step = step // control_decimation
            heights.append(float(data.qpos[2]))
            pitch, roll = root_pitch_roll_deg(data.qpos)
            pitches.append(pitch)
            rolls.append(roll)
            ncons.append(int(data.ncon))
            if height_step is None and criteria.height_violation(data.qpos):
                height_step = control_step
            if rotation_step is None and criteria.rotation_violation(data.qpos):
                rotation_step = control_step
            if fall_step is None and (height_step == control_step
                                      or rotation_step == control_step):
                fall_step = control_step
            if topple_step is None and (
                    float(data.qpos[2]) < topple_height
                    or abs(pitch) > 45.0 or abs(roll) > 45.0):
                topple_step = control_step

    control_hz = 1.0 / (model.opt.timestep * control_decimation)
    return {
        "mode": mode,
        "integrator_diverged": unstable,
        "root_height_init_m": float(qpos_init[2]),
        "root_height_min_m": float(np.min(heights)) if heights else None,
        "root_height_final_m": float(heights[-1]) if heights else None,
        "root_height_drop_m": (float(qpos_init[2] - np.min(heights))
                               if heights else None),
        "pitch_deg": {"min": float(np.min(pitches)), "max": float(np.max(pitches)),
                      "final": float(pitches[-1])} if pitches else None,
        "roll_deg": {"min": float(np.min(rolls)), "max": float(np.max(rolls)),
                     "final": float(rolls[-1])} if rolls else None,
        "ncon": {"init": ncons[0] if ncons else None,
                 "min": int(np.min(ncons)) if ncons else None,
                 "median": float(np.median(ncons)) if ncons else None,
                 "final": int(ncons[-1]) if ncons else None,
                 "frames_with_zero_contacts_pct": (
                     float(100.0 * np.mean(np.asarray(ncons) == 0))
                     if ncons else None)},
        "stayed_in_height_band": height_step is None,
        "stayed_in_rotation_band": rotation_step is None,
        "first_height_violation_step": height_step,
        "first_rotation_violation_step": rotation_step,
        "terminal_steps_survived": (int(sim_steps / control_decimation)
                                    if fall_step is None else fall_step),
        "time_to_fall_s": (None if fall_step is None
                           else float(fall_step) / control_hz),
        "toppled": topple_step is not None,
        "time_to_topple_s": (None if topple_step is None
                             else float(topple_step) / control_hz),
        "control_hz": control_hz,
        "root_height_series_m": [round(h, 4) for h in heights[::10]],
        "pitch_series_deg": [round(p, 2) for p in pitches[::10]],
    }


def run_body(env, robot: str, label: str, morphology, args) -> list[dict]:
    model = body_model(env, robot, morphology)
    offset = float(np.asarray(env.root_height_offset(jnp.asarray(morphology))))
    criteria = TerminalCriteria(env, offset)
    p_base, d_base, gain_source = pd_gains(robot, model)

    frame0 = np.asarray(env.th.traj.data.qpos[0], dtype=np.float64)
    qpos0 = np.asarray(model.qpos0, dtype=np.float64)
    flat, ankle_offset, flat_spread = flat_foot_pose(model, qpos0)
    poses = {
        "reference_frame0": frame0,
        "model_qpos0": qpos0,
        "flat_foot_qpos0": flat,
    }
    geoms = collidable_floor_geoms(model)
    scratch = mujoco.MjData(model)
    spreads = {
        name: float(np.ptp(geom_clearances(model, scratch, q, geoms)))
        for name, q in poses.items()
    }

    modes = [("zero", 0.0)] + [(f"pd_{s:g}x", float(s)) for s in args.gain_scales]

    rows = []
    for pose_name, qpos in poses.items():
        grounded, lift = ground_qpos(model, qpos, collidable_only=True)
        _, lift_all = ground_qpos(model, qpos, collidable_only=False)
        for mode, scale in modes:
            record = run_case(
                model, grounded, criteria, mode,
                p_base * scale, d_base * scale,
                args.seconds, args.control_decimation)
            record.update({
                "robot": robot, "body": label, "init_pose": pose_name,
                "morphology": [float(v) for v in np.asarray(morphology)],
                "gain_scale": scale, "gain_source": gain_source,
                "root_height_offset_m": offset,
                "grounding_lift_m": lift,
                "grounding_lift_all_geoms_m": lift_all,
                "collidable_vs_all_geom_grounding_gap_m": lift - lift_all,
                "n_collidable_floor_geoms": int(geoms.size),
                "foot_geom_clearance_spread_m": spreads[pose_name],
                "flat_foot_ankle_offset_rad": (ankle_offset
                                               if pose_name == "flat_foot_qpos0"
                                               else None),
                "flat_foot_residual_spread_m": (flat_spread
                                                if pose_name == "flat_foot_qpos0"
                                                else None),
                "terminal_height_band_m": list(criteria.height_range),
                "terminal_rotation_threshold_deg": float(
                    np.degrees(criteria.rot_threshold)),
            })
            rows.append(record)
            print(f"[stand] {robot:<3} {label:<11} {pose_name:<17} {mode:<6} "
                  f"z {record['root_height_init_m']:.3f} -> "
                  f"{record['root_height_final_m']:.3f} "
                  f"(min {record['root_height_min_m']:.3f}) "
                  f"pitch {record['pitch_deg']['max']:+.0f} deg "
                  f"ncon {record['ncon']['init']}->{record['ncon']['final']} "
                  + ("UPRIGHT" if record["time_to_topple_s"] is None
                     else f"TOPPLED @ {record['time_to_topple_s']:.2f}s")
                  + (" | terminal ok" if record["time_to_fall_s"] is None
                     else f" | terminal @ {record['time_to_fall_s']:.2f}s"),
                  flush=True)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robots", nargs="+", default=["h1", "g1"])
    parser.add_argument("--clip-window", default=DEFAULT_CLIP)
    parser.add_argument("--reference-root", type=Path,
                        default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--control-decimation", type=int, default=5,
                        help="simulation steps per control step (env n_substeps)")
    parser.add_argument("--gain-scales", type=float, nargs="+",
                        default=[1.0, 3.0, 10.0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for robot in args.robots:
        env = build_env(robot, args.clip_window, args.reference_root)
        for label, morphology in BODIES.items():
            rows.extend(run_body(env, robot, label, morphology, args))
        del env

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "clip_window": args.clip_window,
        "seconds": args.seconds,
        "control_decimation": args.control_decimation,
        "gain_scales": list(args.gain_scales),
        "notes": {
            "collision": "H1 and G1 set contype=conaffinity=0 on every geom; "
                         "contact exists only through explicit <pair> elements "
                         "(4 foot spheres vs the floor plane per robot). Nothing "
                         "above the ankle can touch the ground, so a toppled "
                         "robot's pelvis sinks through the floor plane.",
            "toppled": "root dropped >0.30 m below its initial height, or root "
                       "pitch/roll exceeded 45 deg -- criterion-independent.",
            "terminal": "the production MorphologyAwareRootPoseTrajTerminal"
                        "StateHandler thresholds, read off the env's own handler.",
        },
        "cases": rows,
    }, indent=2), encoding="utf-8")
    print(f"[stand] -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
