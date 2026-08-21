"""Put the reference's feet on the floor, per frame, for any humanoid family.

Why this is needed
------------------
The cross-humanoid references produced by ``cross_humanoid_retarget.py`` are
never grounded: measured on ``dance2_subject4``, **100% of frames have no ground
contact at all** on both H1 and G1, while the same model reports 8 contacts when
the robot is actually standing. The policy is therefore asked to imitate a
motion in which the robot never touches the floor. Every reward term is
maximised by an airborne pose, so tracking the reference and standing up are
in direct conflict, and the episode ends in about a second.

The pipeline-v2 validation reached the same conclusion for the single-body H1
work and rejected a reference whose "feet float above the floor in 68% of
frames"; its gold reference was per-frame re-grounded to within +/-2 cm. That
gold reference is what the good no-fall H1 results were trained on.

Two existing helpers were not usable here:

* ``h1md.c6_reward_discrimination.reground`` is per-frame but hardcodes H1's
  ``left_foot`` / ``right_foot`` geom names, so it cannot ground G1.
* ``morphology_deepmimic.ground_trajectory_constant`` is family-generic but
  applies a single constant lift, which by construction cannot fix per-frame
  floating -- exactly the limitation v2 recorded.

This module is the family-generic per-frame version. It finds the true
clearance with ``mj_geomDistance`` over every collidable robot geom, so it needs
no per-family geom names, and it keeps the trajectory self-consistent by
recomputing the derived fields from the corrected pose rather than shifting them.
"""

from __future__ import annotations

from typing import Sequence

import jax.numpy as jnp
import mujoco
import numpy as np

from loco_mujoco.trajectory import Trajectory

#: Distance to hold between the lowest robot geom and the floor.
DEFAULT_CLEARANCE = 0.002
#: mj_geomDistance search cutoff; beyond this the exact value does not matter.
DISTANCE_CUTOFF = 10.0


def _floor_geom(model) -> int:
    for name in ("floor", "ground", "plane"):
        index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if index >= 0:
            return index
    # fall back to the first geom attached to the world body
    world = np.nonzero(np.asarray(model.geom_bodyid) == 0)[0]
    if world.size == 0:
        raise ValueError("No floor geom found and no world-body geom to use.")
    return int(world[0])


def _robot_geoms(model, floor: int) -> np.ndarray:
    return np.asarray(
        [g for g in range(model.ngeom)
         if g != floor and int(model.geom_bodyid[g]) != 0],
        dtype=np.int64,
    )


def floor_clearance(model, data, floor: int, geoms: Sequence[int]) -> float:
    """Signed distance from the lowest robot geom to the floor.

    Positive means floating, negative means penetrating. Uses MuJoCo's own
    geom-pair distance, so mesh and capsule extents are exact rather than
    approximated by a bounding radius.
    """
    return min(
        mujoco.mj_geomDistance(model, data, floor, int(g), DISTANCE_CUTOFF, None)
        for g in geoms
    )


def ground_trajectory_per_frame(
    model, traj: Trajectory, clearance: float = DEFAULT_CLEARANCE
) -> tuple[Trajectory, dict]:
    """Shift each frame's root height so the lowest geom sits at ``clearance``.

    Only root z changes. Root vertical velocity is re-derived from the corrected
    heights so the qvel term stays consistent with the qpos term, and every
    derived field (xpos, site_xpos, ...) is recomputed by forward kinematics
    instead of being translated, so the trajectory remains internally exact.
    """
    data = mujoco.MjData(model)
    floor = _floor_geom(model)
    geoms = _robot_geoms(model, floor)
    if geoms.size == 0:
        raise ValueError("Model exposes no robot geoms to ground against.")

    qpos = np.array(traj.data.qpos, dtype=np.float64, copy=True)
    qvel = np.array(traj.data.qvel, dtype=np.float64, copy=True)
    n = qpos.shape[0]
    shifts = np.zeros(n)
    before = np.zeros(n)

    for i in range(n):
        data.qpos[:] = qpos[i]
        mujoco.mj_forward(model, data)
        before[i] = floor_clearance(model, data, floor, geoms)
        shifts[i] = clearance - before[i]
        qpos[i, 2] += shifts[i]

    # Root linear z velocity must follow the corrected height, or the qvel term
    # rewards the old (floating) vertical motion.
    frequency = float(traj.info.frequency)
    qvel[:, 2] += np.gradient(shifts) * frequency

    # Recompute every derived field from the corrected state.
    xpos = np.zeros_like(np.asarray(traj.data.xpos))
    xquat = np.zeros_like(np.asarray(traj.data.xquat))
    cvel = np.zeros_like(np.asarray(traj.data.cvel))
    subtree_com = np.zeros_like(np.asarray(traj.data.subtree_com))
    site_xpos = np.zeros_like(np.asarray(traj.data.site_xpos))
    site_xmat = np.zeros_like(np.asarray(traj.data.site_xmat))
    after = np.zeros(n)
    for i in range(n):
        data.qpos[:] = qpos[i]
        data.qvel[:] = qvel[i]
        mujoco.mj_forward(model, data)
        after[i] = floor_clearance(model, data, floor, geoms)
        xpos[i] = data.xpos
        xquat[i] = data.xquat
        cvel[i] = data.cvel
        subtree_com[i] = data.subtree_com
        site_xpos[i] = data.site_xpos
        site_xmat[i] = data.site_xmat

    corrected = Trajectory(
        info=traj.info,
        data=traj.data.replace(
            qpos=jnp.asarray(qpos, dtype=jnp.float32),
            qvel=jnp.asarray(qvel, dtype=jnp.float32),
            xpos=jnp.asarray(xpos, dtype=jnp.float32),
            xquat=jnp.asarray(xquat, dtype=jnp.float32),
            cvel=jnp.asarray(cvel, dtype=jnp.float32),
            subtree_com=jnp.asarray(subtree_com, dtype=jnp.float32),
            site_xpos=jnp.asarray(site_xpos, dtype=jnp.float32),
            site_xmat=jnp.asarray(site_xmat, dtype=jnp.float32),
        ),
    )
    stats = {
        "frames": int(n),
        "clearance_target_m": float(clearance),
        "shift_mean_m": float(shifts.mean()),
        "shift_max_abs_m": float(np.abs(shifts).max()),
        "shift_rms_m": float(np.sqrt((shifts ** 2).mean())),
        "clearance_before_m": {
            "median": float(np.median(before)), "min": float(before.min()),
            "max": float(before.max()),
        },
        "clearance_after_m": {
            "median": float(np.median(after)), "min": float(after.min()),
            "max": float(after.max()),
        },
        "pct_frames_floating_over_2cm_before": float(
            100.0 * np.mean(before > 0.02)),
        "pct_frames_within_2cm_after": float(100.0 * np.mean(np.abs(after) <= 0.02)),
    }
    return corrected, stats


def ground_trajectory_constant(
    model, traj: Trajectory, clearance: float = DEFAULT_CLEARANCE
) -> tuple[Trajectory, dict]:
    """Single constant lift so the clip's lowest frame clears the floor.

    Preserves the motion and velocities exactly, and therefore cannot fix
    per-frame floating -- kept as the control arm for the per-frame version.
    """
    data = mujoco.MjData(model)
    floor = _floor_geom(model)
    geoms = _robot_geoms(model, floor)
    qpos = np.array(traj.data.qpos, dtype=np.float64, copy=True)
    clearances = np.zeros(qpos.shape[0])
    for i in range(qpos.shape[0]):
        data.qpos[:] = qpos[i]
        mujoco.mj_forward(model, data)
        clearances[i] = floor_clearance(model, data, floor, geoms)
    offset = float(clearance - clearances.min())
    qpos[:, 2] += offset
    corrected = Trajectory(
        info=traj.info,
        data=traj.data.replace(qpos=jnp.asarray(qpos, dtype=jnp.float32)),
    )
    return corrected, {
        "frames": int(qpos.shape[0]), "constant_offset_m": offset,
        "clearance_before_m": {"median": float(np.median(clearances)),
                               "min": float(clearances.min()),
                               "max": float(clearances.max())},
        "pct_frames_floating_over_2cm_before": float(
            100.0 * np.mean(clearances > 0.02)),
    }


def apply(model, traj: Trajectory, mode: str, clearance: float = DEFAULT_CLEARANCE):
    if mode in (None, "none"):
        return traj, {"mode": "none"}
    if mode == "per_frame":
        corrected, stats = ground_trajectory_per_frame(model, traj, clearance)
    elif mode == "constant":
        corrected, stats = ground_trajectory_constant(model, traj, clearance)
    else:
        raise ValueError(
            f"reference grounding mode must be none/constant/per_frame, got {mode!r}")
    stats["mode"] = mode
    return corrected, stats
