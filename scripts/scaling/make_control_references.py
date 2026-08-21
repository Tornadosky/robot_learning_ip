"""Positive-control reference clips for the cross-humanoid trainer.

Two questions the production clip cannot answer on its own:

``stand_still``
    Can this trainer learn *anything*? Every frame is the robot's own standing
    pose, grounded, at zero velocity. Tracking it perfectly means standing up
    and not moving. If one policy cannot drive both families to the full
    horizon on this clip, the defect is in the trainer, the observation, the
    reward or the terminal -- not in the motion, the budget or the morphology,
    and no result on a real clip is interpretable until it is fixed.

``dance_slowN``
    Is the clip simply too dynamic for these robots? The same
    ``dance2_subject4`` poses, played N times slower, so required joint
    accelerations fall by N^2 while the pose sequence is unchanged. A policy
    that tracks the slow version and not the fast one is telling you the
    reference is outside the robot's dynamic envelope, which is a different
    problem from a learning failure.

Both are written in the exact layout the trainer expects:

    <reference-root>/h1_source/<clip>/<robot>/start0_<frames>f_direct.npz

Derived fields are recomputed by forward kinematics rather than copied, the
same discipline ``reference_grounding`` uses, so the saved trajectory is
internally consistent for the reward, the goal and the FK targets alike.
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

from loco_mujoco.trajectory import Trajectory  # noqa: E402

from scaling.parallel_cross_humanoid_train import HUMANOIDS  # noqa: E402
from scaling.reference_grounding import (  # noqa: E402
    DEFAULT_CLEARANCE,
    _floor_geom,
    _robot_geoms,
    floor_clearance,
)


def _recompute(model, qpos: np.ndarray, qvel: np.ndarray, template: Trajectory):
    """Rebuild every derived field of a trajectory from (qpos, qvel) by FK."""
    data = mujoco.MjData(model)
    n = qpos.shape[0]
    fields = {
        name: np.zeros((n,) + np.asarray(getattr(template.data, name)).shape[1:],
                       dtype=np.float32)
        for name in ("xpos", "xquat", "cvel", "subtree_com", "site_xpos", "site_xmat")
    }
    for i in range(n):
        data.qpos[:] = qpos[i]
        data.qvel[:] = qvel[i]
        mujoco.mj_forward(model, data)
        fields["xpos"][i] = data.xpos
        fields["xquat"][i] = data.xquat
        fields["cvel"][i] = data.cvel
        fields["subtree_com"][i] = data.subtree_com
        fields["site_xpos"][i] = data.site_xpos
        fields["site_xmat"][i] = data.site_xmat
    return Trajectory(
        info=template.info,
        data=template.data.replace(
            qpos=jnp.asarray(qpos, dtype=jnp.float32),
            qvel=jnp.asarray(qvel, dtype=jnp.float32),
            **{k: jnp.asarray(v) for k, v in fields.items()},
        ),
        transitions=template.transitions,
    )


def _ground_pose(model, qpos: np.ndarray, clearance: float = DEFAULT_CLEARANCE):
    """Shift root z so the lowest geom sits `clearance` above the floor."""
    data = mujoco.MjData(model)
    floor = _floor_geom(model)
    geoms = _robot_geoms(model, floor)
    out = np.array(qpos, dtype=np.float64, copy=True)
    data.qpos[:] = out
    mujoco.mj_forward(model, data)
    before = floor_clearance(model, data, floor, geoms)
    out[2] += clearance - before
    data.qpos[:] = out
    mujoco.mj_forward(model, data)
    return out, float(before), float(floor_clearance(model, data, floor, geoms))


def _slerp(q0: np.ndarray, q1: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Batched quaternion slerp; w-first (MuJoCo) convention, shortest arc."""
    q0 = q0 / np.linalg.norm(q0, axis=-1, keepdims=True)
    q1 = q1 / np.linalg.norm(q1, axis=-1, keepdims=True)
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.abs(dot).clip(-1.0, 1.0)
    theta = np.arccos(dot)
    small = theta[..., 0] < 1e-6
    sin_theta = np.sin(theta)
    t = t[..., None]
    out = np.empty_like(q0)
    if np.any(~small):
        a = np.sin((1.0 - t) * theta) / np.where(sin_theta == 0, 1.0, sin_theta)
        b = np.sin(t * theta) / np.where(sin_theta == 0, 1.0, sin_theta)
        out[~small] = (a * q0 + b * q1)[~small]
    if np.any(small):
        out[small] = ((1.0 - t) * q0 + t * q1)[small]
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


def _free_joint_qpos_layout(model):
    """(root position slice, root quaternion slice) for the free joint."""
    assert int(model.jnt_type[0]) == mujoco.mjtJoint.mjJNT_FREE, \
        "expected a free root joint"
    adr = int(model.jnt_qposadr[0])
    return slice(adr, adr + 3), slice(adr + 3, adr + 7)


def _finite_difference_qvel(model, qpos: np.ndarray, frequency: float) -> np.ndarray:
    """qvel from consecutive qpos with `mj_differentiatePos` (quaternion-safe)."""
    n = qpos.shape[0]
    qvel = np.zeros((n, model.nv))
    dt = 1.0 / frequency
    for i in range(n - 1):
        mujoco.mj_differentiatePos(model, qvel[i], dt, qpos[i], qpos[i + 1])
    qvel[-1] = qvel[-2] if n > 1 else 0.0
    return qvel


def build_stand_still(model, template: Trajectory, frames: int, pose: str):
    if pose == "qpos0":
        base = np.array(model.qpos0, dtype=np.float64)
    elif pose == "frame0":
        base = np.array(template.data.qpos[0], dtype=np.float64)
    else:  # keyframe name
        key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, pose)
        if key < 0:
            raise ValueError(f"no keyframe named {pose!r}")
        base = np.array(model.key_qpos[key], dtype=np.float64)
    grounded, before, after = _ground_pose(model, base)
    qpos = np.repeat(grounded[None, :], frames, axis=0)
    qvel = np.zeros((frames, model.nv))
    stats = {"pose": pose, "clearance_before_m": before, "clearance_after_m": after,
             "root_height_m": float(grounded[2])}
    return _recompute(model, qpos, qvel, template), stats


def build_slow(model, template: Trajectory, factor: int, frames: int):
    src_qpos = np.asarray(template.data.qpos, dtype=np.float64)
    frequency = float(template.info.frequency)
    n_src = int(np.ceil(frames / factor)) + 1
    if n_src > src_qpos.shape[0]:
        raise ValueError(
            f"slow factor {factor} needs {n_src} source frames, clip has "
            f"{src_qpos.shape[0]}")
    src = src_qpos[:n_src]
    # Sample the source at 1/factor of real time; the pose path is identical,
    # the time parameterisation is stretched.
    u = np.arange(frames) / float(factor)
    i0 = np.floor(u).astype(int)
    i1 = np.minimum(i0 + 1, n_src - 1)
    t = (u - i0)
    qpos = (1.0 - t)[:, None] * src[i0] + t[:, None] * src[i1]
    pos_s, quat_s = _free_joint_qpos_layout(model)
    qpos[:, pos_s] = (1.0 - t)[:, None] * src[i0][:, pos_s] + t[:, None] * src[i1][:, pos_s]
    qpos[:, quat_s] = _slerp(src[i0][:, quat_s], src[i1][:, quat_s], t)
    qvel = _finite_difference_qvel(model, qpos, frequency)
    stats = {"factor": factor, "source_frames_used": int(n_src),
             "qvel_p95_before": float(np.percentile(np.abs(np.asarray(
                 template.data.qvel)), 95)),
             "qvel_p95_after": float(np.percentile(np.abs(qvel), 95))}
    return _recompute(model, qpos, qvel, template), stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robots", nargs="+", default=["h1", "g1"])
    parser.add_argument("--reference-root", type=Path,
                        default=WORKSPACE / "external_data" / "cross_humanoid")
    parser.add_argument("--source-clip", default="dance2_subject4")
    parser.add_argument("--source-start", type=int, default=19482)
    parser.add_argument("--source-frames", type=int, default=800)
    parser.add_argument("--frames", type=int, default=800)
    parser.add_argument("--kind", choices=["stand_still", "slow"], required=True)
    parser.add_argument("--pose", default="qpos0",
                        help="stand_still only: qpos0 | frame0 | <keyframe name>")
    parser.add_argument("--factor", type=int, default=2, help="slow only")
    parser.add_argument("--clip-name", default=None)
    args = parser.parse_args()

    clip = args.clip_name or (
        "stand_still" if args.kind == "stand_still" else f"dance_slow{args.factor}")
    report = {"clip": clip, "kind": args.kind, "robots": {}}
    for robot in args.robots:
        src = (args.reference_root / "h1_source" / args.source_clip / robot /
               f"start{args.source_start}_{args.source_frames}f_direct.npz")
        template = Trajectory.load(str(src))
        model = mujoco.MjModel.from_xml_path(str(HUMANOIDS[robot].xml_path))
        if args.kind == "stand_still":
            traj, stats = build_stand_still(model, template, args.frames, args.pose)
        else:
            traj, stats = build_slow(model, template, args.factor, args.frames)
        out = (args.reference_root / "h1_source" / clip / robot /
               f"start0_{args.frames}f_direct.npz")
        out.parent.mkdir(parents=True, exist_ok=True)
        traj.save(str(out))
        stats["path"] = str(out)
        stats["source"] = str(src)
        report["robots"][robot] = stats
        print(f"[control-ref] {robot} {clip} -> {out}", flush=True)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
