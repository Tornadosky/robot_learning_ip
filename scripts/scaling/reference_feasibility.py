"""Is the reference motion physically executable by this robot at all?

Before blaming a policy for not tracking a clip, check that the clip is inside
the robot's actuation limits. Inverse dynamics answers that directly: put the
robot in the reference state, ask what generalized force would be needed to
produce the reference acceleration, and compare it against the actuator's own
force range.

Method, per frame:

1. ``qpos``/``qvel`` from the reference; ``qacc`` by central difference of
   ``qvel`` at the trajectory's own frequency.
2. ``mj_forward`` first, so contacts and constraint forces exist -- without it
   the required torque is attributed entirely to the actuators and every
   stance frame looks impossible.
3. ``mj_inverse`` -> ``qfrc_inverse``, compared per actuated joint against
   ``actuator_forcerange`` (rescaled by the sampled body's strength coordinate).

The numbers are indicative rather than exact: contact forces come from the
reference pose rather than from a solved contact schedule, so stance frames
carry the larger error. A joint that needs several times its limit across a
large share of the clip is nonetheless not a tuning problem.
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


def _env(robot: str, reference_root: Path, clip_window: str):
    args = _ensure_latent_defaults(SimpleNamespace(
        source="h1", reference_mode="direct", reference_root=reference_root,
        clip=None, start_frame=None, frames=None, clip_windows=[clip_window],
        morphology="continuous", use_mjwarp=False,
        reward_type="MorphMimicReward", goal_type="GoalTrajMimic",
    ))
    return _build_robot_env(args, robot)[0]


def analyse(env, model, robot: str, label: str, stride: int,
            height_offset: float = 0.0) -> dict:
    traj = env.th.traj.data
    frequency = float(env.th.traj.info.frequency)
    n = int(np.asarray(traj.split_points)[1])

    actuated = []
    for a in range(model.nu):
        joint = int(model.actuator_trnid[a, 0])
        actuated.append((a, int(model.jnt_dofadr[joint]), joint))
    # These models leave actuator_forcerange at [0, 0] with forcelimited=False,
    # so the real authority is the CONTROL range: for a motor actuator the
    # applied torque is gear * gainprm[0] * ctrl, and ctrl is clipped to
    # ctrlrange. Reading forcerange here would report every joint as unlimited.
    limits = []
    for a, _, _ in actuated:
        force = max(abs(model.actuator_forcerange[a]))
        if not model.actuator_forcelimited[a] or force <= 0:
            force = (abs(model.actuator_gear[a, 0])
                     * abs(model.actuator_gainprm[a, 0])
                     * max(abs(model.actuator_ctrlrange[a])))
        limits.append(force if force > 0 else np.inf)
    limits = np.array(limits)
    dofs = np.array([d for _, d, _ in actuated])
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
             for _, _, j in actuated]

    data = mujoco.MjData(model)
    required = []
    contact_depth = []
    frames = list(range(1, n - 1, stride))
    for t in frames:
        prev = np.asarray(traj.get(0, t - 1, jnp).qvel)
        nxt = np.asarray(traj.get(0, t + 1, jnp).qvel)
        sample = traj.get(0, t, jnp)
        data.qpos[:] = np.asarray(sample.qpos)
        # The reset raises a longer-legged body by exactly this much; without it
        # the reference pose drives the feet through the floor and mj_inverse
        # reports the contact solver's reaction as required actuator torque
        # (11,206 Nm on an H1 hip, against a 260 Nm limit).
        data.qpos[2] += height_offset
        data.qvel[:] = np.asarray(sample.qvel)
        mujoco.mj_forward(model, data)
        if data.ncon:
            contact_depth.append(float(-np.min(data.contact.dist[:data.ncon])))
        data.qacc[:] = (nxt - prev) * frequency / 2.0
        mujoco.mj_inverse(model, data)
        required.append(np.abs(data.qfrc_inverse[dofs]))

    required = np.stack(required)
    ratio = required / limits[None, :]
    return {
        "robot": robot,
        "body": label,
        "frames_analysed": len(frames),
        "trajectory_frequency_hz": frequency,
        "actuated_joints": len(actuated),
        "torque_ratio_p50": float(np.median(ratio)),
        "torque_ratio_p95": float(np.percentile(ratio, 95)),
        "torque_ratio_max": float(ratio.max()),
        "frames_with_any_joint_over_limit_pct": float(
            100.0 * np.mean(ratio.max(axis=1) > 1.0)),
        "joint_frame_pairs_over_limit_pct": float(100.0 * np.mean(ratio > 1.0)),
        "worst_joints": [
            {"joint": names[j], "limit_Nm": float(limits[j]),
             "p95_required_Nm": float(np.percentile(required[:, j], 95)),
             "ratio_p95": float(np.percentile(ratio[:, j], 95))}
            for j in np.argsort(-np.percentile(ratio, 95, axis=0))[:5]
        ],
        "max_reference_penetration_m": (
            float(np.max(contact_depth)) if contact_depth else None),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robots", nargs="+", default=["h1", "g1"])
    parser.add_argument("--clip-window", default="dance2_subject4:19482:800")
    parser.add_argument(
        "--compare-window", default=None,
        help="A second clip to contrast against, e.g. walk1_subject1:10521:800")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument(
        "--reference-root", type=Path,
        default=WORKSPACE / "external_data" / "cross_humanoid")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    windows = [args.clip_window] + (
        [args.compare_window] if args.compare_window else [])
    results = []
    for window in windows:
        for robot in args.robots:
            try:
                env = _env(robot, args.reference_root, window)
            except FileNotFoundError as exc:
                print(f"[feas] skipping {robot} {window}: {exc}")
                continue
            bodies = {
                "nominal": np.ones(4, dtype=np.float32),
                "low_corner": FAMILY_MORPHOLOGY_LOW.astype(np.float32),
                "high_corner": FAMILY_MORPHOLOGY_HIGH.astype(np.float32),
            }
            for label, morphology in bodies.items():
                model = (env._model if label == "nominal"
                         else cpu_morphology_model(env._model, robot, morphology))
                offset = float(np.asarray(
                    env.root_height_offset(jnp.asarray(morphology))))
                entry = analyse(env, model, robot, label, args.stride, offset)
                entry["root_height_offset_m"] = offset
                entry["clip_window"] = window
                results.append(entry)
                print(f"[feas] {window.split(':')[0]:<16} {robot:<3} {label:<12} "
                      f"median {entry['torque_ratio_p50']:.2f}x limit, "
                      f"p95 {entry['torque_ratio_p95']:.2f}x, "
                      f"{entry['frames_with_any_joint_over_limit_pct']:.0f}% of "
                      f"frames need more torque than the robot has")
            del env

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"cells": results}, indent=2),
                           encoding="utf-8")
    print(f"[feas] -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
