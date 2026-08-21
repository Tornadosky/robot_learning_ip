"""Does the box foot make H1 able to hold a pose? A/B, stock vs box.

`t02_stand_test.json` established that H1 topples in ~1-1.7 s under PD hold at
every gain from 1x to 30x, on its own `qpos0` and on the reference's frame 0.
`foot_contact_fix` argues the cause is upstream's two centre-line capsules,
whose 9 mm height offset leaves only a lateral toe roller bearing load.

This is the intervention test for that claim: identical model, identical pose,
identical controller, one thing changed. If H1 stands with the box foot and
falls without it, the contact geometry is the cause and not a correlate.
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

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from scaling.parallel_cross_humanoid_train import (  # noqa: E402
    _build_robot_env,
    _ensure_latent_defaults,
)


def build(robot: str, foot_model: str, reference_root: Path, clip_window: str):
    args = _ensure_latent_defaults(SimpleNamespace(
        source="h1", reference_mode="direct", reference_root=reference_root,
        clip=None, start_frame=None, frames=None, clip_windows=[clip_window],
        morphology=None, use_mjwarp=False, foot_model=foot_model,
        reward_type="MorphMimicReward", goal_type="GoalTrajMimic",
    ))
    return _build_robot_env(args, robot)[0]


def _cpu_model(env):
    for attr in ("_model", "model", "_mjmodel"):
        m = getattr(env, attr, None)
        if isinstance(m, mujoco.MjModel):
            return m
    raise AttributeError("no MjModel on env")


def foot_geometry(model) -> dict:
    """Which geoms can hit the floor, and what footprint do they present."""
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    pairs = []
    for p in range(model.npair):
        g1, g2 = int(model.pair_geom1[p]), int(model.pair_geom2[p])
        if floor in (g1, g2):
            other = g2 if g1 == floor else g1
            pairs.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other))
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    mujoco.mj_forward(model, data)
    lows = {}
    for name in pairs:
        g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        # lowest point of the geom's world AABB, adequate for box/capsule/sphere
        rbound = float(model.geom_rbound[g])
        lows[name] = float(data.geom_xpos[g][2] - rbound)
    return {"floor_pairs": pairs, "lowest_point_m": lows,
            "height_spread_m": (max(lows.values()) - min(lows.values())
                                if lows else None)}


def pd_hold(model, qpos0: np.ndarray, gain_scale: float, seconds: float,
            decimation: int = 5) -> dict:
    data = mujoco.MjData(model)
    data.qpos[:] = qpos0
    mujoco.mj_forward(model, data)
    # Same convention the stand test used: gains derived from the control
    # range, scaled. Actuators here are motors, so this is a torque servo.
    target = np.array(qpos0)
    jnt_of_act, dof_of_act, qpos_of_act = [], [], []
    for a in range(model.nu):
        j = int(model.actuator_trnid[a, 0])
        jnt_of_act.append(j)
        dof_of_act.append(int(model.jnt_dofadr[j]))
        qpos_of_act.append(int(model.jnt_qposadr[j]))
    ctrl_hi = np.abs(model.actuator_ctrlrange[:, 1])
    kp = 100.0 * gain_scale * np.ones(model.nu)
    kd = 5.0 * gain_scale * np.ones(model.nu)
    steps = int(seconds / (model.opt.timestep * decimation))
    z0 = float(data.qpos[2])
    heights, tilts = [], []
    for _ in range(steps):
        err = target[qpos_of_act] - data.qpos[qpos_of_act]
        derr = -data.qvel[dof_of_act]
        tau = kp * err + kd * derr
        data.ctrl[:] = np.clip(tau, -ctrl_hi, ctrl_hi)
        for _ in range(decimation):
            mujoco.mj_step(model, data)
        heights.append(float(data.qpos[2]))
        quat = data.qpos[3:7]
        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, quat)
        tilts.append(float(np.degrees(np.arccos(np.clip(mat[8], -1, 1)))))
    heights, tilts = np.array(heights), np.array(tilts)
    fallen = (heights < z0 - 0.30) | (tilts > 45.0)
    idx = int(np.argmax(fallen)) if fallen.any() else None
    dt = model.opt.timestep * decimation
    return {
        "gain_scale": gain_scale,
        "root_height_init_m": z0,
        "root_height_final_m": float(heights[-1]),
        "root_height_min_m": float(heights.min()),
        "max_tilt_deg": float(tilts.max()),
        "toppled": bool(fallen.any()),
        "time_to_topple_s": (idx * dt) if idx is not None else None,
        "seconds": seconds,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robots", nargs="+", default=["h1"])
    parser.add_argument("--gains", type=float, nargs="+",
                        default=[1.0, 3.0, 10.0, 30.0])
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--reference-root", type=Path,
                        default=WORKSPACE / "external_data" / "cross_humanoid")
    parser.add_argument("--clip-window", default="dance2_subject4:19482:800")
    parser.add_argument("--out", type=Path, default=WORKSPACE / "experiments" /
                        "failure_rootcause_20260817" / "metrics" /
                        "t03b_box_foot_ab.json")
    args = parser.parse_args()

    report = {"clip_window": args.clip_window, "robots": {}}
    for robot in args.robots:
        entry = {}
        for foot_model in ("stock", "box"):
            env = build(robot, foot_model, args.reference_root, args.clip_window)
            model = _cpu_model(env)
            geom = foot_geometry(model)
            qpos0 = np.array(model.qpos0)
            runs = [pd_hold(model, qpos0, g, args.seconds) for g in args.gains]
            entry[foot_model] = {"geometry": geom, "pd_hold": runs}
            print(f"[box-foot] {robot}/{foot_model} pairs={geom['floor_pairs']} "
                  f"spread={geom['height_spread_m']}", flush=True)
            for r in runs:
                print(f"    gain {r['gain_scale']:>5}: toppled={r['toppled']} "
                      f"t={r['time_to_topple_s']} final_z={r['root_height_final_m']:.3f}",
                      flush=True)
        report["robots"][robot] = entry
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"[box-foot] wrote {args.out}")


if __name__ == "__main__":
    main()
