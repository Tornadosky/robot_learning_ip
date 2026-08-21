"""Is the reference expressed in the same joint frame the trainer simulates?

The references are generated on loco-mujoco's **CPU** env
(``cross_humanoid_retarget._direct_reference`` -> ``ImitationFactory.make(
spec.cpu_env_name, ...)``) and consumed by the **MJX** env
(``spec.mjx_env_name``). Those are different classes with different
``_modify_spec_for_mjx`` treatments, and this repository has a recorded bug of
exactly that shape: same joint names, negated axes, on 13 of H1's 19 and **15 of
G1's 23** joints, plus a G1 mirroring bug in the urma2 loader that was found and
fixed once already.

If a joint's axis is negated between the two models, the reference commands the
mirror image of the intended motion on that joint. A policy chasing it would
score *worse than emitting zeros* — which is exactly what G1 does (return 4.97
vs zero action's 13.66 at 300M) while H1 is 7x better than zero.

This compares, per actuated joint, between the CPU and MJX models of the same
family: joint id order, axis direction, range, and the actuator's gear sign.
Nothing is inferred from names alone.
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

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from loco_mujoco.environments.base import LocoEnv  # noqa: E402

from scaling.cross_humanoid_retarget import HUMANOIDS  # noqa: E402


def _model_of(env_name: str):
    cls = LocoEnv.registered_envs[env_name]
    env = cls()
    for attr in ("_model", "model"):
        m = getattr(env, attr, None)
        if isinstance(m, mujoco.MjModel):
            return m, env
    raise AttributeError(f"no MjModel on {env_name}")


def _joint_table(model) -> dict:
    out = {}
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if name is None:
            continue
        out[name] = {
            "id": j,
            "type": int(model.jnt_type[j]),
            "axis": [round(float(x), 6) for x in model.jnt_axis[j]],
            "range": [round(float(x), 6) for x in model.jnt_range[j]],
            "qposadr": int(model.jnt_qposadr[j]),
            "limited": bool(model.jnt_limited[j]),
        }
    for a in range(model.nu):
        j = int(model.actuator_trnid[a, 0])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if name in out:
            out[name]["actuator_index"] = a
            out[name]["gear0"] = round(float(model.actuator_gear[a, 0]), 6)
            out[name]["ctrlrange"] = [round(float(x), 6)
                                      for x in model.actuator_ctrlrange[a]]
    return out


def compare(family: str) -> dict:
    spec = HUMANOIDS[family]
    cpu_model, _ = _model_of(spec.cpu_env_name)
    mjx_model, _ = _model_of(spec.mjx_env_name)
    cpu, mjx = _joint_table(cpu_model), _joint_table(mjx_model)

    shared = [n for n in cpu if n in mjx]
    only_cpu = sorted(set(cpu) - set(mjx))
    only_mjx = sorted(set(mjx) - set(cpu))

    flipped, range_diff, order_diff, gear_flipped = [], [], [], []
    for n in shared:
        a, b = cpu[n], mjx[n]
        ax_a, ax_b = np.array(a["axis"]), np.array(b["axis"])
        if np.linalg.norm(ax_a) > 0 and np.linalg.norm(ax_b) > 0:
            dot = float(ax_a @ ax_b)
            if dot < -1e-6:
                flipped.append({"joint": n, "cpu_axis": a["axis"],
                                "mjx_axis": b["axis"], "dot": dot})
        if a["range"] != b["range"]:
            range_diff.append({"joint": n, "cpu": a["range"], "mjx": b["range"]})
        if a["qposadr"] != b["qposadr"]:
            order_diff.append({"joint": n, "cpu_qposadr": a["qposadr"],
                               "mjx_qposadr": b["qposadr"]})
        if ("gear0" in a and "gear0" in b
                and np.sign(a["gear0"]) != np.sign(b["gear0"])):
            gear_flipped.append({"joint": n, "cpu": a["gear0"], "mjx": b["gear0"]})

    # actuator ORDER, which is what the per-joint action head writes into
    def act_order(model):
        return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT,
                                  int(model.actuator_trnid[a, 0]))
                for a in range(model.nu)]

    cpu_acts, mjx_acts = act_order(cpu_model), act_order(mjx_model)

    return {
        "family": family,
        "cpu_env": spec.cpu_env_name,
        "mjx_env": spec.mjx_env_name,
        "n_joints_cpu": len(cpu),
        "n_joints_mjx": len(mjx),
        "joints_only_in_cpu": only_cpu,
        "joints_only_in_mjx": only_mjx,
        "axis_flipped_count": len(flipped),
        "axis_flipped": flipped,
        "range_differs": range_diff,
        "qposadr_differs": order_diff,
        "actuator_gear_sign_differs": gear_flipped,
        "actuator_order_matches": cpu_acts == mjx_acts,
        "cpu_actuator_order": cpu_acts,
        "mjx_actuator_order": mjx_acts,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--families", nargs="+", default=["h1", "g1"])
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    report = {}
    for family in args.families:
        r = compare(family)
        report[family] = r
        print(f"[axes] {family}: cpu={r['cpu_env']} mjx={r['mjx_env']} "
              f"joints {r['n_joints_cpu']}/{r['n_joints_mjx']} "
              f"axis_flipped={r['axis_flipped_count']} "
              f"range_diff={len(r['range_differs'])} "
              f"qposadr_diff={len(r['qposadr_differs'])} "
              f"gear_sign_diff={len(r['actuator_gear_sign_differs'])} "
              f"actuator_order_matches={r['actuator_order_matches']}", flush=True)
        for f in r["axis_flipped"][:8]:
            print(f"    FLIP {f['joint']}: cpu {f['cpu_axis']} vs mjx {f['mjx_axis']}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"[axes] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
