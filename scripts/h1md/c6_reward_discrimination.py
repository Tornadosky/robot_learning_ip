"""C6 -- can ``MimicReward`` even tell a body-specific reference from a shared one?

C2 measured that morphology moves the five mimic sites by 11-104 mm. This script
converts centimetres into *reward*, which is the only unit that decides whether
per-body retargeting is worth building.

Setup, per catalog body ``b``:

* the robot is body ``b`` and tracks **its own** reference perfectly;
* reference A = "shared nominal": site quantities computed on the *nominal* H1;
* reference B = "body-specific": same joint angles applied to body ``b``, with
  the root height re-grounded per frame from ``b``'s own forward kinematics.

Because both references carry identical joint angles, the qpos/qvel terms are
almost unchanged and the comparison isolates exactly the spatial terms that
per-body retargeting exists to fix. The script reports the reward each choice
earns, using the real upstream term formulas, plus a sensitivity curve mapping
site error in cm to reward.

It also records which reward weights the repository's own trainers actually use,
since a zeroed weight makes the whole question moot.

Run under WSL dance_env (needs the LAFAN1 dataset).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE / "scripts"))

from h1_morphology_variants import H1MorphologyPreset, create_h1_variant_xml  # noqa: E402
from loco_mujoco.core.utils.math import (  # noqa: E402
    calculate_relative_site_quatities,
    quaternion_angular_distance,
)

MIMIC_SITES = (
    "upper_body_mimic",
    "left_hand_mimic",
    "left_foot_mimic",
    "right_hand_mimic",
    "right_foot_mimic",
)

# Weights from scripts/train_deepmimic_dance.py (the stock DeepMimic arm).
W_DANCE = dict(qpos_w_sum=0.4, qvel_w_sum=0.2, rpos_w_sum=0.5, rquat_w_sum=0.3, rvel_w_sum=0.1)
# Weights from scripts/scaling/online_h1_train.py (the multi-body morphology arm).
W_SCALING = dict(qpos_w_sum=0.6, qvel_w_sum=0.4, rpos_w_sum=0.0, rquat_w_sum=0.0, rvel_w_sum=0.0)
# Upstream exponential sharpnesses (MimicReward defaults).
E = dict(qpos_w_exp=10.0, qvel_w_exp=2.0, rpos_w_exp=100.0, rquat_w_exp=10.0, rvel_w_exp=0.1)

DYNAMIC_DIMS = ("leg_length_scale", "arm_length_scale", "shoulder_width_scale", "torso_mass_scale")


def catalog(seed_base: int = 1000, n: int = 4):
    low = np.array([0.85, 0.85, 0.85, 0.70])
    high = np.array([1.20, 1.20, 1.20, 1.50])
    out = [("body00_nominal", dict(zip(DYNAMIC_DIMS, [1.0] * 4)))]
    for i in range(1, n + 1):
        vals = np.random.default_rng(seed_base + i).uniform(low, high)
        out.append((f"body{i:02d}_seed{seed_base + i}", dict(zip(DYNAMIC_DIMS, vals.tolist()))))
    return out


def build_model(name: str, morph: dict, root: Path) -> mujoco.MjModel:
    preset = H1MorphologyPreset(
        name=f"c2_{name}", label=name,
        leg_length_scale=morph["leg_length_scale"],
        arm_length_scale=morph["arm_length_scale"],
        shoulder_width_scale=morph["shoulder_width_scale"],
        torso_mass_scale=morph["torso_mass_scale"],
    )
    return mujoco.MjModel.from_xml_path(str(create_h1_variant_xml(preset, output_root=root)))


def reground(model, qpos_batch: np.ndarray, clearance: float = 0.002) -> tuple[np.ndarray, dict]:
    """Per-frame root-height correction from the body's own FK (cheap, see Finding 6)."""
    data = mujoco.MjData(model)
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    feet = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("left_foot", "right_foot")]
    out = qpos_batch.copy()
    shifts = np.zeros(len(qpos_batch))
    for i, q in enumerate(qpos_batch):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        d = min(mujoco.mj_geomDistance(model, data, floor, f, 10.0, None) for f in feet)
        shifts[i] = clearance - d
        out[i, 2] += shifts[i]
    return out, {"mean_m": float(shifts.mean()), "max_abs_m": float(np.abs(shifts).max()),
                 "rms_m": float(np.sqrt((shifts ** 2).mean()))}


def site_quantities(model, qpos_batch: np.ndarray, qvel_batch: np.ndarray):
    """Reproduce exactly what MimicReward compares: upper-body-relative site pos/angles/vel."""
    data = mujoco.MjData(model)
    site_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n) for n in MIMIC_SITES])
    body_ids = np.array([model.site_bodyid[s] for s in site_ids])
    rpos, rang, rvel = [], [], []
    for q, v in zip(qpos_batch, qvel_batch):
        data.qpos[:] = q
        data.qvel[:] = v
        mujoco.mj_forward(model, data)
        p, a, vv = calculate_relative_site_quatities(data, site_ids, body_ids, model.body_rootid, np)
        rpos.append(p); rang.append(a); rvel.append(vv)
    return np.array(rpos), np.array(rang), np.array(rvel)


def terms(qpos_d, qvel_d, rpos_d, rang_d, rvel_rot_d, rvel_lin_d) -> dict:
    return {
        "qpos_reward": float(np.exp(-E["qpos_w_exp"] * qpos_d)),
        "qvel_reward": float(np.exp(-E["qvel_w_exp"] * qvel_d)),
        "rpos_reward": float(np.exp(-E["rpos_w_exp"] * rpos_d)),
        "rquat_reward": float(np.exp(-E["rquat_w_exp"] * rang_d)),
        "rvel_rot_reward": float(np.exp(-E["rvel_w_exp"] * rvel_rot_d)),
        "rvel_lin_reward": float(np.exp(-E["rvel_w_exp"] * rvel_lin_d)),
    }


def total(t: dict, w: dict) -> float:
    return (w["qpos_w_sum"] * t["qpos_reward"] + w["qvel_w_sum"] * t["qvel_reward"]
            + w["rpos_w_sum"] * t["rpos_reward"] + w["rquat_w_sum"] * t["rquat_reward"]
            + w["rvel_w_sum"] * (t["rvel_rot_reward"] + t["rvel_lin_reward"]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--start", type=int, default=19482, help="frame index at 100 Hz env rate")
    ap.add_argument("--frames", type=int, default=800)
    ap.add_argument("--stride", type=int, default=4, help="subsample the window for speed")
    ap.add_argument("--xml-root", type=Path,
                    default=WORKSPACE / "experiments" / "h1_morphology_deepmimic_20260808" / "body_catalog" / "xml")
    args = ap.parse_args()

    from loco_mujoco.task_factories import ImitationFactory, LAFAN1DatasetConf

    env = ImitationFactory.make("UnitreeH1", lafan1_dataset_conf=LAFAN1DatasetConf(["dance2_subject4"]))
    th = env.th.traj
    qpos = np.asarray(th.data.qpos)[args.start:args.start + args.frames][:: args.stride]
    qvel = np.asarray(th.data.qvel)[args.start:args.start + args.frames][:: args.stride]
    print(f"window {args.start}:{args.start + args.frames} @ {float(th.info.frequency)} Hz, "
          f"using {len(qpos)} frames (stride {args.stride})")

    bodies = catalog()
    nominal = build_model(*bodies[0], args.xml_root)
    qpos_nom, ground_nom = reground(nominal, qpos)
    rpos_nom, rang_nom, rvel_nom = site_quantities(nominal, qpos_nom, qvel)

    records = []
    for name, morph in bodies:
        model = build_model(name, morph, args.xml_root)
        qpos_b, ground_b = reground(model, qpos)
        rpos_b, rang_b, rvel_b = site_quantities(model, qpos_b, qvel)

        # The robot IS body b tracking its own reference perfectly.
        # Against reference B the spatial error is zero by construction.
        # Against reference A (shared nominal) it is the difference below.
        d_rpos = float(np.mean((rpos_b - rpos_nom) ** 2))
        d_rang = float(np.mean((rang_b - rang_nom) ** 2))
        d_rvrot = float(np.mean((rvel_b[:, :, :3] - rvel_nom[:, :, :3]) ** 2))
        d_rvlin = float(np.mean((rvel_b[:, :, 3:] - rvel_nom[:, :, 3:]) ** 2))
        # joint angles identical; only the re-grounded root height differs
        d_qpos = float(np.mean((qpos_b[:, :7][:, ~np.isin(np.arange(7), [3, 4, 5, 6])]
                                - qpos_nom[:, :7][:, ~np.isin(np.arange(7), [3, 4, 5, 6])]) ** 2)
                       * 3 / qpos.shape[1])

        t_own = terms(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        t_shared = terms(d_qpos, 0.0, d_rpos, d_rang, d_rvrot, d_rvlin)

        site_err = np.linalg.norm(rpos_b - rpos_nom, axis=-1)  # (frames, 4 sites)
        rel_names = [n for n in MIMIC_SITES if n != "upper_body_mimic"]

        records.append({
            "body_id": name,
            "morphology": morph,
            "regrounding": {"nominal": ground_nom, "body": ground_b},
            "site_error_vs_nominal_cm": {
                "rms": float(np.sqrt((site_err ** 2).mean()) * 100),
                "max": float(site_err.max() * 100),
                "per_site_rms": {n: float(np.sqrt((site_err[:, i] ** 2).mean()) * 100)
                                 for i, n in enumerate(rel_names)},
            },
            "reward_terms_own_reference": t_own,
            "reward_terms_shared_nominal_reference": t_shared,
            "total_dance_weights": {
                "own": total(t_own, W_DANCE), "shared": total(t_shared, W_DANCE),
                "loss": total(t_own, W_DANCE) - total(t_shared, W_DANCE),
                "loss_fraction": (total(t_own, W_DANCE) - total(t_shared, W_DANCE)) / max(total(t_own, W_DANCE), 1e-9),
            },
            "total_scaling_weights": {
                "own": total(t_own, W_SCALING), "shared": total(t_shared, W_SCALING),
                "loss": total(t_own, W_SCALING) - total(t_shared, W_SCALING),
            },
        })
        r = records[-1]
        print(f"{name}: site rms {r['site_error_vs_nominal_cm']['rms']:.2f} cm "
              f"max {r['site_error_vs_nominal_cm']['max']:.2f} cm | "
              f"rpos_reward {t_shared['rpos_reward']:.4f} | "
              f"dance total {r['total_dance_weights']['shared']:.4f} vs {r['total_dance_weights']['own']:.4f} "
              f"({r['total_dance_weights']['loss_fraction'] * 100:.2f}% loss) | "
              f"scaling total loss {r['total_scaling_weights']['loss']:.6f}")

    # Sensitivity curve: uniform site error of magnitude e on all 4 relative sites.
    curve = []
    for cm in (0.5, 1, 2, 3, 5, 7.5, 10, 15, 20, 30):
        e = cm / 100.0
        d = e ** 2 / 3.0  # isotropic error, mean over 3 coords
        curve.append({
            "site_error_cm": cm,
            "rpos_reward": float(np.exp(-E["rpos_w_exp"] * d)),
            "rpos_contribution_dance": W_DANCE["rpos_w_sum"] * float(np.exp(-E["rpos_w_exp"] * d)),
            "fraction_of_max_total_dance": (
                W_DANCE["rpos_w_sum"] * float(np.exp(-E["rpos_w_exp"] * d))
                + sum(W_DANCE.values()) - W_DANCE["rpos_w_sum"] + W_DANCE["rvel_w_sum"]
            ) / (sum(W_DANCE.values()) + W_DANCE["rvel_w_sum"]),
        })

    out = {
        "component": "C6_reward_discrimination",
        "question": "does MimicReward distinguish a body-specific reference from the shared nominal one?",
        "window": {"start_frame_100hz": args.start, "frames": args.frames, "stride": args.stride,
                   "frames_used": int(len(qpos))},
        "reward_weights": {"train_deepmimic_dance": W_DANCE, "scaling_online_h1_train": W_SCALING,
                           "exponents": E},
        "reference_construction": (
            "same joint angles on every body, root height re-grounded per frame from that "
            "body's own FK; this is the Phase-8 'body-normalized task-space' baseline, not an IK retarget"
        ),
        "bodies": records,
        "sensitivity_curve": curve,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
