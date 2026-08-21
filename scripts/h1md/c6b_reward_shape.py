"""C6b -- is the site reward term redundant with the joint term, or body-specific?

C6 built each body's reference by applying the nominal joint angles to that body
and re-grounding the root. Under that construction the sites are a deterministic
function of the joints, so perfect joint tracking implies perfect site tracking
and the spatial terms add no *new optimum*. That does not settle whether they
add useful **shape**: a given joint error may translate into very different site
error depending on limb lengths, in which case the spatial terms still carry
body-specific gradient information that a shared nominal reference would get
wrong.

This script perturbs the joint angles by a controlled magnitude and measures,
per body:

* the site error the same joint error produces (the amplification factor);
* the resulting reward under the body's own reference vs the shared nominal one.

If the amplification factor is body-dependent, per-body spatial references
matter even when the reference is only an FK construction. If it is not, the
joint terms alone are sufficient and the whole spatial arm can be dropped.

Run under WSL dance_env.
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
sys.path.insert(0, str(WORKSPACE / "scripts" / "h1md"))

from c6_reward_discrimination import (  # noqa: E402
    E, W_DANCE, build_model, catalog, reground, site_quantities, terms, total,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--start", type=int, default=19482)
    ap.add_argument("--frames", type=int, default=800)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--xml-root", type=Path,
                    default=WORKSPACE / "experiments" / "h1_morphology_deepmimic_20260808" / "body_catalog" / "xml")
    args = ap.parse_args()

    from loco_mujoco.task_factories import ImitationFactory, LAFAN1DatasetConf

    env = ImitationFactory.make("UnitreeH1", lafan1_dataset_conf=LAFAN1DatasetConf(["dance2_subject4"]))
    th = env.th.traj
    qpos = np.asarray(th.data.qpos)[args.start:args.start + args.frames][:: args.stride]
    qvel = np.asarray(th.data.qvel)[args.start:args.start + args.frames][:: args.stride]
    n_joint = qpos.shape[1] - 7
    print(f"{len(qpos)} frames, {n_joint} actuated-joint qpos entries")

    bodies = catalog()
    nominal = build_model(*bodies[0], args.xml_root)
    qpos_nom, _ = reground(nominal, qpos)
    rpos_nom_ref, _, _ = site_quantities(nominal, qpos_nom, qvel)

    rng = np.random.default_rng(args.seed)
    # one fixed unit-norm joint perturbation direction, reused for every body so
    # the comparison is about the body, not about the noise draw
    direction = rng.normal(size=(len(qpos), n_joint))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)

    deltas_rad = [0.0, 0.02, 0.05, 0.10, 0.20, 0.40]
    records = []
    for name, morph in bodies:
        model = build_model(name, morph, args.xml_root)
        qpos_b, _ = reground(model, qpos)
        rpos_ref, _, _ = site_quantities(model, qpos_b, qvel)

        rows = []
        for d in deltas_rad:
            perturbed = qpos_b.copy()
            perturbed[:, 7:] += direction * d
            perturbed, _ = reground(model, perturbed)
            rpos_p, _, _ = site_quantities(model, perturbed, qvel)

            # joint distance exactly as MimicReward computes it (non-quat entries)
            qpos_dist = float(np.mean((perturbed[:, 7:] - qpos_b[:, 7:]) ** 2))
            d_own = float(np.mean((rpos_p - rpos_ref) ** 2))
            d_shared = float(np.mean((rpos_p - rpos_nom_ref) ** 2))
            site_err_cm = float(np.sqrt((np.linalg.norm(rpos_p - rpos_ref, axis=-1) ** 2).mean()) * 100)

            rows.append({
                "joint_perturbation_rad": d,
                "qpos_dist": qpos_dist,
                "qpos_reward": float(np.exp(-E["qpos_w_exp"] * qpos_dist)),
                "site_error_cm_vs_own_ref": site_err_cm,
                "amplification_cm_per_rad": site_err_cm / d if d > 0 else 0.0,
                "rpos_reward_own_ref": float(np.exp(-E["rpos_w_exp"] * d_own)),
                "rpos_reward_shared_nominal_ref": float(np.exp(-E["rpos_w_exp"] * d_shared)),
                "total_dance_own_ref": total(
                    terms(qpos_dist, 0.0, d_own, 0.0, 0.0, 0.0), W_DANCE),
                "total_dance_shared_ref": total(
                    terms(qpos_dist, 0.0, d_shared, 0.0, 0.0, 0.0), W_DANCE),
            })

        amps = [r["amplification_cm_per_rad"] for r in rows if r["joint_perturbation_rad"] > 0]
        records.append({
            "body_id": name, "morphology": morph,
            "rows": rows,
            "mean_amplification_cm_per_rad": float(np.mean(amps)),
        })
        print(f"{name}: amplification {np.mean(amps):6.2f} cm/rad | "
              f"@0.10 rad site err {rows[3]['site_error_cm_vs_own_ref']:5.2f} cm, "
              f"rpos own {rows[3]['rpos_reward_own_ref']:.3f} vs shared {rows[3]['rpos_reward_shared_nominal_ref']:.3f}")

    base = records[0]["mean_amplification_cm_per_rad"]
    spread = [r["mean_amplification_cm_per_rad"] / base for r in records]
    summary = {
        "amplification_relative_to_nominal": {r["body_id"]: s for r, s in zip(records, spread)},
        "max_relative_spread": float(max(spread) - min(spread)),
    }
    print(f"\namplification spread across bodies: {summary['max_relative_spread'] * 100:.1f}% "
          f"of the nominal value")

    out = {
        "component": "C6b_reward_shape",
        "question": "is the site term redundant with the joint term under an FK-constructed reference?",
        "joint_perturbation_direction": "fixed unit-norm gaussian per frame, shared across bodies",
        "bodies": records,
        "summary": summary,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
