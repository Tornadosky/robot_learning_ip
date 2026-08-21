"""C19 -- log MimicReward term by term along a trained rollout.

C18 measured that trained policies drift 0.67-2.75 m against a motion that
travels 0.66 m, and argued from the upstream source that the reward cannot see
it: rpos/rquat/rvel are relative to `upper_body_mimic` and therefore
translation-invariant, so root position survives only as 3 of 22 entries inside
`qpos`'s mean-square.

That was a reading of the code plus a consistency check against total return.
This measures it: each term, per step, along the actual rollout. The prediction
is that `qpos_reward` has collapsed while the relative terms stay high -- i.e.
the policy bought 1.0 of available reward by surrendering 0.4.

CPU only; safe alongside a trainer.
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

from c6_reward_discrimination import E, W_DANCE, build_model, catalog, reground  # noqa: E402
from c8_reference_feasibility import reference_qvel  # noqa: E402
from c9_shared_policy import MIMIC_SITES, make_complete_trajectory, register_variant  # noqa: E402
from c15b_policy_videos import rollout_qpos  # noqa: E402


def decompose(model, seq_qpos, ref_qpos, ref_qvel, traj_site_xpos, rel_site_ids):
    """Per-step MimicReward terms, using the upstream formulas."""
    from loco_mujoco.core.utils.math import calculate_relative_site_quatities

    body_ids = np.array([model.site_bodyid[s] for s in rel_site_ids])
    data = mujoco.MjData(model)
    ref_data = mujoco.MjData(model)

    quat_idx = np.array([3, 4, 5, 6])
    non_quat = np.setdiff1d(np.arange(model.nq), quat_idx)

    rows = []
    for i, q in enumerate(seq_qpos):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        ref_data.qpos[:] = ref_qpos[i]
        ref_data.qvel[:] = ref_qvel[i]
        mujoco.mj_forward(model, ref_data)

        p, a, _ = calculate_relative_site_quatities(data, rel_site_ids, body_ids, model.body_rootid, np)
        pr, ar, _ = calculate_relative_site_quatities(ref_data, rel_site_ids, body_ids, model.body_rootid, np)

        qpos_dist = float(np.mean((q[non_quat] - ref_qpos[i][non_quat]) ** 2))
        # the same distance with the root coordinates removed, to attribute it
        root_free = np.setdiff1d(non_quat, np.array([0, 1, 2]))
        qpos_dist_joints_only = float(np.mean((q[root_free] - ref_qpos[i][root_free]) ** 2))
        rpos_dist = float(np.mean((p - pr) ** 2))
        rang_dist = float(np.mean((a - ar) ** 2))

        rows.append({
            "qpos_reward": float(np.exp(-E["qpos_w_exp"] * qpos_dist)),
            "qpos_reward_joints_only": float(np.exp(-E["qpos_w_exp"] * qpos_dist_joints_only)),
            "rpos_reward": float(np.exp(-E["rpos_w_exp"] * rpos_dist)),
            "rquat_reward": float(np.exp(-E["rquat_w_exp"] * rang_dist)),
            "root_xy_err_m": float(np.linalg.norm(q[:2] - ref_qpos[i][:2])),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--body", default="body04_seed1004")
    ap.add_argument("--arms", nargs="+", default=["fk", "ik_scaled", "shared_nominal"])
    ap.add_argument("--start", type=int, default=19482)
    ap.add_argument("--frames", type=int, default=800)
    ap.add_argument("--xml-root", type=Path,
                    default=WORKSPACE / "experiments" / "h1_morphology_deepmimic_20260808" / "body_catalog" / "xml")
    ap.add_argument("--checkpoint-root", type=Path,
                    default=WORKSPACE / "experiments" / "h1_morphology_deepmimic_20260808" / "checkpoints")
    args = ap.parse_args()

    import jax.numpy as jnp
    from c3_reference_methods import IKKernel, LIMB_SCALE_INDEX
    from loco_mujoco.algorithms import PPOJax
    from loco_mujoco.task_factories import CustomDatasetConf, ImitationFactory, LAFAN1DatasetConf

    src = ImitationFactory.make("UnitreeH1", lafan1_dataset_conf=LAFAN1DatasetConf(["dance2_subject4"]))
    th = src.th.traj
    freq = float(th.info.frequency)
    dt = 1.0 / freq
    qpos = np.asarray(th.data.qpos)[args.start:args.start + args.frames].astype(np.float64)
    qvel = np.asarray(th.data.qvel)[args.start:args.start + args.frames].astype(np.float64)

    bodies = dict(catalog())
    model = build_model(args.body, bodies[args.body], args.xml_root)
    nominal = build_model("body00_nominal", bodies["body00_nominal"], args.xml_root)
    xml_path = args.xml_root / f"h1_morphology_c2_{args.body}" / "h1.xml"
    kern = IKKernel(args.xml_root / "h1_morphology_c2_body00_nominal" / "h1.xml")

    def reference_for(arm):
        if arm == "fk":
            return reground(model, qpos)[0]
        if arm == "shared_nominal":
            return reground(nominal, qpos)[0]
        mo = bodies[args.body]
        m = jnp.asarray(np.array([[mo["leg_length_scale"], mo["arm_length_scale"],
                                   mo["shoulder_width_scale"]]], dtype=np.float32))
        Q = jnp.asarray(qpos.astype(np.float32))
        nom = np.asarray(kern.fk_bt(jnp.ones((1, 3), jnp.float32), Q))[0]
        nt = jnp.asarray(nom[:, kern.TARGET_SITES, :])
        rn = jnp.asarray(qpos[:, 0:3].astype(np.float32))
        rel = nt - rn[:, None, :]
        s = m[0][LIMB_SCALE_INDEX][None, :, None]
        root = rn.at[:, 2].multiply(m[0][0])
        xs, _ = kern.ik(m, Q, (root[:, None, :] + rel * s)[None])
        xs = np.asarray(xs)[0]
        q = qpos.copy()
        q[:, kern.ACT_QADR] = np.clip(xs[:, : kern.NU], kern.jnt_low, kern.jnt_high)
        q[:, 0:3] += xs[:, kern.NU: kern.NU + 3]
        return q

    rel_site_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n) for n in MIMIC_SITES])
    results = {}
    for arm in args.arms:
        ckpt = args.checkpoint_root / f"{args.body}__{arm}" / "PPOJax_saved.pkl"
        if not ckpt.exists():
            print(f"skip {arm}: no checkpoint")
            continue
        q_ref = reference_for(arm)
        v_ref = reference_qvel(qvel, q_ref, dt)
        env_name = f"C19_{arm}"
        register_variant(env_name, xml_path, mjx=False)
        traj = make_complete_trajectory(env_name, q_ref, v_ref, freq, th)
        env = ImitationFactory.make(
            env_name, custom_dataset_conf=CustomDatasetConf(traj), headless=True, horizon=1000,
            th_params=dict(random_start=False, fixed_start_conf=(0, 0)),
            goal_type="GoalTrajMimic", goal_params=dict(visualize_goal=False),
            reward_type="MimicReward", reward_params=dict(**W_DANCE, sites_for_mimic=MIMIC_SITES))
        agent_conf, agent_state = PPOJax.load_agent(ckpt)
        seq = rollout_qpos(env, agent_conf, agent_state, args.frames)

        traj_qpos = np.asarray(traj.data.qpos)[: len(seq)]
        traj_qvel = np.asarray(traj.data.qvel)[: len(seq)]
        rows = decompose(model, seq, traj_qpos, traj_qvel,
                         np.asarray(traj.data.site_xpos), rel_site_ids)

        mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        contrib = {
            "qpos": W_DANCE["qpos_w_sum"] * mean["qpos_reward"],
            "rpos": W_DANCE["rpos_w_sum"] * mean["rpos_reward"],
            "rquat": W_DANCE["rquat_w_sum"] * mean["rquat_reward"],
        }
        results[arm] = {"steps": len(seq), "mean_terms": mean, "weighted_contribution": contrib,
                        "max_weighted": {"qpos": W_DANCE["qpos_w_sum"], "rpos": W_DANCE["rpos_w_sum"],
                                         "rquat": W_DANCE["rquat_w_sum"]}}
        print(f"{arm:15s} steps {len(seq):3d} | qpos {mean['qpos_reward']:.4f} "
              f"(joints-only {mean['qpos_reward_joints_only']:.4f}) | "
              f"rpos {mean['rpos_reward']:.4f} | rquat {mean['rquat_reward']:.4f} | "
              f"root err {mean['root_xy_err_m']:.2f} m | "
              f"earned {contrib['qpos']:.3f}/{W_DANCE['qpos_w_sum']} qpos, "
              f"{contrib['rpos'] + contrib['rquat']:.3f}/{W_DANCE['rpos_w_sum'] + W_DANCE['rquat_w_sum']} spatial")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "component": "C19_reward_decomposition",
        "prediction": "qpos_reward collapses (it is the only term containing root position) while "
                      "the upper-body-relative terms stay high",
        "body": args.body,
        "results": results,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
