"""C10 -- score every trained arm against ONE criterion, not against its own teacher.

C9 compared arms by training return, which is confounded: each arm's reward is
computed against its own reference, so an arm can look better merely by having
an easier teacher. This evaluates every checkpoint against a single fixed
criterion that none of them was trained on:

    **task-space site error versus the nominal body's world motion.**

That is "did the robot reproduce the choreography", asked identically of every
arm. Reported alongside it: episode length, fall rate, phase coverage, and a
matched zero-action control on the same body.

Do not run this while a trainer is on the GPU -- eval compile has OOM-killed
training runs on this machine before. It is CPU-only by default for that reason.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE / "scripts"))
sys.path.insert(0, str(WORKSPACE / "scripts" / "h1md"))

from c6_reward_discrimination import build_model, catalog, reground  # noqa: E402
from c8_reference_feasibility import reference_qvel  # noqa: E402
from c9_shared_policy import (  # noqa: E402
    MIMIC_SITES, W_DANCE, make_complete_trajectory, register_variant,
)


def nominal_yardstick(nominal_model, qpos, qvel, dt):
    """Upper-body-relative mimic-site offsets of the NOMINAL body's motion.

    This is the common target: the choreography as the nominal robot performs it.
    """
    import mujoco
    from loco_mujoco.core.utils.math import calculate_relative_site_quatities

    q, _ = reground(nominal_model, qpos)
    v = reference_qvel(qvel, q, dt)
    sids = np.array([mujoco.mj_name2id(nominal_model, mujoco.mjtObj.mjOBJ_SITE, n) for n in MIMIC_SITES])
    bids = np.array([nominal_model.site_bodyid[s] for s in sids])
    d = mujoco.MjData(nominal_model)
    out = []
    for qq, vv in zip(q, v):
        d.qpos[:] = qq
        d.qvel[:] = vv
        mujoco.mj_forward(nominal_model, d)
        p, _, _ = calculate_relative_site_quatities(d, sids, bids, nominal_model.body_rootid, np)
        out.append(p)
    return np.asarray(out)


def rollout_policy(env, agent_conf, agent_state, model, yardstick, n_steps, zero_action=False):
    """Deterministic CPU rollout recording the common-yardstick error each step."""
    import mujoco
    from loco_mujoco.core.utils.math import calculate_relative_site_quatities

    sids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n) for n in MIMIC_SITES])
    bids = np.array([model.site_bodyid[s] for s in sids])

    if agent_conf is not None:
        ts = agent_state.train_state
        if agent_conf.config.experiment.n_seeds > 1:
            ts = jax.tree.map(lambda x: x[0], ts)
        ts.params["log_std"] = np.ones_like(ts.params["log_std"]) * -np.inf

        def act(ts, obs, key):
            y, upd = agent_conf.network.apply(
                {"params": ts.params, "run_stats": ts.run_stats}, obs, mutable=["run_stats"])
            pi, _ = y
            return pi.sample(seed=key), ts.replace(run_stats=upd["run_stats"])

        act = jax.jit(act)

    obs = env.reset()
    rng = jax.random.key(0)
    errs, rewards = [], []
    steps = 0
    for i in range(min(n_steps, len(yardstick))):
        if zero_action or agent_conf is None:
            action = np.zeros((1, env.info.action_space.shape[0]))
        else:
            rng, sub = jax.random.split(rng)
            a, ts = act(ts, jnp.atleast_2d(obs), sub)
            action = np.asarray(jnp.atleast_2d(a))
        obs, reward, absorbing, done, info = env.step(action)
        rewards.append(float(np.asarray(reward).item()))
        data = env._data
        p, _, _ = calculate_relative_site_quatities(data, sids, bids, model.body_rootid, np)
        errs.append(np.linalg.norm(p - yardstick[i], axis=-1))
        steps += 1
        if bool(np.asarray(done).item()) or bool(np.asarray(absorbing).item()):
            break

    errs = np.asarray(errs)
    return {
        "episode_length": steps,
        "phase_coverage": steps / len(yardstick),
        "terminated_early": steps < min(n_steps, len(yardstick)),
        "yardstick_site_error_cm_mean": float(errs.mean() * 100) if errs.size else float("nan"),
        "yardstick_site_error_cm_final": float(errs[-1].mean() * 100) if errs.size else float("nan"),
        "yardstick_per_site_cm": (
            {n: float(errs[:, i].mean() * 100) for i, n in enumerate(
                [s for s in MIMIC_SITES if s != "upper_body_mimic"])} if errs.size else {}),
        "own_reward_mean": float(np.mean(rewards)) if rewards else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--checkpoint-root", type=Path,
                    default=WORKSPACE / "experiments" / "h1_morphology_deepmimic_20260808" / "checkpoints")
    ap.add_argument("--start", type=int, default=19482)
    ap.add_argument("--frames", type=int, default=800)
    ap.add_argument("--n-steps", type=int, default=800)
    ap.add_argument("--xml-root", type=Path,
                    default=WORKSPACE / "experiments" / "h1_morphology_deepmimic_20260808" / "body_catalog" / "xml")
    args = ap.parse_args()

    from c3_reference_methods import IKKernel, LIMB_SCALE_INDEX
    from loco_mujoco.algorithms import PPOJax
    from loco_mujoco.task_factories import CustomDatasetConf, ImitationFactory, LAFAN1DatasetConf

    src = ImitationFactory.make("UnitreeH1", lafan1_dataset_conf=LAFAN1DatasetConf(["dance2_subject4"]))
    th = src.th.traj
    freq = float(th.info.frequency)
    qpos = np.asarray(th.data.qpos)[args.start:args.start + args.frames].astype(np.float64)
    qvel = np.asarray(th.data.qvel)[args.start:args.start + args.frames].astype(np.float64)
    dt = 1.0 / freq

    bodies = dict(catalog())
    nominal_model = build_model("body00_nominal", bodies["body00_nominal"], args.xml_root)
    yard = nominal_yardstick(nominal_model, qpos, qvel, dt)
    print(f"yardstick built: {yard.shape} (nominal body's choreography)")

    kern = IKKernel(args.xml_root / "h1_morphology_c2_body00_nominal" / "h1.xml")

    def reference_for(name, arm):
        model = build_model(name, bodies[name], args.xml_root)
        if arm == "fk":
            return reground(model, qpos)[0]
        if arm == "shared_nominal":
            return reground(nominal_model, qpos)[0]
        morph = bodies[name]
        m = jnp.asarray(np.array([[morph["leg_length_scale"], morph["arm_length_scale"],
                                   morph["shoulder_width_scale"]]], dtype=np.float32))
        Q = jnp.asarray(qpos.astype(np.float32))
        nom = np.asarray(kern.fk_bt(jnp.ones((1, 3), jnp.float32), Q))[0]
        nom_t = jnp.asarray(nom[:, kern.TARGET_SITES, :])
        root_nom = jnp.asarray(qpos[:, 0:3].astype(np.float32))
        rel = nom_t - root_nom[:, None, :]
        s = m[0][LIMB_SCALE_INDEX][None, :, None]
        root = root_nom.at[:, 2].multiply(m[0][0])
        xs, _ = kern.ik(m, Q, (root[:, None, :] + rel * s)[None])
        xs = np.asarray(xs)[0]
        q = qpos.copy()
        q[:, kern.ACT_QADR] = np.clip(xs[:, : kern.NU], kern.jnt_low, kern.jnt_high)
        q[:, 0:3] += xs[:, kern.NU: kern.NU + 3]
        return q

    results = {}
    ckpts = sorted(p for p in args.checkpoint_root.glob("*/PPOJax_saved.pkl"))
    if not ckpts:
        raise SystemExit(f"no checkpoints under {args.checkpoint_root}")
    print(f"found {len(ckpts)} checkpoints")

    for ckpt in ckpts:
        tag = ckpt.parent.name
        name, arm = tag.split("__")
        model = build_model(name, bodies[name], args.xml_root)
        xml_path = args.xml_root / f"h1_morphology_c2_{name}" / "h1.xml"
        cpu_name = f"C10Cpu_{tag}"
        register_variant(cpu_name, xml_path, mjx=False)

        q_ref = reference_for(name, arm)
        v_ref = reference_qvel(qvel, q_ref, dt)
        traj = make_complete_trajectory(cpu_name, q_ref, v_ref, freq, th)

        # A deterministic start is required, not merely tidy: the yardstick is
        # indexed from frame 0, and the default handler uses reference-state
        # initialisation at a RANDOM phase. Evaluating against a random phase
        # offset measures nothing but the offset.
        env = ImitationFactory.make(
            cpu_name, custom_dataset_conf=CustomDatasetConf(traj),
            headless=True, horizon=1000,
            th_params=dict(random_start=False, fixed_start_conf=(0, 0)),
            goal_type="GoalTrajMimic", goal_params=dict(visualize_goal=False),
            reward_type="MimicReward",
            reward_params=dict(**W_DANCE, sites_for_mimic=MIMIC_SITES),
        )
        agent_conf, agent_state = PPOJax.load_agent(ckpt)
        policy = rollout_policy(env, agent_conf, agent_state, env._model, yard, args.n_steps)
        zero = rollout_policy(env, None, None, env._model, yard, args.n_steps, zero_action=True)
        results[tag] = {"body": name, "arm": arm, "checkpoint": str(ckpt),
                        "policy": policy, "zero_action": zero}
        print(f"{tag:38s} len {policy['episode_length']:4d} (zero {zero['episode_length']:3d}) | "
              f"yardstick {policy['yardstick_site_error_cm_mean']:6.2f} cm "
              f"(zero {zero['yardstick_site_error_cm_mean']:6.2f}) | "
              f"own R {policy['own_reward_mean']:.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "component": "C10_common_yardstick",
        "criterion": "upper-body-relative mimic-site error vs the NOMINAL body's choreography",
        "why": "training return is scored against each arm's own reference and is not comparable across arms",
        "window": {"start_frame_100hz": args.start, "frames": args.frames},
        "results": results,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
