"""C37 -- score policies against the targets that are CORRECT for their body.

Training return cannot compare these arms: one was trained against the nominal
body's site targets and the other against per-body targets, so they optimise
different objectives and their returns are on different scales.

The common criterion is the one retargeting is supposed to improve: the distance
between the robot's mimic sites and the targets computed by forward kinematics
of the reference on **that robot's own body**. Both policies are judged by it,
regardless of what either was trained on.

Root error is reported alongside, indexed by each environment's own trajectory
phase, so a drifting policy is not mistaken for a mistracking one.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE / "scripts"))
sys.path.insert(0, str(WORKSPACE / "scripts" / "scaling"))
sys.path.insert(0, str(WORKSPACE / "scripts" / "h1md"))

from c9_shared_policy import MIMIC_SITES  # noqa: E402
from online_h1_train import build_online_env  # noqa: E402


def evaluate(ckpt: Path, num_envs: int, horizon: int, seed: int) -> dict:
    from loco_mujoco.algorithms import PPOJax
    from loco_mujoco.core.utils.math import calculate_relative_site_quatities
    from loco_mujoco.core.wrappers import LogWrapper, VecEnv
    from mujoco import mjx

    manifest = json.loads((ckpt.parent.parent / "manifest.json").read_text(encoding="utf-8"))
    freq = float(manifest["frequency_hz"])
    a = SimpleNamespace(
        clip=manifest["clip"], duration=manifest["window_frames"] / freq,
        start_frame=manifest["window_start_frame"], run_tag=f"c37_{ckpt.parent.parent.name}",
        use_mjwarp=False, backbone=manifest.get("backbone", "mlp"), resample_per_episode=False,
        morph_low=manifest["morphology_low"], morph_high=manifest["morphology_high"],
        catalog=(Path(manifest["catalog_path"]) if manifest.get("catalog_path") else None),
        catalog_mode=manifest.get("catalog_mode", "continuous"), catalog_stride=1,
        keep_morph_bounds=True, reward_weights=manifest.get("reward_weights_preset", "dance"),
        terminal_handler=manifest.get("terminal_handler"),
        goal_type=manifest.get("goal_type", "GoalTrajMimic"),
        reward_type=manifest.get("reward_type", "MimicReward"),
        max_root_deviation=None,
    )
    env, _ = build_online_env(a)
    traj_qpos = jnp.asarray(np.asarray(env.th.traj.data.qpos))

    site_ids = np.array([mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_SITE, n)
                         for n in MIMIC_SITES])
    body_ids = np.array([env._model.site_bodyid[s] for s in site_ids])
    rootid = env.sys.body_rootid
    d0 = mjx.make_data(env.sys)

    agent_conf, agent_state = PPOJax.load_agent(ckpt)
    ts = agent_state.train_state
    if agent_conf.config.experiment.n_seeds > 1:
        ts = jax.tree.map(lambda x: x[0], ts)
    variables = {"params": ts.params, "run_stats": ts.run_stats}

    def act(obs):
        (pi, _), _ = agent_conf.network.apply(variables, obs, mutable=["run_stats"])
        return pi.mean()

    def body_correct_sites(morph, qpos_ref):
        """Targets for THIS body: FK of the reference on its own morphology."""
        m = env._apply_morphology(env.sys, morph)
        ref = mjx.kinematics(m, d0.replace(qpos=qpos_ref))
        p, ang, _ = calculate_relative_site_quatities(ref, site_ids, body_ids, rootid, jnp)
        return p

    wrapped = VecEnv(LogWrapper(env))
    keys = jax.random.split(jax.random.PRNGKey(seed), num_envs)
    obs, state = jax.jit(wrapped.reset)(keys)

    def step(carry, _):
        obs, state, alive = carry
        a = act(obs)
        obs2, _, _, done, _, state2 = wrapped.step(state, a)
        alive = alive & ~done.astype(jnp.bool_)
        es = state2.env_state
        phase = es.additional_carry.traj_state.subtraj_step_no
        morph = es.additional_carry.morphology
        qref = traj_qpos[jnp.clip(phase, 0, traj_qpos.shape[0] - 1)]
        tgt = jax.vmap(body_correct_sites)(morph, qref)
        cur = jax.vmap(lambda d: calculate_relative_site_quatities(
            d, site_ids, body_ids, rootid, jnp)[0])(es.data)
        site_err = jnp.linalg.norm(cur - tgt, axis=-1).mean(axis=-1)
        root_err = jnp.linalg.norm(es.data.qpos[:, :2] - qref[:, :2], axis=-1)
        return (obs2, state2, alive), (site_err, root_err, alive)

    init = (obs, state, jnp.ones((num_envs,), dtype=jnp.bool_))
    _, (site_err, root_err, alive) = jax.jit(
        lambda c: jax.lax.scan(step, c, None, horizon))(init)
    site_err, root_err, alive = np.asarray(site_err), np.asarray(root_err), np.asarray(alive)
    live = alive.astype(bool)
    return {
        "site_error_cm": float(site_err[live].mean() * 100),
        "root_error_m": float(root_err[live].mean()),
        "alive_at_horizon": float(alive[-1].mean()),
        "clip": manifest["clip"],
        "trained_reward": manifest.get("reward_type", "MimicReward"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--horizon", type=int, default=300)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    CK = WORKSPACE / "experiments" / "h1_morphology_deepmimic_20260808" / "checkpoints"
    arms = {
        "dance · shared nominal targets": ["c13_spatial", "c34_nodev_seed1"],
        "dance · RETARGETED per body": ["c36_morph_seed0", "c36_morph_seed1"],
        # NOTE: the only existing walk baseline (c26) is a SINGLE-BODY run, so it
        # is not a matched control for the multi-body retargeted walk arm. It is
        # reported for reference, not as a comparison.
        "walk · single body, shared targets (unmatched)": ["c26_walk20m_seed0"],
        "walk · multi-body, RETARGETED": ["c36_morph_walk_seed0"],
    }
    out = {}
    for label, tags in arms.items():
        rows = []
        for tag in tags:
            ckpt = CK / tag / "checkpoint_final" / "PPOJax_saved.pkl"
            if not ckpt.exists():
                print(f"  {tag}: MISSING")
                continue
            r = evaluate(ckpt, args.num_envs, args.horizon, args.seed)
            rows.append(r)
            print(f"  {tag:24s} site {r['site_error_cm']:6.2f} cm | "
                  f"root {r['root_error_m']:.2f} m | alive {r['alive_at_horizon']:.2f}")
        if rows:
            s = [r["site_error_cm"] for r in rows]
            out[label] = {"per_seed_site_cm": s, "mean_site_cm": st.mean(s),
                          "sd_site_cm": st.stdev(s) if len(s) > 1 else 0.0,
                          "mean_root_m": st.mean([r["root_error_m"] for r in rows]),
                          "rows": rows}
            print(f"{label}: site {out[label]['mean_site_cm']:.2f} "
                  f"+/- {out[label]['sd_site_cm']:.2f} cm | "
                  f"root {out[label]['mean_root_m']:.2f} m\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "component": "C37_body_correct_eval",
        "criterion": "mimic-site error against targets computed by FK on the robot's OWN body",
        "arms": out,
    }, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
