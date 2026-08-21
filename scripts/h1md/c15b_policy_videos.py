"""C15b -- trained-policy videos with the reward's own targets drawn.

C15 produced the reference-validation half of the goal document's video
requirement (kinematic playback + target spheres). This produces the other half:
a deterministic rollout of a trained checkpoint on its own body, with the same
spheres, read from the same array the reward indexes.

The two videos together are what the document asks for per body: one showing the
reference is valid, one showing what the policy actually does about it.

Start phase is pinned (`random_start=False`), for the reason C10 found the hard
way: with the default random RSI the robot and the drawn targets are at
different phases and the video shows a policy that appears not to track.

Runs on CPU with EGL; safe alongside a trainer on the GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE / "scripts"))
sys.path.insert(0, str(WORKSPACE / "scripts" / "h1md"))

from c6_reward_discrimination import build_model, catalog, reground  # noqa: E402
from c8_reference_feasibility import reference_qvel  # noqa: E402
from c9_shared_policy import MIMIC_SITES, W_DANCE, make_complete_trajectory, register_variant  # noqa: E402
from c15_render_targets import add_target_spheres, render_clip  # noqa: E402
from refbuild import build_reference  # noqa: E402


def rollout_qpos(env, agent_conf, agent_state, n_steps, zero_action=False):
    """Deterministic rollout; returns the qpos actually visited, per step."""
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
    qpos_seq = []
    for _ in range(n_steps):
        if zero_action or agent_conf is None:
            action = np.zeros((1, env.info.action_space.shape[0]))
        else:
            rng, sub = jax.random.split(rng)
            a, ts = act(ts, jnp.atleast_2d(obs), sub)
            action = np.asarray(jnp.atleast_2d(a))
        obs, reward, absorbing, done, info = env.step(action)
        qpos_seq.append(np.array(env._data.qpos, copy=True))
        if bool(np.asarray(done).item()) or bool(np.asarray(absorbing).item()):
            break
    return np.asarray(qpos_seq)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--checkpoint-root", type=Path,
                    default=WORKSPACE / "experiments" / "h1_morphology_deepmimic_20260808" / "checkpoints")
    ap.add_argument("--video-dir", type=Path,
                    default=WORKSPACE / "experiments" / "h1_morphology_deepmimic_20260808" / "videos")
    ap.add_argument("--start", type=int, default=19482)
    ap.add_argument("--frames", type=int, default=800)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--xml-root", type=Path,
                    default=WORKSPACE / "experiments" / "h1_morphology_deepmimic_20260808" / "body_catalog" / "xml")
    args = ap.parse_args()

    from loco_mujoco.algorithms import PPOJax
    from loco_mujoco.task_factories import CustomDatasetConf, ImitationFactory, LAFAN1DatasetConf

    src = ImitationFactory.make("UnitreeH1", lafan1_dataset_conf=LAFAN1DatasetConf(["dance2_subject4"]))
    th = src.th.traj
    freq = float(th.info.frequency)
    qpos = np.asarray(th.data.qpos)[args.start:args.start + args.frames].astype(np.float64)
    qvel = np.asarray(th.data.qvel)[args.start:args.start + args.frames].astype(np.float64)
    dt = 1.0 / freq

    bodies = dict(catalog())
    # only the per-body C9b checkpoints carry a single known body; the
    # continuous-morphology runs have no fixed body to render against.
    ckpts = sorted(p for p in args.checkpoint_root.glob("*__*/PPOJax_saved.pkl"))
    if not ckpts:
        raise SystemExit(f"no per-body checkpoints under {args.checkpoint_root}")
    print(f"{len(ckpts)} per-body checkpoints")

    records = []
    for ckpt in ckpts:
        tag = ckpt.parent.name
        name, arm = tag.split("__")
        # Render every per-body checkpoint. Note for `ik_scaled`: those policies
        # were trained against the pre-Finding-32 reference, which was never
        # re-grounded and floated ~1.2 cm, so their video shows a policy trained
        # on a defective reference. Kept and labelled rather than dropped.
        model_plain = build_model(name, bodies[name], args.xml_root)
        xml_path = args.xml_root / f"h1_morphology_c2_{name}" / "h1.xml"
        cpu_name = f"C15bCpu_{tag}"
        register_variant(cpu_name, xml_path, mjx=False)

        # The reference MUST match the arm the checkpoint was trained on.
        # Building the fk reference for every arm reported 166 steps for the
        # shared_nominal policy that actually completes 799.
        q_ref = build_reference(name, arm, qpos, args.xml_root)
        v_ref = reference_qvel(qvel, q_ref, dt)
        traj = make_complete_trajectory(cpu_name, q_ref, v_ref, freq, th)

        rel_site_ids = np.array([mujoco.mj_name2id(model_plain, mujoco.mjtObj.mjOBJ_SITE, n)
                                 for n in MIMIC_SITES])
        targets = np.asarray(traj.data.site_xpos)[:, rel_site_ids, :]

        env = ImitationFactory.make(
            cpu_name, custom_dataset_conf=CustomDatasetConf(traj),
            headless=True, horizon=1000,
            th_params=dict(random_start=False, fixed_start_conf=(0, 0)),
            goal_type="GoalTrajMimic", goal_params=dict(visualize_goal=False),
            reward_type="MimicReward",
            reward_params=dict(**W_DANCE, sites_for_mimic=MIMIC_SITES),
        )
        agent_conf, agent_state = PPOJax.load_agent(ckpt)
        n = min(args.frames, len(targets))
        seq = rollout_qpos(env, agent_conf, agent_state, n)

        viz_xml = add_target_spheres(xml_path)
        viz_model = mujoco.MjModel.from_xml_path(str(viz_xml))
        sl = slice(None, None, args.stride)
        rec = render_clip(
            viz_model, seq[sl], targets[: len(seq)][sl],
            args.video_dir / f"{name}_policy_{arm}_with_targets.mp4",
            fps=max(1, int(round(freq / args.stride))),
            width=args.width, height=args.height,
            label=f"{name} policy ({arm}) + reward targets")
        rec.update({"body_id": name, "arm": arm, "mode": "policy",
                    "checkpoint": str(ckpt), "rollout_steps": int(len(seq)),
                    "completed_clip": bool(len(seq) >= n)})
        records.append(rec)
        print(f"{tag}: {len(seq)} steps, {rec['frames']} frames -> {rec['path']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "component": "C15b_policy_videos",
        "start_phase": "pinned (random_start=False); see C10 Finding 19 note",
        "target_source": "traj.data.site_xpos[rel_site_ids] -- the array MimicReward indexes",
        "videos": records,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
