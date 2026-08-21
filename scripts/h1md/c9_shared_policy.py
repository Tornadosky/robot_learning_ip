"""C9 -- does the reference construction change what a policy can learn?

Every cheap probe so far agrees the bodies are equivalent (C2 kinematically, C8
dynamically) and that reference *semantics* dominate reference *personalisation*
(C3 Finding 13). What none of them can settle is whether those differences
survive contact with PPO. This runs the smallest experiment that can.

Arms, at matched steps and matched everything else:

  ik_scaled       the body trains against its own scale-normalized IK reference
  fk              the body trains against its own FK + re-grounded reference
  shared_nominal  the body trains against the *nominal* body's reference

plus a nominal-body control. Site reward terms are ON (the stock dance weights),
because C6 Finding 8 showed the repo's multi-body trainer had them zeroed, which
makes every reference arm identical by construction.

The missing pipeline link this script also exercises: a raw `(qpos, qvel)`
reference is not something `MimicReward` can consume -- it needs site positions,
orientations and velocities computed *on the target body*. `extend_motion` does
that, and it must be run against the variant env, not the nominal one.

Run under WSL dance_env with the GPU visible.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE / "scripts"))
sys.path.insert(0, str(WORKSPACE / "scripts" / "h1md"))

from c6_reward_discrimination import build_model, catalog, reground  # noqa: E402
from c8_reference_feasibility import reference_qvel  # noqa: E402

MIMIC_SITES = ["upper_body_mimic", "left_hand_mimic", "left_foot_mimic",
               "right_hand_mimic", "right_foot_mimic"]
W_DANCE = dict(qpos_w_sum=0.4, qvel_w_sum=0.2, rpos_w_sum=0.5, rquat_w_sum=0.3, rvel_w_sum=0.1)


def register_variant(env_name: str, xml_path: Path, mjx: bool) -> None:
    """Register an H1 subclass whose default XML is the variant body."""
    from loco_mujoco.environments import LocoEnv
    from loco_mujoco.environments.humanoids import MjxUnitreeH1, UnitreeH1

    base = MjxUnitreeH1 if mjx else UnitreeH1
    LocoEnv.registered_envs[env_name] = type(
        env_name, (base,),
        {"get_default_xml_file_path": classmethod(lambda cls: str(xml_path))},
    )


def make_complete_trajectory(cpu_env_name: str, q_ref: np.ndarray, v_ref: np.ndarray,
                             freq: float, template, n_substeps: int = 10):
    """Raw (qpos, qvel) -> a Trajectory carrying the site data MimicReward needs.

    The site quantities must be computed on the *target* body, so extend_motion
    is called with the variant env name.

    ``n_substeps`` matters more than it looks: ``extend_motion`` resamples to the
    env it builds from ``conf.env_params``, and the SMPL robot conf ships
    ``n_substeps: 25`` (40 Hz) while the training envs default to 10 (100 Hz).
    Left alone it silently downsamples a 100 Hz reference by 2.5x. Default here
    is 10 so the reference matches the trainer.
    """
    from copy import deepcopy

    from loco_mujoco.smpl.retargeting import extend_motion, load_robot_conf_file
    from loco_mujoco.trajectory import Trajectory, TrajectoryData, TrajectoryInfo

    info = TrajectoryInfo(
        joint_names=template.info.joint_names,
        model=template.info.model,
        frequency=freq,
    )
    data = TrajectoryData(
        qpos=jnp.asarray(q_ref, dtype=jnp.float32),
        qvel=jnp.asarray(v_ref, dtype=jnp.float32),
        split_points=jnp.array([0, len(q_ref)]),
    )
    env_params = deepcopy(load_robot_conf_file("UnitreeH1").env_params)
    env_params["n_substeps"] = n_substeps
    return extend_motion(cpu_env_name, env_params, Trajectory(info=info, data=data))


def build_config(num_envs, num_steps, total_timesteps):
    return OmegaConf.create({"experiment": {
        "hidden_layers": [512, 256], "lr": 1e-4,
        "num_envs": num_envs, "num_steps": num_steps,
        "total_timesteps": float(total_timesteps), "update_epochs": 4,
        "proportion_env_reward": 0.0, "num_minibatches": 32,
        "gamma": 0.99, "gae_lambda": 0.95, "clip_eps": 0.2,
        "init_std": 0.2, "learnable_std": False, "ent_coef": 0.0, "vf_coef": 0.5,
        "max_grad_norm": 0.5, "activation": "tanh", "anneal_lr": False,
        "weight_decay": 0.0, "normalize_env": True, "debug": False,
        "n_seeds": 1, "vmap_across_seeds": True,
        "validation": {"active": False, "num_steps": 100, "num_envs": 100, "num": 1},
    }})


def run_arm(mjx_env_name, traj, num_envs, num_steps, total_timesteps, seed, save_dir=None):
    from loco_mujoco.algorithms import PPOJax
    from loco_mujoco.task_factories import CustomDatasetConf, ImitationFactory

    env = ImitationFactory.make(
        mjx_env_name,
        custom_dataset_conf=CustomDatasetConf(traj),
        nconmax=7000, headless=True, horizon=1000,
        goal_type="GoalTrajMimic", goal_params=dict(visualize_goal=False),
        reward_type="MimicReward",
        reward_params=dict(**W_DANCE, sites_for_mimic=MIMIC_SITES),
    )
    config = build_config(num_envs, num_steps, total_timesteps)
    agent_conf = PPOJax.init_agent_conf(env, config)
    train_fn = jax.jit(PPOJax.build_train_fn(env, agent_conf, mh=None))
    t0 = time.perf_counter()
    out = train_fn(jax.random.PRNGKey(seed))
    jax.block_until_ready(out["agent_state"])
    wall = time.perf_counter() - t0
    m = out["training_metrics"]
    returns = np.asarray(m.mean_episode_return)
    lengths = np.asarray(m.mean_episode_length)
    saved = None
    if save_dir is not None:
        # returns are scored against each arm's OWN reference and are therefore
        # not comparable across arms; the checkpoint is what makes a common
        # yardstick evaluation possible later.
        save_dir.mkdir(parents=True, exist_ok=True)
        saved = str(PPOJax.save_agent(str(save_dir), agent_conf, out["agent_state"]))
    return {
        "checkpoint": saved,
        "wall_s": wall,
        "steps_per_s": total_timesteps / wall,
        "return_first": float(returns[0]), "return_last": float(returns[-1]),
        "return_best": float(returns.max()),
        "length_first": float(lengths[0]), "length_last": float(lengths[-1]),
        "length_best": float(lengths.max()),
        "return_curve": [float(x) for x in returns[:: max(1, len(returns) // 40)]],
        "length_curve": [float(x) for x in lengths[:: max(1, len(lengths) // 40)]],
        "finite": bool(np.isfinite(returns).all()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--clip", default="dance2_subject4")
    ap.add_argument("--start", type=int, default=19482)
    ap.add_argument("--frames", type=int, default=800)
    ap.add_argument("--bodies", nargs="+", default=["body04_seed1004"])
    ap.add_argument("--arms", nargs="+", default=["ik_scaled", "fk", "shared_nominal"])
    ap.add_argument("--nominal-control", action="store_true", default=True)
    ap.add_argument("--num-envs", type=int, default=512)
    ap.add_argument("--num-steps", type=int, default=100)
    ap.add_argument("--total-timesteps", type=float, default=20e6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--xml-root", type=Path,
                    default=WORKSPACE / "experiments" / "h1_morphology_deepmimic_20260808" / "body_catalog" / "xml")
    args = ap.parse_args()

    from c3_reference_methods import IKKernel, LIMB_SCALE_INDEX
    from loco_mujoco.task_factories import ImitationFactory, LAFAN1DatasetConf

    src_env = ImitationFactory.make("UnitreeH1", lafan1_dataset_conf=LAFAN1DatasetConf([args.clip]))
    th = src_env.th.traj
    freq = float(th.info.frequency)
    qpos = np.asarray(th.data.qpos)[args.start:args.start + args.frames].astype(np.float64)
    qvel = np.asarray(th.data.qvel)[args.start:args.start + args.frames].astype(np.float64)
    dt = 1.0 / freq
    print(f"window {args.frames} frames @ {freq} Hz, backend={jax.default_backend()}")

    bodies = dict(catalog())
    kern = IKKernel(args.xml_root / "h1_morphology_c2_body00_nominal" / "h1.xml")

    def ik_reference(name):
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
        tgt = (root[:, None, :] + rel * s)[None]
        xs, _ = kern.ik(m, Q, tgt)
        xs = np.asarray(xs)[0]
        q = qpos.copy()
        q[:, kern.ACT_QADR] = np.clip(xs[:, : kern.NU], kern.jnt_low, kern.jnt_high)
        q[:, 0:3] += xs[:, kern.NU: kern.NU + 3]
        return q

    results = {}
    todo = []
    for name in args.bodies:
        for arm in args.arms:
            todo.append((name, arm))
    if args.nominal_control:
        todo.append(("body00_nominal", "ik_scaled"))

    for name, arm in todo:
        tag = f"{name}__{arm}"
        model = build_model(name, bodies[name], args.xml_root)
        xml_path = args.xml_root / f"h1_morphology_c2_{name}" / "h1.xml"
        cpu_name, mjx_name = f"C9Cpu_{tag}", f"C9Mjx_{tag}"
        register_variant(cpu_name, xml_path, mjx=False)
        register_variant(mjx_name, xml_path, mjx=True)

        if arm == "fk":
            q_ref, _ = reground(model, qpos)
        elif arm == "ik_scaled":
            q_ref = ik_reference(name)
        elif arm == "shared_nominal":
            q_ref, _ = reground(build_model("body00_nominal", bodies["body00_nominal"], args.xml_root), qpos)
        else:
            raise ValueError(arm)
        v_ref = reference_qvel(qvel, q_ref, dt)

        t0 = time.perf_counter()
        traj = make_complete_trajectory(cpu_name, q_ref, v_ref, freq, th)
        extend_s = time.perf_counter() - t0
        print(f"[{tag}] reference extended in {extend_s:.1f} s "
              f"(complete={bool(traj.data.is_complete)}, n={int(traj.data.n_samples)})")

        r = run_arm(mjx_name, traj, args.num_envs, args.num_steps, args.total_timesteps, args.seed,
                    save_dir=args.out.parent.parent / "checkpoints" / tag)
        r["extend_motion_s"] = extend_s
        r["reference_frequency_hz"] = float(traj.info.frequency)
        r["reference_samples"] = int(traj.data.n_samples)
        r["body"] = name
        r["arm"] = arm
        results[tag] = r
        print(f"[{tag}] return {r['return_first']:.2f} -> {r['return_last']:.2f} "
              f"(best {r['return_best']:.2f}) | len {r['length_first']:.1f} -> {r['length_last']:.1f} "
              f"(best {r['length_best']:.1f}) | {r['wall_s']:.0f} s @ {r['steps_per_s']:,.0f} steps/s")

        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "component": "C9_reference_arm_comparison",
            "question": "does the reference construction change what PPO can learn?",
            "window": {"start_frame_100hz": args.start, "frames": args.frames},
            "reward_weights": W_DANCE,
            "budget": {"num_envs": args.num_envs, "num_steps": args.num_steps,
                       "total_timesteps": args.total_timesteps, "seed": args.seed,
                       "seeds_note": "single seed -- feasibility result, not a robustness claim"},
            "results": results,
        }, indent=2), encoding="utf-8")

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
