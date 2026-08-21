"""C4 -- does the stock LocoMuJoCo DeepMimic stack compile and run, and how fast?

Phase 1 of the goal document asks for "the smallest smoke budget that proves
reset, one compiled update, evaluation, and rendering". This does the first
three and, just as importantly, reports the numbers that decide whether the
plan is affordable at all:

* env construction time;
* JIT compile time for one PPO update (this is the cost that a per-body XLA
  recompile would multiply by the number of bodies);
* steps/second once compiled;
* projected wall time for a given step budget.

Nothing here is a training run -- the total budget is a handful of updates.

Run under WSL dance_env with the GPU visible.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import numpy as np
import yaml
from omegaconf import OmegaConf

import loco_mujoco
from loco_mujoco.algorithms import PPOJax
from loco_mujoco.task_factories import CustomDatasetConf, ImitationFactory, LAFAN1DatasetConf
from loco_mujoco.trajectory import Trajectory, TrajectoryData

MIMIC_SITES = [
    "upper_body_mimic", "left_hand_mimic", "left_foot_mimic",
    "right_hand_mimic", "right_foot_mimic",
]


def crop(traj: Trajectory, start: int, n: int) -> Trajectory:
    import jax.numpy as jnp
    return Trajectory(info=traj.info,
                      data=TrajectoryData.dynamic_slice_in_dim(traj.data, 0, start, n, backend=jnp))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--robot", default="MjxUnitreeH1")
    ap.add_argument("--clip", default="dance2_subject4")
    ap.add_argument("--start", type=int, default=19482, help="window start at 100 Hz env rate")
    ap.add_argument("--frames", type=int, default=800)
    ap.add_argument("--num-envs", type=int, default=512)
    ap.add_argument("--num-steps", type=int, default=100)
    ap.add_argument("--updates", type=int, default=3)
    ap.add_argument("--zero-site-weights", action="store_true",
                    help="Use the scaling trainer's weights (site terms off) instead of the dance weights.")
    args = ap.parse_args()

    result = {"component": "C4_stock_env_smoke", "args": vars(args) | {"out": str(args.out)}}
    result["jax"] = {"version": jax.__version__, "backend": jax.default_backend(),
                     "devices": [str(d) for d in jax.devices()]}
    print(f"backend={jax.default_backend()} devices={jax.devices()}")

    t0 = time.perf_counter()
    # Load the clip through the CPU env so the window indices are the 100 Hz ones
    # frozen in C1, then hand the cropped trajectory to the Mjx env explicitly.
    cpu_env = ImitationFactory.make("UnitreeH1", lafan1_dataset_conf=LAFAN1DatasetConf([args.clip]))
    full = cpu_env.th.traj
    freq = float(full.info.frequency)
    assert args.start + args.frames <= int(full.data.n_samples), "window out of range"
    traj = crop(full, args.start, args.frames)
    result["window"] = {"start": args.start, "frames": args.frames, "frequency_hz": freq,
                        "duration_s": args.frames / freq}
    result["timings_s"] = {"load_and_crop": time.perf_counter() - t0}
    print(f"clip loaded @ {freq} Hz, window {args.frames} frames "
          f"({result['timings_s']['load_and_crop']:.1f} s)")

    reward_params = dict(
        qpos_w_sum=0.6, qvel_w_sum=0.4, rpos_w_sum=0.0, rquat_w_sum=0.0, rvel_w_sum=0.0,
        sites_for_mimic=MIMIC_SITES,
    ) if args.zero_site_weights else dict(
        qpos_w_sum=0.4, qvel_w_sum=0.2, rpos_w_sum=0.5, rquat_w_sum=0.3, rvel_w_sum=0.1,
        sites_for_mimic=MIMIC_SITES,
    )
    result["reward_params"] = {k: v for k, v in reward_params.items() if k != "sites_for_mimic"}

    t0 = time.perf_counter()
    env = ImitationFactory.make(
        args.robot,
        custom_dataset_conf=CustomDatasetConf(traj),
        nconmax=7000, headless=True, horizon=1000,
        goal_type="GoalTrajMimic", goal_params=dict(visualize_goal=False),
        reward_type="MimicReward", reward_params=reward_params,
    )
    result["timings_s"]["env_make"] = time.perf_counter() - t0
    result["env"] = {
        "observation_dim": int(env.info.observation_space.shape[0]),
        "action_dim": int(env.info.action_space.shape[0]),
    }
    print(f"env built in {result['timings_s']['env_make']:.1f} s | "
          f"obs {result['env']['observation_dim']} act {result['env']['action_dim']}")

    total_timesteps = args.num_envs * args.num_steps * args.updates
    config = OmegaConf.create({"experiment": {
        "hidden_layers": [512, 256], "lr": 1e-4,
        "num_envs": args.num_envs, "num_steps": args.num_steps,
        "total_timesteps": float(total_timesteps), "update_epochs": 4,
        "proportion_env_reward": 0.0, "num_minibatches": 32,
        "gamma": 0.99, "gae_lambda": 0.95, "clip_eps": 0.2,
        "init_std": 0.2, "learnable_std": False, "ent_coef": 0.0, "vf_coef": 0.5,
        "max_grad_norm": 0.5, "activation": "tanh", "anneal_lr": False,
        "weight_decay": 0.0, "normalize_env": True, "debug": False,
        "n_seeds": 1, "vmap_across_seeds": True,
        # validation_interval = num_updates // validation.num upstream, so
        # validation.num must not exceed the (tiny) smoke update count.
        "validation": {"active": False, "num_steps": 100, "num_envs": 100, "num": 1},
    }})

    agent_conf = PPOJax.init_agent_conf(env, config)
    t0 = time.perf_counter()
    train_fn = jax.jit(PPOJax.build_train_fn(env, agent_conf, mh=None))
    result["timings_s"]["build_train_fn"] = time.perf_counter() - t0

    # First call pays the XLA compile; the second is pure runtime, so the
    # difference separates the per-graph compile cost from the step rate. That
    # split is the whole argument for one shared graph over one graph per body.
    t0 = time.perf_counter()
    out = train_fn(jax.random.PRNGKey(0))
    jax.block_until_ready(out["agent_state"])
    wall_first = time.perf_counter() - t0

    t0 = time.perf_counter()
    out2 = train_fn(jax.random.PRNGKey(1))
    jax.block_until_ready(out2["agent_state"])
    wall_second = time.perf_counter() - t0

    wall = wall_first
    result["timings_s"]["first_call_compile_plus_run"] = wall_first
    result["timings_s"]["second_call_run_only"] = wall_second
    result["timings_s"]["xla_compile_estimate"] = wall_first - wall_second
    result["total_timesteps"] = total_timesteps
    result["steps_per_second_including_compile"] = total_timesteps / wall_first
    result["steps_per_second_compiled"] = total_timesteps / wall_second
    print(f"compile ~{wall_first - wall_second:.1f} s | compiled rate "
          f"{total_timesteps / wall_second:,.0f} steps/s")

    metrics = out["training_metrics"]
    returns = np.asarray(metrics.mean_episode_return)
    lengths = np.asarray(metrics.mean_episode_length)
    result["metrics"] = {
        "mean_episode_return": [float(x) for x in returns],
        "mean_episode_length": [float(x) for x in lengths],
        "finite": bool(np.isfinite(returns).all() and np.isfinite(lengths).all()),
    }
    print(f"{args.updates} updates / {total_timesteps} steps in {wall:.1f} s "
          f"(includes compile) -> {total_timesteps / wall:,.0f} steps/s")
    print(f"returns  {np.round(returns, 3).tolist()}")
    print(f"ep_lens  {np.round(lengths, 1).tolist()}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
