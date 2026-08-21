"""C31 -- can ANY policy on this stack translate? A plain locomotion baseline.

Finding 43 closed the imitation investigation: no configuration tracks world-frame
position, and reward weighting, budget, observability, clip contact quality and
their combinations were all eliminated by test. The first follow-up it names is
this one, and it is the cheapest possible discriminator:

    train a standard velocity-command locomotion policy on the same stack,
    same robot, same actuators, same PPO -- no imitation at all.

If it walks, the failure is specific to the imitation path and worth chasing
there. **If it also cannot translate, the problem is upstream of imitation
entirely** -- in the actuation, control mode or environment -- and every drift
result in this audit is explained by one fact that has nothing to do with
references, rewards or morphology.

Metric is the honest one for a velocity task: achieved root speed against the
commanded speed, plus the same stand-still comparison used throughout.

Run under WSL dance_env with the GPU visible.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf


def build_config(num_envs, num_steps, total_timesteps):
    return OmegaConf.create({"experiment": {
        "hidden_layers": [512, 256], "lr": 3e-4,
        "num_envs": num_envs, "num_steps": num_steps,
        "total_timesteps": float(total_timesteps), "update_epochs": 4,
        "proportion_env_reward": 1.0, "num_minibatches": 32,
        "gamma": 0.99, "gae_lambda": 0.95, "clip_eps": 0.2,
        "init_std": 0.2, "learnable_std": False, "ent_coef": 0.0, "vf_coef": 0.5,
        "max_grad_norm": 0.5, "activation": "tanh", "anneal_lr": False,
        "weight_decay": 0.0, "normalize_env": True, "debug": False,
        "n_seeds": 1, "vmap_across_seeds": True,
        "validation": {"active": False, "num_steps": 100, "num_envs": 100, "num": 1},
    }})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--robot", default="MjxUnitreeH1")
    ap.add_argument("--num-envs", type=int, default=512)
    ap.add_argument("--num-steps", type=int, default=100)
    ap.add_argument("--total-timesteps", type=float, default=20e6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-envs", type=int, default=64)
    ap.add_argument("--eval-horizon", type=int, default=400)
    args = ap.parse_args()

    from loco_mujoco.algorithms import PPOJax
    from loco_mujoco.core.wrappers import LogWrapper, VecEnv
    from loco_mujoco.task_factories import RLFactory

    t0 = time.perf_counter()
    env = RLFactory.make(
        args.robot,
        headless=True, horizon=1000,
        goal_type="GoalRandomRootVelocity",
        reward_type="LocomotionReward",
    )
    build_s = time.perf_counter() - t0
    print(f"env built in {build_s:.1f} s | obs {env.info.observation_space.shape[0]} "
          f"act {env.info.action_space.shape[0]} | backend {jax.default_backend()}")

    config = build_config(args.num_envs, args.num_steps, args.total_timesteps)
    agent_conf = PPOJax.init_agent_conf(env, config)
    train_fn = jax.jit(PPOJax.build_train_fn(env, agent_conf, mh=None))

    t0 = time.perf_counter()
    out = train_fn(jax.random.PRNGKey(args.seed))
    jax.block_until_ready(out["agent_state"])
    wall = time.perf_counter() - t0
    metrics = out["training_metrics"]
    returns = np.asarray(metrics.mean_episode_return)
    lengths = np.asarray(metrics.mean_episode_length)
    print(f"trained {args.total_timesteps:.0f} steps in {wall:.0f} s "
          f"({args.total_timesteps / wall:,.0f} steps/s)")
    print(f"return {returns[0]:.2f} -> {returns[-1]:.2f} | "
          f"ep_len {lengths[0]:.1f} -> {lengths[-1]:.1f}")

    # --- does it actually move? achieved speed vs commanded speed
    agent_state = out["agent_state"]
    ts = agent_state.train_state
    if agent_conf.config.experiment.n_seeds > 1:
        ts = jax.tree.map(lambda x: x[0], ts)
    variables = {"params": ts.params, "run_stats": ts.run_stats}

    def act(obs):
        (pi, _), _ = agent_conf.network.apply(variables, obs, mutable=["run_stats"])
        return pi.mean()

    wrapped = VecEnv(LogWrapper(env))
    keys = jax.random.split(jax.random.PRNGKey(args.seed + 991), args.eval_envs)
    obs, state = jax.jit(wrapped.reset)(keys)

    def step(carry, _):
        obs, state, alive = carry
        a = act(obs)
        obs2, _, _, done, _, state2 = wrapped.step(state, a)
        alive = alive & ~done.astype(jnp.bool_)
        return (obs2, state2, alive), (state2.env_state.data.qpos[:, :3], alive)

    init = (obs, state, jnp.ones((args.eval_envs,), dtype=jnp.bool_))
    _, (roots, alives) = jax.jit(lambda c: jax.lax.scan(step, c, None, args.eval_horizon))(init)
    roots, alives = np.asarray(roots), np.asarray(alives)

    dt = env.dt
    # Measure only while alive. Path length after a fall is the robot sliding and
    # flailing; the smoke run scored 3.90 m of "path" at 0.05 m of net
    # displacement with every env dead, which would have read as walking.
    n_alive = alives.sum(axis=0)                       # steps survived, per env
    last = np.clip(n_alive - 1, 0, args.eval_horizon - 1)
    idx = np.arange(alives.shape[1])
    disp = np.linalg.norm(roots[last, idx, :2] - roots[0, :, :2], axis=-1)
    step_d = np.linalg.norm(np.diff(roots[:, :, :2], axis=0), axis=-1)
    path = (step_d * alives[1:]).sum(axis=0)
    alive_s = np.maximum(n_alive, 1) * dt              # seconds survived, per env
    duration = args.eval_horizon * dt
    result = {
        "component": "C31_locomotion_baseline",
        "question": "can any policy on this stack translate, with no imitation involved?",
        "robot": args.robot, "goal": "GoalRandomRootVelocity", "reward": "LocomotionReward",
        "total_timesteps": args.total_timesteps, "wall_s": wall,
        "steps_per_second": args.total_timesteps / wall,
        "return_first": float(returns[0]), "return_last": float(returns[-1]),
        "ep_len_first": float(lengths[0]), "ep_len_last": float(lengths[-1]),
        "eval": {
            "horizon_steps": args.eval_horizon, "dt": float(dt), "duration_s": float(duration),
            "steps_survived_mean": float(n_alive.mean()),
            "seconds_survived_mean": float(alive_s.mean()),
            "net_displacement_while_alive_m_mean": float(disp.mean()),
            "net_displacement_while_alive_m_max": float(disp.max()),
            "path_length_while_alive_m_mean": float(path.mean()),
            "net_speed_m_s": float((disp / alive_s).mean()),
            "alive_fraction_at_horizon": float(alives[-1].mean()),
        },
    }
    # Speed must come from net displacement WHILE ALIVE, matching the JSON.
    # Path length over the full horizon counts post-fall sliding and reported
    # "DOES NOT TRANSLATE" for a policy that in fact walks 0.90 m in 1.5 s.
    net_speed = float((disp / alive_s).mean())
    print(f"\nsurvived {n_alive.mean():.0f}/{args.eval_horizon} steps "
          f"({alive_s.mean():.2f} s) | net displacement while alive {disp.mean():.2f} m "
          f"(max {disp.max():.2f}) | path {path.mean():.2f} m | "
          f"net speed {net_speed:.2f} m/s | alive at horizon {alives[-1].mean():.2f}")
    print("VERDICT:", "TRANSLATES" if net_speed > 0.3 else "DOES NOT TRANSLATE")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.out.parent.parent / "checkpoints" / "c31_locomotion"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    result["checkpoint"] = str(PPOJax.save_agent(str(ckpt_dir), agent_conf, agent_state))
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
