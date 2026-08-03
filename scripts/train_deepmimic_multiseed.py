"""Train N seeds of one DeepMimic cell IN PARALLEL (jax.vmap) and keep the best.

Why: crossing the balance transition is a stochastic, bifurcation-sensitive event
(only ~1 in 5 single-seed G1 runs cross), so a lone run may never converge. Training
several seeds at once on the GPU and selecting the best by final tracking return
reliably yields a converged "crosser" in roughly one run's wall-clock -- far cheaper
than re-rolling single seeds sequentially.

This vmaps PPOJax.build_train_fn over a stack of PRNG keys (the same pattern as the
upstream jax_rl_mimic example), runs one shot to --total-timesteps per seed, then
saves the winning seed's agent + a manifest in the SAME schema the renderer and the
recipe plotter already consume. Reuses all env/reference/config setup from
train_deepmimic_morphology so the recipe is identical to the single-seed trainer.

GPU JAX only (WSL2 / conda env `locodm`). Memory scales ~linearly with --n-seeds at
fixed --num-envs, so reduce one if you hit OOM.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from loco_mujoco.algorithms import PPOJax

from morphology_deepmimic import (
    cell_dir,
    control_config,
    get_robot,
    make_mimic_env,
    prepare_variant,
)
from train_deepmimic_morphology import NUM_STEPS_PER_UPDATE, build_config, resolve_reference


def main() -> None:
    args = parse_args()
    robot = get_robot(args.robot)

    variant = prepare_variant(robot, args.preset, args.cache_tag)
    traj, start_frame, n_frames, frequency, ref_desc = resolve_reference(robot, args, variant)
    print(f"[multiseed] {robot.key}/{args.clip}/{args.preset}: ref={ref_desc} "
          f"({int(traj.data.n_samples)} samples @ {frequency:.0f}Hz)")

    ctrl_params = control_config(robot.key, args.control)
    if args.control == "pd" and args.pd_gain_scale != 1.0:
        cp = ctrl_params["control_params"]
        cp["p_gain"] = [g * args.pd_gain_scale for g in cp["p_gain"]]
        cp["d_gain"] = [g * args.pd_gain_scale for g in cp["d_gain"]]
    env = make_mimic_env(
        variant["mjx_env_name"], traj,
        use_mjwarp=args.use_mjwarp, nconmax=7000, headless=True,
        **ctrl_params,
    )

    steps_per_update = NUM_STEPS_PER_UPDATE * args.num_envs
    total_updates = max(1, int(args.total_timesteps // steps_per_update))
    total_timesteps = total_updates * steps_per_update

    config = build_config(args, total_timesteps)
    agent_conf = PPOJax.init_agent_conf(env, config)

    # vmap the single-seed train fn over a stack of PRNG keys -> trains all seeds at once.
    seeds = list(range(args.seed, args.seed + args.n_seeds))
    rngs = jnp.stack([jax.random.PRNGKey(s) for s in seeds])
    print(f"[multiseed] backend={jax.default_backend()} seeds={seeds} "
          f"envs/seed={args.num_envs} pd_gain_scale={args.pd_gain_scale} lr={args.lr} "
          f"init_std={args.init_std} -> {total_timesteps:,} steps/seed in one shot")

    train_fn = jax.jit(jax.vmap(PPOJax.build_train_fn(env, agent_conf, mh=None)))
    t_start = time.time()
    out = train_fn(rngs)
    jax.block_until_ready(out["agent_state"])
    elapsed = time.time() - t_start

    # Per-seed curves: training_metrics fields are [n_seeds, n_updates].
    returns = np.asarray(out["training_metrics"].mean_episode_return)
    lengths = np.asarray(out["training_metrics"].mean_episode_length)
    # Rank seeds by the average return over the last few % of updates (robust to noise).
    tail = max(1, returns.shape[1] // 20)
    seed_score = returns[:, -tail:].mean(axis=1)
    best = int(np.argmax(seed_score))
    print("[multiseed] per-seed final (avg last %d updates):" % tail)
    for i, s in enumerate(seeds):
        print(f"    seed {s}: return={seed_score[i]:7.1f} len_last={lengths[i, -1]:7.1f}"
              + ("  <-- BEST" if i == best else ""))

    # Extract and save the winning seed's agent (index the batched pytree).
    best_state = jax.tree_util.tree_map(lambda x: x[best], out["agent_state"])
    out_dir = cell_dir(robot.key, args.clip, args.preset)
    if args.out_suffix:
        out_dir = out_dir.parent / f"{args.preset}__{args.out_suffix}"
    ckpt_dir = out_dir / "checkpoints" / f"ckpt_best_seed{seeds[best]}_{total_timesteps}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    agent_path = PPOJax.save_agent(str(ckpt_dir), agent_conf, best_state)
    ref_save_path = out_dir / "reference.npz"
    traj.save(str(ref_save_path))

    best_curve = returns[best]
    manifest = {
        "robot": robot.mjx_env_name,
        "robot_key": robot.key,
        "preset": args.preset,
        "clip": args.clip,
        "reference": ref_desc,
        "raw_reference": bool(args.raw_reference),
        "control": args.control,
        "pd_gain_scale": args.pd_gain_scale,
        "lr": args.lr,
        "init_std": args.init_std,
        "out_suffix": args.out_suffix,
        "reference_path": str(ref_save_path),
        "window_start_frame": int(start_frame),
        "window_frames": int(n_frames),
        "frequency_hz": frequency,
        "num_envs": args.num_envs,
        "n_seeds": args.n_seeds,
        "seeds": seeds,
        "best_seed": seeds[best],
        "total_timesteps": int(total_timesteps),
        "training_minutes": elapsed / 60.0,
        "mean_episode_return_first": float(best_curve[0]),
        "mean_episode_return_last": float(best_curve[-1]),
        "mean_episode_length_last": float(lengths[best, -1]),
        "per_seed_score": {str(s): float(seed_score[i]) for i, s in enumerate(seeds)},
        "per_seed_length_last": {str(s): float(lengths[i, -1]) for i, s in enumerate(seeds)},
        # Single checkpoint entry (best seed) so the existing renderer works unchanged.
        "checkpoints": [{
            "index": 0,
            "cumulative_steps": int(total_timesteps),
            "agent_path": str(agent_path),
            "mean_episode_return": float(best_curve[-1]),
            "mean_episode_length": float(lengths[best, -1]),
        }],
        "return_curve_every_update": [
            float(r) for r in best_curve[:: max(1, len(best_curve) // 200)]
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[multiseed] done in {elapsed / 60:.1f} min -> {out_dir / 'manifest.json'}")
    print(json.dumps({k: v for k, v in manifest.items()
                      if k not in ("checkpoints", "return_curve_every_update")}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", required=True, choices=["h1", "g1"])
    parser.add_argument("--preset", required=True)
    parser.add_argument("--clip", default="dance2_subject4")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--cache-tag", default="dance")
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--n-seeds", type=int, default=4)
    parser.add_argument("--total-timesteps", type=float, default=300e6)
    parser.add_argument("--seed", type=int, default=1, help="First seed; uses seed..seed+n_seeds-1.")
    parser.add_argument("--control", choices=["torque", "pd"], default="pd")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--init-std", type=float, default=0.2)
    parser.add_argument("--learnable-std", action="store_true")
    parser.add_argument("--pd-gain-scale", type=float, default=3.0)
    parser.add_argument("--out-suffix", default="multiseed")
    parser.add_argument("--raw-reference", action="store_true")
    parser.add_argument("--no-retarget", action="store_true")
    parser.add_argument("--use-mjwarp", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
