"""Experiment 1 -- train ONE DeepMimic policy on several motions (one body).

Stock G1 (nominal) + 3x PD, trained on dance2_subject1..4 simultaneously (the
trajectory handler samples a clip per episode; the policy reads the current
reference from its goal obs). Mirrors the proven single-cell PPO loop from
train_deepmimic_morphology, with K equal segments + checkpoints, and resumes from
the latest checkpoint if restarted.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import numpy as np
from omegaconf import OmegaConf

from loco_mujoco.algorithms import PPOJax

import common as C
import mm_common as MM

NUM_STEPS_PER_UPDATE = 200


def build_config(args, seg_timesteps):
    return OmegaConf.create({"experiment": {
        "hidden_layers": list(args.hidden), "lr": args.lr, "num_envs": args.num_envs,
        "num_steps": NUM_STEPS_PER_UPDATE, "total_timesteps": float(seg_timesteps),
        "update_epochs": 4, "proportion_env_reward": 0.0, "num_minibatches": 32,
        "gamma": 0.99, "gae_lambda": 0.95, "clip_eps": 0.2, "init_std": args.init_std,
        "learnable_std": bool(args.learnable_std), "ent_coef": 0.0, "vf_coef": 0.5, "max_grad_norm": 0.5,
        "activation": "tanh", "anneal_lr": False, "weight_decay": 0.0,
        "normalize_env": True, "debug": False, "n_seeds": 1, "vmap_across_seeds": True,
        "validation": {"active": False, "num_steps": 100, "num_envs": 100, "num": 10},
    }})


def latest_checkpoint(ckpt_root: Path):
    cks = sorted(ckpt_root.glob("ckpt_*/PPOJax_saved.pkl"),
                 key=lambda p: int(p.parent.name.split("_")[-1]))
    return cks[-1] if cks else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--total-timesteps", type=float, default=600e6)
    ap.add_argument("--num-checkpoints", type=int, default=12)
    ap.add_argument("--num-envs", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--init-std", type=float, default=0.2)
    ap.add_argument("--learnable-std", action="store_true",
                    help="Let the policy learn its action std (helps multi-motion robustness).")
    ap.add_argument("--hidden", type=int, nargs="+", default=[512, 256])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--weighted", action="store_true",
                    help="Difficulty-weighted clip sampling (over-sample the worst-tracked clips).")
    args = ap.parse_args()

    if C.phase_done(args.outdir, "mm_train"):
        C.log("phase mm_train already done; skipping")
        return

    out = Path(args.outdir) / "mm_train"
    ckpt_root = out / "checkpoints"
    ckpt_root.mkdir(parents=True, exist_ok=True)

    train_clips = MM.WEIGHTED_TRAIN_CLIPS if args.weighted else MM.TRAIN_CLIPS
    C.log(f"building multi-clip env: body={MM.BODY_ENV} clips={train_clips} "
          f"weighted={args.weighted} pd={MM.PD_SCALE}x")
    env = MM.make_multiclip_env(train_clips)
    C.log(f"n_trajectories={env.th.n_trajectories} obs={env.info.observation_space.shape} "
          f"act={env.info.action_space.shape}")

    steps_per_update = NUM_STEPS_PER_UPDATE * args.num_envs
    total_updates = max(args.num_checkpoints, int(args.total_timesteps // steps_per_update))
    updates_per_segment = max(1, total_updates // args.num_checkpoints)
    seg_timesteps = updates_per_segment * steps_per_update

    config = build_config(args, seg_timesteps)
    agent_conf = PPOJax.init_agent_conf(env, config)
    train_fn = jax.jit(PPOJax.build_train_fn(env, agent_conf, mh=None))
    resume_fn = jax.jit(PPOJax.build_resume_train_fn(env, agent_conf, mh=None))

    # resume from latest checkpoint if present (robust to restarts)
    agent_state = None
    done_steps = 0
    start_k = 0
    last = latest_checkpoint(ckpt_root)
    if last is not None:
        _, agent_state = PPOJax.load_agent(str(last))
        done_steps = int(last.parent.name.split("_")[-1])
        start_k = done_steps // seg_timesteps
        C.log(f"resuming from {last} ({done_steps:,} steps, segment {start_k})")

    rng = jax.random.PRNGKey(args.seed)
    return_curve, checkpoints = [], []
    t0 = time.time()
    for k in range(start_k, args.num_checkpoints):
        rng, seg_rng = jax.random.split(rng)
        out_seg = train_fn(seg_rng) if agent_state is None else resume_fn(seg_rng, agent_state)
        jax.block_until_ready(out_seg["agent_state"])
        agent_state = out_seg["agent_state"]
        rets = np.asarray(out_seg["training_metrics"].mean_episode_return)
        lens = np.asarray(out_seg["training_metrics"].mean_episode_length)
        return_curve.extend(float(r) for r in rets)
        cum = (k + 1) * seg_timesteps
        ck = ckpt_root / f"ckpt_{k:02d}_{cum}"
        ck.mkdir(parents=True, exist_ok=True)
        PPOJax.save_agent(str(ck), agent_conf, agent_state)
        checkpoints.append({"index": k, "cumulative_steps": int(cum),
                            "agent_path": str(ck / "PPOJax_saved.pkl"),
                            "mean_episode_return": float(rets[-1]),
                            "mean_episode_length": float(lens[-1])})
        C.log(f"ckpt {k+1}/{args.num_checkpoints} @ {cum:,}: "
              f"return={rets[-1]:.1f} len={lens[-1]:.1f}")

    best = max(checkpoints, key=lambda c: c["mean_episode_return"]) if checkpoints else None
    manifest = {
        "experiment": "multi_motion_single_body", "body": MM.BODY_ENV,
        "train_clips": MM.TRAIN_CLIPS, "heldout_clip": MM.HELDOUT_CLIP,
        "pd_scale": MM.PD_SCALE, "lr": args.lr, "init_std": args.init_std,
        "hidden": list(args.hidden), "num_envs": args.num_envs,
        "total_timesteps": int(args.num_checkpoints * seg_timesteps),
        "training_minutes": (time.time() - t0) / 60.0,
        "checkpoints": checkpoints, "best_checkpoint": best,
        "return_curve": [float(r) for r in np.asarray(return_curve)[::max(1, len(return_curve)//200)]],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    C.log(f"mm_train done: best return {best['mean_episode_return']:.1f} @ {best['cumulative_steps']:,} steps")
    C.mark_done(args.outdir, "mm_train",
                {"best_return": best["mean_episode_return"], "best_steps": best["cumulative_steps"]})


if __name__ == "__main__":
    main()
