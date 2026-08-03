"""Phase 2 -- collect the expert rollout dataset.

Roll out each expert on its OWN body (the demonstrations a shared policy must
reproduce) and store every transition. Episodes start at random reference phases,
so the union covers the whole dance motion. Deterministic (mean) actions = clean
expert demonstrations.

Per variant -> expert_dataset/<variant>.npz with arrays:
  obs (N,450) action (N,23) reward (N,) done (N,) step_idx (N,) episode_id (N,)
Plus expert_dataset/meta.json and dataset_stats.json (obs/action mean+std for BC).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import common as C


def collect_variant(variant: dict, var_idx: int, target: int, n_envs: int) -> dict:
    """GPU MJX collection: N parallel envs, expert deterministic policy, auto-reset.

    Each env is an independent trajectory; we derive an episode_id per transition by
    cumulatively counting `done` flags within each env column so the BC train/val
    split can still be BY TRAJECTORY.
    """
    import gpu_roll as G
    import jax
    wenv = G.build_mjx_env(variant)
    policy = G.load_jax_policy(variant)
    chunk = max(200, C.HORIZON + 200)      # so most episodes complete inside a chunk
    d = G.collect_dataset(wenv, policy, n_envs, chunk_steps=chunk, target=target,
                          key=jax.random.PRNGKey(var_idx))
    n = len(d["obs"])
    # episode_id: a global counter incremented at every done (order is by step then env;
    # good enough to keep whole rollouts together for the trajectory split).
    done = d["done"].astype(bool)
    episode_id = np.cumsum(np.concatenate([[0], done[:-1]])).astype(np.int32)
    out = dict(obs=d["obs"], action=d["action"], reward=d["reward"], done=d["done"],
               step_idx=d["step_idx"], episode_id=episode_id,
               variant_idx=np.full(n, var_idx, dtype=np.int32))
    C.log(f"  {variant['preset']}: collected {n:,} samples, mean reward/step "
          f"{float(d['reward'].mean()):.3f}, ~{int(done.sum())} episode-ends")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--samples-per-variant", type=int, default=400_000)
    ap.add_argument("--envs", type=int, default=1024)
    args = ap.parse_args()

    if C.phase_done(args.outdir, "collect"):
        C.log("phase collect already done; skipping")
        return

    selection = C.load_selection(args.outdir)
    order = selection["variants_order"]
    ds_dir = args.outdir / "expert_dataset"
    ds_dir.mkdir(parents=True, exist_ok=True)

    descriptors = json.loads((args.outdir / "morphology_descriptors.json").read_text())
    meta = {"variants_order": order, "clip": C.CLIP,
            "descriptor_keys": descriptors["feature_keys"],
            "descriptor_normalized": descriptors["normalized"], "variants": {}}

    total = 0
    for i, preset in enumerate(order):
        out_npz = ds_dir / f"{preset}.npz"
        if out_npz.exists():  # resume: per-variant granularity
            d = np.load(out_npz)
            n = len(d["obs"])
            C.log(f"{preset}: already collected ({n:,}); skipping")
        else:
            C.log(f"collecting {preset} (target {args.samples_per_variant:,})")
            data = collect_variant(selection["variants"][preset], i,
                                   args.samples_per_variant, args.envs)
            np.savez_compressed(out_npz, **data)
            n = len(data["obs"])
        meta["variants"][preset] = {"variant_idx": i, "n_samples": int(n),
                                    "npz": str(out_npz.name)}
        total += n

    # global obs/action stats for BC input/target normalization
    obs_sum = obs_sq = None
    act_sum = act_sq = None
    N = 0
    for preset in order:
        d = np.load(ds_dir / f"{preset}.npz")
        o = d["obs"].astype(np.float64); a = d["action"].astype(np.float64)
        if obs_sum is None:
            obs_sum = o.sum(0); obs_sq = (o**2).sum(0)
            act_sum = a.sum(0); act_sq = (a**2).sum(0)
        else:
            obs_sum += o.sum(0); obs_sq += (o**2).sum(0)
            act_sum += a.sum(0); act_sq += (a**2).sum(0)
        N += len(o)
    obs_mean = obs_sum / N; obs_std = np.sqrt(np.maximum(obs_sq / N - obs_mean**2, 1e-8))
    act_mean = act_sum / N; act_std = np.sqrt(np.maximum(act_sq / N - act_mean**2, 1e-8))

    np.savez(ds_dir / "norm_stats.npz",
             obs_mean=obs_mean.astype(np.float32), obs_std=obs_std.astype(np.float32),
             act_mean=act_mean.astype(np.float32), act_std=act_std.astype(np.float32))
    C.write_json(ds_dir / "meta.json", meta)
    C.write_json(args.outdir / "dataset_stats.json", {
        "total_samples": int(total),
        "per_variant": {p: meta["variants"][p]["n_samples"] for p in order},
        "obs_dim": int(obs_mean.shape[0]), "action_dim": int(act_mean.shape[0]),
        "obs_mean_l2": float(np.linalg.norm(obs_mean)),
        "action_abs_mean": float(np.abs(act_mean).mean()),
        "samples_per_variant_target": args.samples_per_variant,
    })
    C.log(f"dataset complete: {total:,} samples -> {ds_dir}")
    C.mark_done(args.outdir, "collect", {"total_samples": int(total)})


if __name__ == "__main__":
    main()
