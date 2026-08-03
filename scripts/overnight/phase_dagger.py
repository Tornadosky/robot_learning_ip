"""Phase 5 (optional) -- DAgger fine-tuning of the best morphology-conditioned BC policy.

PPO fine-tuning needs GPU MJX; JAX here is CPU-only, so PPO across thousands of
parallel envs is impractical overnight. DAgger is the CPU-feasible online remedy
for the BC distribution-shift failure mode and reuses the already-loaded experts:
roll out the current shared policy on each body, relabel the states it actually
visits with that body's expert, aggregate, and retrain. This directly attacks
"offline MSE low but closed-loop falls".

Operates on shared_morphology_descriptor (the cross-embodiment candidate).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import common as C
from models import build_model, TorchActor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--episodes-per-variant", type=int, default=12)
    ap.add_argument("--epochs-per-round", type=int, default=8)
    ap.add_argument("--base-model", default="shared_morphology_descriptor")
    args = ap.parse_args()

    if C.phase_done(args.outdir, "dagger"):
        C.log("phase dagger already done; skipping")
        return

    import torch, jax, jax.numpy as jnp
    import gpu_roll as G
    device = "cuda" if torch.cuda.is_available() else "cpu"
    selection = C.load_selection(args.outdir)
    order = selection["variants_order"]
    summary = json.loads((args.outdir / "bc_offline_summary.json").read_text())
    base_ckpt = summary[args.base_model]["checkpoint"]
    ck = torch.load(base_ckpt, map_location=device, weights_only=False)
    cfg = ck["cfg"]
    model = build_model(cfg).to(device)
    model.load_state_dict(ck["state_dict"])

    obs_mean = np.asarray(cfg["obs_mean"], np.float32); obs_std = np.asarray(cfg["obs_std"], np.float32)
    act_mean = np.asarray(cfg["act_mean"], np.float32); act_std = np.asarray(cfg["act_std"], np.float32)
    desc_mat = np.asarray(cfg["variant_descriptors"], np.float32)

    # batched JAX expert relabelers (obs (M,450) np -> action (M,23) np)
    relabel = {}
    for v in order:
        pol = G.load_jax_policy(selection["variants"][v])
        relabel[v] = jax.jit(jax.vmap(pol))
    out_ckpt = args.outdir / "bc_checkpoints" / "shared_morphology_descriptor_dagger.pt"

    opt = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
    loss_fn = torch.nn.MSELoss()
    buf_obs, buf_cond, buf_act = [], [], []
    history = []
    n_envs = args.episodes_per_variant * 16   # parallel MJX envs in lieu of sequential episodes
    n_steps = C.HORIZON + 200
    key = jax.random.PRNGKey(0)

    def actor_from_current(vi):
        torch.save({"cfg": cfg, "state_dict": model.state_dict()}, out_ckpt)
        return TorchActor(out_ckpt, vi, device=device)

    for rnd in range(args.rounds):
        round_returns = {}
        for vi, v in enumerate(order):
            wenv = G.build_mjx_env(selection["variants"][v])
            actor = actor_from_current(vi)
            key, k = jax.random.split(key)
            stats, frames = G.torch_rollout(wenv, actor.batch, n_envs, n_steps, k, collect=True)
            round_returns[v] = float(stats["mean_return"])
            obs = frames["obs"]
            if len(obs) == 0:
                continue
            cap = 40000                                   # bound the aggregated buffer
            if len(obs) > cap:
                sel_idx = np.random.default_rng(rnd * 100 + vi).choice(len(obs), cap, replace=False)
                obs = obs[sel_idx]
            labels = np.asarray(relabel[v](jnp.asarray(obs)), np.float32)  # expert relabel (batched)
            buf_obs.append(((obs - obs_mean) / obs_std).astype(np.float32))
            buf_cond.append(np.tile(desc_mat[vi], (len(obs), 1)).astype(np.float32))
            buf_act.append(((labels - act_mean) / act_std).astype(np.float32))
        # retrain on aggregated buffer
        X = torch.tensor(np.concatenate(buf_obs), device=device)
        Cnd = torch.tensor(np.concatenate(buf_cond), device=device)
        Y = torch.tensor(np.concatenate(buf_act), device=device)
        n = X.shape[0]; bs = 4096
        model.train()
        for ep in range(args.epochs_per_round):
            perm = torch.randperm(n, device=device)
            for s in range(0, n, bs):
                idx = perm[s:s + bs]
                pred = model(X[idx], Cnd[idx])
                loss = loss_fn(pred, Y[idx])
                opt.zero_grad(); loss.backward(); opt.step()
        history.append({"round": rnd, "buffer": int(n), "mean_returns": round_returns})
        C.log(f"[dagger] round {rnd}: buffer={n:,} returns={ {k: round(v) for k,v in round_returns.items()} }")

    torch.save({"cfg": cfg, "state_dict": model.state_dict()}, out_ckpt)
    C.write_json(args.outdir / "dagger_results.json",
                 {"base_model": args.base_model, "rounds": args.rounds,
                  "history": history, "checkpoint": str(out_ckpt)})
    C.mark_done(args.outdir, "dagger", {"rounds": args.rounds})


if __name__ == "__main__":
    main()
