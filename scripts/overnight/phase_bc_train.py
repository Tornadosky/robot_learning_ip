"""Phase 3 -- train the shared behaviour-cloning policies.

Trains, on the pooled expert dataset:
  A shared_no_descriptor              obs -> action
  B shared_variant_onehot             obs + variant one-hot -> action
  C shared_morphology_descriptor      obs + morphology descriptor -> action
  D shared_film_descriptor            obs, FiLM(descriptor) -> action
  C' shared_morphology_descriptor_heldout  C trained on 4 variants (held-out probe)

Train/val split is BY TRAJECTORY (whole episodes), never by individual frame, so
val MSE measures generalization to unseen rollouts rather than frame interpolation.
Targets are z-scored actions; MSE is reported in that normalized space.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import common as C
from models import build_model


def load_dataset(outdir: Path):
    ds_dir = outdir / "expert_dataset"
    meta = json.loads((ds_dir / "meta.json").read_text())
    order = meta["variants_order"]
    desc = json.loads((outdir / "morphology_descriptors.json").read_text())
    keys = desc["feature_keys"]
    desc_mat = np.array([[desc["normalized"][v][k] for k in keys] for v in order],
                        dtype=np.float32)
    ns = np.load(ds_dir / "norm_stats.npz")
    obs_mean, obs_std = ns["obs_mean"], ns["obs_std"]
    act_mean, act_std = ns["act_mean"], ns["act_std"]

    obs_l, act_l, vidx_l, traj_l = [], [], [], []
    traj_base = 0
    for i, v in enumerate(order):
        d = np.load(ds_dir / f"{v}.npz")
        obs_l.append(d["obs"]); act_l.append(d["action"])
        vidx_l.append(np.full(len(d["obs"]), i, dtype=np.int32))
        # globally-unique trajectory id = base + episode_id
        traj_l.append(traj_base + d["episode_id"].astype(np.int64))
        traj_base += int(d["episode_id"].max()) + 1
    obs = np.concatenate(obs_l); act = np.concatenate(act_l)
    vidx = np.concatenate(vidx_l); traj = np.concatenate(traj_l)

    obs_n = ((obs - obs_mean) / obs_std).astype(np.float32)
    act_n = ((act - act_mean) / act_std).astype(np.float32)
    return dict(order=order, desc_mat=desc_mat, obs=obs_n, act=act_n, vidx=vidx,
                traj=traj, obs_mean=obs_mean, obs_std=obs_std,
                act_mean=act_mean, act_std=act_std, keys=keys)


def split_by_trajectory(traj, val_frac, seed=0):
    rng = np.random.default_rng(seed)
    uniq = np.unique(traj)
    val_set = set(rng.choice(uniq, size=max(1, int(len(uniq) * val_frac)), replace=False).tolist())
    is_val = np.array([t in val_set for t in traj])
    return is_val


def train_one(name, cond_type, data, is_val, train_variant_mask, args, device, outdir):
    order = data["order"]
    obs = data["obs"]; act = data["act"]; vidx = data["vidx"]
    n_var = len(order)

    # conditioning source
    if cond_type == "onehot":
        cond_all = np.eye(n_var, dtype=np.float32)[vidx]
        cond_dim = n_var
    elif cond_type in ("descriptor", "film"):
        cond_all = data["desc_mat"][vidx]
        cond_dim = data["desc_mat"].shape[1]
    else:
        cond_all = None
        cond_dim = 0

    keep = train_variant_mask[vidx]                 # restrict to allowed variants
    tr = keep & (~is_val)
    va = keep & is_val
    C.log(f"[{name}] train {tr.sum():,} / val {va.sum():,} "
          f"(variants={[order[i] for i in range(n_var) if train_variant_mask[i]]})")

    def to_t(a):
        return torch.tensor(a, device=device)
    Xtr, Ytr = to_t(obs[tr]), to_t(act[tr])
    Xva, Yva = to_t(obs[va]), to_t(act[va])
    Ctr = to_t(cond_all[tr]) if cond_all is not None else None
    Cva = to_t(cond_all[va]) if cond_all is not None else None
    vidx_va = vidx[va]

    cfg = dict(cond_type=cond_type, cond_dim=cond_dim, obs_dim=obs.shape[1],
               action_dim=act.shape[1], hidden=list(args.hidden),
               obs_mean=data["obs_mean"].tolist(), obs_std=data["obs_std"].tolist(),
               act_mean=data["act_mean"].tolist(), act_std=data["act_std"].tolist(),
               variant_descriptors=data["desc_mat"].tolist(), variants_order=order)
    model = build_model(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = torch.nn.MSELoss()

    n = Xtr.shape[0]
    bs = args.batch_size
    curve = {"train_mse": [], "val_mse": []}
    best_val = float("inf")
    ckpt_dir = outdir / "bc_checkpoints"; ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{name}.pt"
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            xb = Xtr[idx]; yb = Ytr[idx]
            cb = Ctr[idx] if Ctr is not None else None
            pred = model(xb) if cond_type == "none" else model(xb, cb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        tr_mse = tot / n
        # val
        model.eval()
        with torch.no_grad():
            pv = model(Xva) if cond_type == "none" else model(Xva, Cva)
            v_mse = float(loss_fn(pv, Yva).item())
        curve["train_mse"].append(tr_mse); curve["val_mse"].append(v_mse)
        if np.isfinite(v_mse) and v_mse < best_val:
            best_val = v_mse
            torch.save({"cfg": cfg, "state_dict": model.state_dict()}, ckpt_path)
        if ep % 5 == 0 or ep == args.epochs - 1:
            C.log(f"[{name}] epoch {ep:3d} train_mse={tr_mse:.4f} val_mse={v_mse:.4f}")
    # always leave a usable checkpoint, even if val was empty/degenerate
    if not ckpt_path.exists():
        torch.save({"cfg": cfg, "state_dict": model.state_dict()}, ckpt_path)
        best_val = float(tr_mse)

    # per-variant val MSE (memorization probe)
    model.eval()
    with torch.no_grad():
        pv = (model(Xva) if cond_type == "none" else model(Xva, Cva)).cpu().numpy()
    yv = Yva.cpu().numpy()
    per_var = {}
    se = ((pv - yv) ** 2).mean(axis=1)
    for i, v in enumerate(order):
        mask = vidx_va == i
        per_var[v] = float(se[mask].mean()) if mask.any() else None

    return dict(name=name, cond_type=cond_type, best_val_mse=best_val,
                final_val_mse=curve["val_mse"][-1], curve=curve,
                per_variant_val_mse=per_var, train_minutes=(time.time() - t0) / 60.0,
                checkpoint=str(ckpt_path), n_train=int(tr.sum()), n_val=int(va.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, nargs="+", default=[512, 512])
    ap.add_argument("--val-frac", type=float, default=0.1)
    args = ap.parse_args()

    if C.phase_done(args.outdir, "bc_train"):
        C.log("phase bc_train already done; skipping")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    C.log(f"BC training on {device}")
    data = load_dataset(args.outdir)
    is_val = split_by_trajectory(data["traj"], args.val_frac)
    order = data["order"]
    n_var = len(order)
    all_mask = np.ones(n_var, dtype=bool)
    heldout_idx = order.index(C.HELDOUT_VARIANT)
    train4_mask = all_mask.copy(); train4_mask[heldout_idx] = False

    jobs = [
        ("shared_no_descriptor", "none", all_mask),
        ("shared_variant_onehot", "onehot", all_mask),
        ("shared_morphology_descriptor", "descriptor", all_mask),
        ("shared_film_descriptor", "film", all_mask),
        ("shared_morphology_descriptor_heldout", "descriptor", train4_mask),
    ]
    results = {}
    for name, ct, mask in jobs:
        results[name] = train_one(name, ct, data, is_val, mask, args, device, args.outdir)

    summary = {n: {k: r[k] for k in ("cond_type", "best_val_mse", "final_val_mse",
                                     "per_variant_val_mse", "train_minutes",
                                     "checkpoint", "n_train", "n_val")}
               for n, r in results.items()}
    C.write_json(args.outdir / "bc_offline_results.json", {"models": results})
    C.write_json(args.outdir / "bc_offline_summary.json", summary)

    # markdown
    md = ["# Offline BC results (normalized-action MSE, lower is better)", "",
          "| model | conditioning | best val MSE | train min |",
          "|---|---|---|---|"]
    for n, r in summary.items():
        md.append(f"| {n} | {r['cond_type']} | {r['best_val_mse']:.4f} | {r['train_minutes']:.1f} |")
    md += ["", "## Per-variant val MSE", "",
           "| model | " + " | ".join(order) + " |", "|" + "---|" * (n_var + 1)]
    for n, r in summary.items():
        row = [f"{(r['per_variant_val_mse'][v] if r['per_variant_val_mse'][v] is not None else float('nan')):.4f}"
               for v in order]
        md.append(f"| {n} | " + " | ".join(row) + " |")
    (args.outdir / "bc_offline_results.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # plot curves (guarded)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        for n, r in results.items():
            plt.plot(r["curve"]["val_mse"], label=n)
        plt.xlabel("epoch"); plt.ylabel("val MSE (norm action)"); plt.yscale("log")
        plt.legend(fontsize=7); plt.title("BC validation MSE"); plt.tight_layout()
        (args.outdir / "plots").mkdir(exist_ok=True)
        plt.savefig(args.outdir / "plots" / "bc_val_mse.png", dpi=120)
    except Exception as exc:
        C.log(f"plot skipped: {exc!r}")

    C.mark_done(args.outdir, "bc_train", summary)


if __name__ == "__main__":
    main()
