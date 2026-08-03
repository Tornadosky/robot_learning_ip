"""Phase 6 (optional, lowest priority) -- conditional VAE over expert actions.

Encoder:  (obs, action)            -> latent z
Decoder:  (z, obs, descriptor)     -> action
A CVAE of the expert action conditioned on proprio+goal obs and morphology. Swept
over latent_dim in {8,16,32} and compared to deterministic BC by reconstruction
MSE on the same trajectory-held-out val split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import common as C
from phase_bc_train import load_dataset, split_by_trajectory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--latents", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--beta", type=float, default=1e-3)
    args = ap.parse_args()

    if C.phase_done(args.outdir, "vae"):
        C.log("phase vae already done; skipping")
        return

    import torch
    import torch.nn as nn
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_dataset(args.outdir)
    is_val = split_by_trajectory(data["traj"], 0.1)
    obs = data["obs"]; act = data["act"]; vidx = data["vidx"]
    cond = data["desc_mat"][vidx]
    od, ad, cd = obs.shape[1], act.shape[1], cond.shape[1]

    tr = ~is_val; va = is_val
    to_t = lambda a: torch.tensor(a, device=device)
    Xtr, Atr, Ctr = to_t(obs[tr]), to_t(act[tr]), to_t(cond[tr])
    Xva, Ava, Cva = to_t(obs[va]), to_t(act[va]), to_t(cond[va])

    class CVAE(nn.Module):
        def __init__(self, z):
            super().__init__()
            self.enc = nn.Sequential(nn.Linear(od + ad, 512), nn.ReLU(),
                                     nn.Linear(512, 256), nn.ReLU())
            self.mu = nn.Linear(256, z); self.lv = nn.Linear(256, z)
            self.dec = nn.Sequential(nn.Linear(z + od + cd, 512), nn.ReLU(),
                                     nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, ad))

        def forward(self, x, a, c):
            h = self.enc(torch.cat([x, a], -1))
            mu, lv = self.mu(h), self.lv(h)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
            recon = self.dec(torch.cat([z, x, c], -1))
            return recon, mu, lv

        def sample(self, x, c):
            z = torch.randn(x.shape[0], self.mu.out_features, device=x.device)
            return self.dec(torch.cat([z, x, c], -1))

    results = {}
    ckdir = args.outdir / "bc_checkpoints"; ckdir.mkdir(parents=True, exist_ok=True)
    n = Xtr.shape[0]; bs = 4096
    for z in args.latents:
        model = CVAE(z).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for ep in range(args.epochs):
            model.train(); perm = torch.randperm(n, device=device)
            for s in range(0, n, bs):
                idx = perm[s:s + bs]
                recon, mu, lv = model(Xtr[idx], Atr[idx], Ctr[idx])
                rec = ((recon - Atr[idx]) ** 2).mean()
                kld = -0.5 * (1 + lv - mu ** 2 - lv.exp()).mean()
                loss = rec + args.beta * kld
                opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            recon, mu, lv = model(Xva, Ava, Cva)
            recon_mse = float(((recon - Ava) ** 2).mean().item())   # posterior recon
            prior = model.sample(Xva, Cva)
            prior_mse = float(((prior - Ava) ** 2).mean().item())   # prior sample (deploy-like)
        torch.save({"latent": z, "state_dict": model.state_dict()}, ckdir / f"cvae_z{z}.pt")
        results[f"z{z}"] = {"posterior_recon_mse": recon_mse, "prior_sample_mse": prior_mse}
        C.log(f"[vae] z={z}: posterior_recon_mse={recon_mse:.4f} prior_sample_mse={prior_mse:.4f}")

    bc = json.loads((args.outdir / "bc_offline_summary.json").read_text())
    det_mse = bc["shared_morphology_descriptor"]["best_val_mse"]
    C.write_json(args.outdir / "vae_results.json",
                 {"latents": results, "deterministic_bc_descriptor_val_mse": det_mse,
                  "note": "MSE in normalized-action space; comparable to BC val MSE."})
    C.mark_done(args.outdir, "vae", results)


if __name__ == "__main__":
    main()
