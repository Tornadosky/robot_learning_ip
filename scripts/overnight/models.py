"""Shared torch policy definitions for the shared-policy BC/eval phases.

Four conditioning modes, all mapping (proprio+goal) obs -> joint targets:
  none       : obs only                       (shared_no_descriptor)
  onehot     : obs + variant one-hot          (shared_variant_onehot)
  descriptor : obs + morphology descriptor     (shared_morphology_descriptor)
  film       : obs, FiLM-modulated by descriptor (shared_film_descriptor)

Inputs are z-scored with dataset stats; the network predicts z-scored actions and
the caller denormalizes. A checkpoint stores both weights and the full config so
the closed-loop eval (a different process) can reconstruct the policy exactly.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class MLPPolicy(nn.Module):
    """obs (+ optional concatenated conditioning) -> action."""

    def __init__(self, obs_dim, cond_dim, act_dim, hidden=(512, 512)):
        super().__init__()
        layers = []
        d = obs_dim + cond_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, act_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, obs, cond=None):
        x = obs if cond is None else torch.cat([obs, cond], dim=-1)
        return self.net(x)


class FiLMPolicy(nn.Module):
    """obs -> action, each hidden layer FiLM-modulated by the descriptor.

    h <- ReLU(gamma(desc) * Linear(h) + beta(desc)) per block.
    """

    def __init__(self, obs_dim, cond_dim, act_dim, hidden=(512, 512)):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.films = nn.ModuleList()
        d = obs_dim
        for h in hidden:
            self.blocks.append(nn.Linear(d, h))
            self.films.append(nn.Linear(cond_dim, 2 * h))
            d = h
        self.head = nn.Linear(d, act_dim)

    def forward(self, obs, cond):
        h = obs
        for block, film in zip(self.blocks, self.films):
            h = block(h)
            gamma, beta = film(cond).chunk(2, dim=-1)
            h = torch.relu(gamma * h + beta)
        return self.head(h)


def build_model(cfg) -> nn.Module:
    obs_dim = cfg["obs_dim"]; act_dim = cfg["action_dim"]
    hidden = tuple(cfg.get("hidden", (512, 512)))
    ct = cfg["cond_type"]
    if ct == "film":
        return FiLMPolicy(obs_dim, cfg["cond_dim"], act_dim, hidden)
    cond_dim = 0 if ct == "none" else cfg["cond_dim"]
    return MLPPolicy(obs_dim, cond_dim, act_dim, hidden)


def cond_for_variant(cfg, variant_idx: int) -> np.ndarray | None:
    """Build the conditioning vector for a given variant index (eval time)."""
    ct = cfg["cond_type"]
    if ct == "none":
        return None
    if ct == "onehot":
        v = np.zeros(cfg["cond_dim"], dtype=np.float32)
        v[variant_idx] = 1.0
        return v
    # descriptor / film
    return np.asarray(cfg["variant_descriptors"][variant_idx], dtype=np.float32)


class TorchActor:
    """Wrap a trained checkpoint as an act_fn(obs)->action for closed-loop rollout."""

    def __init__(self, ckpt_path, variant_idx, device="cpu"):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.cfg = ck["cfg"]
        self.model = build_model(self.cfg).to(device)
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()
        self.device = device
        self.obs_mean = torch.tensor(self.cfg["obs_mean"], device=device)
        self.obs_std = torch.tensor(self.cfg["obs_std"], device=device)
        self.act_mean = np.asarray(self.cfg["act_mean"], dtype=np.float32)
        self.act_std = np.asarray(self.cfg["act_std"], dtype=np.float32)
        c = cond_for_variant(self.cfg, variant_idx)
        self.cond = None if c is None else torch.tensor(c, device=device).unsqueeze(0)
        self.uses_film = self.cfg["cond_type"] == "film"

    @torch.no_grad()
    def __call__(self, obs):
        o = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        return self.batch(o)[0]

    @torch.no_grad()
    def batch(self, obs):
        """Batched: obs (N, obs_dim) -> action (N, act_dim) numpy (for MJX rollouts)."""
        o = np.asarray(obs, dtype=np.float32)
        ot = (torch.tensor(o, device=self.device) - self.obs_mean) / self.obs_std
        if self.cfg["cond_type"] == "none":
            out = self.model(ot)
        else:
            cond = self.cond.expand(ot.shape[0], -1)
            out = self.model(ot, cond)
        a = out.cpu().numpy()
        return a * self.act_std + self.act_mean
