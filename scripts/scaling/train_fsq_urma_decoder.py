"""FSQ design B — canonical encoder + embodiment-conditioned decoder.

The encoder sees exactly the same canonical future-motion window as design A.
The decoder is URMA-style: it receives the (quantized) motion code plus one
22-dim structural description PER JOINT of the target robot and predicts that
joint's future position — the same code must therefore serve H1 (19), G1 (23)
and Atlas (27) through their descriptions, which is the property the RL
pipeline needs from z.  Paired data: the per-family retargets of the SAME
canonical clip windows (never independently learned code semantics).

Subcommands:
  build — assemble the paired dataset npz from the cross_humanoid crop caches
          (needs loco-mujoco envs; run locally, ship the npz to Viper).
  train — train/evaluate; val split by CLIP WINDOW, never random frames.

No AMP.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[2]
SCRIPTS = WORKSPACE / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import flax.linen as nn  # noqa: E402
import flax.serialization  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402

from scaling.fsq_motion import (  # noqa: E402
    FSQ,
    NormalizationStats,
    future_window,
)

DEFAULT_LEVELS = (8, 5, 5, 5)
FUTURE_HORIZON_FRAMES = 30  # 0.3 s at the canonical 100 Hz


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


def cmd_build(args):
    from loco_mujoco.trajectory import Trajectory

    from scaling.cross_humanoid_retarget import HUMANOIDS
    from scaling.joint_descriptions import build_joint_block_spec
    from morphology_deepmimic import make_mimic_env

    payload = np.load(args.canonical_dataset, allow_pickle=False)
    features = payload["features"]
    split_points = payload["split_points"]
    clip_names = [str(x) for x in payload["clip_names"]]

    window_specs = []
    for spec in args.windows:
        clip, start, frames = spec.split(":")
        window_specs.append((clip, int(start), int(frames)))

    # canonical windows, per crop window (identical rows for every family)
    window_blocks, window_splits = [], [0]
    for clip, start, frames in window_specs:
        c = clip_names.index(clip)
        rows = slice(
            int(split_points[c]) + start, int(split_points[c]) + start + frames
        )
        window_blocks.append(
            future_window(
                features[rows], np.asarray([0, frames]), args.window, args.stride
            )
        )
        window_splits.append(window_splits[-1] + frames)
    windows = np.concatenate(window_blocks, axis=0)

    families, targets, descriptions = [], [], []
    j_max = 0
    family_joint_counts = {}
    for robot in args.robots:
        blocks = []
        for clip, start, frames in window_specs:
            crop = (
                WORKSPACE
                / "external_data"
                / "cross_humanoid"
                / f"{args.source}_source"
                / clip
                / robot
                / f"start{start}_{frames}f_direct.npz"
            )
            trajectory = Trajectory.load(str(crop))
            qpos = np.asarray(trajectory.data.qpos, dtype=np.float32)[:, 7:]
            # future joint target, clamped at the window end
            idx = np.minimum(
                np.arange(frames) + FUTURE_HORIZON_FRAMES, frames - 1
            )
            blocks.append(qpos[idx])
        family_target = np.concatenate(blocks, axis=0)
        family_joint_counts[robot] = family_target.shape[1]
        j_max = max(j_max, family_target.shape[1])
        targets.append(family_target)
        families.append(robot)

        # structural descriptions from the family's nominal model
        spec_ = HUMANOIDS[robot]
        env = make_mimic_env(
            spec_.mjx_env_name,
            Trajectory.load(
                str(
                    WORKSPACE
                    / "external_data"
                    / "cross_humanoid"
                    / f"{args.source}_source"
                    / window_specs[0][0]
                    / robot
                    / f"start{window_specs[0][1]}_{window_specs[0][2]}f_direct.npz"
                )
            ),
            headless=True,
        )
        block_spec = build_joint_block_spec(env, robot)
        descriptions.append(block_spec.descriptions)
        del env
        print(f"[build-b] {robot}: joints={family_target.shape[1]}", flush=True)

    padded_targets = np.zeros((len(families), windows.shape[0], j_max), np.float32)
    padded_masks = np.zeros((len(families), j_max), np.float32)
    padded_descriptions = np.zeros(
        (len(families), j_max, descriptions[0].shape[1]), np.float32
    )
    for f, (target, description) in enumerate(zip(targets, descriptions, strict=True)):
        j = target.shape[1]
        padded_targets[f, :, :j] = target
        padded_masks[f, :j] = 1.0
        padded_descriptions[f, :j] = description

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        windows=windows,
        window_splits=np.asarray(window_splits, dtype=np.int64),
        window_specs=np.asarray(args.windows),
        families=np.asarray(families),
        targets=padded_targets,
        masks=padded_masks,
        descriptions=padded_descriptions,
        window=args.window,
        stride=args.stride,
        horizon=FUTURE_HORIZON_FRAMES,
    )
    print(f"[build-b] -> {out} windows={windows.shape} j_max={j_max}")


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


class EncoderB(nn.Module):
    hidden: tuple[int, ...]
    code_dim: int

    @nn.compact
    def __call__(self, x):
        for width in self.hidden:
            x = nn.elu(nn.Dense(width)(x))
        return nn.Dense(self.code_dim)(x)


class URMADecoderB(nn.Module):
    """code + per-joint description -> per-joint future target (masked)."""

    embed_dim: int = 64
    joint_hidden: tuple[int, ...] = (128, 128)

    @nn.compact
    def __call__(self, code, descriptions, mask):
        # code: (n, d); descriptions: (F, J, 22); mask: (F, J)
        embedded = nn.elu(nn.Dense(self.embed_dim)(code))  # (n, e)
        n = embedded.shape[0]
        f, j, ddim = descriptions.shape
        desc = jnp.broadcast_to(descriptions[None], (n, f, j, ddim))
        emb = jnp.broadcast_to(embedded[:, None, None, :], (n, f, j, self.embed_dim))
        x = jnp.concatenate([emb, desc], axis=-1)
        for width in self.joint_hidden:
            x = nn.elu(nn.Dense(width)(x))
        prediction = nn.Dense(1)(x)[..., 0]  # (n, F, J)
        return prediction * mask[None]


class ModelB(nn.Module):
    hidden: tuple[int, ...]
    code_dim: int
    levels: tuple[int, ...] | None

    def setup(self):
        self.encoder = EncoderB(self.hidden, self.code_dim)
        self.decoder = URMADecoderB()
        self.fsq = FSQ(list(self.levels)) if self.levels is not None else None

    def __call__(self, x, descriptions, mask):
        z = self.encoder(x)
        code = self.fsq.quantize(z) if self.fsq is not None else jnp.tanh(z)
        return self.decoder(code, descriptions, mask), code


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


def cmd_train(args):
    payload = np.load(args.dataset, allow_pickle=False)
    windows = payload["windows"]
    window_splits = payload["window_splits"]
    window_specs = [str(x) for x in payload["window_specs"]]
    families = [str(x) for x in payload["families"]]
    targets = payload["targets"]  # (F, N, J)
    masks = payload["masks"]  # (F, J)
    descriptions = payload["descriptions"]  # (F, J, 22)

    val_windows = set(args.val_windows)
    unknown = val_windows - set(window_specs)
    if unknown:
        raise ValueError(f"--val-windows not in dataset: {sorted(unknown)}")
    train_rows, val_rows = [], []
    for w, spec in enumerate(window_specs):
        rows = np.arange(window_splits[w], window_splits[w + 1])
        (val_rows if spec in val_windows else train_rows).append(rows)
    train_rows = np.concatenate(train_rows)
    val_rows = np.concatenate(val_rows)

    stats = NormalizationStats.fit(windows[train_rows])
    x_train = stats.apply(windows[train_rows])
    x_val = stats.apply(windows[val_rows])
    target_stats = NormalizationStats.fit(
        targets[:, train_rows].transpose(1, 0, 2).reshape(len(train_rows), -1)
    )
    # normalize targets per family-joint for a fair masked MSE
    t_mean = target_stats.mean.reshape(len(families), -1)
    t_std = target_stats.std.reshape(len(families), -1)
    y_train = (targets[:, train_rows] - t_mean[:, None]) / t_std[:, None]
    y_val = (targets[:, val_rows] - t_mean[:, None]) / t_std[:, None]
    y_train = jnp.asarray(y_train.transpose(1, 0, 2))  # (n, F, J)
    y_val = np.asarray(y_val.transpose(1, 0, 2))

    mask = jnp.asarray(masks)
    desc = jnp.asarray(descriptions)
    levels = None if args.continuous else tuple(args.levels)
    model = ModelB(
        hidden=tuple(args.hidden), code_dim=len(args.levels), levels=levels
    )
    params = model.init(
        jax.random.PRNGKey(args.seed), jnp.zeros((1, x_train.shape[1])), desc, mask
    )
    tx = optax.adamw(args.lr, weight_decay=1e-5)
    opt_state = tx.init(params)

    def masked_mse(prediction, target):
        error = ((prediction - target) ** 2) * mask[None]
        return error.sum() / (mask.sum() * target.shape[0])

    @jax.jit
    def step(params, opt_state, x, y):
        def loss_fn(p):
            prediction, _ = model.apply(p, x, desc, mask)
            return masked_mse(prediction, y)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    @jax.jit
    def evaluate(params, x, y):
        prediction, code = model.apply(params, x, desc, mask)
        per_family = (
            ((prediction - y) ** 2) * mask[None]
        ).sum(axis=(0, 2)) / (mask.sum(axis=1) * y.shape[0])
        return masked_mse(prediction, y), per_family, code

    data_x = jnp.asarray(x_train)
    n = data_x.shape[0]
    key = jax.random.PRNGKey(args.seed + 1)
    steps_per_epoch = max(1, n // args.batch_size)
    started = time.perf_counter()
    history = []
    for epoch in range(args.epochs):
        key, perm_key = jax.random.split(key)
        order = jax.random.permutation(perm_key, n)
        epoch_loss = 0.0
        for i in range(steps_per_epoch):
            batch = order[i * args.batch_size : (i + 1) * args.batch_size]
            params, opt_state, loss = step(
                params, opt_state, data_x[batch], y_train[batch]
            )
            epoch_loss += float(loss)
        val_loss, per_family, _ = evaluate(params, jnp.asarray(x_val), jnp.asarray(y_val))
        history.append(
            {
                "epoch": epoch,
                "train": epoch_loss / steps_per_epoch,
                "val": float(val_loss),
            }
        )
        print(
            f"[train-b] epoch {epoch:3d} train={epoch_loss / steps_per_epoch:.5f} "
            f"val={float(val_loss):.5f}",
            flush=True,
        )

    val_loss, per_family, val_codes = evaluate(
        params, jnp.asarray(x_val), jnp.asarray(y_val)
    )
    # constant predictor: train-mean normalized target = 0 after normalization
    constant = float(np.mean((y_val**2) * np.asarray(mask)[None]) /
                     (np.asarray(mask).mean()))
    report = {
        "val_masked_mse": float(val_loss),
        "constant_predictor_mse": constant,
        "per_family_val_mse": {
            family: float(per_family[f]) for f, family in enumerate(families)
        },
        "train_seconds": time.perf_counter() - started,
        "quantized": levels is not None,
        "history": history[-5:],
    }
    if levels is not None:
        fsq = FSQ(list(levels))
        indices = np.asarray(fsq.codes_to_indexes(val_codes))
        counts = np.bincount(indices, minlength=fsq.codebook_size).astype(float)
        probabilities = counts / counts.sum()
        nonzero = probabilities[probabilities > 0]
        report.update(
            {
                "codebook_size": fsq.codebook_size,
                "unique_codes_val": int((counts > 0).sum()),
                "code_entropy_bits_val": float(
                    -(nonzero * np.log2(nonzero)).sum()
                ),
            }
        )

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "params.msgpack").write_bytes(flax.serialization.to_bytes(params))
    np.savez(out / "normalization.npz", mean=stats.mean, std=stats.std,
             target_mean=t_mean, target_std=t_std)
    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "experiment": "fsq_motion_design_b_urma_decoder",
                "dataset": str(args.dataset),
                "val_windows": sorted(val_windows),
                "levels": list(args.levels),
                "continuous_control": bool(args.continuous),
                "hidden": list(args.hidden),
                "epochs": args.epochs,
                "seed": args.seed,
                "jax_backend": jax.default_backend(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out / "DONE").write_text("ok\n", encoding="utf-8")
    print(f"[train-b] -> {out} val={float(val_loss):.5f} constant={constant:.5f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build")
    p.add_argument("--canonical-dataset", required=True)
    p.add_argument("--robots", nargs="+", default=["h1", "g1", "atlas"])
    p.add_argument("--source", default="h1")
    p.add_argument("--windows", nargs="+", required=True,
                   help="clip:start:frames with existing direct crops")
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--output", required=True)
    p.set_defaults(fn=cmd_build)

    p = sub.add_parser("train")
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--val-windows", nargs="+", required=True)
    p.add_argument("--levels", type=int, nargs="+", default=list(DEFAULT_LEVELS))
    p.add_argument("--continuous", action="store_true")
    p.add_argument("--hidden", type=int, nargs="+", default=[256, 128])
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_train)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
