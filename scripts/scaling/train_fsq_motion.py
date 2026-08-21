"""Offline canonical FSQ autoencoder (design A) for motion-latent commands.

Subcommands:
  build   — extract canonical features from converted LAFAN1 clips into one
            portable npz (runs locally; the npz ships to Viper).
  train   — train encoder→FSQ→decoder on future-motion windows, split by
            CLIP, report reconstruction per feature group + code health.
  encode  — write a canonical token cache (codes per timestamp) for the RL
            trajectory layout (list of clip windows), loadable by
            `fsq_motion.buffer_from_codes_npz`.

No AMP anywhere.  The FSQ quantizer is the official Google implementation
(see fsq_motion.py provenance note).
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
    FRAME_FEATURE_DIM,
    FSQ,
    NormalizationStats,
    canonical_frame_features,
    feature_group_slices,
    future_window,
)

DEFAULT_LEVELS = (8, 5, 5, 5)


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


def _load_clip(args, clip):
    """One clip on the canonical grid.

    ``factory`` loads through ImitationFactory, which resamples to the env
    control frequency (100 Hz for UnitreeH1) — the SAME grid the RL
    references were cropped from (start19482_800f is a 100 Hz index; the raw
    converted cache is 40 Hz, so raw-npz features would misalign with every
    RL cursor).
    """
    if args.loader == "factory":
        from loco_mujoco.task_factories import (
            ImitationFactory,
            LAFAN1DatasetConf,
        )

        env = ImitationFactory.make(
            args.env_name, lafan1_dataset_conf=LAFAN1DatasetConf([clip])
        )
        return env.th.traj
    from loco_mujoco.trajectory import Trajectory

    return Trajectory.load(str(Path(args.clip_root) / f"{clip}.npz"))


def cmd_build(args):
    features, split_points, names = [], [0], []
    frequency = None
    for clip in args.clips:
        trajectory = _load_clip(args, clip)
        if frequency is None:
            frequency = float(trajectory.info.frequency)
        elif float(trajectory.info.frequency) != frequency:
            raise ValueError(
                f"{clip}: frequency {trajectory.info.frequency} != {frequency}"
            )
        clip_features = canonical_frame_features(trajectory)
        features.append(clip_features)
        split_points.append(split_points[-1] + clip_features.shape[0])
        names.append(clip)
        print(f"[build] {clip}: {clip_features.shape[0]} frames", flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        features=np.concatenate(features, axis=0),
        split_points=np.asarray(split_points, dtype=np.int64),
        clip_names=np.asarray(names),
        frequency=np.float64(frequency),
    )
    print(f"[build] -> {out} ({split_points[-1]} frames, {frequency} Hz)")


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


class Encoder(nn.Module):
    hidden: tuple[int, ...]
    code_dim: int

    @nn.compact
    def __call__(self, x):
        for width in self.hidden:
            x = nn.elu(nn.Dense(width)(x))
        return nn.Dense(self.code_dim)(x)


class Decoder(nn.Module):
    hidden: tuple[int, ...]
    out_dim: int

    @nn.compact
    def __call__(self, z):
        for width in self.hidden:
            z = nn.elu(nn.Dense(width)(z))
        return nn.Dense(self.out_dim)(z)


class AutoEncoder(nn.Module):
    hidden: tuple[int, ...]
    code_dim: int
    out_dim: int
    levels: tuple[int, ...] | None  # None => continuous bottleneck control

    def setup(self):
        self.encoder = Encoder(self.hidden, self.code_dim)
        self.decoder = Decoder(tuple(reversed(self.hidden)), self.out_dim)
        self.fsq = FSQ(list(self.levels)) if self.levels is not None else None

    def __call__(self, x):
        z = self.encoder(x)
        code = self.fsq.quantize(z) if self.fsq is not None else jnp.tanh(z)
        return self.decoder(code), code


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


def _split_by_clip(split_points, clip_names, val_clips):
    train_idx, val_idx = [], []
    for c, name in enumerate(clip_names):
        rows = np.arange(split_points[c], split_points[c + 1])
        (val_idx if name in val_clips else train_idx).append(rows)
    if not val_idx:
        raise ValueError(f"No validation clips matched {val_clips}")
    if not train_idx:
        raise ValueError("All clips went to validation")
    return np.concatenate(train_idx), np.concatenate(val_idx)


def _per_group_mse(reconstruction, target, window):
    """MSE per canonical feature group, averaged over window steps."""
    groups = feature_group_slices()
    frame_dim = FRAME_FEATURE_DIM
    per_dim = np.mean((reconstruction - target) ** 2, axis=0)  # (window*D,)
    per_dim = per_dim.reshape(window, frame_dim).mean(axis=0)  # (D,)
    return {
        name: float(per_dim[sl].mean()) for name, sl in groups.items()
    }


def cmd_train(args):
    print(f"[train] backend={jax.default_backend()} dataset={args.dataset}", flush=True)
    payload = np.load(args.dataset, allow_pickle=False)
    features = payload["features"]
    split_points = payload["split_points"]
    clip_names = [str(x) for x in payload["clip_names"]]
    print(f"[train] features {features.shape} clips={clip_names}", flush=True)
    val_clips = set(args.val_clips)
    unknown = val_clips - set(clip_names)
    if unknown:
        raise ValueError(f"--val-clips not in dataset: {sorted(unknown)}")

    windows = future_window(features, split_points, args.window, args.stride)
    train_rows, val_rows = _split_by_clip(split_points, clip_names, val_clips)
    stats = NormalizationStats.fit(windows[train_rows])
    x_train = stats.apply(windows[train_rows])
    x_val = stats.apply(windows[val_rows])

    levels = None if args.continuous else tuple(args.levels)
    model = AutoEncoder(
        hidden=tuple(args.hidden),
        code_dim=len(args.levels),
        out_dim=x_train.shape[1],
        levels=levels,
    )
    rng = jax.random.PRNGKey(args.seed)
    params = model.init(rng, jnp.zeros((1, x_train.shape[1])))
    tx = optax.adamw(args.lr, weight_decay=1e-5)
    opt_state = tx.init(params)

    @jax.jit
    def step(params, opt_state, batch):
        def loss_fn(p):
            reconstruction, _ = model.apply(p, batch)
            return jnp.mean((reconstruction - batch) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    @jax.jit
    def evaluate(params, batch):
        reconstruction, code = model.apply(params, batch)
        return jnp.mean((reconstruction - batch) ** 2), reconstruction, code

    data = jnp.asarray(x_train)
    n = data.shape[0]
    key = jax.random.PRNGKey(args.seed + 1)
    started = time.perf_counter()
    history = []
    steps_per_epoch = max(1, n // args.batch_size)
    for epoch in range(args.epochs):
        key, perm_key = jax.random.split(key)
        order = jax.random.permutation(perm_key, n)
        epoch_loss = 0.0
        for i in range(steps_per_epoch):
            batch = data[order[i * args.batch_size : (i + 1) * args.batch_size]]
            params, opt_state, loss = step(params, opt_state, batch)
            epoch_loss += float(loss)
        val_loss = float(evaluate(params, jnp.asarray(x_val))[0])
        history.append(
            {"epoch": epoch, "train": epoch_loss / steps_per_epoch, "val": val_loss}
        )
        print(
            f"[train] epoch {epoch:3d} train={epoch_loss / steps_per_epoch:.5f} "
            f"val={val_loss:.5f}",
            flush=True,
        )
    train_seconds = time.perf_counter() - started

    # ---- evaluation report -------------------------------------------------
    val_loss, reconstruction, val_codes = evaluate(params, jnp.asarray(x_val))
    reconstruction = np.asarray(reconstruction)
    val_codes = np.asarray(val_codes)
    constant_prediction = x_train.mean(axis=0, keepdims=True)
    constant_mse = float(np.mean((x_val - constant_prediction) ** 2))

    report = {
        "val_mse": float(val_loss),
        "constant_predictor_mse": constant_mse,
        "val_over_constant": float(val_loss) / max(constant_mse, 1e-12),
        "per_group_mse": _per_group_mse(reconstruction, x_val, args.window),
        "train_seconds": train_seconds,
        "history": history,
        "quantized": levels is not None,
    }

    if levels is not None:
        fsq = FSQ(list(levels))
        _, train_codes = evaluate(params, jnp.asarray(x_train))[1:]
        all_codes = np.concatenate([np.asarray(train_codes), val_codes], axis=0)
        indices = np.asarray(fsq.codes_to_indexes(jnp.asarray(all_codes)))
        half_width = np.asarray(levels) // 2
        discrete = np.round(all_codes * half_width + half_width).astype(np.int64)
        utilization = [
            {
                "level_count": int(levels[d]),
                "levels_used": int(np.unique(discrete[:, d]).size),
                "saturated_fraction": float(
                    np.mean(
                        (discrete[:, d] == 0) | (discrete[:, d] == levels[d] - 1)
                    )
                ),
            }
            for d in range(len(levels))
        ]
        counts = np.bincount(indices, minlength=fsq.codebook_size).astype(np.float64)
        probabilities = counts / counts.sum()
        nonzero = probabilities[probabilities > 0]
        # temporal stability on the validation rows (contiguous by clip)
        changes = float(np.mean(indices[len(train_codes) + 1 :]
                                != indices[len(train_codes) : -1]))
        report.update(
            {
                "codebook_size": fsq.codebook_size,
                "unique_codes": int((counts > 0).sum()),
                "code_entropy_bits": float(-(nonzero * np.log2(nonzero)).sum()),
                "per_dim_utilization": utilization,
                "val_code_change_rate": changes,
            }
        )

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "params.msgpack").write_bytes(flax.serialization.to_bytes(params))
    np.savez(
        out / "normalization.npz", mean=stats.mean, std=stats.std
    )
    (out / "metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "experiment": "fsq_motion_design_a",
                "dataset": str(args.dataset),
                "clips": clip_names,
                "val_clips": sorted(val_clips),
                "window": args.window,
                "stride": args.stride,
                "levels": list(args.levels),
                "continuous_control": bool(args.continuous),
                "hidden": list(args.hidden),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "seed": args.seed,
                "jax_backend": jax.default_backend(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    # completion marker LAST
    (out / "DONE").write_text("ok\n", encoding="utf-8")
    print(f"[train] -> {out} val_mse={float(val_loss):.5f} "
          f"constant={constant_mse:.5f}")


# ---------------------------------------------------------------------------
# encode — canonical token cache for the RL trajectory layout
# ---------------------------------------------------------------------------


def cmd_encode(args):
    payload = np.load(args.dataset, allow_pickle=False)
    features = payload["features"]
    split_points = payload["split_points"]
    clip_names = [str(x) for x in payload["clip_names"]]

    run_dir = Path(args.model_dir)
    stats_payload = np.load(run_dir / "normalization.npz")
    stats = NormalizationStats(stats_payload["mean"], stats_payload["std"])
    manifest = json.loads((run_dir / "manifest.json").read_text())
    model = AutoEncoder(
        hidden=tuple(manifest["hidden"]),
        code_dim=len(manifest["levels"]),
        out_dim=int(stats.mean.shape[0]),
        levels=None if manifest["continuous_control"] else tuple(manifest["levels"]),
    )
    template = model.init(
        jax.random.PRNGKey(0), jnp.zeros((1, stats.mean.shape[0]))
    )
    params = flax.serialization.from_bytes(
        template, (run_dir / "params.msgpack").read_bytes()
    )

    # windows: "clip:start:frames" — must match the RL reference crops in order
    codes_list, cache_splits = [], [0]
    for spec in args.windows:
        clip, start, frames = spec.split(":")
        start, frames = int(start), int(frames)
        c = clip_names.index(clip)
        clip_start, clip_stop = int(split_points[c]), int(split_points[c + 1])
        if clip_start + start + frames > clip_stop:
            raise ValueError(f"{spec}: window exceeds clip length")
        rows = slice(clip_start + start, clip_start + start + frames)
        clip_features = features[rows]
        local_windows = future_window(
            clip_features,
            np.asarray([0, frames]),
            manifest["window"],
            manifest["stride"],
        )
        _, code = model.apply(params, jnp.asarray(stats.apply(local_windows)))
        codes_list.append(np.asarray(code, dtype=np.float32))
        cache_splits.append(cache_splits[-1] + frames)
        print(f"[encode] {spec}: codes {codes_list[-1].shape}", flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        codes=np.concatenate(codes_list, axis=0),
        split_points=np.asarray(cache_splits, dtype=np.int32),
        windows=np.asarray(args.windows),
    )
    print(f"[encode] -> {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build")
    p.add_argument("--loader", choices=["npz", "factory"], default="factory")
    p.add_argument("--clip-root", default=None,
                   help="npz loader only: directory of converted clips")
    p.add_argument("--env-name", default="UnitreeH1",
                   help="factory loader: canonical source environment")
    p.add_argument("--clips", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(fn=cmd_build)

    p = sub.add_parser("train")
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--val-clips", nargs="+", required=True)
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--levels", type=int, nargs="+", default=list(DEFAULT_LEVELS))
    p.add_argument("--continuous", action="store_true",
                   help="equally sized continuous bottleneck control (no FSQ)")
    p.add_argument("--hidden", type=int, nargs="+", default=[256, 128])
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("encode")
    p.add_argument("--dataset", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--windows", nargs="+", required=True,
                   help="clip:start:frames, in RL trajectory order")
    p.add_argument("--output", required=True)
    p.set_defaults(fn=cmd_encode)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
