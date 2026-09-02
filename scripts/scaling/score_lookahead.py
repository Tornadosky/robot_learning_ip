"""F0 -- score what the token has NEVER been scored on: its LOOKAHEAD.

Every tokenizer number this project owns was computed from `recon[:, 1]`, and
because `build_windows` duplicates frame t (rows 0 AND 1 are both frame t),
row 1 IS the current frame. So the enc/dec cells, the width sweep, the epoch
sweep and every gate measured how well the token reproduces the frame it
already has, and said nothing about the 9 future frames it was trained to
predict.

This script reports reconstruction RMSE PER WINDOW ROW: row 1 ("reproduces
now") separately from rows 2..N ("predicts next", offsets +1..+9 at 40 fps).
It changes NO training code and does NOT fix the build_windows duplicate --
fixing the bug first would silently redefine row 1 as t+1 and change the
meaning of every historical number (plan §3.2, warning).

Row semantics (verified against build_windows): row 0 = frame t,
row r (r>=1) = frame t+r-1 clamped at the clip end. Frames whose target would
be clamped are EXCLUDED from that row's RMSE, so the tail of a clip cannot
flatter the future rows with copies of the present.

Registered decision rule (plan §3.2 F0):
  * future-row error ~ current-row error  -> the pooled token already carries
    the window; the mean-pool is NOT the bottleneck; F1 is cancelled.
  * future-row error much worse, flat across checkpoints -> the pool IS the
    bottleneck -> F1 (replace the pool) is worth training.

Runs in the Windows .venv on CPU, like the tokenizer fits themselves:
  .venv/Scripts/python.exe scripts/scaling/score_lookahead.py \
      --tokenizer experiments/fsq_khaendler/kevin_tokenizer_denoise2 [...] \
      --out experiments/fsq_khaendler/f0_lookahead.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

_spec = importlib.util.spec_from_file_location(
    "khaendler_fsq_clip", HERE / "khaendler_fsq_clip.py")
kfc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kfc)


def score_tokenizer(tok_dir: Path, clip_dir: Path, batch_size: int):
    import jax
    import jax.numpy as jnp
    import flax.serialization

    config = json.loads((tok_dir / "config.json").read_text())
    lookahead = config["lookahead"]

    class _A:
        latent_dim = config["latent_dim"]
        fsq_levels = config["fsq_levels"]
        width = config["width"]

    model = kfc.make_model(_A, lookahead)
    desc_store = np.load(tok_dir / "descriptions.npz", allow_pickle=True)

    clips = kfc.resolve_clips(config.get("clips") or config.get("clip"))
    heldout = set(config.get("heldout_clips") or [])
    test_fraction = float(config.get("test_fraction", 0.1))

    out = {"config_epochs": config.get("epochs"), "robots": {}}
    for robot in config["robots"]:
        desc = desc_store[f"{robot}_desc"]
        joint_names = [str(n) for n in desc_store[f"{robot}_joints"]]
        dummy = np.zeros((1, lookahead + 1, desc.shape[0], 2), dtype=np.float32)
        params = flax.serialization.from_bytes(
            model.init(
                jax.random.PRNGKey(0),
                jnp.asarray(dummy),
                jnp.asarray(desc[None]),
                jnp.zeros((1, 1), dtype=jnp.float32),
            )["params"],
            (tok_dir / "params.msgpack").read_bytes(),
        )

        @jax.jit
        def apply(w, d):
            return model.apply({"params": params}, w, d,
                               jnp.zeros((w.shape[0], 1)))

        per_clip = {}
        for clip in clips:
            src = clip_dir / robot / clip
            if not src.exists():
                continue
            qpos, qvel, _ = kfc.load_clip_joints(src, joint_names)
            windows = kfc.build_windows(qpos, qvel, lookahead)
            T = qpos.shape[0]

            rec_rows = []
            for start in range(0, T, batch_size):
                w = jnp.asarray(windows[start:start + batch_size])
                d = jnp.broadcast_to(jnp.asarray(desc),
                                     (w.shape[0],) + desc.shape)
                recon, _aux = apply(w, d)
                rec_rows.append(np.asarray(recon))
            rec = np.concatenate(rec_rows, axis=0)  # (T, lookahead+1, J, 2)

            # Frame split: a fitted clip's held-out frames are its test TAIL
            # (same convention as training); a clip in heldout_clips is
            # held out in full.
            if clip in heldout:
                eval_mask = np.ones(T, dtype=bool)
            else:
                n_test = max(1, int(T * test_fraction))
                eval_mask = np.zeros(T, dtype=bool)
                eval_mask[max(1, T - n_test):] = True

            rows = {}
            for r in range(1, lookahead + 1):
                offset = r - 1          # row r decodes frame t + r - 1
                tgt_idx = np.arange(T) + offset
                valid = (tgt_idx <= T - 1) & eval_mask   # exclude clamped
                if valid.sum() < 8:
                    continue
                err = rec[valid, r, :, 0] - qpos[tgt_idx[valid]]
                rows[f"row{r}_offset+{offset}"] = float(
                    np.sqrt(np.mean(err ** 2)))
            future = [v for k, v in rows.items()
                      if not k.startswith("row1_")]
            per_clip[clip] = {
                "held_out_clip": clip in heldout,
                "eval_frames": int(eval_mask.sum()),
                "qpos_rmse_current": rows.get("row1_offset+0"),
                "qpos_rmse_future_mean": float(np.mean(future)) if future else None,
                "per_row": rows,
            }
        out["robots"][robot] = per_clip
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokenizer", nargs="+", required=True,
                   help="tokenizer dirs (config.json + params.msgpack)")
    p.add_argument("--clip-dir", default=str(kfc.DEFAULT_CLIP_DIR))
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    report = {}
    for tok in args.tokenizer:
        tok_dir = Path(tok)
        name = tok_dir.name
        print(f"=== {name} ===", flush=True)
        report[name] = score_tokenizer(tok_dir, Path(args.clip_dir),
                                       args.batch_size)
        for robot, per_clip in report[name]["robots"].items():
            for clip, row in per_clip.items():
                cur, fut = row["qpos_rmse_current"], row["qpos_rmse_future_mean"]
                if cur is None or fut is None:
                    continue
                tag = " HELD-OUT" if row["held_out_clip"] else ""
                print(f"  {robot:12s} {clip:28s}{tag:9s} "
                      f"now {cur:.4f}  future {fut:.4f}  "
                      f"ratio {fut / max(cur, 1e-9):.2f}x", flush=True)

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
