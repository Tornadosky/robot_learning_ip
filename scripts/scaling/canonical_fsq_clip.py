"""Canonical motion-token FSQ (our design B) applied to RL clips.

The third arm of the A/B/C: unlike the khaendler variant (per-joint,
embodiment-conditioned tokens — each robot's own reference encoded), here the
token IS the motion: one FSQ code stream per canonical timestep, computed from
the SOURCE (H1) clip's task-space features (root height/attitude/velocities +
4 end-effector positions in the root frame, the verified design-A feature
set from fsq_motion.py), decoded per robot through the same URMA-style
decoder conditioned on that robot's 47-dim joint descriptions.  H1 and G1
therefore share the exact same z for the same moment of the dance.

Everything else mirrors khaendler_fsq_clip.py so the three RL arms differ in
ONE thing only: which clip npz the tracking pipeline reads.

Subcommands:
  build       — (local) dataset npz + vendored flax modules -> portable bundle
  train       — (anywhere with jax/flax/optax; e.g. Viper CPU) fit the model
  reconstruct — (local) write reconstructed clips for both robots from the
                SHARED token stream + metrics

The vendored `nn_vendor/` package (decoder/fsq/scaled_width from the
loco-mujoco integration branch) keeps `train` free of any loco-mujoco or
mujoco dependency.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[2]
SCRIPTS = WORKSPACE / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

DEFAULT_CLIP_DIR = WORKSPACE / "external_data" / "amass_converted" / "LAFAN1"
DEFAULT_LEVELS = (8, 5, 5, 5)  # the verified design-A codebook (1000 codes)

CANONICAL_EE_SITES = (
    "left_foot_mimic",
    "right_foot_mimic",
    "left_hand_mimic",
    "right_hand_mimic",
)


# 2026-08-26: every clip carries 16 mimic sites on BOTH robots, knees and
# elbows among them. The original 4-end-effector set was the reason the encoder
# could not see the joints its reconstruction was worst at; --sites makes the
# set a parameter so that claim is testable rather than structural.
CANONICAL_SITES_RICH = (
    "left_hip_mimic", "left_knee_mimic", "left_foot_mimic",
    "right_hip_mimic", "right_knee_mimic", "right_foot_mimic",
    "upper_body_mimic", "head_mimic",
    "left_shoulder_mimic", "left_elbow_mimic", "left_hand_mimic",
    "right_shoulder_mimic", "right_elbow_mimic", "right_hand_mimic",
)
SITE_SETS = {"ee4": CANONICAL_EE_SITES, "rich14": CANONICAL_SITES_RICH}


def canonical_features_from_npz(d, include_source_joints: bool = False,
                                sites=CANONICAL_EE_SITES) -> np.ndarray:
    """(T, 22) canonical task-space features straight from a clip npz.

    Same math as fsq_motion.canonical_frame_features, but reading the npz's
    stored per-frame site_xpos instead of a Trajectory object.

    include_source_joints appends the SOURCE robot's joint angles (T, J).
    Measured 2026-08-22: the 22 task-space dims underdetermine the pose —
    canonical reconstruction is pinned at ~0.3 rad across data size, epochs
    AND a 1M-code codebook (7445/9025 unique codes used, RMSE unchanged), with
    the error concentrated in elbows/knees, the joints task-space cannot see.
    The token stays embodiment-portable: consumers only ever see z.
    """
    from scaling.fsq_motion import _quat_to_rotmat

    qpos = np.asarray(d["qpos"], dtype=np.float64)
    qvel = np.asarray(d["qvel"], dtype=np.float64)
    site_names = [str(n) for n in d["site_names"]]
    site_xpos = np.asarray(d["site_xpos"], dtype=np.float64)
    missing = [s for s in sites if s not in site_names]
    if missing:
        raise ValueError(f"clip lacks canonical sites: {missing}")
    ee_idx = [site_names.index(s) for s in sites]

    root_pos = qpos[:, 0:3]
    rot = _quat_to_rotmat(qpos[:, 3:7])
    world_to_root = np.transpose(rot, (0, 2, 1))
    gravity = np.einsum("nij,nj->ni", world_to_root, np.tile([0.0, 0.0, -1.0], (qpos.shape[0], 1)))
    linvel = np.einsum("nij,nj->ni", world_to_root, qvel[:, 0:3])
    angvel = qvel[:, 3:6]
    ee_rel = site_xpos[:, ee_idx, :] - root_pos[:, None, :]
    ee_in_root = np.einsum("nij,nkj->nki", world_to_root, ee_rel)
    parts = [root_pos[:, 2:3], gravity, linvel, angvel,
             ee_in_root.reshape(qpos.shape[0], -1)]
    if include_source_joints:
        parts.append(qpos[:, 7:])
    return np.concatenate(parts, axis=-1).astype(np.float32)


def feature_windows(features: np.ndarray, lookahead: int) -> np.ndarray:
    """(T, (lookahead+1)*D) windows [t, t, t+1..t+lookahead-1] (clamped),
    mirroring khaendler build_windows row semantics."""
    T = features.shape[0]
    rows = [features]
    for k in range(lookahead):
        idx = np.minimum(np.arange(T) + k, T - 1)
        rows.append(features[idx])
    return np.stack(rows, axis=1).reshape(T, -1)


# ---------------------------------------------------------------------------
# build (local)
# ---------------------------------------------------------------------------


def cmd_build(args):
    from scaling.khaendler_fsq_clip import (
        build_robot_assets,
        load_clip_joints,
        build_windows,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Windows are built PER CLIP and then concatenated, so lookahead clamping
    # stays clip-local and no window straddles a clip boundary.
    clips = list(args.clip)
    fwins, frames_per_clip = [], []
    feature_dim = None
    for clip in clips:
        src = np.load(Path(args.clip_dir) / args.source_robot / clip, allow_pickle=True)
        features = canonical_features_from_npz(
            src, include_source_joints=args.include_source_joints,
            sites=SITE_SETS[args.sites])
        feature_dim = int(features.shape[1])
        frames_per_clip.append(int(features.shape[0]))
        fwins.append(feature_windows(features, args.lookahead))
    fwin = np.concatenate(fwins, axis=0)

    data = {"features": fwin.astype(np.float32)}
    meta = {
        "source_robot": args.source_robot,
        "clip": ",".join(clips),
        "clips": clips,
        "frames_per_clip": frames_per_clip,
        "clip_dir": str(args.clip_dir),
        "lookahead": args.lookahead,
        "feature_dim": feature_dim,
        "sites": args.sites,
        "robots": args.robots,
        "frames": int(fwin.shape[0]),
    }
    for robot in args.robots:
        desc, joint_names = build_robot_assets(robot)
        targets = []
        for ci, clip in enumerate(clips):
            qpos, qvel, _ = load_clip_joints(Path(args.clip_dir) / robot / clip, joint_names)
            if qpos.shape[0] != frames_per_clip[ci]:
                raise ValueError(
                    f"{robot} {clip} frame count {qpos.shape[0]} != source {frames_per_clip[ci]}")
            targets.append(build_windows(qpos, qvel, args.lookahead).astype(np.float32))
        data[f"{robot}_targets"] = np.concatenate(targets, axis=0)
        data[f"{robot}_desc"] = desc
        data[f"{robot}_joints"] = np.array(joint_names)
    np.savez_compressed(out / "dataset.npz", **data)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    vendor = out / "nn_vendor"
    vendor.mkdir(exist_ok=True)
    nn_src = WORKSPACE / "loco-mujoco" / "loco_mujoco" / "algorithms" / "autoencoder" / "nn"
    for f in ("decoder.py", "fsq.py", "scaled_width.py"):
        shutil.copy2(nn_src / f, vendor / f)
    (vendor / "__init__.py").write_text("")
    shutil.copy2(Path(__file__), out / "canonical_fsq_clip.py")
    print(f"bundle ready: {out} (dataset {fwin.shape}, robots {args.robots})")


# ---------------------------------------------------------------------------
# model (train + reconstruct share it)
# ---------------------------------------------------------------------------


def make_model(bundle_dir: Path, levels, latent_dim, lookahead, width):
    import flax.linen as nn

    sys.path.insert(0, str(bundle_dir))
    from nn_vendor.decoder import URMADecoder
    from nn_vendor.fsq import FSQ

    class CanonicalMotionAutoencoder(nn.Module):
        levels: tuple
        latent_dim: int
        lookahead: int
        width: float

        def setup(self):
            self.enc_hidden0 = nn.Dense(256)
            self.enc_hidden1 = nn.Dense(128)
            self.enc_out = nn.Dense(self.latent_dim)
            self.fsq = FSQ(levels=list(self.levels))
            self.decoder = URMADecoder(
                output_dim=2,
                n_step_lookahead=self.lookahead,
                softmax_temperature=1.0,
                softmax_temperature_min=0.01,
                stability_epsilon=1.0e-5,
                network_width_multiplier=self.width,
            )

        def encode(self, feature_window):
            import jax.numpy as jnp
            x = nn.gelu(self.enc_hidden0(feature_window))
            x = nn.gelu(self.enc_hidden1(x))
            z = self.enc_out(x)
            z_q, aux = self.fsq(z)
            return z_q, aux

        def __call__(self, feature_window, joint_description):
            import jax.numpy as jnp
            z_q, aux = self.encode(feature_window)
            # ONE motion token, broadcast to every joint of whichever robot
            # is being decoded — embodiment enters only via the description.
            n_joints = joint_description.shape[-2]
            joint_latent = jnp.broadcast_to(
                z_q[:, None, :], (z_q.shape[0], n_joints, z_q.shape[-1]))
            recon = self.decoder(joint_latent, joint_description, None)
            return recon, aux

    return CanonicalMotionAutoencoder(
        levels=tuple(levels), latent_dim=latent_dim, lookahead=lookahead, width=width)


# ---------------------------------------------------------------------------
# train (portable: jax + flax + optax + numpy only)
# ---------------------------------------------------------------------------


def cmd_train(args):
    import jax
    import jax.numpy as jnp
    import optax
    import flax.serialization
    from flax.training import train_state

    bundle = Path(args.bundle)
    meta = json.loads((bundle / "meta.json").read_text())
    ds = np.load(bundle / "dataset.npz", allow_pickle=True)
    robots = meta["robots"]
    lookahead = meta["lookahead"]

    features = ds["features"]
    mean, std = features.mean(0), features.std(0) + 1e-6
    features_n = (features - mean) / std
    T = features_n.shape[0]
    n_test = max(1, int(T * args.test_fraction))
    train_end = max(1, T - n_test - lookahead)

    # --levels is authoritative when given: latent_dim = len(levels). The
    # single-token codebook (1000 codes) is the canonical design's measured
    # bottleneck (repeatedly ~900/1000 codes used, RMSE stuck ~0.3 rad), so
    # scaling it is a first-class experiment, not a tweak.
    levels = tuple(args.levels) if args.levels else DEFAULT_LEVELS[: args.latent_dim]
    latent_dim = len(levels)
    model = make_model(bundle, levels, latent_dim, lookahead, args.width)

    rng = jax.random.PRNGKey(args.seed)
    r0 = robots[0]
    variables = model.init(rng, jnp.asarray(features_n[:2]),
                           jnp.asarray(np.broadcast_to(ds[f"{r0}_desc"], (2,) + ds[f"{r0}_desc"].shape)))
    steps_per_epoch = max(1, (train_end // args.batch_size) * len(robots))
    schedule = optax.cosine_decay_schedule(args.lr, args.epochs * steps_per_epoch, alpha=0.05)
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=variables["params"], tx=optax.adamw(schedule))

    @jax.jit
    def train_step(state, fwin, desc, target):
        def loss_fn(params):
            recon, _ = state.apply_fn({"params": params}, fwin, desc)
            return jnp.mean(jnp.square(recon - target))
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        return state.apply_gradients(grads=grads), loss

    @jax.jit
    def eval_step(state, fwin, desc, target):
        recon, _ = state.apply_fn({"params": state.params}, fwin, desc)
        return jnp.mean(jnp.square(recon - target))

    history = {"train_loss": [], "eval_loss": {r: [] for r in robots}}
    t0 = time.time()
    for epoch in range(args.epochs):
        batches = []
        for robot in robots:
            rng, prng = jax.random.split(rng)
            perm = np.asarray(jax.random.permutation(prng, train_end))
            nb = train_end // args.batch_size
            for i in range(nb):
                batches.append((robot, perm[i * args.batch_size:(i + 1) * args.batch_size]))
        rng, srng = jax.random.split(rng)
        order = np.asarray(jax.random.permutation(srng, len(batches)))
        losses = []
        for bi in order:
            robot, idx = batches[int(bi)]
            desc = np.broadcast_to(ds[f"{robot}_desc"], (idx.shape[0],) + ds[f"{robot}_desc"].shape)
            state, loss = train_step(state, jnp.asarray(features_n[idx]),
                                     jnp.asarray(desc),
                                     jnp.asarray(ds[f"{robot}_targets"][idx]))
            losses.append(float(loss))
        history["train_loss"].append(float(np.mean(losses)))
        for robot in robots:
            desc = np.broadcast_to(ds[f"{robot}_desc"], (n_test,) + ds[f"{robot}_desc"].shape)
            history["eval_loss"][robot].append(float(eval_step(
                state, jnp.asarray(features_n[T - n_test:]), jnp.asarray(desc),
                jnp.asarray(ds[f"{robot}_targets"][T - n_test:]))))
        if (epoch + 1) % args.log_every == 0 or epoch == args.epochs - 1:
            evals = " ".join(f"{r}={history['eval_loss'][r][-1]:.5f}" for r in robots)
            print(f"epoch {epoch + 1}/{args.epochs} train={history['train_loss'][-1]:.5f} "
                  f"eval[{evals}] ({time.time() - t0:.0f}s)", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "params.msgpack").write_bytes(flax.serialization.to_bytes(state.params))
    np.savez(out / "feature_norm.npz", mean=mean, std=std)
    (out / "config.json").write_text(json.dumps({
        "levels": list(levels), "latent_dim": latent_dim,
        "lookahead": lookahead, "width": args.width, "epochs": args.epochs,
        "batch_size": args.batch_size, "lr": args.lr, "seed": args.seed,
        "test_fraction": args.test_fraction,
        "final_train_loss": history["train_loss"][-1],
        "final_eval_loss": {r: history["eval_loss"][r][-1] for r in robots},
    }, indent=2))
    (out / "history.json").write_text(json.dumps(history, indent=2))
    print(f"saved canonical tokenizer to {out}")


# ---------------------------------------------------------------------------
# reconstruct (local)
# ---------------------------------------------------------------------------


def cmd_reconstruct(args):
    import jax
    import jax.numpy as jnp
    import flax.serialization

    bundle = Path(args.bundle)
    tok = Path(args.tokenizer)
    dst_dir = Path(args.out)
    meta = json.loads((bundle / "meta.json").read_text())
    config = json.loads((tok / "config.json").read_text())
    ds = np.load(bundle / "dataset.npz", allow_pickle=True)
    lookahead = meta["lookahead"]
    robots = meta["robots"]

    norm = np.load(tok / "feature_norm.npz")
    features_n = (ds["features"] - norm["mean"]) / norm["std"]
    T = features_n.shape[0]

    model = make_model(bundle, tuple(config["levels"]), config["latent_dim"],
                       lookahead, config["width"])
    r0 = robots[0]
    params = flax.serialization.from_bytes(
        model.init(jax.random.PRNGKey(0), jnp.asarray(features_n[:2]),
                   jnp.asarray(np.broadcast_to(ds[f"{r0}_desc"], (2,) + ds[f"{r0}_desc"].shape)))["params"],
        (tok / "params.msgpack").read_bytes())

    @jax.jit
    def apply(fwin, desc):
        return model.apply({"params": params}, fwin, desc)

    report = {}
    zq_all = None
    for robot in robots:
        desc0 = ds[f"{robot}_desc"]
        joint_names = [str(n) for n in ds[f"{robot}_joints"]]
        rec_rows, zq_rows = [], []
        for start in range(0, T, args.batch_size):
            fwin = jnp.asarray(features_n[start:start + args.batch_size])
            desc = jnp.broadcast_to(jnp.asarray(desc0), (fwin.shape[0],) + desc0.shape)
            recon, aux = apply(fwin, desc)
            rec_rows.append(np.asarray(recon[:, 1]))
            zq_rows.append(np.asarray(aux["z_q"], dtype=np.float32))
        rec = np.concatenate(rec_rows, axis=0)
        zq_all = np.concatenate(zq_rows, axis=0)  # identical across robots

        from scaling.khaendler_fsq_clip import load_clip_joints
        src = Path(meta["clip_dir"]) / robot / meta["clip"]
        qpos, qvel, perm = load_clip_joints(src, joint_names)
        rec_qpos, rec_qvel = rec[..., 0], rec[..., 1]
        inv = np.argsort(perm)

        d = dict(np.load(src, allow_pickle=True))
        new_qpos = np.array(d["qpos"]); new_qpos[:, 7:] = rec_qpos[:, inv]
        new_qvel = np.array(d["qvel"]); new_qvel[:, 6:] = rec_qvel[:, inv]
        d["qpos"] = new_qpos.astype(np.float32)
        d["qvel"] = new_qvel.astype(np.float32)
        out_robot = dst_dir / robot
        out_robot.mkdir(parents=True, exist_ok=True)
        np.savez(out_robot / meta["clip"], **d)

        err = rec_qpos - qpos
        report[robot] = {
            "frames": int(T),
            "qpos_rmse_rad": float(np.sqrt(np.mean(err ** 2))),
            "qpos_max_abs_err_rad": float(np.max(np.abs(err))),
            "per_joint_rmse_rad": {joint_names[j]: float(np.sqrt(np.mean(err[:, j] ** 2)))
                                   for j in range(len(joint_names))},
            "qvel_rmse": float(np.sqrt(np.mean((rec_qvel - qvel) ** 2))),
        }
        print(f"{robot}: qpos RMSE {report[robot]['qpos_rmse_rad']:.4f} rad, "
              f"max {report[robot]['qpos_max_abs_err_rad']:.4f} rad")

    report["unique_codes_used"] = int(np.unique(zq_all, axis=0).shape[0])
    report["code_slots"] = int(T)
    np.savez(dst_dir / "z_canonical.npz", z_q=zq_all)

    if args.copy_other_robots:
        src_dir = Path(meta["clip_dir"])
        for sub in src_dir.iterdir():
            if sub.is_dir() and sub.name not in robots and (sub / meta["clip"]).exists():
                out_robot = dst_dir / sub.name
                out_robot.mkdir(parents=True, exist_ok=True)
                shutil.copy2(sub / meta["clip"], out_robot / meta["clip"])
    (dst_dir / "reconstruction_report.json").write_text(json.dumps(report, indent=2))
    print(f"reconstructed clips in {dst_dir} "
          f"({report['unique_codes_used']}/{T} unique canonical codes)")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--robots", nargs="+", default=["UnitreeH1", "UnitreeG1"])
    b.add_argument("--source-robot", default="UnitreeH1")
    b.add_argument("--clip", nargs="+", default=["dance2_subject4.npz"])
    b.add_argument("--clip-dir", default=str(DEFAULT_CLIP_DIR))
    b.add_argument("--lookahead", type=int, default=10)
    b.add_argument("--include-source-joints", action="store_true", default=False)
    b.add_argument("--sites", default="ee4", choices=sorted(SITE_SETS),
                   help="which task-space sites the token encodes. ee4 = the two "
                        "feet and two hands (the 22-dim design that reconstructs "
                        "at 0.37 rad); rich14 adds hips, knees, shoulders, elbows, "
                        "chest and head -- the joints ee4 cannot see.")
    b.add_argument("--out", required=True)
    b.set_defaults(fn=cmd_build)

    t = sub.add_parser("train")
    t.add_argument("--bundle", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--latent-dim", type=int, default=4)
    t.add_argument("--levels", type=int, nargs="+", default=None,
                   help="FSQ levels per latent dim; overrides --latent-dim")
    t.add_argument("--width", type=float, default=1.0)
    t.add_argument("--epochs", type=int, default=500)
    t.add_argument("--batch-size", type=int, default=256)
    t.add_argument("--lr", type=float, default=1.5e-3)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--test-fraction", type=float, default=0.1)
    t.add_argument("--log-every", type=int, default=20)
    t.set_defaults(fn=cmd_train)

    r = sub.add_parser("reconstruct")
    r.add_argument("--bundle", required=True)
    r.add_argument("--tokenizer", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--batch-size", type=int, default=512)
    r.add_argument("--copy-other-robots", action="store_true", default=True)
    r.set_defaults(fn=cmd_reconstruct)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
