"""Per-joint FSQ motion tokenizer (khaendler autoencoder) applied to RL clips.

Trains the colleague's UniversalMotionAutoencoder (loco-mujoco integration
branch: per-joint encoder -> FSQ -> URMA decoder conditioned on 47-dim joint
descriptions) directly on the converted-LAFAN1 clip npz frames that the
loco_mjx/urma2 tracking pipeline consumes, then writes FSQ-RECONSTRUCTED
copies of those clips in the identical npz format.

The RL A/B is then: tracking_clip on the original clips (baseline) vs
tracking_clip on the reconstructed clips (FSQ arm) — everything downstream
(loader, axis mapping, per-variant fit, FK reward) is byte-identical code.

Window layout mirrors his MultiEnvironment.joint_obs exactly:
row 0 = current frame t, rows 1..N = goal frames t .. t+N-1 (clamped at clip
end), each entry (qpos_j, qvel_j) in ACTUATOR order, raw units (his training
uses no normalization).  Reconstructed frame t = decoder output row 1 (the
goal channel for frame t — the quantity the RL command serves as reference).

Subcommands:
  train        — fit the autoencoder on one or more robots' clips
  reconstruct  — write reconstructed npz clips + z_q sidecar + metrics

Runs in the Windows .venv (CPU jax is fine; the model is small).
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

import jax
import jax.numpy as jnp
import flax.serialization

WORKSPACE = Path(__file__).resolve().parents[2]

DEFAULT_CLIP_DIR = WORKSPACE / "external_data" / "amass_converted" / "LAFAN1"

# loco-mujoco env name per clip subdirectory name.
ROBOT_ENVS = {
    "UnitreeH1": "UnitreeH1",
    "UnitreeG1": "UnitreeG1",
}


# ---------------------------------------------------------------------------
# Robot assets: model + his 47-dim joint descriptions
# ---------------------------------------------------------------------------


def build_robot_assets(robot: str):
    """Compile the loco-mujoco model and compute the colleague's joint
    descriptions at the nominal pose.

    His ``_init_internal_state`` requires the two IMU sensors; the plain
    RLFactory models ship without sensors, so they are injected into the
    mjspec at the pelvis_mimic site (matching his comment that the IMU is
    attached there) before compiling.  Only the description vector is used —
    the general observation (which is what actually reads the IMU) is
    discarded, matching the decoder, which ignores general_state.
    """
    import mujoco
    from loco_mujoco.task_factories import RLFactory
    from loco_mujoco.environments.mulitenvironment import MultiEnvironment

    made = RLFactory.make(ROBOT_ENVS[robot])
    env = made[0] if isinstance(made, tuple) else made
    spec = env.mjspec

    have = {s.name for s in spec.sensors}
    for name, stype in (
        ("imu_linear_velocity", mujoco.mjtSensor.mjSENS_FRAMELINVEL),
        ("imu_angular_velocity", mujoco.mjtSensor.mjSENS_FRAMEANGVEL),
    ):
        if name not in have:
            s = spec.add_sensor()
            s.name = name
            s.type = stype
            s.objtype = mujoco.mjtObj.mjOBJ_SITE
            s.objname = "pelvis_mimic"

    model = spec.compile()
    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    internal = MultiEnvironment._init_internal_state(None, model)
    # The method is jit-wrapped (static self); MjModel is not a pytree, so
    # call the eager underlying function instead.
    compute = MultiEnvironment.compute_observations_and_joint_descriptions.__wrapped__
    _, descriptions = compute(None, model, data, internal)
    descriptions = np.asarray(descriptions, dtype=np.float32)

    actuator_joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(model.actuator_trnid[a, 0]))
        for a in range(model.nu)
    ]
    return descriptions, actuator_joint_names


# ---------------------------------------------------------------------------
# Clip windows
# ---------------------------------------------------------------------------


def load_clip_joints(npz_path: Path, actuator_joint_names: list[str]):
    """(T, J) qpos / qvel joint columns of one clip, permuted to actuator order."""
    d = np.load(npz_path, allow_pickle=True)
    joint_names = [str(n) for n in d["joint_names"]]
    if joint_names[0] != "root":
        raise ValueError(f"{npz_path}: expected joint_names[0]=='root', got {joint_names[0]}")
    clip_joints = joint_names[1:]
    if sorted(clip_joints) != sorted(actuator_joint_names):
        raise ValueError(
            f"{npz_path}: clip joints do not biject onto actuators.\n"
            f"clip only: {sorted(set(clip_joints) - set(actuator_joint_names))}\n"
            f"model only: {sorted(set(actuator_joint_names) - set(clip_joints))}"
        )
    perm = np.array([clip_joints.index(n) for n in actuator_joint_names], dtype=np.int64)
    qpos = np.asarray(d["qpos"], dtype=np.float32)[:, 7:][:, perm]
    qvel = np.asarray(d["qvel"], dtype=np.float32)[:, 6:][:, perm]
    return qpos, qvel, perm


def build_windows(qpos: np.ndarray, qvel: np.ndarray, lookahead: int) -> np.ndarray:
    """(T, lookahead+1, J, 2) windows: row 0 = frame t, rows 1..N = frames
    t..t+N-1 clamped at the clip end (his goal-clamp semantics)."""
    T = qpos.shape[0]
    frames = np.stack([qpos, qvel], axis=-1)  # (T, J, 2)
    rows = [frames]
    for k in range(lookahead):
        idx = np.minimum(np.arange(T) + k, T - 1)
        rows.append(frames[idx])
    return np.stack(rows, axis=1)


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


def make_model(args, n_step_lookahead: int):
    from loco_mujoco.algorithms.autoencoder import UniversalMotionAutoencoder

    return UniversalMotionAutoencoder(
        input_dim=2,
        n_step_lookahead=n_step_lookahead,
        latent_dim=args.latent_dim,
        levels=tuple([args.fsq_levels] * args.latent_dim),
        softmax_temperature=1.0,
        softmax_temperature_min=0.01,
        stability_epsilon=1.0e-5,
        encoder_network_width_multiplier=args.width,
        decoder_network_width_multiplier=args.width,
    )


def cmd_train(args):
    from loco_mujoco.algorithms.autoencoder.train.step import (
        create_train_state,
        train_step,
        eval_step,
    )

    rng = jax.random.PRNGKey(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    per_robot = {}
    for robot in args.robots:
        desc, joint_names = build_robot_assets(robot)
        qpos, qvel, _ = load_clip_joints(Path(args.clip_dir) / robot / args.clip, joint_names)
        windows = build_windows(qpos, qvel, args.lookahead)
        T = windows.shape[0]
        n_test = max(1, int(T * args.test_fraction))
        # Temporal tail split; drop `lookahead` frames before the boundary so
        # no training window peeks into the test tail.
        train_end = max(1, T - n_test - args.lookahead)
        per_robot[robot] = {
            "desc": desc,
            "joints": joint_names,
            "train": windows[:train_end],
            "test": windows[T - n_test:],
        }
        print(f"{robot}: {T} frames, {desc.shape[0]} joints, desc dim {desc.shape[1]}, "
              f"train {train_end} / test {n_test}")

    first = per_robot[args.robots[0]]
    model = make_model(args, args.lookahead)
    rng, init_rng = jax.random.split(rng)
    b0 = first["train"][: args.batch_size]
    d0 = np.broadcast_to(first["desc"], (b0.shape[0],) + first["desc"].shape)
    g0 = np.zeros((b0.shape[0], 1), dtype=np.float32)  # decoder ignores it
    state = create_train_state(
        model=model,
        joint_obs=jnp.asarray(b0),
        joint_description=jnp.asarray(d0),
        general_observation=jnp.asarray(g0),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        rng=init_rng,
    )

    history = {"train_loss": [], "eval_loss": {r: [] for r in args.robots}}
    t0 = time.time()
    for epoch in range(args.epochs):
        batches = []
        for robot in args.robots:
            tr = per_robot[robot]["train"]
            rng, prng = jax.random.split(rng)
            perm = np.asarray(jax.random.permutation(prng, tr.shape[0]))
            nb = tr.shape[0] // args.batch_size
            for i in range(nb):
                batches.append((robot, perm[i * args.batch_size:(i + 1) * args.batch_size]))
        rng, srng = jax.random.split(rng)
        order = np.asarray(jax.random.permutation(srng, len(batches)))

        losses = []
        for bi in order:
            robot, idx = batches[int(bi)]
            tr = per_robot[robot]["train"][idx]
            desc = np.broadcast_to(per_robot[robot]["desc"], (tr.shape[0],) + per_robot[robot]["desc"].shape)
            g = np.zeros((tr.shape[0], 1), dtype=np.float32)
            state, loss, _ = train_step(
                state, jnp.asarray(tr), jnp.asarray(desc), jnp.asarray(g), jnp.asarray(tr)
            )
            losses.append(float(loss))
        history["train_loss"].append(float(np.mean(losses)))

        for robot in args.robots:
            te = per_robot[robot]["test"]
            desc = np.broadcast_to(per_robot[robot]["desc"], (te.shape[0],) + per_robot[robot]["desc"].shape)
            g = np.zeros((te.shape[0], 1), dtype=np.float32)
            loss, _, _, _ = eval_step(
                state, jnp.asarray(te), jnp.asarray(desc), jnp.asarray(g), jnp.asarray(te)
            )
            history["eval_loss"][robot].append(float(loss))

        if (epoch + 1) % args.log_every == 0 or epoch == args.epochs - 1:
            evals = " ".join(f"{r}={history['eval_loss'][r][-1]:.5f}" for r in args.robots)
            print(f"epoch {epoch + 1}/{args.epochs} train={history['train_loss'][-1]:.5f} "
                  f"eval[{evals}] ({time.time() - t0:.0f}s)", flush=True)

    (out / "params.msgpack").write_bytes(flax.serialization.to_bytes(state.params))
    config = {
        "robots": args.robots,
        "clip": args.clip,
        "clip_dir": str(args.clip_dir),
        "lookahead": args.lookahead,
        "latent_dim": args.latent_dim,
        "fsq_levels": args.fsq_levels,
        "width": args.width,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "test_fraction": args.test_fraction,
        "final_train_loss": history["train_loss"][-1],
        "final_eval_loss": {r: history["eval_loss"][r][-1] for r in args.robots},
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))
    np.savez(
        out / "descriptions.npz",
        **{f"{r}_desc": per_robot[r]["desc"] for r in args.robots},
        **{f"{r}_joints": np.array(per_robot[r]["joints"]) for r in args.robots},
    )
    (out / "history.json").write_text(json.dumps(history, indent=2))
    print(f"saved tokenizer to {out}")


# ---------------------------------------------------------------------------
# reconstruct
# ---------------------------------------------------------------------------


def cmd_reconstruct(args):
    src_dir = Path(args.clip_dir)
    tok = Path(args.tokenizer)
    dst_dir = Path(args.out)
    config = json.loads((tok / "config.json").read_text())
    lookahead = config["lookahead"]

    class _A:
        latent_dim = config["latent_dim"]
        fsq_levels = config["fsq_levels"]
        width = config["width"]

    model = make_model(_A, lookahead)
    desc_store = np.load(tok / "descriptions.npz", allow_pickle=True)

    report = {}
    for robot in args.robots:
        desc = desc_store[f"{robot}_desc"]
        joint_names = [str(n) for n in desc_store[f"{robot}_joints"]]
        src = src_dir / robot / args.clip
        qpos, qvel, perm = load_clip_joints(src, joint_names)
        windows = build_windows(qpos, qvel, lookahead)
        T, J = qpos.shape

        params = flax.serialization.from_bytes(
            model.init(
                jax.random.PRNGKey(0),
                jnp.asarray(windows[:1]),
                jnp.asarray(desc[None]),
                jnp.zeros((1, 1), dtype=jnp.float32),
            )["params"],
            (tok / "params.msgpack").read_bytes(),
        )

        @jax.jit
        def apply(w, d):
            return model.apply({"params": params}, w, d, jnp.zeros((w.shape[0], 1)))

        rec_rows = []
        zq_rows = []
        for start in range(0, T, args.batch_size):
            w = jnp.asarray(windows[start:start + args.batch_size])
            d = jnp.broadcast_to(jnp.asarray(desc), (w.shape[0],) + desc.shape)
            recon, aux = apply(w, d)
            # row 1 = the goal channel for frame t (see module docstring).
            rec_rows.append(np.asarray(recon[:, 1]))
            zq_rows.append(np.asarray(aux["z_q"], dtype=np.float32))
        rec = np.concatenate(rec_rows, axis=0)  # (T, J, 2)
        zq = np.concatenate(zq_rows, axis=0)    # (T, J, latent)

        rec_qpos, rec_qvel = rec[..., 0], rec[..., 1]
        inv = np.argsort(perm)

        d = dict(np.load(src, allow_pickle=True))
        new_qpos = np.array(d["qpos"])
        new_qvel = np.array(d["qvel"])
        new_qpos[:, 7:] = rec_qpos[:, inv]
        new_qvel[:, 6:] = rec_qvel[:, inv]
        d["qpos"] = new_qpos.astype(np.float32)
        d["qvel"] = new_qvel.astype(np.float32)

        out_robot = dst_dir / robot
        out_robot.mkdir(parents=True, exist_ok=True)
        np.savez(out_robot / args.clip, **d)
        np.savez(out_robot / (Path(args.clip).stem + "_zq.npz"),
                 z_q=zq, joint_names=np.array(joint_names))

        err = rec_qpos - qpos
        codes = zq.reshape(T * J, -1)
        report[robot] = {
            "frames": int(T),
            "joints": int(J),
            "qpos_rmse_rad": float(np.sqrt(np.mean(err ** 2))),
            "qpos_max_abs_err_rad": float(np.max(np.abs(err))),
            "per_joint_rmse_rad": {
                joint_names[j]: float(np.sqrt(np.mean(err[:, j] ** 2))) for j in range(J)
            },
            "qvel_rmse": float(np.sqrt(np.mean((rec_qvel - qvel) ** 2))),
            "unique_codes_used": int(np.unique(codes, axis=0).shape[0]),
            "code_slots": int(T * J),
        }
        print(f"{robot}: qpos RMSE {report[robot]['qpos_rmse_rad']:.4f} rad, "
              f"max {report[robot]['qpos_max_abs_err_rad']:.4f} rad, "
              f"{report[robot]['unique_codes_used']}/{T * J} unique per-joint codes")

    # Copy untouched robots' clips so the reconstructed dir is a drop-in
    # replacement for tracking_clip_dir (loader resolves other families too).
    if args.copy_other_robots:
        for sub in src_dir.iterdir():
            if sub.is_dir() and sub.name not in args.robots and (sub / args.clip).exists():
                out_robot = dst_dir / sub.name
                out_robot.mkdir(parents=True, exist_ok=True)
                shutil.copy2(sub / args.clip, out_robot / args.clip)

    (dst_dir / "reconstruction_report.json").write_text(json.dumps(report, indent=2))
    print(f"reconstructed clips in {dst_dir}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--robots", nargs="+", default=["UnitreeH1", "UnitreeG1"])
    t.add_argument("--clip", default="dance2_subject4.npz")
    t.add_argument("--clip-dir", default=str(DEFAULT_CLIP_DIR))
    t.add_argument("--out", required=True)
    t.add_argument("--lookahead", type=int, default=10)
    t.add_argument("--latent-dim", type=int, default=32)
    t.add_argument("--fsq-levels", type=int, default=8)
    t.add_argument("--width", type=float, default=1.0)
    t.add_argument("--epochs", type=int, default=500)
    t.add_argument("--batch-size", type=int, default=256)
    t.add_argument("--lr", type=float, default=1.5e-3)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--test-fraction", type=float, default=0.1)
    t.add_argument("--log-every", type=int, default=10)
    t.set_defaults(fn=cmd_train)

    r = sub.add_parser("reconstruct")
    r.add_argument("--robots", nargs="+", default=["UnitreeH1", "UnitreeG1"])
    r.add_argument("--clip", default="dance2_subject4.npz")
    r.add_argument("--clip-dir", default=str(DEFAULT_CLIP_DIR))
    r.add_argument("--tokenizer", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--batch-size", type=int, default=512)
    r.add_argument("--copy-other-robots", action="store_true", default=True)
    r.set_defaults(fn=cmd_reconstruct)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
