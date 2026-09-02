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
    # The scaling bundles need descriptors for every body they decode onto, and
    # loco-mujoco names these envs identically to the clip directories. Only
    # dance2_subject4 is retargeted to all seven, so these five matter for the
    # robot-count curve, not the motion-count one.
    "UnitreeH1v2": "UnitreeH1v2",
    "BoosterT1": "BoosterT1",
    "Atlas": "Atlas",
    "Talos": "Talos",
    "ToddlerBot": "ToddlerBot",
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
    # The free joint is the first entry and its NAME is not standardised across
    # loco-mujoco robots: H1/G1/Atlas/BoosterT1 call it "root", Talos calls it
    # "reference". The check exists to catch a clip whose first column is a real
    # hinge (which would silently shift every joint by 7), so widen it to the
    # known free-joint names rather than excluding a whole family. Verified on
    # Talos 2026-08-27: qpos is 7 + 44 and all 30 descriptor joints are present.
    # "floating_base_joint" is UnitreeH1v2 (verified 2026-08-31: qpos 28 = 7 + 21).
    if joint_names[0] not in ("root", "reference", "floating_base_joint"):
        raise ValueError(
            f"{npz_path}: expected a free joint first, got {joint_names[0]!r}")
    clip_joints = joint_names[1:]
    # An actuator with NO clip column is fatal -- the permutation below would
    # raise, or worse, silently take a wrong column. Extra CLIP columns are a
    # different matter: Talos ships 14 gripper/fingertip joints that no
    # locomotion actuator drives, and dropping those is correct, not lossy. So
    # allow a clip superset, refuse a clip subset, and always say what was
    # dropped rather than truncating in silence.
    missing = sorted(set(actuator_joint_names) - set(clip_joints))
    if missing:
        raise ValueError(
            f"{npz_path}: {len(missing)} actuator joints have no clip column: {missing}")
    extra = sorted(set(clip_joints) - set(actuator_joint_names))
    if extra:
        print(f"[load_clip_joints] {npz_path}: ignoring {len(extra)} unactuated "
              f"clip joints ({', '.join(extra[:3])}{', ...' if len(extra) > 3 else ''})",
              flush=True)
    perm = np.array([clip_joints.index(n) for n in actuator_joint_names], dtype=np.int64)
    qpos = np.asarray(d["qpos"], dtype=np.float32)[:, 7:][:, perm]
    qvel = np.asarray(d["qvel"], dtype=np.float32)[:, 6:][:, perm]
    return qpos, qvel, perm


def resolve_clips(raw) -> list[str]:
    """Normalise --clip into a list.

    Accepts a single name, several names, or one comma-separated string (the
    form the 2026-08-27 multi-clip bundles recorded in their config.json, so an
    existing bundle stays readable).
    """
    items = raw if isinstance(raw, (list, tuple)) else [raw]
    out = []
    for item in items:
        out.extend(part for part in str(item).split(",") if part)
    return out


def load_clip_feet(npz_path: Path) -> np.ndarray:
    """(T, 2) left/right foot height over ground from the clip's mimic sites.

    Height is the site z minus that foot's own per-clip minimum, so 'on the
    floor' reads ~0 regardless of the retarget's absolute z convention. A5's
    channel: what the qpos/qvel window under-represents (airborne frames are
    ~13% of a dance) made an explicit reconstruction target.
    """
    d = np.load(npz_path, allow_pickle=True)
    sn = d.get("site_names")
    xp = d.get("site_xpos")
    # The re-issued BoosterT1 clips (LAFAN1_fixed pipeline) store site_names as
    # a 0-d None and site_xpos empty -- no site record at all. Zero channels
    # keep input_dim uniform across robots; that robot's code simply carries no
    # foot info until its clips are re-emitted with FK sites.
    if sn is None or (hasattr(sn, "ndim") and sn.ndim == 0 and sn.item() is None) \
            or xp is None or np.asarray(xp).size == 0:
        T = np.asarray(d["qpos"]).shape[0]
        print(f"[load_clip_feet] {npz_path}: NO site data -> zero foot channels",
              flush=True)
        return np.zeros((T, 2), dtype=np.float32)
    names = [str(s) for s in (sn.item() if getattr(sn, "ndim", 1) == 0 else sn)]
    idx = [names.index("left_foot_mimic"), names.index("right_foot_mimic")]
    z = np.asarray(xp, dtype=np.float32)[:, idx, 2]
    return z - z.min(axis=0, keepdims=True)


def build_windows(qpos: np.ndarray, qvel: np.ndarray, lookahead: int,
                  feet: np.ndarray | None = None) -> np.ndarray:
    """(T, lookahead+1, J, C) windows: row 0 = frame t, rows 1..N = frames
    t..t+N-1 clamped at the clip end (his goal-clamp semantics).

    C=2 (qpos, qvel); with `feet` (T, 2), C=4 -- both foot heights broadcast
    to every joint row, so each per-joint code must carry the contact context
    (the two-head/A5 variant: the decoder reconstructs them, forcing swing
    information through the bottleneck; z_q shape is unchanged)."""
    T = qpos.shape[0]
    chans = [qpos, qvel]
    if feet is not None:
        J = qpos.shape[1]
        chans.append(np.broadcast_to(feet[:, None, 0:1], (T, J, 1))[..., 0])
        chans.append(np.broadcast_to(feet[:, None, 1:2], (T, J, 1))[..., 0])
    frames = np.stack(chans, axis=-1)  # (T, J, C)
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
        input_dim=4 if getattr(args, "foot_channels", False) else 2,
        n_step_lookahead=n_step_lookahead,
        latent_dim=args.latent_dim,
        levels=tuple([args.fsq_levels] * args.latent_dim),
        softmax_temperature=1.0,
        softmax_temperature_min=0.01,
        stability_epsilon=1.0e-5,
        encoder_network_width_multiplier=args.width,
        decoder_network_width_multiplier=args.width,
    )


def load_desc(robot: str, desc_cache):
    """Joint descriptions for `robot`.

    The 47-dim description vector depends only on the nominal model, so it is
    identical across every fit — which lets a cached `descriptions.npz` stand in
    for the whole loco-mujoco stack on machines that only carry the trainer
    (Viper has loco_mjx, not loco-mujoco).  Verified identical across the four
    tokenizers fitted locally.
    """
    if not desc_cache:
        return build_robot_assets(robot)
    store = np.load(desc_cache, allow_pickle=True)
    return store[f"{robot}_desc"], [str(n) for n in store[f"{robot}_joints"]]


def cmd_train(args):
    from loco_mujoco.algorithms.autoencoder.train.step import (
        create_train_state,
        train_step,
        eval_step,
    )

    rng = jax.random.PRNGKey(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    clips = resolve_clips(args.clip)
    heldout = set(resolve_clips(args.heldout_clips)) if args.heldout_clips else set()
    unknown = heldout - set(clips)
    if unknown:
        raise ValueError(f"--heldout-clips names clips that are not in --clip: {sorted(unknown)}")
    fit_clips = [c for c in clips if c not in heldout]
    if not fit_clips:
        raise ValueError("every clip was held out; nothing to fit on")

    per_robot = {}
    for robot in args.robots:
        desc, joint_names = load_desc(robot, getattr(args, 'desc_cache', None))
        train_parts, test_parts, used, skipped = [], [], [], []
        for clip in clips:
            src = Path(args.clip_dir) / robot / clip
            if not src.exists():
                # Retarget coverage is not uniform (G1 has 10 clips where H1 has
                # 11). Skipping loudly beats either crashing or silently fitting
                # a different clip set per robot.
                skipped.append(clip)
                continue
            qpos, qvel, _ = load_clip_joints(src, joint_names)
            feet = load_clip_feet(src) if getattr(args, "foot_channels", False) else None
            windows = build_windows(qpos, qvel, args.lookahead, feet)
            T = windows.shape[0]
            if clip in heldout:
                # A whole clip held out: the honest generalisation split, since
                # a tail split leaves the tokenizer scored on the same motion.
                test_parts.append(windows)
                used.append(f"{clip}(heldout {T})")
                continue
            n_test = max(1, int(T * args.test_fraction))
            # Temporal tail split; drop `lookahead` frames before the boundary so
            # no training window peeks into the test tail.
            train_end = max(1, T - n_test - args.lookahead)
            train_parts.append(windows[:train_end])
            if not heldout:
                test_parts.append(windows[T - n_test:])
            used.append(f"{clip}({train_end}/{n_test})")
        if not train_parts:
            raise ValueError(f"{robot}: none of {clips} exist under {args.clip_dir}")
        per_robot[robot] = {
            "desc": desc,
            "joints": joint_names,
            "train": np.concatenate(train_parts, axis=0),
            "test": np.concatenate(test_parts, axis=0),
            "clips_used": used,
            "clips_skipped": skipped,
        }
        print(f"{robot}: {desc.shape[0]} joints, desc dim {desc.shape[1]}, "
              f"train {per_robot[robot]['train'].shape[0]} / test "
              f"{per_robot[robot]['test'].shape[0]} frames over {len(used)} clips")
        print(f"  {' '.join(used)}")
        if skipped:
            print(f"  SKIPPED (no retarget): {' '.join(skipped)}")

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
            # BATCHED. This used to score the whole test set in one call, which
            # was survivable while the split was a 10% tail of one clip (~900
            # frames) and is not once a WHOLE clip is held out (~10 500 frames):
            # the decoder's activations are (N, lookahead+1, J, width) and the
            # process was killed mid-epoch, silently, with an empty stderr.
            # Sample-weighted so the number means the same as the unbatched one.
            total, seen = 0.0, 0
            for start in range(0, te.shape[0], args.batch_size):
                tb = te[start:start + args.batch_size]
                desc = np.broadcast_to(per_robot[robot]["desc"], (tb.shape[0],) + per_robot[robot]["desc"].shape)
                g = np.zeros((tb.shape[0], 1), dtype=np.float32)
                loss, _, _, _ = eval_step(
                    state, jnp.asarray(tb), jnp.asarray(desc), jnp.asarray(g), jnp.asarray(tb)
                )
                total += float(loss) * tb.shape[0]
                seen += tb.shape[0]
            history["eval_loss"][robot].append(total / max(seen, 1))

        if (epoch + 1) % args.log_every == 0 or epoch == args.epochs - 1:
            evals = " ".join(f"{r}={history['eval_loss'][r][-1]:.5f}" for r in args.robots)
            print(f"epoch {epoch + 1}/{args.epochs} train={history['train_loss'][-1]:.5f} "
                  f"eval[{evals}] ({time.time() - t0:.0f}s)", flush=True)

    (out / "params.msgpack").write_bytes(flax.serialization.to_bytes(state.params))
    config = {
        "robots": args.robots,
        # `clip` stays the comma-joined string older readers expect; `clips` is
        # the authoritative list.
        "clip": ",".join(clips),
        "clips": clips,
        "heldout_clips": sorted(heldout),
        "clips_per_robot": {r: per_robot[r]["clips_used"] for r in args.robots},
        "clips_skipped_per_robot": {r: per_robot[r]["clips_skipped"] for r in args.robots},
        "clip_dir": str(args.clip_dir),
        "lookahead": args.lookahead,
        "foot_channels": bool(getattr(args, "foot_channels", False)),
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
        foot_channels = bool(config.get("foot_channels", False))

    model = make_model(_A, lookahead)
    desc_store = np.load(tok / "descriptions.npz", allow_pickle=True)

    report = {}
    clips = resolve_clips(args.clip)
    heldout = set(config.get("heldout_clips") or [])
    for robot in args.robots:
        desc = desc_store[f"{robot}_desc"]
        joint_names = [str(n) for n in desc_store[f"{robot}_joints"]]

        # Params depend only on shapes, so they are loaded once per robot and
        # reused across that robot's clips.
        dummy = np.zeros((1, lookahead + 1, desc.shape[0],
                          4 if _A.foot_channels else 2), dtype=np.float32)
        params = flax.serialization.from_bytes(
            model.init(
                jax.random.PRNGKey(0),
                jnp.asarray(dummy),
                jnp.asarray(desc[None]),
                jnp.zeros((1, 1), dtype=jnp.float32),
            )["params"],
            (tok / "params.msgpack").read_bytes(),
        )

        @jax.jit
        def apply(w, d):
            return model.apply({"params": params}, w, d, jnp.zeros((w.shape[0], 1)))

        for clip in clips:
            src = src_dir / robot / clip
            if not src.exists():
                print(f"{robot}/{clip}: no retarget, skipped", flush=True)
                continue
            qpos, qvel, perm = load_clip_joints(src, joint_names)
            feet = load_clip_feet(src) if _A.foot_channels else None
            windows = build_windows(qpos, qvel, lookahead, feet)
            T, J = qpos.shape

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
            np.savez(out_robot / clip, **d)
            np.savez(out_robot / (Path(clip).stem + "_zq.npz"),
                     z_q=zq, joint_names=np.array(joint_names))

            err = rec_qpos - qpos
            # Same contamination as the canonical script had: the loop above
            # scores EVERY frame while --test-fraction defaults to 0.1, so
            # qpos_rmse_rad is ~90% training data. Confirmed 2026-08-27 -- which
            # means the published "per-joint 0.0508 vs canonical 0.1774"
            # compared two contaminated numbers. Gate on qpos_rmse_rad_heldout;
            # the mixed figure stays only so older reports stay comparable.
            # A clip named in --heldout-clips was never fitted at all, so there
            # every frame is held out and the two figures coincide.
            if clip in heldout:
                err_h = err
            else:
                n_test = max(1, int(T * float(config.get("test_fraction", 0.1))))
                err_h = err[max(1, T - n_test):]
            codes = zq.reshape(T * J, -1)
            key = f"{robot}/{clip}"
            report[key] = {
                "robot": robot,
                "clip": clip,
                "heldout_clip": clip in heldout,
                "frames": int(T),
                "joints": int(J),
                "heldout_frames": int(err_h.shape[0]),
                "qpos_rmse_rad_heldout": float(np.sqrt(np.mean(err_h ** 2))),
                "qpos_rmse_rad": float(np.sqrt(np.mean(err ** 2))),
                "qpos_max_abs_err_rad": float(np.max(np.abs(err))),
                "per_joint_rmse_rad": {
                    joint_names[j]: float(np.sqrt(np.mean(err[:, j] ** 2))) for j in range(J)
                },
                "qvel_rmse": float(np.sqrt(np.mean((rec_qvel - qvel) ** 2))),
                "unique_codes_used": int(np.unique(codes, axis=0).shape[0]),
                "code_slots": int(T * J),
            }
            print(f"{key}: qpos RMSE {report[key]['qpos_rmse_rad']:.4f} rad "
                  f"(held-out {report[key]['qpos_rmse_rad_heldout']:.4f}), max "
                  f"{report[key]['qpos_max_abs_err_rad']:.4f} rad, "
                  f"{report[key]['unique_codes_used']}/{T * J} unique per-joint codes",
                  flush=True)

    # Copy untouched robots' clips so the reconstructed dir is a drop-in
    # replacement for tracking_clip_dir (loader resolves other families too).
    if args.copy_other_robots:
        for sub_dir in src_dir.iterdir():
            if not sub_dir.is_dir() or sub_dir.name in args.robots:
                continue
            for clip in clips:
                if (sub_dir / clip).exists():
                    out_robot = dst_dir / sub_dir.name
                    out_robot.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(sub_dir / clip, out_robot / clip)

    (dst_dir / "reconstruction_report.json").write_text(json.dumps(report, indent=2))
    print(f"reconstructed clips in {dst_dir}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--robots", nargs="+", default=["UnitreeH1", "UnitreeG1"])
    t.add_argument("--clip", nargs="+", default=["dance2_subject4.npz"],
                   help="one or more clip filenames, or one comma-separated string")
    t.add_argument("--heldout-clips", nargs="+", default=None,
                   help="clips excluded from the fit entirely and scored as a "
                        "generalisation set. Without this the split is a 10%% "
                        "tail of each clip, which scores the tokenizer on "
                        "motions it was fitted on.")
    t.add_argument("--clip-dir", default=str(DEFAULT_CLIP_DIR))
    t.add_argument("--out", required=True)
    t.add_argument("--lookahead", type=int, default=10)
    t.add_argument("--foot-channels", action="store_true",
                   help="A5/two-head: append both feet's clip heights to every "
                        "joint window row (input_dim 2 -> 4); the decoder must "
                        "reconstruct them, so the code carries swing/contact "
                        "info. z_q layout is unchanged.")
    t.add_argument("--latent-dim", type=int, default=32)
    t.add_argument("--fsq-levels", type=int, default=8)
    t.add_argument("--width", type=float, default=1.0)
    t.add_argument("--epochs", type=int, default=500)
    t.add_argument("--batch-size", type=int, default=256)
    t.add_argument("--lr", type=float, default=1.5e-3)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--test-fraction", type=float, default=0.1)
    t.add_argument("--log-every", type=int, default=10)
    t.add_argument("--desc-cache", default=None,
                   help="descriptions.npz from a previous fit; skips the "
                        "loco-mujoco model compile (nominal bodies only)")
    t.set_defaults(fn=cmd_train)

    r = sub.add_parser("reconstruct")
    r.add_argument("--robots", nargs="+", default=["UnitreeH1", "UnitreeG1"])
    r.add_argument("--clip", nargs="+", default=["dance2_subject4.npz"],
                   help="one or more clip filenames, or one comma-separated string")
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
