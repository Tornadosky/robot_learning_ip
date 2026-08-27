"""Cross-evaluate a tracking policy's EXECUTED motion against the RAW clip.

The A/B/C arms each report reward against their own (FSQ-smoothed) reference,
so their returns are not comparable as motion quality. This script rolls out a
checkpoint in its own training condition (its clip dir, nominal body, no DR)
and scores the joints the robot ACTUALLY executed against the raw LAFAN1 clip
at the same reference phase — one number, comparable across all arms.

Usage:
  python crosseval_motion.py --model_path <latest.model> --robot unitree_h1 \
      --clip_dir <arm clip dir> --raw_clip_dir <LAFAN1 dir> --out out.json
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "disable")

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

ROBOT_TO_CLIP_SUBDIR = {"unitree_h1": "UnitreeH1", "unitree_g1": "UnitreeG1"}


class _DummyWriter:
    def add_scalar(self, *args, **kwargs):
        pass

    def close(self):
        pass


def make_eval_checkpoint(model_path: str, work_dir: str) -> str:
    # Strip resume-only files so URMA2.load() treats this as a plain test load
    # (same trick as eval_heldout.py).
    src = Path(model_path)
    dst = Path(work_dir) / "model"
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.parent.iterdir():
        if f.name in ("training_progress.json", "resume_manifest.json", "resume_state.npz"):
            continue
        shutil.copy2(f, dst / f.name)
    return str(dst / src.name)


def load_raw_joints(raw_clip_dir: str, robot: str, clip: str, mj_model):
    """Raw clip joints for the subset of actuators the clip covers.

    Returns (raw_joints (T, K), actuator_ids (K,), names (K,)). The env's model
    can have more actuated joints than the clip (G1 waist_roll/pitch); those
    are excluded from the metric on every arm equally.
    """
    import mujoco

    from loco_mjx.environments.locomotion.urma2.mjx.clip_reference import (
        H1_CLIP_SIGNS, G1_CLIP_SIGNS,
    )

    d = np.load(Path(raw_clip_dir) / ROBOT_TO_CLIP_SUBDIR[robot] / clip, allow_pickle=True)
    joint_names = [str(n) for n in d["joint_names"]][1:]
    act_ids, cols, names, signs = [], [], [], []
    for a in range(mj_model.nu):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, int(mj_model.actuator_trnid[a, 0]))
        if name in joint_names:
            act_ids.append(a)
            cols.append(joint_names.index(name))
            names.append(name)
            signs.append(H1_CLIP_SIGNS.get(name, G1_CLIP_SIGNS.get(name, 1.0)))
    raw = np.asarray(d["qpos"], dtype=np.float64)[:, 7:][:, np.array(cols, dtype=np.int64)]
    # Same per-joint sign correction the env applies when loading a clip
    # (clip_reference.load_clip); without it, flipped joints score ~1.5 rad of
    # pure convention mismatch. Returned ABSOLUTE (uncentered): the anchor-fix
    # era scores an absolute metric too, and the shape metric re-centers at
    # scoring time from these same angles.
    raw = raw * np.array(signs, dtype=np.float64)[None, :]
    return raw, np.array(act_ids, dtype=np.int64), names



def _heading_stats(d):
    """Absolute root-heading error against the clip's commanded heading.

    Wrapped: if any run configuration lacks the yaw bookkeeping this reads, the
    crosseval must still produce its joint numbers. Missing is reported as None,
    never as zero -- a zero here would read as perfect tracking.
    """
    try:
        import numpy as _np
        chunks = [c for c in d.get("heading_err", []) if len(c)]
        if not chunks:
            return {"heading_error_deg_mean": None, "heading_error_deg_p95": None}
        he = _np.degrees(_np.concatenate(chunks, axis=0))
        return {
            "heading_error_deg_mean": float(he.mean()),
            "heading_error_deg_p95": float(_np.percentile(he, 95)),
        }
    except Exception as exc:  # noqa: BLE001
        print(f"heading metric unavailable: {type(exc).__name__}: {exc}")
        return {"heading_error_deg_mean": None, "heading_error_deg_p95": None}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--clip_dir", required=True, help="the ARM's clip dir (training condition)")
    p.add_argument("--raw_clip_dir", required=True, help="raw LAFAN1 dir to score against")
    p.add_argument("--clip", default="dance2_subject4.npz")
    p.add_argument("--nr_envs", type=int, default=32)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    # LADDER: this defaulted to 1.0 while every arm of the last three campaigns
    # trained with REFBIAS=0.0. Under refbias=1.0 the PD position target is
    # `reference + action` instead of `nominal + action`, so a refbias-0
    # checkpoint evaluated at 1.0 is executing completely different action
    # semantics from the ones it learned. There is no safe default here -- the
    # value MUST match the arm -- so it is required rather than defaulted.
    p.add_argument("--refbias", type=float, required=True,
                   help="the ARM's tracking_reference_action_bias; must match or the "
                        "checkpoint is evaluated under different action semantics")
    p.add_argument("--anchor", default="centered", choices=("centered", "absolute"),
                   help="the ARM's tracking_clip_anchor; changes the env reference")
    p.add_argument("--fitvariant", default="True", choices=("True", "False"),
                   help="the ARM's tracking_clip_fit_per_variant")
    p.add_argument("--cyclic", default="False", choices=("True", "False"))
    p.add_argument("--refroot", default="False", choices=("True", "False"),
                   help="tracking_clip_root_height_from_pose, as trained")
    # TRAP (goal doc): this script never set root_height_pose_as_floor, so every
    # foot_height_error / ref_foot_* number it has ever printed for a
    # REFROOT_FLOOR=True arm was scored against the wrong reference.
    p.add_argument("--refroot_floor", default="False", choices=("True", "False"),
                   help="tracking_clip_root_height_pose_as_floor, as trained")
    # The checkpoint's manifest pins the robot set. L0/L1 arms train unitree_h1
    # ALONE, and hardcoding the pair here made this script unusable on them.
    # REFVEL_OBS adds a SIXTH per-joint observation channel. A checkpoint
    # trained with it cannot be loaded without it -- orchestrated as
    # "Requested shape: (5, 8) is not compatible with the stored shape: (6, 8)".
    # Same class as --latent: the evaluation must reproduce the arm's
    # observation layout exactly.
    p.add_argument("--refvel_obs", default="False", choices=("True", "False"),
                   help="tracking_clip_observe_velocity, as trained")
    p.add_argument("--robots", default="unitree_h1:unitree_g1",
                   help="colon-separated robot set, EXACTLY as the arm trained")
    p.add_argument("--latent_hold", type=int, default=1,
                   help="tracking_clip_latent_hold, as trained")
    p.add_argument("--latent_replaces", default="True", choices=("True", "False"),
                   help="tracking_clip_latent_replaces_reference, as trained. False "
                        "means the arm saw BOTH the reference and the token, which is "
                        "a WIDER joint observation -- hardcoding True (as this script "
                        "did until 26-08) makes such a checkpoint unloadable.")
    p.add_argument("--latent_dim", type=int, default=32,
                   help="tracking_clip_latent_dim, as trained. Kevin's per-joint "
                        "tokenizer is 32; the CANONICAL shared-stream tokenizer is "
                        "4 (codebook 8x5x5x5, one code per frame). A mismatch "
                        "changes the observation width and the checkpoint will "
                        "not load.")
    p.add_argument("--reference_hold", type=int, default=1,
                   help="freeze the OBSERVED reference to every K-th clip frame; "
                        "the reward target stays fresh (K=1 disables)")
    p.add_argument("--dump_render", default=None,
                   help="write <path>__<robot>.npz with the fields rf_render_dance.py "
                        "reads (qpos, reference_joint_targets, reference_root_yaw_delta, "
                        "root_yaw_origin, heading_error, dt). This script already builds "
                        "the arm's EXACT env -- latent width, hold, anchor, refroot -- so "
                        "it is the only rollout that can render a token arm.")
    p.add_argument("--record_envs", type=int, default=4,
                   help="how many envs to keep full qpos for, when dumping")
    p.add_argument("--zero_action", action="store_true",
                   help="roll out with a ZERO action instead of the policy. The "
                        "same-body floor every tracking claim must clear -- an "
                        "earlier zero-shot result died to its absence.")
    p.add_argument("--latent", action="store_true",
                   help="checkpoint was trained with z-token obs replacing the reference channel")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    import importlib

    import jax
    import mujoco
    from ml_collections import config_dict
    from rl_x.algorithms.algorithm_manager import get_algorithm_config, get_algorithm_model_class
    from rl_x.environments.environment_manager import get_environment_config
    from rl_x.runner.default_config import get_config as get_runner_config
    from rl_x.runner.runner_mode import RunnerMode

    from loco_mjx.environments.locomotion.urma2.mjx.create_env import create_env

    algorithm_name = "urma2.mjx"
    environment_name = "locomotion.urma2.mjx"
    importlib.import_module(f"loco_mjx.environments.{environment_name}")
    importlib.import_module(f"loco_mjx.algorithms.{algorithm_name}")

    config = config_dict.ConfigDict()
    config.runner = get_runner_config(RunnerMode.TRAIN)
    config.algorithm = get_algorithm_config(algorithm_name)
    config.environment = get_environment_config(environment_name)

    work_dir = tempfile.mkdtemp()
    config.runner.mode = "test"
    config.runner.load_model = make_eval_checkpoint(args.model_path, work_dir)
    config.runner.save_model = False
    config.runner.track_wandb = False
    config.runner.track_tb = False
    config.runner.track_console = False
    config.runner.exp_name = "crosseval"
    config.runner.run_name = "crosseval"
    config.runner.project_name = "loco_mjx"

    config.algorithm.name = algorithm_name
    config.algorithm.evaluation_active = False

    # The checkpoint's manifest pins the exact robot set; evaluate exactly that
    # set and split the metrics per robot afterwards.
    robots = tuple(r for r in args.robots.replace(",", ":").split(":") if r)
    config.environment.name = environment_name
    config.environment.train_robots = robots
    config.environment.eval_robots = ()
    config.environment.nr_envs = args.nr_envs
    config.environment.nr_eval_envs = 0
    config.environment.render = False
    config.environment.seed = args.seed

    # Must match the training run or the checkpoint's obs sizes won't line up.
    config.environment.terrain.type = "plane"
    config.environment.critic_exteroceptive_observation_type = "none"
    config.environment.command.type = "tracking_clip"
    config.environment.reward.type = "tracking"
    config.environment.reward.log_info = True
    config.environment.reward.nominal_diff_target = "reference"
    config.environment.reward.joint_tracking_coeff = 30.0
    config.environment.reward.joint_tracking_temperature = 0.25
    config.environment.reward.deepmimic_enabled = True
    config.environment.command.tracking_clip_dir = args.clip_dir
    config.environment.command.tracking_clip_file = args.clip
    config.environment.command.tracking_clip_fit_per_variant = (args.fitvariant == "True")
    config.environment.command.tracking_clip_anchor = args.anchor
    config.environment.command.tracking_clip_cyclic = (args.cyclic == "True")
    config.environment.command.tracking_clip_root_height_from_pose = (args.refroot == "True")
    config.environment.command.tracking_clip_root_height_pose_as_floor = (args.refroot_floor == "True")
    config.environment.command.tracking_clip_observe_velocity = (args.refvel_obs == "True")
    config.environment.command.tracking_clip_velocity_command = True
    config.environment.command.tracking_reference_action_bias = args.refbias
    config.environment.command.tracking_clip_reference_hold = args.reference_hold
    if args.latent:
        config.environment.command.tracking_clip_latent_obs = True
        config.environment.command.tracking_clip_latent_dim = args.latent_dim
        config.environment.command.tracking_clip_latent_replaces_reference = (
            args.latent_replaces == "True")
        config.environment.command.tracking_clip_latent_hold = args.latent_hold
    config.environment.domain_randomization.initial_state.type = "reference"
    # Nominal body, no randomization: motion quality only.
    config.environment.domain_randomization.sampling_type = "none"
    config.environment.domain_randomization.perturbation.sampling_type = "none"
    config.environment.domain_randomization.observation_noise.type = "none"

    started = time.time()
    train_env, eval_env = create_env(config)
    model_class = get_algorithm_model_class(algorithm_name)
    model = model_class.load(config, train_env, eval_env, tempfile.mkdtemp(), _DummyWriter(), [])

    import jax.numpy as jnp

    nr_robots = len(robots)
    envs_per_robot = model.nr_envs_per_train_robot
    per = []
    for i, robot in enumerate(robots):
        mj_model = train_env.train_envs[i].initial_mj_model
        raw, act_ids, names = load_raw_joints(args.raw_clip_dir, robot, args.clip, mj_model)
        qposadr = np.array(
            [int(mj_model.jnt_qposadr[int(mj_model.actuator_trnid[a, 0])]) for a in act_ids],
            dtype=np.int64,
        )
        raw_root = np.asarray(
            np.load(Path(args.raw_clip_dir) / ROBOT_TO_CLIP_SUBDIR[robot] / args.clip,
                    allow_pickle=True)["qpos"], dtype=np.float64)[:, 0:7]
        per.append({
            "robot": robot, "names": names, "qposadr": qposadr, "raw": raw,
            "raw_root": raw_root, "nu": len(act_ids),
            "act_ids": act_ids,
            # Samples are collected and mean-centered per joint at the end, so
            # the metric scores motion SHAPE, not rest-pose convention.
            "exec_samples": [], "ref_samples": [], "raw_samples": [],
            "heading_err": [],
            "alive": 0, "total": 0,
            # FEETFIX: the physical foot channels, accumulated over the SAME
            # alive mask as the tracking samples. Read straight out of the env's
            # own info dict, so the number here is computed by the identical
            # code that logs it during training and the two are comparable.
            "foot": {}, "foot_n": 0,
            # render dump: full qpos + the reference pose, for the first few envs
            "r_qpos": [], "r_ref": [], "r_delta": [], "r_origin": [], "r_root": [],
        })

    key = jax.random.PRNGKey(args.seed)
    keys = jax.random.split(key, model.nr_envs)
    train_env.init(keys)
    multi_state = train_env.reset(keys)

    for _ in range(args.steps):
        obs_r = multi_state["next_observation"].reshape(nr_robots, envs_per_robot, -1)
        processed_actions = []
        for i in range(nr_robots):
            jd, jo, gs = model._decode_train_obs(obs_r[i], i)[:3]
            if args.zero_action:
                # as_shape[0] is the padded action width; the per-robot head is
                # that minus the padding, exactly as the concatenate below assumes.
                action_mean = jnp.zeros(
                    (envs_per_robot, model.as_shape[0] - model.missing_nr_of_actions[i]))
            else:
                action_mean, _ = model.policy.apply(model.policy_state.params, jd, jo, gs)
            action = jnp.concatenate(
                [action_mean, jnp.zeros((envs_per_robot, model.missing_nr_of_actions[i]))], axis=1)
            processed_actions.append(model.get_processed_action(action))
        multi_state = train_env.step(multi_state, jnp.concatenate(processed_actions, axis=0))

        terminated_all = np.asarray(jax.device_get(multi_state["terminated"])).astype(bool).reshape(
            nr_robots, envs_per_robot)
        for i, d in enumerate(per):
            state = multi_state["env_states"][i]
            qpos = np.asarray(jax.device_get(state.data.qpos))
            phase = np.asarray(jax.device_get(state.internal_state["reference_phase"])).reshape(-1)
            exec_joints = qpos[:, d["qposadr"]]
            T_clip = d["raw"].shape[0]
            # LADDER FIX. This was a NEAREST-frame lookup, round(phase*(T-1)),
            # while the environment does frame = floor(phase * clip_length) with
            # a LINEAR BLEND to the next frame (tracking_clip._reference_offsets)
            # -- the clip is 40 Hz and the controller 50 Hz, so the two never
            # line up. At dance2_subject4's 2.45 rad/s joint velocity, half a
            # frame of index error is 0.031 rad, and it lands in BOTH the
            # executed error and the reference floor. It is why this script
            # reported a 0.048 rad reference-vs-raw floor where the reference is
            # actually 0.009 rad from the clip (REPORT_ladder A4).
            fpos = np.clip(phase * T_clip, 0.0, T_clip - 1.0)
            f0 = np.floor(fpos).astype(np.int64)
            f1 = np.minimum(f0 + 1, T_clip - 1)
            blend = (fpos - f0)[:, None]
            ref = (1.0 - blend) * d["raw"][f0] + blend * d["raw"][f1]
            ok = ~terminated_all[i]
            ref_targets = np.asarray(jax.device_get(state.internal_state["reference_joint_targets"]))
            if args.dump_render:
                n = args.record_envs
                ist = state.internal_state
                # The clip's OWN root pose at this phase. Without it the render
                # has to borrow the policy's root, which makes the reference pane
                # slide sideways with no leg motion and hold the torso flat --
                # an artefact of the drawing, not of the reference.
                rr = (1.0 - blend) * d["raw_root"][f0] + blend * d["raw_root"][f1]
                q = rr[:, 3:7]
                rr[:, 3:7] = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-9)
                d["r_root"].append(rr[:n].copy())
                d["r_qpos"].append(qpos[:n].copy())
                d["r_ref"].append(ref_targets[:n].copy())
                d["r_delta"].append(np.asarray(jax.device_get(
                    ist["reference_root_yaw_delta"])).reshape(-1)[:n].copy())
                d["r_origin"].append(np.asarray(jax.device_get(
                    ist["root_yaw_origin"])).reshape(-1)[:n].copy())
            d["exec_samples"].append(exec_joints[ok])
            d["ref_samples"].append(ref_targets[:, d["act_ids"]][ok])
            d["raw_samples"].append(ref[ok])
            try:
                _q = qpos[:, 3:7]
                _w, _x, _y, _z = _q[:, 0], _q[:, 1], _q[:, 2], _q[:, 3]
                _yaw = np.arctan2(2 * (_w * _z + _x * _y),
                                  1 - 2 * (_y * _y + _z * _z))
                _og = np.asarray(jax.device_get(
                    state.internal_state["root_yaw_origin"])).reshape(-1)
                _dl = np.asarray(jax.device_get(
                    state.internal_state["reference_root_yaw_delta"])).reshape(-1)
                _e = _yaw - _og - _dl
                d["heading_err"].append(
                    np.abs(np.arctan2(np.sin(_e), np.cos(_e)))[ok])
            except Exception:  # noqa: BLE001
                pass
            d["alive"] += int(ok.sum())
            d["total"] += int(ok.shape[0])
            n_ok = int(ok.sum())
            if n_ok:
                for k, v in state.info.items():
                    if not (k.startswith("env_info/foot") or k.startswith("env_info/ref_foot")):
                        continue
                    vals = np.asarray(jax.device_get(v)).reshape(-1)
                    if vals.shape[0] != ok.shape[0]:
                        continue
                    d["foot"][k] = d["foot"].get(k, 0.0) + float(vals[ok].sum())
                d["foot_n"] += n_ok

    result = {"model_path": args.model_path, "clip_dir": args.clip_dir,
              # The 24-08 FSQ control was destroyed by two evaluations of one
              # checkpoint writing the same ${EXP}.json. Stamping the condition
              # INTO the artifact means a collision is at least detectable.
              "eval_condition": {
                  "clip": args.clip, "raw_clip_dir": args.raw_clip_dir,
                  "robots": list(robots), "refbias": args.refbias,
                  "anchor": args.anchor, "fitvariant": args.fitvariant,
                  "refroot": args.refroot, "refroot_floor": args.refroot_floor,
                  "refvel_obs": args.refvel_obs,
                  "latent": bool(args.latent), "latent_hold": args.latent_hold,
                  "latent_dim": args.latent_dim, "zero_action": bool(args.zero_action),
                  "latent_replaces": args.latent_replaces,
                  "reference_hold": args.reference_hold,
              },
              "nr_envs": model.nr_envs, "steps": args.steps,
              "wall_time_s": round(time.time() - started, 1), "robots": {}}
    for d in per:
        ex_abs = np.concatenate(d["exec_samples"], axis=0)
        rf_abs = np.concatenate(d["ref_samples"], axis=0)
        rw_abs = np.concatenate(d["raw_samples"], axis=0)
        # Shape metric (historical, comparable across every arm ever run):
        # raw centered by the CLIP's own mean, exec/ref by their sample mean.
        clip_mean = d["raw"].mean(0, keepdims=True)
        rw = rw_abs - clip_mean
        ex = ex_abs - ex_abs.mean(0, keepdims=True)
        rf = rf_abs - rf_abs.mean(0, keepdims=True)
        diff = ex - rw
        ref_diff = rf - rw
        rmse = float(np.sqrt(np.mean(diff**2)))
        # Absolute metric (anchor-fix era): no centering anywhere. A centered-
        # anchor arm keeps its constant rest-pose bias here, which is the point.
        diff_abs = ex_abs - rw_abs
        result["robots"][d["robot"]] = {
            "raw_rmse_rad": rmse,
            "raw_rmse_rad_absolute": float(np.sqrt(np.mean(diff_abs**2))),
            "per_joint_rmse_rad": {
                name: float(np.sqrt(np.mean(diff[:, k] ** 2)))
                for k, name in enumerate(d["names"])
            },
            # LADDER: per_joint_rmse_rad above is the mean-CENTRED shape metric,
            # which is what every historical table was built on. Under
            # ANCHOR=absolute the reference is absolute, so the metric that
            # matches the recipe is the uncentred one -- and the two disagree
            # about WHICH LIMB is failing (centred: arms worst; absolute: legs
            # worst, by 2x). Emit both rather than pick, so a limb verdict can
            # never be read off the wrong metric by accident.
            "per_joint_rmse_rad_absolute": {
                name: float(np.sqrt(np.mean(diff_abs[:, k] ** 2)))
                for k, name in enumerate(d["names"])
            },
            "samples": int(ex.shape[0]),
            "alive_fraction": d["alive"] / max(1, d["total"]),
            **_heading_stats(d),
            "reference_vs_raw_rmse_rad": float(np.sqrt(np.mean(ref_diff**2))),
            "reference_vs_raw_rmse_rad_absolute": float(np.sqrt(np.mean((rf_abs - rw_abs) ** 2))),
            "foot_metrics": {
                k.split("/")[1]: v / max(1, d["foot_n"]) for k, v in sorted(d["foot"].items())
            },
        }
        r = result["robots"][d["robot"]]
        print(f'{d["robot"]} vs raw: shape RMSE {rmse:.4f} rad / absolute {r["raw_rmse_rad_absolute"]:.4f} rad '
              f'(alive {r["alive_fraction"]:.2%}, ref-vs-raw {r["reference_vs_raw_rmse_rad"]:.4f}'
              f'/{r["reference_vs_raw_rmse_rad_absolute"]:.4f})')
        fm = r["foot_metrics"]
        if fm:
            print(f'{d["robot"]} feet: pen {fm.get("foot_penetration_m", 0):.4f} m  '
                  f'clear {fm.get("foot_clearance_m", 0):.4f} m  '
                  f'airborne {fm.get("foot_airborne", 0):.3f} (ref {fm.get("ref_foot_airborne", 0):.3f})  '
                  f'zspeed^2 {fm.get("foot_z_speed_sq", 0):.3f}  '
                  f'slip^2 {fm.get("foot_slip_speed_sq", 0):.4f}  '
                  f'ref_pen {fm.get("ref_foot_penetration_m", 0):.4f} m  '
                  f'height_err {fm.get("foot_height_error", 0):.5f}')
    if args.dump_render:
        for i, d in enumerate(per):
            if not d["r_qpos"]:
                continue
            qp = np.array(d["r_qpos"])                     # (T, n, nq)
            dl = np.array(d["r_delta"])
            og = np.array(d["r_origin"])
            w, x, y, z = qp[..., 3], qp[..., 4], qp[..., 5], qp[..., 6]
            yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            e = yaw - og - dl
            dst = f"{args.dump_render}__{d['robot']}.npz"
            np.savez_compressed(
                dst, qpos=qp, reference_joint_targets=np.array(d["r_ref"]),
                reference_root_yaw_delta=dl, root_yaw_origin=og,
                heading_error=np.arctan2(np.sin(e), np.cos(e)),
                reference_root=np.array(d["r_root"]),
                dt=float(train_env.train_envs[i].dt),
            )
            print(f"RENDER DUMP {dst} qpos{qp.shape}")

    Path(args.out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
