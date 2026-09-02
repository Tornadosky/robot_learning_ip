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
import zipfile
import tempfile
import time
from pathlib import Path

import numpy as np

# This is a SECOND copy of the map that lives in urma2/mjx/default_config.py as
# tracking_clip_robot_map. Adding a family means editing both, and forgetting
# this one costs a full rollout: the env builds happily and the scorer then dies
# on KeyError after the 3-robot graph has already compiled for ten minutes.
ROBOT_TO_CLIP_SUBDIR = {
    "unitree_h1": "UnitreeH1", "h1": "UnitreeH1",
    "unitree_g1": "UnitreeG1", "g1": "UnitreeG1",
    "booster_t1": "BoosterT1", "t1": "BoosterT1",
    "atlas": "Atlas", "talos": "Talos", "toddlerbot": "ToddlerBot",
    "unitree_h1v2": "UnitreeH1v2", "h1v2": "UnitreeH1v2",
    "at": "Atlas", "tl": "Talos", "tdlb": "ToddlerBot",
    "apptronik_apollo": "Apollo", "apo": "Apollo", "fourier_gr1t2": "FourierGR1T2", "gr1t2": "FourierGR1T2",
}


class _DummyWriter:
    def add_scalar(self, *args, **kwargs):
        pass

    def close(self):
        pass


def make_eval_checkpoint(model_path: str, work_dir: str) -> str:
    """Copy only the selected model and remove resume-only archive members.

    Intermediate checkpoints can share the source directory with ``latest.model``.
    Copying the whole directory makes each evaluation duplicate every checkpoint,
    while filtering sibling files cannot remove resume metadata stored *inside*
    the model ZIP.  Rewriting one archive gives URMA2.load() a plain evaluation
    checkpoint and bounds temporary disk use to one model.
    """
    src = Path(model_path)
    if not src.is_file():
        raise FileNotFoundError(src)
    dst = Path(work_dir) / "model"
    dst.mkdir(parents=True, exist_ok=True)
    target = dst / src.name
    resume_members = {"training_progress.json", "resume_manifest.json", "resume_state.npz"}
    with zipfile.ZipFile(src, "r") as source, zipfile.ZipFile(target, "w") as output:
        for info in source.infolist():
            if Path(info.filename).name in resume_members:
                continue
            output.writestr(info, source.read(info.filename))
    return str(target)


def load_raw_joints(raw_clip_dir: str, robot: str, clip: str, mj_model):
    """Raw clip joints for the subset of actuators the clip covers.

    Returns (raw_joints (T, K), actuator_ids (K,), names (K,)). The env's model
    can have more actuated joints than the clip (G1 waist_roll/pitch); those
    are excluded from the metric on every arm equally.
    """
    import mujoco

    # THIRD copy of the sign lookup (env: clip_reference.load_clip; robot->subdir
    # map above; here). T1_CLIP_SIGNS was missing until 2026-08-27, so every
    # booster_t1 joint scored at +1.0 while 14 of its 23 are flipped: the first
    # 3-topology crosseval reported t1 at 0.888 rad with ref-vs-raw 0.964, i.e.
    # the "reference" was a rad off the clip it was being compared to. That is
    # the convention mismatch this function's own comment warns about, not a
    # policy failure. The three name spaces are disjoint (verified), so the
    # chained lookup below is safe.
    from loco_mjx.environments.locomotion.urma2.mjx.clip_reference import (
        H1_CLIP_SIGNS, G1_CLIP_SIGNS, T1_CLIP_SIGNS,
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
            signs.append(H1_CLIP_SIGNS.get(
                name, G1_CLIP_SIGNS.get(name, T1_CLIP_SIGNS.get(name, 1.0))))
    raw = np.asarray(d["qpos"], dtype=np.float64)[:, 7:][:, np.array(cols, dtype=np.int64)]
    # Same per-joint sign correction the env applies when loading a clip
    # (clip_reference.load_clip); without it, flipped joints score ~1.5 rad of
    # pure convention mismatch. Returned ABSOLUTE (uncentered): the anchor-fix
    # era scores an absolute metric too, and the shape metric re-centers at
    # scoring time from these same angles.
    raw = raw * np.array(signs, dtype=np.float64)[None, :]
    return raw, np.array(act_ids, dtype=np.int64), names



_AGE_BINS = (0, 5, 10, 20, 40, 80, 160, 320, 640, 10 ** 9)


def _age_binned(d, diff):
    """Tracking RMSE conditioned on WITHIN-EPISODE step index.

    The env auto-resets, so a crosseval's `steps` frames are many stitched
    episodes and every sample is bounded by how long the policy survives. An arm
    that survives longer is therefore sampled from further into the drift, and a
    plain per-frame RMSE comparison between two arms silently mixes "tracks
    better" with "died sooner". This is the same failure family as the
    per-episode-mean heading confound (F10), one level deeper: a FIXED horizon
    does not remove it.

    Compare arms bin-for-bin. Wrapped like _heading_stats: instrumentation must
    never be what breaks a crosseval.
    """
    try:
        import numpy as _np
        chunks = [c for c in d.get("age_samples", []) if len(c)]
        if not chunks:
            return {"age_binned_rmse_rad": None}
        age = _np.concatenate(chunks, axis=0)
        n = min(age.shape[0], diff.shape[0])
        age, dd = age[:n], diff[:n]
        out = {}
        for lo, hi in zip(_AGE_BINS[:-1], _AGE_BINS[1:]):
            m = (age >= lo) & (age < hi)
            k = f"{lo}-{hi - 1}" if hi < 10 ** 9 else f"{lo}+"
            out[k] = ({"n": int(m.sum()),
                       "rmse_rad": float(_np.sqrt(_np.mean(dd[m] ** 2)))}
                      if m.any() else {"n": 0, "rmse_rad": None})
        return {"age_binned_rmse_rad": out,
                "age_mean": float(age.mean()), "age_p95": float(_np.percentile(age, 95))}
    except Exception:  # noqa: BLE001
        return {"age_binned_rmse_rad": None}


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


def build_arg_parser() -> argparse.ArgumentParser:
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
    p.add_argument("--root_heading_obs", default="False", choices=("True", "False"),
                   help="tracking_clip_observe_root_heading, as trained; must match checkpoint width")
    p.add_argument("--robots", default="unitree_h1:unitree_g1",
                   help="colon-separated robot set, EXACTLY as the arm trained")
    p.add_argument("--latent_hold", type=int, default=1,
                   help="tracking_clip_latent_hold, as trained")
    p.add_argument("--latent_replaces", default="True", choices=("True", "False"),
                   help="tracking_clip_latent_replaces_reference, as trained. False "
                        "means the arm saw BOTH the reference and the token, which is "
                        "a WIDER joint observation -- hardcoding True (as this script "
                        "did until 26-08) makes such a checkpoint unloadable.")
    p.add_argument("--jlat_enc_dim", type=int, default=4,
                   help="algorithm.joint_latent_encoder_dim, as trained. The "
                        "separate-encoder arms changed the NETWORK, so an eval "
                        "that leaves this at the default rebuilds the wrong "
                        "policy and still returns a number.")
    p.add_argument("--latent_scope", default="per_joint",
                   choices=("per_joint", "global", "both"),
                   help="tracking_clip_latent_scope, as trained. Changes the "
                        "observation LAYOUT, so a mismatch loads a checkpoint "
                        "against the wrong channels.")
    p.add_argument("--latent_divisor", type=float, default=1.0,
                   help="tracking_clip_latent_obs_divisor, as trained.")
    p.add_argument("--latent_dim", type=int, default=32,
                   help="tracking_clip_latent_dim, as trained. Kevin's per-joint "
                        "tokenizer is 32; the CANONICAL shared-stream tokenizer is "
                        "4 (codebook 8x5x5x5, one code per frame). A mismatch "
                        "changes the observation width and the checkpoint will "
                        "not load.")
    p.add_argument("--latent_sidecar", default="_zq",
                   help="tracking_clip_latent_sidecar_suffix, as trained (_win for co-training arms)")
    p.add_argument("--reference_hold", type=int, default=1,
                   help="freeze the OBSERVED reference to every K-th clip frame; "
                        "the reward target stays fresh (K=1 disables)")
    p.add_argument("--contact_timeconst", type=float, default=0.0,
                   help="terrain.contact_solref_timeconst; must match the training arm")
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
    p.add_argument("--morphology_coeff", type=float, default=0.0,
                   help="reset-time seen-body morphology magnitude; 0 evaluates nominal bodies")
    p.add_argument("--torque_scaling_exponent", type=float, default=1.0,
                   help="seen-body actuator torque scaling exponent used by the checkpoint recipe")
    p.add_argument("--exact_inertia_rescale", default="False", choices=("True", "False"),
                   help="use exact inertia re-diagonalization for randomized evaluation bodies")
    p.add_argument("--body_pool_size", type=int, default=0,
                   help="optional deterministic finite randomized-body pool; 0 samples continuously")
    p.add_argument("--out", required=True)
    return p


def _as_bool(value: str) -> bool:
    return value == "True"


def apply_evaluation_settings(config, args) -> None:
    """Reproduce observation width and configure nominal/reset-only morphology eval."""
    if not 0.0 <= args.morphology_coeff <= 1.0:
        raise ValueError("morphology_coeff must be in [0, 1]")
    if args.torque_scaling_exponent <= 0.0:
        raise ValueError("torque_scaling_exponent must be positive")
    if args.body_pool_size < 0:
        raise ValueError("body_pool_size must be >= 0")
    config.environment.command.tracking_clip_observe_root_heading = _as_bool(
        args.root_heading_obs)
    dr = config.environment.domain_randomization
    seen = dr.seen_robot
    seen.morphology_coeff_mode = "fixed"
    seen.morphology_coeff_value = float(args.morphology_coeff)
    seen.torque_scaling_exponent = float(args.torque_scaling_exponent)
    seen.exact_inertia_rescale = _as_bool(args.exact_inertia_rescale)
    seen.body_pool_size = int(args.body_pool_size)
    if args.morphology_coeff > 0.0:
        # Probability zero means no mid-episode mutation; the "and_reset"
        # component still samples a body at each reset.
        dr.sampling_type = "step_probability_and_reset"
        dr.sampling_probability = 0.0
    else:
        dr.sampling_type = "none"


def make_eval_condition(args, robots) -> dict:
    return {
        "clip": args.clip, "raw_clip_dir": args.raw_clip_dir,
        "robots": list(robots), "refbias": args.refbias,
        "anchor": args.anchor, "fitvariant": args.fitvariant,
        "refroot": args.refroot, "refroot_floor": args.refroot_floor,
        "refvel_obs": args.refvel_obs, "root_heading_obs": args.root_heading_obs,
        "latent": bool(args.latent), "latent_hold": args.latent_hold,
        "latent_dim": args.latent_dim, "zero_action": bool(args.zero_action),
        "latent_scope": args.latent_scope, "latent_divisor": float(args.latent_divisor),
        "jlat_enc_dim": int(args.jlat_enc_dim),
        "latent_replaces": args.latent_replaces,
        "latent_sidecar": str(args.latent_sidecar),
        "reference_hold": args.reference_hold,
        "morphology_coeff": float(args.morphology_coeff),
        "torque_scaling_exponent": float(args.torque_scaling_exponent),
        "exact_inertia_rescale": args.exact_inertia_rescale,
        "body_pool_size": int(args.body_pool_size),
    }


def main() -> None:
    args = build_arg_parser().parse_args()

    import importlib

    import jax
    import mujoco
    from os import stat as _os_stat
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
    config.algorithm.joint_latent_encoder_dim = int(args.jlat_enc_dim)

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
    config.environment.terrain.contact_solref_timeconst = args.contact_timeconst
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
        config.environment.command.tracking_clip_latent_scope = args.latent_scope
        config.environment.command.tracking_clip_latent_obs_divisor = float(args.latent_divisor)
        config.environment.command.tracking_clip_latent_sidecar_suffix = str(args.latent_sidecar)
    config.environment.domain_randomization.initial_state.type = "reference"
    apply_evaluation_settings(config, args)
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
            # 2026-08-28 (F16/F17): the env AUTO-RESETS, so these `steps` frames
            # are many stitched episodes, not one trajectory -- and no sample is
            # ever drawn from deeper into an episode than the policy survives.
            # An arm that survives longer therefore samples from further into the
            # drift, which biases any per-frame RMSE comparison between arms.
            # `age` is the within-episode step index of each env, so the error
            # can be conditioned on it. `n_term` counts resets, which is the
            # honest survival number: alive_fraction is ~1.0 by construction
            # because only the single terminal frame is masked, and must NEVER be
            # quoted as survival.
            "age": None, "age_samples": [], "n_term": 0,
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
            # Age is recorded for the SAME rows the samples above were taken
            # from, then advanced; terminated envs restart at 0 because the env
            # has already auto-reset them.
            if d["age"] is None:
                d["age"] = np.zeros(ok.shape[0], dtype=np.int64)
            d["age_samples"].append(d["age"][ok].copy())
            d["n_term"] += int(terminated_all[i].sum())
            d["age"] = np.where(terminated_all[i], 0, d["age"] + 1)
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

    # A crosseval submitted while its arm is still TRAINING silently scores a
    # partially-trained policy, because the launcher resolves latest.model at
    # JOB START and that path is MUTABLE. It leaves no error and model_path is
    # byte-identical between a contaminated run and a clean one, so the artifact
    # gives no signal at all -- the only tell was comparing sacct timestamps.
    # Recording the resolved file's mtime and size turns that silent failure
    # into a visible one: two runs of "the same" checkpoint that disagree here
    # did not evaluate the same weights.
    try:
        _st = _os_stat(args.model_path)
        _mtime, _size = float(_st.st_mtime), int(_st.st_size)
    except Exception:  # noqa: BLE001
        _mtime, _size = None, None
    result = {"model_path": args.model_path,
              "model_mtime": _mtime, "model_size_bytes": _size,
              "clip_dir": args.clip_dir,
              # The 24-08 FSQ control was destroyed by two evaluations of one
              # checkpoint writing the same ${EXP}.json. Stamping the condition
              # INTO the artifact means a collision is at least detectable.
              "eval_condition": make_eval_condition(args, robots),
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
        # F20 (2026-08-28): the shape metric above centres its two sides by
        # DIFFERENT constants -- rw by the FULL-CLIP mean, ex and rf by their own
        # SAMPLE means. A perfect tracker (ex_abs == rw_abs) therefore leaves a
        # residual of clip_mean - mean(rw_abs): the gap between the whole clip and
        # the phases actually VISITED. Measured by resampling the real clip at the
        # observed episode lengths, that manufactures 0.0206 rad on H1 and 0.0125
        # on G1, and it falls to exactly 0 when episodes cover the clip. It is why
        # shape exceeds absolute on both robots, which consistent centring makes
        # impossible.
        #
        # ADDITIVE by decision: every historical table is built on the keys above,
        # so they are left exactly as they are and the consistently-centred
        # versions are emitted alongside. Both on one artifact is the only way to
        # learn how much the defect actually moved anything.
        rw_self = rw_abs - rw_abs.mean(0, keepdims=True)
        diff_self = ex - rw_self
        ref_diff_self = rf - rw_self
        # Absolute metric (anchor-fix era): no centering anywhere. A centered-
        # anchor arm keeps its constant rest-pose bias here, which is the point.
        diff_abs = ex_abs - rw_abs
        result["robots"][d["robot"]] = {
            "raw_rmse_rad": rmse,
            "raw_rmse_rad_absolute": float(np.sqrt(np.mean(diff_abs**2))),
            # F20: all three sides centred by their own sample mean. Compare
            # against raw_rmse_rad to see how much the centring mismatch moved.
            "raw_rmse_rad_selfcentred": float(np.sqrt(np.mean(diff_self**2))),
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
            # F20: per-joint, consistently centred. The manufactured term is a
            # CONSTANT in radians, so as a fraction of a joint's own signal it
            # lands hardest on LOW-amplitude chains -- 21.7% of trunk's reference
            # std against 3.8% of the arms' on H1. That is exactly where F15's
            # amplitude-normalised conclusions live, so the chain analysis has to
            # be re-read on this key rather than on the centred one.
            "per_joint_rmse_rad_selfcentred": {
                name: float(np.sqrt(np.mean(diff_self[:, k] ** 2)))
                for k, name in enumerate(d["names"])
            },
            "samples": int(ex.shape[0]),
            # SIGNED per-joint bias, executed minus reference. The implied bias
            # sqrt(absolute^2 - centred^2) recovers only a MAGNITUDE, which left
            # the direction of the 29-degree H1 ankle offset (F17) unresolvable
            # from the JSON. This is the signed quantity, one line, and it says
            # which way the joint is displaced.
            "per_joint_bias_rad": {
                name: float(np.mean(diff_abs[:, k]))
                for k, name in enumerate(d["names"])
            },
            # Pearson correlation, executed vs raw clip, PER JOINT. RMSE cannot
            # tell "actively moving the wrong way" from "not moving with the
            # reference at all", and on 2026-08-28 the hips scored WORSE than
            # zero action in both metric variants -- which a sign flip and a
            # learning failure both predict. This separates them:
            #   corr ~ -1 : anti-tracked. A sign/axis defect (this repo has had
            #               13/19 H1 axes reversed once) or an inverted target.
            #   corr ~  0 : not tracked. The reference carries no influence.
            #   corr >  0 : tracked, badly. RMSE is then amplitude or phase.
            # Zero marginal cost: the arrays are already in hand.
            "per_joint_corr": {
                name: (
                    float(
                        np.corrcoef(ex_abs[:, k], rw_abs[:, k])[0, 1]
                    )
                    if float(np.std(ex_abs[:, k])) > 1e-9
                    and float(np.std(rw_abs[:, k])) > 1e-9
                    else None
                )
                for k, name in enumerate(d["names"])
            },
            # The PIPELINE's own per-joint discrepancy: the env's internal
            # reference against the raw clip, with nothing to do with the policy.
            # Aggregate `reference_vs_raw_rmse_rad` is 0.0283 rad absolute on H1
            # but only 0.0050 on G1 (2026-08-28), so it is neither negligible nor
            # uniform across robots -- and because per_joint_rmse_rad scores
            # executed-vs-RAW, whatever sits here is inside every per-joint number
            # we quote. Emitting it per joint is what lets the 29-degree H1 ankle
            # offset (F17) be split into "the policy stands differently" and "the
            # reference the policy was given already differed from the clip".
            "per_joint_ref_vs_raw_rmse_rad": {
                name: float(np.sqrt(np.mean(ref_diff[:, k] ** 2)))
                for k, name in enumerate(d["names"])
            },
            "per_joint_ref_bias_rad": {
                name: float(np.mean((rf_abs - rw_abs)[:, k]))
                for k, name in enumerate(d["names"])
            },
            # Survival, honestly. alive_fraction below is ~1.0 BY CONSTRUCTION
            # (only the terminal frame is masked) and is not a survival number.
            "n_terminations": int(d["n_term"]),
            "mean_episode_length_steps": (
                float(d["total"]) / max(1, d["n_term"])),
            **_age_binned(d, diff),
            "alive_fraction": d["alive"] / max(1, d["total"]),
            **_heading_stats(d),
            "reference_vs_raw_rmse_rad": float(np.sqrt(np.mean(ref_diff**2))),
            # F20: the FLOOR is where the centring mismatch bit hardest -- ~40% of
            # H1's reported shape floor and ~58% of G1's was manufactured. Use
            # this one, or the absolute variant, whenever quoting a floor.
            "reference_vs_raw_rmse_rad_selfcentred": float(
                np.sqrt(np.mean(ref_diff_self**2))),
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
