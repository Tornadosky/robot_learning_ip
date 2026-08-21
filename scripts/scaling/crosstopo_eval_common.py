"""Shared rollout / scoring machinery for the cross-topology failure audit.

Three probes (``reward_discrimination_crosstopo``, ``reference_causality``,
``phase_failure_map``) all need the same four things, and each one of them is a
place a previous audit went wrong by re-deriving it:

* **the production environment, rebuilt from the checkpoint's own manifest** --
  goal class, reward class, terminal handler and deviation threshold included.
  A checkpoint replayed under a different goal is a network of the wrong shape;
  replayed under a different terminal rule it measures a different question.
* **a logged rollout** whose per-step arrays are read at the TOP of the step, so
  the auto-reset that ``mjx_step`` performs inside itself cannot overwrite the
  pose that ended the episode.
* **an offline, numpy re-implementation of ``MorphMimicReward``** that returns
  the per-term split the production class does not expose.  It is validated
  against the environment's own scalar reward on every run rather than trusted.
* **a reward-free site tracking error**, computed by CPU forward kinematics on
  the same sampled body for both the achieved pose and its target, through the
  one shared provider (``scaling.body_correct_reference``).

Nothing here edits vendored code or ``scripts/h1md/morph_mimic_reward.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

WORKSPACE = Path(__file__).resolve().parents[2]
for _p in (str(WORKSPACE / "scripts"), str(WORKSPACE / "scripts" / "h1md")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from loco_mujoco.core.utils.math import (  # noqa: E402
    calculate_relative_site_quatities,
    quaternion_angular_distance,
)

from scaling.evaluate_cross_humanoid_policy import (  # noqa: E402
    _env_args,
    _find_manifest,
    _resolve_checkpoint,
)
from scaling.parallel_cross_humanoid_train import (  # noqa: E402
    build_cross_humanoid_env,
    trainer_for,
)

DEFAULT_CHECKPOINT = (
    WORKSPACE
    / "experiments"
    / "urma_h1g1_singlemotion_dance2_subject4_s3"
    / "checkpoints"
    / "continuous_60m_s3"
    / "checkpoint_final"
)
DEFAULT_REFERENCE_ROOT = WORKSPACE / "external_data" / "cross_humanoid"


# --------------------------------------------------------------------------- #
# environment + agent
# --------------------------------------------------------------------------- #
def load_manifest(checkpoint: Path):
    checkpoint = _resolve_checkpoint(Path(checkpoint))
    manifest_path, manifest = _find_manifest(checkpoint)
    return checkpoint, manifest_path, manifest


def build_env(
    manifest,
    *,
    envs_per_robot: int,
    reference_root: Path = DEFAULT_REFERENCE_ROOT,
    morphology: str | None = None,
    root_frame: str = "episode_start",
    robots=None,
):
    """The production environment for this checkpoint, on CPU MJX.

    ``root_frame`` is NOT recorded in this checkpoint's manifest (the field
    postdates it), so it is an explicit argument here and is echoed into every
    output rather than silently defaulted.
    """
    args = SimpleNamespace(
        robots=robots,
        reference_root=Path(reference_root),
        use_mjwarp=False,
        envs_per_robot=int(envs_per_robot),
    )
    env_args = _env_args(args, manifest)
    if morphology is not None:
        env_args.morphology = morphology
    env_args.root_frame = str(root_frame)
    env, metadata = build_cross_humanoid_env(env_args)
    return env, env_args, metadata


def load_policy(checkpoint: Path, manifest):
    trainer = trainer_for(str(manifest.get("backbone", "urmav2")))
    agent_conf, agent_state = trainer.load_agent(Path(checkpoint))
    variables = {
        "params": agent_state.train_state.params,
        "run_stats": agent_state.train_state.run_stats,
    }

    def policy_mean(obs):
        (policy, _), _ = agent_conf.network.apply(
            variables, obs, mutable=["run_stats"]
        )
        return policy.mean()

    return agent_conf, agent_state, policy_mean


# --------------------------------------------------------------------------- #
# per-group static tables (servo gains, reference trajectory, goal slices)
# --------------------------------------------------------------------------- #
def group_tables(env):
    """Static per-group arrays every scripted controller needs.

    ``qpos_adr`` / ``qvel_adr`` are the actuated joints in *actuator* order, so
    a servo command lines up with the action vector without any name matching at
    call time.
    """
    tables = []
    for gi, group in enumerate(env.groups):
        raw = env._raw_envs[gi]
        model = raw._model
        act_idx = np.asarray(raw._action_indices, dtype=int)
        jnt_ids = np.asarray(model.actuator_trnid)[act_idx, 0]
        qpos_adr = np.asarray(model.jnt_qposadr)[jnt_ids]
        qvel_adr = np.asarray(model.jnt_dofadr)[jnt_ids]
        ctrl_low = np.asarray(model.actuator_ctrlrange)[act_idx, 0]
        ctrl_high = np.asarray(model.actuator_ctrlrange)[act_idx, 1]
        control = raw._control_func
        traj = raw.th.traj.data
        tables.append(
            SimpleNamespace(
                name=group.name,
                index=gi,
                start=group.start,
                stop=group.stop,
                size=group.size,
                num_actions=int(env.group_action_dims[gi]),
                obs_dim=int(env.group_observation_dims[gi]),
                model=model,
                qpos_adr=qpos_adr,
                qvel_adr=qvel_adr,
                ctrl_low=ctrl_low,
                ctrl_high=ctrl_high,
                norm_act_mean=np.asarray(control.norm_act_mean, dtype=np.float64),
                norm_act_delta=np.asarray(control.norm_act_delta, dtype=np.float64),
                ref_qpos=np.asarray(traj.qpos, dtype=np.float64),
                ref_qvel=np.asarray(traj.qvel, dtype=np.float64),
                clip_length=int(np.asarray(traj.split_points)[1]),
                reward=raw._reward_function,
                goal=raw._goal,
                terminal=raw._terminal_state_handler,
                raw=raw,
            )
        )
    return tables


#: Shipped PD gains exist for G1 only (``morphology_deepmimic.PD_GAINS``).
#: For H1 there are none in this repository, so the servo derives them from the
#: actuator force limits with the constant that reproduces G1's shipped gains to
#: within a factor of ~2 (G1 hip 88 Nm/kp 100, knee 139/150, ankle 50/40).
#: Reported in every output so the choice is auditable, not hidden.
H1_GAIN_FROM_FORCE_LIMIT = 1.0
H1_DAMPING_FRACTION = 0.025


def servo_gains(table, gain_scale: float = 10.0):
    """(kp, kd, provenance) in actuator order for one group."""
    from morphology_deepmimic import PD_GAINS

    if table.name in PD_GAINS:
        kp = np.asarray(PD_GAINS[table.name]["p_gain"], dtype=np.float64)
        kd = np.asarray(PD_GAINS[table.name]["d_gain"], dtype=np.float64)
        provenance = "morphology_deepmimic.PD_GAINS"
    else:
        force = np.abs(table.ctrl_high)
        kp = H1_GAIN_FROM_FORCE_LIMIT * force
        kd = H1_DAMPING_FRACTION * kp
        provenance = (
            f"derived: kp = {H1_GAIN_FROM_FORCE_LIMIT} * |actuator force limit|, "
            f"kd = {H1_DAMPING_FRACTION} * kp (no shipped gains for this family)"
        )
    return kp * gain_scale, kd * gain_scale, provenance


# --------------------------------------------------------------------------- #
# rollout
# --------------------------------------------------------------------------- #
def _pre_step_log(state, tables):
    out = []
    for t in tables:
        inner = state.group_states[t.index].env_state
        carry = inner.additional_carry
        out.append(
            {
                "qpos": inner.data.qpos,
                "qvel": inner.data.qvel,
                "step_no": carry.traj_state.subtraj_step_no,
                "step_no_init": carry.traj_state.subtraj_step_no_init,
                "traj_no": carry.traj_state.traj_no,
                "morph": getattr(carry, "morphology", jnp.zeros((t.size, 0))),
            }
        )
    return out


def rollout(env, tables, action_fn, steps: int, seed: int, extra_fn=None):
    """Scan a rollout, logging pre-step state and post-step signals.

    ``action_fn(step_index, obs, state, pre)`` returns the padded action.
    ``extra_fn(step_index, obs, state, pre)`` may return an extra pytree to log
    (used by the causality probe to record the goal block it fed).
    """
    keys = jax.random.split(jax.random.PRNGKey(seed), env.num_envs)
    observation, state = jax.jit(env.reset)(keys)

    def body(carry, step_index):
        obs, st = carry
        pre = _pre_step_log(st, tables)
        action = action_fn(step_index, obs, st, pre)
        extra = None if extra_fn is None else extra_fn(step_index, obs, st, pre)
        next_obs, reward, absorbing, done, _info, next_state = env.step(st, action)
        record = {
            "pre": pre,
            "action": action,
            "reward": reward,
            "absorbing": absorbing,
            "done": done,
        }
        if extra is not None:
            record["extra"] = extra
        return (next_obs, next_state), record

    scan = jax.jit(
        lambda c: jax.lax.scan(body, c, jnp.arange(steps, dtype=jnp.int32))
    )
    (_, _), record = scan((observation, state))
    return jax.tree.map(np.asarray, record)


def episode_slices(done):
    """(first_end_index, alive_mask) from a (steps, envs) done array.

    ``alive_mask[t, e]`` is True while episode 1 of env ``e`` is still running,
    inclusive of the step that ends it -- the same convention
    ``evaluate_cross_humanoid_policy`` accumulates return under.
    """
    steps = done.shape[0]
    ended = done.any(axis=0)
    first = np.where(ended, done.argmax(axis=0), steps - 1)
    idx = np.arange(steps)[:, None]
    alive = idx <= first[None, :]
    return first, alive, ended


def per_group(array, tables):
    return {t.name: array[:, t.start : t.stop] for t in tables}


# --------------------------------------------------------------------------- #
# offline reward decomposition (instrumented copy of MorphMimicReward)
# --------------------------------------------------------------------------- #
def _cpu_bundle(table, cpu_model, traj_no, step_no, step_no_init, root_frame):
    from scaling.body_correct_reference import body_correct_reference

    reward = table.reward
    return body_correct_reference(
        table.raw,
        cpu_model,
        None,
        SimpleNamespace(traj_state=None, morphology=None),
        np,
        rel_site_ids=reward._rel_site_ids,
        rel_body_ids=reward._rel_body_ids,
        body_rootid=cpu_model.body_rootid,
        traj_no=int(traj_no),
        subtraj_step_no=int(step_no),
        subtraj_step_no_init=int(step_no_init),
        root_frame=root_frame,
        include_site_velocity=False,
    )


def reward_terms(table, cpu_model, qpos, qvel, action, traj_no, step_no,
                 step_no_init, root_frame):
    """Per-term reward for ONE (post-step) sample, exactly as the class computes it.

    Returns the four exponential terms, their weighted contributions, the
    out-of-bounds penalty and the total, plus the mean site position error the
    ``rpos`` term is a monotone function of.
    """
    reward = table.reward
    bundle = _cpu_bundle(table, cpu_model, traj_no, step_no, step_no_init, root_frame)
    ref_qpos = np.asarray(bundle.reference_qpos_clamped)
    ref_qvel = np.asarray(bundle.reference_qvel)

    qpos_traj = ref_qpos[reward._qpos_ind]
    qvel_traj = ref_qvel[reward._qvel_ind]
    quat_mask = reward._quat_in_qpos
    q = np.asarray(qpos)[reward._qpos_ind]
    v = np.asarray(qvel)[reward._qvel_ind]

    qpos_dist = float(np.mean(np.square(q[~quat_mask] - qpos_traj[~quat_mask])))
    qpos_dist += float(
        np.mean(
            quaternion_angular_distance(
                q[quat_mask].reshape(-1, 4), qpos_traj[quat_mask].reshape(-1, 4), np
            )
        )
    )
    qvel_dist = float(np.mean(np.square(v - qvel_traj)))

    data = mujoco.MjData(cpu_model)
    data.qpos[:] = np.asarray(qpos)
    data.qvel[:] = np.asarray(qvel)
    mujoco.mj_forward(cpu_model, data)
    site_rpos, site_rangles, _ = calculate_relative_site_quatities(
        data, reward._rel_site_ids, reward._rel_body_ids, cpu_model.body_rootid, np
    )
    rpos_traj = np.asarray(bundle.relative_site_position)
    rang_traj = np.asarray(bundle.relative_site_orientation)
    rpos_dist = float(np.mean(np.square(site_rpos - rpos_traj)))
    rang_dist = float(np.mean(np.square(site_rangles - rang_traj)))

    terms = {
        "qpos_term": float(np.exp(-reward._qpos_w_exp * qpos_dist)),
        "qvel_term": float(np.exp(-reward._qvel_w_exp * qvel_dist)),
        "rpos_term": float(np.exp(-reward._rpos_w_exp * rpos_dist)),
        "rquat_term": float(np.exp(-reward._rquat_w_exp * rang_dist)),
    }
    weighted = {
        "qpos": reward._qpos_w_sum * terms["qpos_term"],
        "qvel": reward._qvel_w_sum * terms["qvel_term"],
        "rpos": reward._rpos_w_sum * terms["rpos_term"],
        "rquat": reward._rquat_w_sum * terms["rquat_term"],
    }
    # out_of_bounds_action_cost: sum(((lo-a)_+ + (a-hi)_+)^2) / action_dim,
    # with the action space normalised to [-1, 1] by DefaultControl.
    a = np.asarray(action)[: table.num_actions]
    oob = float(
        np.sum(np.square(np.clip(np.abs(a) - 1.0, 0.0, None))) / table.num_actions
    )
    penalty = max(-reward._action_out_of_bounds_coeff * oob, -1.0)
    total = max(sum(weighted.values()) + penalty, 0.0)
    site_err = np.linalg.norm(site_rpos - rpos_traj, axis=-1)
    return {
        **terms,
        **{f"w_{k}": v for k, v in weighted.items()},
        "penalty": penalty,
        "total": total,
        "site_err_mean_m": float(site_err.mean()),
        "site_err_max_m": float(site_err.max()),
        "qpos_dist": qpos_dist,
        "qvel_dist": qvel_dist,
    }


def score_rollout(record, tables, root_frame, stride: int = 5, max_samples=None):
    """Offline per-term decomposition + site error over the alive part of each episode.

    The decomposition at rollout step ``t`` is evaluated on the state logged at
    ``t + 1`` (which IS the post-step state ``MorphMimicReward`` saw) and the
    action logged at ``t``, so ``total`` is directly comparable with the
    environment's own ``reward[t]``.  That comparison is returned as
    ``validation`` and is the only reason to trust this copy.
    """
    from scaling.body_correct_reference import CpuModelCache

    done = record["done"]
    steps = done.shape[0]
    first, alive, ended = episode_slices(done)
    out = {}
    for t in tables:
        pre = record["pre"][t.index]
        models = CpuModelCache(t.model, t.name)
        rows = []
        errors = []
        for e in range(t.size):
            gi_env = t.start + e
            for step in range(0, min(steps - 1, int(first[gi_env])), stride):
                nxt = step + 1
                cpu_model = models.get(pre["morph"][nxt, e])
                row = reward_terms(
                    t,
                    cpu_model,
                    pre["qpos"][nxt, e],
                    pre["qvel"][nxt, e],
                    record["action"][step, gi_env],
                    pre["traj_no"][nxt, e],
                    pre["step_no"][nxt, e],
                    pre["step_no_init"][nxt, e],
                    root_frame,
                )
                row["env"] = e
                row["t"] = step
                row["env_reward"] = float(record["reward"][step, gi_env])
                rows.append(row)
                errors.append(abs(row["total"] - row["env_reward"]))
                if max_samples is not None and len(rows) >= max_samples:
                    break
            if max_samples is not None and len(rows) >= max_samples:
                break
        keys = [
            "qpos_term", "qvel_term", "rpos_term", "rquat_term",
            "w_qpos", "w_qvel", "w_rpos", "w_rquat", "penalty", "total",
            "site_err_mean_m", "site_err_max_m", "qpos_dist", "qvel_dist",
        ]
        summary = {
            k: (float(np.mean([r[k] for r in rows])) if rows else None) for k in keys
        }
        summary["n_samples"] = len(rows)
        summary["validation"] = {
            "max_abs_diff_vs_env_reward": (
                float(np.max(errors)) if errors else None
            ),
            "mean_abs_diff_vs_env_reward": (
                float(np.mean(errors)) if errors else None
            ),
        }
        out[t.name] = summary
    return out


# --------------------------------------------------------------------------- #
# summaries
# --------------------------------------------------------------------------- #
def rollout_summary(record, tables):
    done = record["done"]
    reward = record["reward"]
    absorbing = record["absorbing"]
    first, alive, ended = episode_slices(done)
    out = {}
    for t in tables:
        sl = slice(t.start, t.stop)
        a = alive[:, sl]
        r = reward[:, sl]
        length = a.sum(axis=0).astype(np.float64)
        ret = (r * a).sum(axis=0)
        fell = (absorbing[:, sl] & a).any(axis=0)
        pre = record["pre"][t.index]
        out[t.name] = {
            "steps_survived_mean": float(length.mean()),
            "steps_survived_median": float(np.median(length)),
            "steps_survived_min": float(length.min()),
            "steps_survived_max": float(length.max()),
            "return_mean": float(ret.mean()),
            "return_std": float(ret.std()),
            "reward_per_step_mean": float(ret.sum() / max(length.sum(), 1.0)),
            "fall_rate": float(fell.mean()),
            "episode_ended_fraction": float(ended[sl].mean()),
            "start_phase_mean": float(np.mean(pre["step_no_init"][0])),
            "num_envs": int(t.size),
        }
    return out
