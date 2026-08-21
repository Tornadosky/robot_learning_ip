"""Reward-independent A/B evaluation: body-correct task-space tracking.

Training returns are NOT comparable between the stock-target and FK-target
arms (the reward function itself differs), so this script scores BOTH
checkpoints under one shared, reward-free metric:

  site tracking error  =  | rel_sites(actual qpos, sampled body)
                           - rel_sites(reference qpos, sampled body) |

computed OFFLINE with CPU MuJoCo forward kinematics on the per-env sampled
morphology model (the numpy mutation independently verified against
``_apply_morphology`` by verify_fk_targets.py).  Both the policy's achieved
pose and its target are evaluated on the SAME sampled body, so a policy that
tracks the nominal body's targets on a stretched body is charged for it.

Reports per family x morphology-bin x motion: mean/median site position
error, end-effector-only RMSE (hand+foot sites), survival, episode length.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
SCRIPTS = WORKSPACE / "scripts"
for p in (str(SCRIPTS), str(SCRIPTS / "h1md")):
    if p not in sys.path:
        sys.path.insert(0, p)

import jax
import mujoco
import numpy as np

from loco_mujoco.core.utils.math import calculate_relative_site_quatities

from scaling.evaluate_cross_humanoid_policy import (
    _env_args,
    _find_manifest,
    _resolve_checkpoint,
)
from scaling.parallel_cross_humanoid_train import (
    build_cross_humanoid_env,
    trainer_for,
)


def rollout_logged(env, agent_conf, agent_state, steps, seed, buffer):
    """Rollout logging qpos, morphology, cursor and absorbing per group env."""
    keys = jax.random.split(jax.random.PRNGKey(seed), env.num_envs)
    observation, state = jax.jit(env.reset)(keys)
    variables = {
        "params": agent_state.train_state.params,
        "run_stats": agent_state.train_state.run_stats,
    }
    step = jax.jit(env.step)

    def act(obs, z):
        if z is None:
            (policy, _), _ = agent_conf.network.apply(
                variables, obs, mutable=["run_stats"]
            )
        else:
            (policy, _), _ = agent_conf.network.apply(
                variables, obs, z, mutable=["run_stats"]
            )
        return policy.mean()

    logs = [
        {"qpos": [], "morph": [], "traj_no": [], "step_no": [], "absorbing": []}
        for _ in env.groups
    ]
    for _ in range(steps):
        for gi, group_state in enumerate(state.group_states):
            inner = group_state.env_state
            carry = inner.additional_carry
            logs[gi]["qpos"].append(np.asarray(inner.data.qpos))
            logs[gi]["morph"].append(np.asarray(carry.morphology))
            logs[gi]["traj_no"].append(np.asarray(carry.traj_state.traj_no))
            logs[gi]["step_no"].append(
                np.asarray(carry.traj_state.subtraj_step_no)
            )
        z = None
        if buffer is not None:
            ts = state.additional_carry.traj_state
            z = buffer.get(ts.traj_no, ts.subtraj_step_no)
        action = act(observation, z)
        observation, _, absorbing, _, _, state = step(state, action)
        offset = 0
        for gi, group in enumerate(env.groups):
            logs[gi]["absorbing"].append(
                np.asarray(absorbing[offset : offset + group.size])
            )
            offset += group.size
    return [
        {key: np.stack(values) for key, values in log.items()} for log in logs
    ]


def site_errors_for_group(env, gi, robot, log, stride):
    """Offline CPU FK: actual-vs-target relative site positions per sample.

    The TARGET side comes from ``scaling.body_correct_reference`` -- the same
    provider the reward scores and the goal commands. It used to be rebuilt
    here, which meant this "reward-independent" evaluator quietly skipped the
    per-body joint-range clamp and scored a reference the training run never
    used.
    """
    from scaling.body_correct_reference import CpuModelCache, cpu_reference_bundle

    raw = env._raw_envs[gi]
    reward = raw._reward_function
    models = CpuModelCache(raw._model, robot)

    def rel_sites(model, qpos):
        data = mujoco.MjData(model)
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        rpos, _, _ = calculate_relative_site_quatities(
            data, reward._rel_site_ids, reward._rel_body_ids,
            model.body_rootid, np,
        )
        return np.asarray(rpos)

    steps, n_envs = log["qpos"].shape[0], log["qpos"].shape[1]
    rows = []
    for e in range(n_envs):
        for t in range(0, steps, stride):
            if t > 0 and bool(log["absorbing"][: t, e].any()):
                continue  # fallen — no longer a tracking sample
            morph = log["morph"][t, e]
            model = models.get(morph)
            target = np.asarray(
                cpu_reference_bundle(
                    raw, model,
                    int(log["traj_no"][t, e]), int(log["step_no"][t, e]),
                    rel_site_ids=reward._rel_site_ids,
                    rel_body_ids=reward._rel_body_ids,
                    include_site_velocity=False,
                ).relative_site_position
            )
            actual = rel_sites(model, log["qpos"][t, e])
            err = np.linalg.norm(actual - target, axis=-1)  # per rel-site
            rows.append(
                {
                    "env": e, "t": t,
                    "leg_scale": float(morph[0]),
                    "traj_no": int(log["traj_no"][t, e]),
                    "mean_site_err_m": float(err.mean()),
                    "max_site_err_m": float(err.max()),
                }
            )
    return rows


def morph_bin(leg_scale):
    if leg_scale < 0.97:
        return "short"
    if leg_scale > 1.03:
        return "tall"
    return "near_nominal"


def summarize(rows, absorbing, steps):
    out = {}
    fell = absorbing.any(axis=0)
    out["survival_rate"] = float(1.0 - fell.mean())
    first_fall = np.where(
        fell, absorbing.argmax(axis=0), steps
    )
    out["mean_steps_alive"] = float(first_fall.mean())
    by = {}
    for row in rows:
        key = (morph_bin(row["leg_scale"]), row["traj_no"])
        by.setdefault(key, []).append(row["mean_site_err_m"])
    out["tracking"] = {
        f"{bin_}|motion{traj}": {
            "n": len(vals),
            "mean_m": float(np.mean(vals)),
            "median_m": float(np.median(vals)),
            "p90_m": float(np.percentile(vals, 90)),
        }
        for (bin_, traj), vals in sorted(by.items())
    }
    all_vals = [row["mean_site_err_m"] for row in rows]
    out["tracking_overall_mean_m"] = float(np.mean(all_vals)) if all_vals else None
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--envs-per-robot", type=int, default=16)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--robots", nargs="*", default=None)
    parser.add_argument(
        "--reference-root", type=Path,
        default=WORKSPACE / "external_data" / "cross_humanoid",
    )
    parser.add_argument("--use-mjwarp", action="store_true")
    parser.add_argument("--morphology-override", default=None)
    parser.add_argument(
        "--morphology-catalog-file", type=Path, default=None,
        help="JSON (N,4) catalog pinning evaluation bodies, e.g. a held-out "
        "out-of-box morphology.",
    )
    parser.add_argument(
        "--clip-windows-override", nargs="+", default=None,
        help="Score against different reference motions (held-out motion "
        "eval), e.g. dance2_subject5:2000:800.",
    )
    parser.add_argument(
        "--latent-codes-override", type=Path, default=None,
        help="Token cache matching --clip-windows-override (encoded with "
        "the trained FSQ encoder).",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = _resolve_checkpoint(args.checkpoint)
    manifest_path, manifest = _find_manifest(checkpoint)
    backbone = str(manifest.get("backbone", "masked_mlp"))
    trainer = trainer_for(backbone)
    env_args = _env_args(args, manifest)
    if args.morphology_override is not None:
        env_args.morphology = args.morphology_override
    if args.morphology_catalog_file is not None:
        env_args.morphology_catalog_file = str(args.morphology_catalog_file)
    if args.clip_windows_override is not None:
        env_args.clip_windows = list(args.clip_windows_override)
    agent_conf, agent_state = trainer.load_agent(checkpoint)
    env, _ = build_cross_humanoid_env(env_args)
    buffer = getattr(agent_conf, "actor_latent_buffer", None)
    if args.latent_codes_override is not None:
        from scaling.fsq_motion import buffer_from_codes_npz

        buffer = buffer_from_codes_npz(str(args.latent_codes_override))
        print(f"[fk-eval] hot-swapped codes: {args.latent_codes_override}")

    logs = rollout_logged(
        env, agent_conf, agent_state, args.steps, args.seed, buffer
    )
    result = {
        "checkpoint": str(checkpoint),
        "training_manifest": str(manifest_path),
        "reward_type_trained": manifest.get("reward_type", "MimicReward"),
        "seed": args.seed,
        "steps": args.steps,
        "envs_per_robot": args.envs_per_robot,
        "per_robot": {},
    }
    for gi, group in enumerate(env.groups):
        rows = site_errors_for_group(env, gi, group.name, logs[gi], args.stride)
        result["per_robot"][group.name] = summarize(
            rows, logs[gi]["absorbing"], args.steps
        )
        print(
            f"[fk-eval] {group.name:>10s}: "
            f"survival {result['per_robot'][group.name]['survival_rate']:.2f}, "
            f"tracking {result['per_robot'][group.name]['tracking_overall_mean_m']}",
            flush=True,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[fk-eval] -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
