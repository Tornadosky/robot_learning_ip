"""T0.5 -- does the production cross-topology reward prefer good behaviour?

The single-body pipeline answered this analytically (``h1md/c6_reward_discrimination``:
perfect tracking vs a deliberately wrong reference, no simulator in the loop).
That form cannot be reused here, because the question is different: the H1/G1
policy is *alive but badly tracking*, so what has to be compared is not two
references but several **behaviours**, each stepped through the real
environment, each scored by the real ``MorphMimicReward``.

Five scripted arms, one environment, identical reset keys (so every arm starts
from the same RSI phases and the same bodies):

1. ``oracle``      -- a stiff PD servo onto the reference joint angles at the
                      current trajectory cursor, torques clipped to the
                      actuator limits.  The best in-family imitator this action
                      space admits.
2. ``stand_still``  -- the same servo holding the episode's own start pose.
3. ``zero_action``  -- no torque at all.
4. ``scrambled``    -- the same servo onto the reference in a fixed random frame
                      permutation: identical pose distribution, destroyed
                      timing.
5. ``policy``       -- the trained checkpoint.

If ``stand_still`` earns as much as or more than ``oracle``, the policy is
behaving near-optimally for the objective it was given and the reward is the
bug.  The ranking is stated explicitly in the output.

Note on ``--root-frame``: this checkpoint's manifest predates the field and its
training command did not pass it, so the run trained under ``absolute``.  The
flag is exposed here and echoed into the output; run both if the ranking is
close.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
for _p in (str(WORKSPACE / "scripts"), str(WORKSPACE / "scripts" / "h1md")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from scaling.crosstopo_eval_common import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_REFERENCE_ROOT,
    build_env,
    group_tables,
    load_manifest,
    load_policy,
    rollout,
    rollout_summary,
    score_rollout,
    servo_gains,
)

ARMS = ("oracle", "stand_still", "zero_action", "scrambled", "policy")


def _servo_context(env, tables, gain_scale, permutation_seed):
    """Static jnp tables for the PD arms, per group."""
    context = []
    rng = np.random.default_rng(permutation_seed)
    for t in tables:
        kp, kd, provenance = servo_gains(t, gain_scale)
        perm = rng.permutation(t.clip_length)
        context.append(
            dict(
                kp=jnp.asarray(kp, dtype=jnp.float32),
                kd=jnp.asarray(kd, dtype=jnp.float32),
                mean=jnp.asarray(t.norm_act_mean, dtype=jnp.float32),
                delta=jnp.asarray(t.norm_act_delta, dtype=jnp.float32),
                ref=jnp.asarray(t.ref_qpos, dtype=jnp.float32),
                qpos_adr=jnp.asarray(t.qpos_adr, dtype=jnp.int32),
                qvel_adr=jnp.asarray(t.qvel_adr, dtype=jnp.int32),
                perm=jnp.asarray(perm, dtype=jnp.int32),
                provenance=provenance,
                kp_values=kp.tolist(),
                kd_values=kd.tolist(),
            )
        )
    return context


def _servo_action(env, tables, context, pre, mode):
    """Padded action for every group from a PD servo onto a chosen reference frame."""
    blocks = []
    for t, ctx, log in zip(tables, context, pre, strict=True):
        if mode == "track":
            idx = log["step_no"]
        elif mode == "hold_start":
            idx = log["step_no_init"]
        elif mode == "scrambled":
            idx = ctx["perm"][log["step_no"]]
        else:
            raise ValueError(mode)
        q_ref = ctx["ref"][idx][:, ctx["qpos_adr"]]
        q = log["qpos"][:, ctx["qpos_adr"]]
        v = log["qvel"][:, ctx["qvel_adr"]]
        torque = ctx["kp"] * (q_ref - q) - ctx["kd"] * v
        action = jnp.clip((torque - ctx["mean"]) / ctx["delta"], -1.0, 1.0)
        pad = env.max_action_dim - t.num_actions
        if pad:
            action = jnp.pad(action, ((0, 0), (0, pad)))
        blocks.append(action)
    return jnp.concatenate(blocks, axis=0)


def build_action_fn(env, tables, context, arm, policy_mean):
    if arm == "zero_action":
        return lambda i, obs, st, pre: jnp.zeros(
            (env.num_envs, env.max_action_dim), dtype=jnp.float32
        )
    if arm == "policy":
        return lambda i, obs, st, pre: policy_mean(obs)
    mode = {"oracle": "track", "stand_still": "hold_start",
            "scrambled": "scrambled"}[arm]
    return lambda i, obs, st, pre: _servo_action(env, tables, context, pre, mode)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--envs-per-robot", type=int, default=16)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--gain-scale", type=float, default=10.0)
    parser.add_argument("--permutation-seed", type=int, default=7)
    parser.add_argument("--morphology", default="nominal")
    parser.add_argument("--root-frame", default="episode_start",
                        choices=("episode_start", "absolute"))
    parser.add_argument("--score-stride", type=int, default=5)
    parser.add_argument("--arms", nargs="*", default=list(ARMS))
    parser.add_argument("--reference-root", type=Path,
                        default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint, manifest_path, manifest = load_manifest(args.checkpoint)
    env, env_args, _meta = build_env(
        manifest,
        envs_per_robot=args.envs_per_robot,
        reference_root=args.reference_root,
        morphology=args.morphology,
        root_frame=args.root_frame,
    )
    tables = group_tables(env)
    _conf, _state, policy_mean = load_policy(checkpoint, manifest)
    context = _servo_context(env, tables, args.gain_scale, args.permutation_seed)

    result = {
        "probe": "T0_5_reward_discrimination_crosstopo",
        "checkpoint": str(checkpoint),
        "training_manifest": str(manifest_path),
        "manifest_records_root_frame": manifest.get("root_frame", None),
        "root_frame_used": args.root_frame,
        "reward_type": manifest.get("reward_type"),
        "goal_type": manifest.get("goal_type"),
        "terminal_handler": manifest.get("terminal_handler"),
        "max_root_pos_deviation": manifest.get("max_root_pos_deviation"),
        "morphology": args.morphology,
        "robots": list(env.names),
        "envs_per_robot": args.envs_per_robot,
        "steps": args.steps,
        "seed": args.seed,
        "gain_scale": args.gain_scale,
        "clip": manifest.get("clip"),
        "clip_window": manifest.get("clip_windows"),
        "servo_gains": {
            t.name: {
                "provenance": ctx["provenance"],
                "kp": ctx["kp_values"],
                "kd": ctx["kd_values"],
            }
            for t, ctx in zip(tables, context, strict=True)
        },
        "max_possible_step_reward": float(
            tables[0].reward._qpos_w_sum + tables[0].reward._qvel_w_sum
            + tables[0].reward._rpos_w_sum + tables[0].reward._rquat_w_sum
        ),
        "arms": {},
    }

    for arm in args.arms:
        action_fn = build_action_fn(env, tables, context, arm, policy_mean)
        record = rollout(env, tables, action_fn, args.steps, args.seed)
        summary = rollout_summary(record, tables)
        decomposition = score_rollout(
            record, tables, args.root_frame, stride=args.score_stride
        )
        for name in summary:
            summary[name]["reward_decomposition"] = decomposition[name]
        result["arms"][arm] = summary
        for name, block in summary.items():
            print(
                f"[t05] {arm:>12s} {name:>3s}: return {block['return_mean']:8.2f} "
                f"r/step {block['reward_per_step_mean']:.4f} "
                f"steps {block['steps_survived_mean']:6.1f} "
                f"site_err {block['reward_decomposition']['site_err_mean_m']} "
                f"(validation max|d| "
                f"{block['reward_decomposition']['validation']['max_abs_diff_vs_env_reward']})",
                flush=True,
            )

    # explicit ranking, per robot, on mean undiscounted return and on r/step
    ranking = {}
    for t in tables:
        for metric in ("return_mean", "reward_per_step_mean"):
            order = sorted(
                result["arms"],
                key=lambda a: result["arms"][a][t.name][metric],
                reverse=True,
            )
            ranking.setdefault(t.name, {})[metric] = [
                {"arm": a, "value": result["arms"][a][t.name][metric]} for a in order
            ]
    result["ranking"] = ranking
    result["verdict"] = {
        name: {
            "oracle_beats_stand_still_on_return": (
                result["arms"]["oracle"][name]["return_mean"]
                > result["arms"]["stand_still"][name]["return_mean"]
                if {"oracle", "stand_still"} <= set(result["arms"]) else None
            ),
            "oracle_beats_stand_still_on_reward_per_step": (
                result["arms"]["oracle"][name]["reward_per_step_mean"]
                > result["arms"]["stand_still"][name]["reward_per_step_mean"]
                if {"oracle", "stand_still"} <= set(result["arms"]) else None
            ),
            "oracle_beats_scrambled_on_reward_per_step": (
                result["arms"]["oracle"][name]["reward_per_step_mean"]
                > result["arms"]["scrambled"][name]["reward_per_step_mean"]
                if {"oracle", "scrambled"} <= set(result["arms"]) else None
            ),
            "policy_beats_zero_action_on_return": (
                result["arms"]["policy"][name]["return_mean"]
                > result["arms"]["zero_action"][name]["return_mean"]
                if {"policy", "zero_action"} <= set(result["arms"]) else None
            ),
        }
        for name in env.names
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[t05] -> {args.output}", flush=True)
    for name, order in ranking.items():
        print(f"[t05] {name} by return: "
              + " > ".join(f"{e['arm']}({e['value']:.1f})"
                           for e in order["return_mean"]), flush=True)
        print(f"[t05] {name} by r/step: "
              + " > ".join(f"{e['arm']}({e['value']:.3f})"
                           for e in order["reward_per_step_mean"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
