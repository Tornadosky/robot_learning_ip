"""Tier 2 -- is the trained policy conditioned on the reference at all?

The repository has a recorded precedent (``realwalk`` session 2) where a policy
that appeared to be imitating scored 1.10x on a scrambled reference versus the
true one, i.e. it was ignoring the target completely and the "tracking" number
measured only the clip's own statistics.  This probe repeats that test on the
cross-topology checkpoint, at EVAL TIME only: nothing about the environment, the
reward, the termination rule or the physics changes, so every arm is scored by
the identical objective.  Only the numbers the *actor* reads are touched.

The manipulated dims are the target side of the goal block, located from the
goal's own index arrays (``body_correct_goal.goal_block_slices``) rather than
hard-coded: ``reference_qpos``, ``reference_qvel``, ``target_site_position``,
``target_site_orientation``, ``target_site_velocity``, ``root_position_error``.
The current-state site block, which is proprioception rather than command, is
left alone -- zeroing it would test a different thing (sensor loss).

Arms:

* ``a_unmodified``      -- the production observation.
* ``b_frozen_frame0``   -- the target block held at the value emitted on the
                           episode's first step (the frame-0 target under RSI).
* ``c_time_shuffled``   -- at every step, the target block recorded at a random
                           other step of the same environment's arm-(a) rollout:
                           same distribution, destroyed timing.
* ``d_zeroed``          -- the target block set to zero.

If (b)/(c)/(d) score close to (a), the actor is not conditioned on the target
and the defect is in the goal/observation path, not in physics or budget.
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

from scaling.body_correct_goal import goal_block_slices  # noqa: E402
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
)

#: Goal sub-blocks that carry the COMMAND (what the reference says to do), as
#: opposed to the current-state blocks that carry proprioception.
TARGET_BLOCKS = (
    "reference_qpos",
    "reference_qvel",
    "target_site_position",
    "target_site_orientation",
    "target_site_velocity",
    "root_position_error",
)

ARMS = ("a_unmodified", "b_frozen_frame0", "c_time_shuffled", "d_zeroed")


def target_indices(env, tables):
    """Absolute indices, in the padded observation, of each group's target block."""
    out = []
    for t in tables:
        goal = t.goal
        base = int(np.asarray(goal.obs_ind)[0])
        obs_ind = np.asarray(goal.obs_ind)
        if not np.all(np.diff(obs_ind) == 1):
            raise ValueError(
                f"{t.name}: goal observation indices are not contiguous; the "
                "block map below would slice the wrong dims."
            )
        slices = goal_block_slices(goal)
        idx = np.concatenate(
            [np.arange(slices[b].start, slices[b].stop) for b in TARGET_BLOCKS]
        )
        out.append(
            dict(
                indices=jnp.asarray(base + idx, dtype=jnp.int32),
                width=int(idx.size),
                first=int(base + idx.min()),
                last=int(base + idx.max()),
                blocks={b: [slices[b].start, slices[b].stop] for b in TARGET_BLOCKS},
                goal_obs_start=base,
                goal_dim=int(goal.dim),
            )
        )
    return out


def _gather_targets(obs, env, tables, target):
    """(num_envs, width) of the current target block, zero-padded across groups."""
    width = max(t["width"] for t in target)
    rows = []
    for t, spec in zip(tables, target, strict=True):
        block = obs[t.start : t.stop][:, spec["indices"]]
        pad = width - spec["width"]
        rows.append(jnp.pad(block, ((0, 0), (0, pad))) if pad else block)
    return jnp.concatenate(rows, axis=0)


def _apply(obs, env, tables, target, replacement):
    """Overwrite each group's target dims with ``replacement`` (num_envs, width)."""
    out = obs
    for t, spec in zip(tables, target, strict=True):
        rows = jnp.arange(t.start, t.stop)[:, None]
        cols = spec["indices"][None, :]
        out = out.at[rows, cols].set(
            replacement[t.start : t.stop, : spec["width"]]
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--envs-per-robot", type=int, default=16)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--shuffle-seed", type=int, default=11)
    parser.add_argument("--morphology", default="nominal")
    parser.add_argument("--root-frame", default="episode_start",
                        choices=("episode_start", "absolute"))
    parser.add_argument("--score-stride", type=int, default=5)
    parser.add_argument("--reference-root", type=Path,
                        default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint, manifest_path, manifest = load_manifest(args.checkpoint)
    env, _env_args, _meta = build_env(
        manifest,
        envs_per_robot=args.envs_per_robot,
        reference_root=args.reference_root,
        morphology=args.morphology,
        root_frame=args.root_frame,
    )
    tables = group_tables(env)
    _conf, _state, policy_mean = load_policy(checkpoint, manifest)
    target = target_indices(env, tables)

    result = {
        "probe": "T2_reference_causality",
        "checkpoint": str(checkpoint),
        "training_manifest": str(manifest_path),
        "root_frame_used": args.root_frame,
        "manifest_records_root_frame": manifest.get("root_frame", None),
        "goal_type": manifest.get("goal_type"),
        "reward_type": manifest.get("reward_type"),
        "morphology": args.morphology,
        "robots": list(env.names),
        "envs_per_robot": args.envs_per_robot,
        "steps": args.steps,
        "seed": args.seed,
        "manipulated_blocks": list(TARGET_BLOCKS),
        "target_block_layout": {
            t.name: {k: v for k, v in spec.items() if k != "indices"}
            for t, spec in zip(tables, target, strict=True)
        },
        "arms": {},
    }

    # ---- arm (a): unmodified, and the bank the other arms are built from -----
    def act_a(i, obs, st, pre):
        return policy_mean(obs)

    def extra_a(i, obs, st, pre):
        return _gather_targets(obs, env, tables, target)

    record_a = rollout(env, tables, act_a, args.steps, args.seed, extra_fn=extra_a)
    bank = jnp.asarray(record_a["extra"])          # (steps, num_envs, width)
    frozen = bank[0]                               # (num_envs, width)

    rng = np.random.default_rng(args.shuffle_seed)
    shuffle = jnp.asarray(
        rng.integers(0, args.steps, size=(args.steps, env.num_envs)), dtype=jnp.int32
    )
    zeros = jnp.zeros_like(frozen)

    def act_b(i, obs, st, pre):
        return policy_mean(_apply(obs, env, tables, target, frozen))

    def act_c(i, obs, st, pre):
        picked = bank[shuffle[i], jnp.arange(env.num_envs)]
        return policy_mean(_apply(obs, env, tables, target, picked))

    def act_d(i, obs, st, pre):
        return policy_mean(_apply(obs, env, tables, target, zeros))

    records = {
        "a_unmodified": record_a,
        "b_frozen_frame0": rollout(env, tables, act_b, args.steps, args.seed),
        "c_time_shuffled": rollout(env, tables, act_c, args.steps, args.seed),
        "d_zeroed": rollout(env, tables, act_d, args.steps, args.seed),
    }

    for arm, record in records.items():
        summary = rollout_summary(record, tables)
        decomposition = score_rollout(
            record, tables, args.root_frame, stride=args.score_stride
        )
        for name in summary:
            summary[name]["reward_decomposition"] = decomposition[name]
        result["arms"][arm] = summary
        for name, block in summary.items():
            print(
                f"[t2] {arm:>16s} {name:>3s}: return {block['return_mean']:8.2f} "
                f"steps {block['steps_survived_mean']:6.1f} "
                f"site_err {block['reward_decomposition']['site_err_mean_m']:.4f} m",
                flush=True,
            )

    ratios = {}
    for name in env.names:
        base = result["arms"]["a_unmodified"][name]
        ratios[name] = {
            arm: {
                "return_ratio_a_over_arm": (
                    base["return_mean"]
                    / max(result["arms"][arm][name]["return_mean"], 1e-9)
                ),
                "steps_ratio_a_over_arm": (
                    base["steps_survived_mean"]
                    / max(result["arms"][arm][name]["steps_survived_mean"], 1e-9)
                ),
                "site_err_ratio_arm_over_a": (
                    result["arms"][arm][name]["reward_decomposition"][
                        "site_err_mean_m"
                    ]
                    / max(base["reward_decomposition"]["site_err_mean_m"], 1e-9)
                ),
            }
            for arm in ARMS
            if arm != "a_unmodified"
        }
    result["ratios_vs_unmodified"] = ratios

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[t2] -> {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
