"""Why did the episode end? Attribute terminations to the criterion that fired.

``absorbing`` is one boolean, but ``RootPoseTrajTerminalStateHandler`` ORs three
independent conditions:

* **root deviation** -- the root strayed further than ``max_root_pos_deviation``
  from the reference's own displacement. That is drifting, not falling.
* **root height** -- outside the clip's height window plus margin. Falling, or
  standing at the wrong height.
* **root rotation** -- further from the clip's orientation centroid than its own
  spread plus margin. Toppling.

Reporting "80% fell" without that split is how a drift problem gets mistaken for
a balance problem. Two independent readings are produced:

1. **Margins.** Every criterion is evaluated on every step, so the state one
   control step before termination shows which one was at its limit. (The
   terminal pose itself is unrecoverable: ``mjx_step`` auto-resets inside
   itself, so the post-step ``data`` on a done step is already the new episode.)
2. **Ablation.** The same checkpoint is re-rolled with the deviation limit
   effectively disabled. If episode length jumps, deviation was binding; if it
   does not, the policy is genuinely losing its balance.

The height check mirrors ``MorphologyAwareRootPoseTrajTerminalStateHandler``:
the sampled body's grounding offset is subtracted first, so a taller body is
judged against its own standing height.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

WORKSPACE = Path(__file__).resolve().parents[2]
for _p in (str(WORKSPACE / "scripts"), str(WORKSPACE / "scripts" / "h1md")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from loco_mujoco.core.utils.math import quat_scalarfirst2scalarlast  # noqa: E402

from scaling.evaluate_cross_humanoid_policy import (  # noqa: E402
    _env_args,
    _find_manifest,
    _resolve_checkpoint,
)
from scaling.parallel_cross_humanoid_train import (  # noqa: E402
    build_cross_humanoid_env,
    trainer_for,
)

CONDITIONS = ("root_deviation", "root_height", "root_rotation")


def _rollout(env, agent_conf, agent_state, steps, seed, zero_action):
    """Per-step root deviation / height / rotation, plus the done flags.

    State is read at the TOP of each iteration, so index t holds the pose the
    policy was in before step t executed -- which is what survives an auto-reset.
    """
    variables = {"params": agent_state.train_state.params,
                 "run_stats": agent_state.train_state.run_stats}

    def act(obs):
        if zero_action:
            return jnp.zeros((obs.shape[0], env.max_action_dim), dtype=obs.dtype)
        (policy, _), _ = agent_conf.network.apply(
            variables, obs, mutable=["run_stats"])
        return policy.mean()

    keys = jax.random.split(jax.random.PRNGKey(seed), env.num_envs)
    observation, state = jax.jit(env.reset)(keys)
    step_fn = jax.jit(env.step)

    logs = [{"deviation": [], "height": [], "rotation": [], "phase": [],
             "done": []} for _ in env.groups]
    for _ in range(steps):
        for gi, group_state in enumerate(state.group_states):
            raw = env._raw_envs[gi]
            handler = raw._terminal_state_handler
            inner = group_state.env_state
            carry = inner.additional_carry
            qpos = np.asarray(inner.data.qpos)
            traj_no = np.asarray(carry.traj_state.traj_no)
            step_no = np.asarray(carry.traj_state.subtraj_step_no)
            step_init = np.asarray(carry.traj_state.subtraj_step_no_init)
            morphology = (np.asarray(carry.morphology)
                          if hasattr(carry, "morphology") else None)

            root_xy = np.asarray(handler.root_xy)
            traj = raw.th.traj.data
            reference = np.stack([
                np.asarray(traj.get(int(t), int(s), jnp).qpos)[root_xy]
                for t, s in zip(traj_no, step_no)])
            initial = np.stack([
                np.asarray(traj.get(int(t), int(s), jnp).qpos)[root_xy]
                for t, s in zip(traj_no, step_init)])
            deviation = np.linalg.norm(
                qpos[:, root_xy] - (reference - initial), axis=-1)

            height = qpos[:, int(handler.root_height_ind)].copy()
            if morphology is not None and hasattr(raw, "root_height_offset"):
                height = height - np.asarray(
                    jax.vmap(raw.root_height_offset)(jnp.asarray(morphology)))

            quat = quat_scalarfirst2scalarlast(
                qpos[:, np.asarray(handler.root_quat_ind)])
            quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
            angle = 2 * np.arccos(
                np.clip(quat @ np.asarray(handler._centroid_quat), -1, 1))

            logs[gi]["deviation"].append(deviation)
            logs[gi]["height"].append(height)
            logs[gi]["rotation"].append(angle)
            logs[gi]["phase"].append(np.asarray(step_no))

        action = act(observation)
        observation, _, absorbing, done, _, state = step_fn(state, action)
        offset = 0
        for gi, group in enumerate(env.groups):
            logs[gi]["done"].append(
                np.asarray(done, dtype=bool)[offset:offset + group.size])
            offset += group.size
    return [{k: np.stack(v) for k, v in log.items()} for log in logs]


def _severity(value, low, high=None):
    """Distance to a limit on one scale: 0 = safe, 1 = exactly at it, >1 = past.

    For a one-sided limit (deviation, rotation) that is just value/limit. For the
    two-sided height window it is the fractional distance from the window centre,
    so all three criteria can be compared directly.
    """
    if high is None:
        return float(value) / max(float(low), 1e-9)
    centre = 0.5 * (low + high)
    half = max(0.5 * (high - low), 1e-9)
    return abs(float(value) - centre) / half


def _outcomes(env, logs, steps):
    """Split episode endings the way the pipeline-v3 dashboard did.

    "Alive at horizon" silently mixes falling with benign reference exhaustion:
    with random start phases an environment that begins near the end of the clip
    reaches the last frame after a few hundred steps and is marked done, having
    failed at nothing. Any survival number that does not separate the two is not
    comparable across runs.
    """
    out = {}
    for gi, group in enumerate(env.groups):
        handler = env._raw_envs[gi]._terminal_state_handler
        low, high = [float(v) for v in handler.root_height_range]
        limit = float(handler.max_root_pos_deviation)
        threshold = float(handler._valid_threshold)
        log = logs[gi]
        clip_length = int(np.asarray(env._raw_envs[gi].th.traj.data.split_points)[1])

        counts = {"survived_horizon": 0, "reference_exhausted_benign": 0,
                  "fell_height": 0, "fell_rotation": 0, "drifted": 0,
                  "unexplained": 0}
        n = log["done"].shape[1]
        for e in range(n):
            ends = np.nonzero(log["done"][:, e])[0]
            if ends.size == 0:
                counts["survived_horizon"] += 1
                continue
            t = int(ends[0])  # classify the FIRST ending only, like v3
            sev_dev = log["deviation"][t, e] / max(limit, 1e-9)
            sev_hgt = abs(log["height"][t, e] - 0.5 * (low + high)) / max(
                0.5 * (high - low), 1e-9)
            sev_rot = log["rotation"][t, e] / max(threshold, 1e-9)
            phase = int(log["phase"][t, e])
            if sev_hgt >= 0.9:
                counts["fell_height"] += 1
            elif sev_rot >= 0.9:
                counts["fell_rotation"] += 1
            elif sev_dev >= 0.9:
                counts["drifted"] += 1
            elif phase >= clip_length - 3:
                counts["reference_exhausted_benign"] += 1
            else:
                counts["unexplained"] += 1
        total = max(n, 1)
        fell = counts["fell_height"] + counts["fell_rotation"]
        out[group.name] = {
            **counts,
            "envs": n,
            "clip_length": clip_length,
            "genuine_fall_rate": round(fell / total, 3),
            "drift_rate": round(counts["drifted"] / total, 3),
            "benign_rate": round(counts["reference_exhausted_benign"] / total, 3),
        }
    return out


def _attribute(env, logs):
    """Which criterion was closest to its limit one control step before the end."""
    out = {}
    for gi, group in enumerate(env.groups):
        handler = env._raw_envs[gi]._terminal_state_handler
        low, high = [float(v) for v in handler.root_height_range]
        limit = float(handler.max_root_pos_deviation)
        threshold = float(handler._valid_threshold)
        log = logs[gi]

        counts = {c: 0 for c in CONDITIONS}
        counts["none_near_limit"] = 0
        severities = {c: [] for c in CONDITIONS}
        episodes = 0

        rows, cols = np.nonzero(log["done"])
        for t, e in zip(rows, cols):
            episodes += 1
            sev = {
                "root_deviation": _severity(log["deviation"][t, e], limit),
                "root_height": _severity(log["height"][t, e], low, high),
                "root_rotation": _severity(log["rotation"][t, e], threshold),
            }
            for key, value in sev.items():
                severities[key].append(value)
            worst = max(sev, key=sev.get)
            # below 0.5 nothing was close, so the episode ended for another
            # reason entirely (horizon, or the reference clip running out)
            counts[worst if sev[worst] >= 0.5 else "none_near_limit"] += 1

        out[group.name] = {
            "episodes_ended": int(episodes),
            "attribution_one_step_before_end": counts,
            "mean_severity_at_end": {
                k: (round(float(np.mean(v)), 3) if v else None)
                for k, v in severities.items()
            },
            "limits": {"max_root_pos_deviation_m": limit,
                       "root_height_range_m": [low, high],
                       "rotation_threshold_rad": threshold},
            "observed": {
                "deviation_p50_m": round(float(np.median(log["deviation"])), 3),
                "deviation_p95_m": round(float(np.percentile(log["deviation"], 95)), 3),
                "height_p05_m": round(float(np.percentile(log["height"], 5)), 3),
                "height_p95_m": round(float(np.percentile(log["height"], 95)), 3),
                "rotation_p95_rad": round(float(np.percentile(log["rotation"], 95)), 3),
            },
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--envs-per-robot", type=int, default=8)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument(
        "--ablate-deviation", action="store_true",
        help="Also roll out with the deviation limit effectively disabled.")
    parser.add_argument(
        "--reference-root", type=Path,
        default=WORKSPACE / "external_data" / "cross_humanoid")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = _resolve_checkpoint(args.checkpoint)
    manifest_path, manifest = _find_manifest(checkpoint)
    trainer = trainer_for(str(manifest.get("backbone", "urmav2")))
    agent_conf, agent_state = trainer.load_agent(checkpoint)

    report = {
        "checkpoint": str(checkpoint),
        "training_manifest": str(manifest_path),
        "steps": args.steps,
        "envs_per_robot": args.envs_per_robot,
        "trained_max_root_pos_deviation": manifest.get("max_root_pos_deviation"),
        "terminal_handler": manifest.get("terminal_handler"),
        "arms": {},
    }

    arms = [("production", None)]
    if args.ablate_deviation:
        arms.append(("deviation_disabled", 1e6))

    for label, override in arms:
        env_args = _env_args(
            SimpleNamespace(robots=None, reference_root=args.reference_root,
                            use_mjwarp=False, envs_per_robot=args.envs_per_robot),
            manifest,
        )
        if override is not None:
            env_args.max_root_deviation = override
        env, _ = build_cross_humanoid_env(env_args)
        logs = _rollout(env, agent_conf, agent_state, args.steps, args.seed,
                        zero_action=False)
        arm = _attribute(env, logs)
        outcomes = _outcomes(env, logs, args.steps)
        for name, block in outcomes.items():
            arm[name]["outcomes"] = block
        for gi, group in enumerate(env.groups):
            done = logs[gi]["done"]
            first = np.where(done.any(axis=0), done.argmax(axis=0), args.steps)
            arm[group.name]["mean_steps_to_first_end"] = float(np.mean(first + 1))
            arm[group.name]["episodes_per_env"] = float(
                done.sum(axis=0).mean())
        report["arms"][label] = arm
        print(f"[term] arm={label}")
        for name, block in arm.items():
            print(f"  {name}: first end at {block['mean_steps_to_first_end']:.0f} "
                  f"steps, {block['episodes_ended']} endings, "
                  f"attribution {block['attribution_one_step_before_end']}")
            print(f"      observed {block['observed']}")
            o = block["outcomes"]
            print(f"      outcomes of {o['envs']}: survived {o['survived_horizon']}"
                  f" | benign clip-end {o['reference_exhausted_benign']}"
                  f" | fell(height) {o['fell_height']}"
                  f" | fell(rot) {o['fell_rotation']}"
                  f" | drifted {o['drifted']}"
                  f" | unexplained {o['unexplained']}"
                  f"  -> genuine fall rate {o['genuine_fall_rate']:.2f}")
        del env

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[term] -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
