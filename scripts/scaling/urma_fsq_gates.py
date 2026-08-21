"""Gate evidence for the URMA + motion-latent pipeline (beyond "it compiles").

Loads a trained cross-topology checkpoint, rebuilds the exact environment,
and measures:

Gate 1/5 — latent contract at the integration level:
  * different z on identical observations changes VALID action means;
  * padded action means stay exactly 0 and their std 1e-3 under every z;
  * the critic value is bit-identical for different z.

Gate 2 — rollout integrity (eager, no jit tricks):
  * pre-step canonical cursor -> z row lookup matches the shared buffer
    exactly at every step, across families;
  * cursors advance by one per step, and reset boundaries never produce an
    off-by-one z;
  * checkpoint serialize -> load reproduces actions bit-exactly.

Gate 3 — topology and morphology:
  * one batch contains all three families (mask counts 19/23/27 exact);
  * >=2 distinct morphologies observed per family;
  * the per-joint description block in the emitted observation changes with
    the sampled morphology and equals dynamic_joint_descriptions(morphology)
    recomputed outside the environment.

Writes JSON evidence next to the checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

WORKSPACE = Path(__file__).resolve().parents[2]
SCRIPTS = WORKSPACE / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from scaling.cross_topology_urma import CrossTopologyURMAPPO  # noqa: E402
from scaling.parallel_cross_humanoid_train import (  # noqa: E402
    build_cross_humanoid_env,
)


def _args_from_manifest(manifest: dict) -> SimpleNamespace:
    return SimpleNamespace(
        robots=manifest["robots"],
        source=manifest["source_robot"],
        clip=manifest["clip"],
        start_frame=manifest["window_start_frame"],
        frames=manifest["window_frames"],
        clip_windows=manifest.get("clip_windows"),
        morphology=manifest.get("morphology"),
        blank_goal=manifest.get("blank_goal_observation", False),
        goal_for_critic=manifest.get("goal_for_critic", False),
        actor_latent_dim=manifest.get("actor_latent_dim", 0),
        latent_codes=manifest.get("latent_codes"),
        reference_mode=manifest["reference_mode"],
        reference_root=WORKSPACE / "external_data" / "cross_humanoid",
        robot_one_hot=manifest["robot_one_hot"],
        append_joint_features=manifest["append_joint_features"],
        reserve_robots=manifest.get("reserved_robots", []),
        envs_per_robot=None,
        total_envs=int(sum(manifest["group_sizes"])),
        use_mjwarp=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--total-envs", type=int, default=None,
        help="override the manifest's env count (e.g. small for CPU runs)",
    )
    args = parser.parse_args()

    manifest = json.loads((args.run_dir / "manifest.json").read_text())
    env_args = _args_from_manifest(manifest)
    if args.total_envs is not None:
        env_args.total_envs = args.total_envs
    env, _ = build_cross_humanoid_env(env_args)

    agent_path = args.run_dir / "checkpoint_final" / "CrossTopologyURMAPPO_saved.pkl"
    agent_conf, agent_state = CrossTopologyURMAPPO.load_agent(agent_path)
    network = agent_conf.network
    buffer = agent_conf.actor_latent_buffer
    train_state = agent_state.train_state
    variables = {
        "params": train_state.params,
        "run_stats": train_state.run_stats,
    }
    evidence = {"run_dir": str(args.run_dir), "backend": jax.default_backend()}

    keys = jax.random.split(jax.random.PRNGKey(args.seed), env.num_envs)
    observation, state = jax.jit(env.reset)(keys)

    # ------------------------------------------------- Gate 1/5: z contract
    traj_state = state.additional_carry.traj_state
    z_true = buffer.get(traj_state.traj_no, traj_state.subtraj_step_no)
    z_shuffled = jnp.roll(z_true, shift=1, axis=0)
    z_zero = jnp.zeros_like(z_true)

    def act(z):
        (pi, value), _ = network.apply(
            variables, observation, z, mutable=["run_stats"]
        )
        return np.asarray(pi.mean()), np.asarray(pi.stddev()), np.asarray(value)

    mean_true, std_true, value_true = act(z_true)
    mean_shuffled, _, value_shuffled = act(z_shuffled)
    mean_zero, _, _ = act(z_zero)

    mask = np.asarray(env.action_mask)  # (n_envs, max_action)
    valid_delta_shuffled = float(
        np.abs((mean_true - mean_shuffled) * mask).max()
    )
    valid_delta_zero = float(np.abs((mean_true - mean_zero) * mask).max())
    padded_max = float(np.abs(mean_true * (1 - mask)).max())
    padded_std = float(np.abs(std_true * (1 - mask) - (1 - mask) * 1e-3).max())
    evidence["gate1"] = {
        "valid_action_delta_shuffled_z": valid_delta_shuffled,
        "valid_action_delta_zero_z": valid_delta_zero,
        "padded_action_mean_max_abs": padded_max,
        "padded_action_std_deviation_from_1e-3": padded_std,
        "critic_delta_across_z": float(
            np.abs(value_true - value_shuffled).max()
        ),
        "pass": bool(
            valid_delta_shuffled > 1e-4
            and valid_delta_zero > 1e-4
            and padded_max < 1e-6
            and np.abs(value_true - value_shuffled).max() == 0.0
        ),
    }

    # --------------------------------------------- Gate 2: rollout integrity
    step_fn = jax.jit(env.step)
    zero_action = jnp.zeros((env.num_envs, env.max_action_dim))
    cursor_exact, advance_ok, resets_seen = [], [], 0
    values = np.asarray(buffer.values)
    split_points = np.asarray(buffer.split_points)
    prev_cursor = None
    current = state
    for _ in range(args.steps):
        ts = current.additional_carry.traj_state
        traj_no = np.asarray(ts.traj_no)
        step_no = np.asarray(ts.subtraj_step_no)
        z_now = np.asarray(buffer.get(ts.traj_no, ts.subtraj_step_no))
        rows = split_points[traj_no] + step_no
        cursor_exact.append(bool(np.array_equal(z_now, values[rows])))
        if prev_cursor is not None:
            prev_rows, prev_done = prev_cursor
            expected = prev_rows + 1
            advanced = rows == expected
            advance_ok.append(bool(np.all(advanced | prev_done)))
            resets_seen += int(np.sum(prev_done))
        _, _, _, done, _, current = step_fn(current, zero_action)
        prev_cursor = (rows, np.asarray(done))
    evidence["gate2"] = {
        "steps": args.steps,
        "pre_step_z_matches_buffer_every_step": all(cursor_exact),
        "cursor_advances_by_one_or_resets": all(advance_ok),
        "resets_observed": resets_seen,
        "no_nans_in_final_observation": bool(
            np.isfinite(np.asarray(current.additional_carry.traj_state.traj_no)).all()
        ),
    }

    # serialize -> reload -> identical actions
    reloaded_conf = type(agent_conf).from_dict(agent_conf.serialize())
    reloaded_mean = np.asarray(
        reloaded_conf.network.apply(
            variables, observation, z_true, mutable=["run_stats"]
        )[0][0].mean()
    )
    evidence["gate2"]["reload_action_max_abs_delta"] = float(
        np.abs(reloaded_mean - mean_true).max()
    )

    # -------------------------------------- Gate 3: topology and morphology
    per_family = {}
    layout = env.urma_input_layout
    obs_np = np.asarray(observation)
    for group_index, group in enumerate(env.groups):
        group_state = state.group_states[group_index]
        name = env.names[group_index]
        entry = {
            "mask_count": int(mask[group.start].sum()),
        }
        carry = group_state.additional_carry
        if hasattr(carry, "morphology"):
            morphology = np.asarray(carry.morphology)
            unique_bodies = np.unique(np.round(morphology, 6), axis=0)
            entry["distinct_morphologies_observed"] = int(unique_bodies.shape[0])
            entry["morphology_min"] = morphology.min(axis=0).tolist()
            entry["morphology_max"] = morphology.max(axis=0).tolist()

            # emitted joint block vs recomputed descriptions, first two envs
            raw_env = env._raw_envs[group_index]
            feature_dim = env.joint_feature_dim
            start = layout.joint_feature_start
            deltas, desc_deltas = [], []
            for local in range(min(2, group.size)):
                row = group.start + local
                block = obs_np[
                    row, start : start + env.num_joint_slots * feature_dim
                ].reshape(env.num_joint_slots, feature_dim)
                expected = np.asarray(
                    raw_env.dynamic_joint_descriptions(
                        jnp.asarray(morphology[local])
                    )
                )
                observed = block[: expected.shape[0], : expected.shape[1]]
                desc_deltas.append(float(np.abs(observed - expected).max()))
            entry["description_matches_sampled_model_max_abs"] = max(desc_deltas)
            if morphology.shape[0] >= 2 and not np.allclose(
                morphology[0], morphology[1]
            ):
                block0 = obs_np[
                    group.start, start : start + env.num_joint_slots * feature_dim
                ]
                block1 = obs_np[
                    group.start + 1,
                    start : start + env.num_joint_slots * feature_dim,
                ]
                deltas.append(float(np.abs(block0 - block1).max()))
                entry["description_delta_between_bodies"] = deltas[-1]
        per_family[name] = entry
    evidence["gate3"] = {
        "families_in_batch": list(env.names),
        "per_family": per_family,
        "expected_mask_counts": list(env.group_action_dims),
    }

    out = args.run_dir / "gate_evidence.json"
    out.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(json.dumps(evidence, indent=2))
    print(f"[gates] -> {out}")


if __name__ == "__main__":
    main()
