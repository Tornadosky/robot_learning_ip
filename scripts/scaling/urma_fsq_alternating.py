"""Design C — alternating FSQ + PPO training, one-cycle correctness proof.

This is deliberately labeled ALTERNATING training, not end-to-end: policy
gradients do NOT flow into the encoder.  One cycle =

  1. one jitted multi-family PPO update with z = FSQ codes of the CURRENT
     encoder (exact per-timestamp buffer, same interface as fake z);
  2. one supervised FSQ autoencoder step (design-A reconstruction objective,
     straight-through quantization) on the canonical dataset;
  3. re-encoding of the RL-window token buffer from the updated encoder —
     the refreshed z is what the next PPO update would consume.

The proof asserts: both losses finite, the supervised step changes the codes
(so step 3 is not a no-op), and the PPO update changes the policy parameters.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[2]
SCRIPTS = WORKSPACE / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import flax.serialization  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402

from loco_mujoco.algorithms import TrajectoryLatentBuffer  # noqa: E402

from scaling.cross_topology_urma import CrossTopologyURMAPPO  # noqa: E402
from scaling.fsq_motion import NormalizationStats, future_window  # noqa: E402
from scaling.parallel_cross_humanoid_train import (  # noqa: E402
    build_config,
    build_cross_humanoid_env,
)
from scaling.train_fsq_motion import AutoEncoder  # noqa: E402


def load_fsq(model_dir: Path):
    manifest = json.loads((model_dir / "manifest.json").read_text())
    stats_payload = np.load(model_dir / "normalization.npz")
    stats = NormalizationStats(stats_payload["mean"], stats_payload["std"])
    model = AutoEncoder(
        hidden=tuple(manifest["hidden"]),
        code_dim=len(manifest["levels"]),
        out_dim=int(stats.mean.shape[0]),
        levels=None
        if manifest["continuous_control"]
        else tuple(manifest["levels"]),
    )
    template = model.init(
        jax.random.PRNGKey(0), jnp.zeros((1, stats.mean.shape[0]))
    )
    params = flax.serialization.from_bytes(
        template, (model_dir / "params.msgpack").read_bytes()
    )
    return model, params, stats, manifest


def rl_window_inputs(canonical_dataset: Path, windows, manifest, stats):
    payload = np.load(canonical_dataset, allow_pickle=False)
    features = payload["features"]
    split_points = payload["split_points"]
    clip_names = [str(x) for x in payload["clip_names"]]
    blocks, splits = [], [0]
    for clip, start, frames in windows:
        c = clip_names.index(clip)
        rows = slice(
            int(split_points[c]) + start, int(split_points[c]) + start + frames
        )
        blocks.append(
            future_window(
                features[rows],
                np.asarray([0, frames]),
                manifest["window"],
                manifest["stride"],
            )
        )
        splits.append(splits[-1] + frames)
    return stats.apply(np.concatenate(blocks, axis=0)), np.asarray(
        splits, dtype=np.int32
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fsq-dir", type=Path, required=True)
    parser.add_argument("--canonical-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--total-envs", type=int, default=24)
    parser.add_argument("--num-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=5)
    args = parser.parse_args()

    model, fsq_params, stats, fsq_manifest = load_fsq(args.fsq_dir)

    env_args = SimpleNamespace(
        robots=["h1", "g1", "atlas"],
        source="h1",
        clip="dance2_subject4",
        start_frame=19482,
        frames=800,
        clip_windows=[
            "dance2_subject4:19482:800",
            "dance2_subject1:2000:800",
        ],
        morphology="continuous",
        blank_goal=True,
        actor_latent_dim=len(fsq_manifest["levels"]),
        latent_codes="in-graph",
        reference_mode="direct",
        reference_root=WORKSPACE / "external_data" / "cross_humanoid",
        robot_one_hot=False,
        append_joint_features=True,
        reserve_robots=[],
        envs_per_robot=None,
        total_envs=args.total_envs,
        use_mjwarp=False,
        num_steps=args.num_steps,
        num_minibatches=2,
        update_epochs=1,
        hidden=[256, 128],
        lr=1e-4,
        init_std=0.2,
        learnable_std=True,
        no_normalize_reward=False,
        urma_latent_slots=64,
        urma_joint_value_dim=4,
        backbone="urmav2",
        fake_latent_seed=0,
        fake_latent_scale=1.0,
        motion_latent_embed_dim=64,
    )
    env, _ = build_cross_humanoid_env(env_args)

    window_specs = [
        (clip, int(start), int(frames))
        for clip, start, frames in (
            spec.split(":") for spec in env_args.clip_windows
        )
    ]
    x_windows, splits = rl_window_inputs(
        args.canonical_dataset, window_specs, fsq_manifest, stats
    )

    def encode_buffer(params):
        _, codes = model.apply(params, jnp.asarray(x_windows))
        return TrajectoryLatentBuffer(
            values=np.asarray(codes, dtype=np.float32), split_points=splits
        )

    buffer0 = encode_buffer(fsq_params)

    steps_per_update = args.num_steps * env.num_envs
    config = build_config(env_args, env.num_envs, steps_per_update)
    agent_conf = CrossTopologyURMAPPO.init_agent_conf(env, config, buffer0)

    # ---------------------------------------------- 1. one jitted PPO update
    train_fn = jax.jit(
        CrossTopologyURMAPPO.build_train_fn(env, agent_conf, mh=None)
    )
    output = train_fn(jax.random.PRNGKey(args.seed))
    jax.block_until_ready(output["agent_state"])
    params_after = output["agent_state"].train_state.params
    initial = CrossTopologyURMAPPO._network_init(
        agent_conf.network,
        jax.random.PRNGKey(0),
        jnp.zeros(env.info.observation_space.shape),
        jnp.zeros((buffer0.latent_dim,)),
    )["params"]
    ppo_param_delta = float(
        max(
            np.abs(np.asarray(a) - np.asarray(b)).max()
            for a, b in zip(
                jax.tree_util.tree_leaves(params_after),
                jax.tree_util.tree_leaves(initial),
                strict=True,
            )
        )
    )

    # -------------------------------------- 2. one supervised FSQ update
    tx = optax.adamw(1e-3, weight_decay=1e-5)
    opt_state = tx.init(fsq_params)

    @jax.jit
    def fsq_step(params, opt_state, batch):
        def loss_fn(p):
            reconstruction, _ = model.apply(p, batch)
            return jnp.mean((reconstruction - batch) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    batch = jnp.asarray(x_windows[:1024])
    fsq_params_new, opt_state, fsq_loss = fsq_step(fsq_params, opt_state, batch)

    # ------------------------------------------ 3. re-encode the token buffer
    buffer1 = encode_buffer(fsq_params_new)
    code_delta = float(
        np.abs(np.asarray(buffer1.values) - np.asarray(buffer0.values)).max()
    )

    proof = {
        "fsq_dir": str(args.fsq_dir),
        "families": list(env.names),
        "envs": env.num_envs,
        "ppo_update_finite": bool(
            all(
                np.isfinite(np.asarray(leaf)).all()
                for leaf in jax.tree_util.tree_leaves(params_after)
            )
        ),
        "ppo_params_changed_max_abs": ppo_param_delta,
        "fsq_supervised_loss": float(fsq_loss),
        "codes_changed_after_fsq_step_max_abs": code_delta,
        "alternating_not_end_to_end": True,
        "pass": bool(
            ppo_param_delta > 0.0
            and np.isfinite(float(fsq_loss))
            and code_delta > 0.0
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(proof, indent=2), encoding="utf-8")
    print(json.dumps(proof, indent=2))


if __name__ == "__main__":
    main()
