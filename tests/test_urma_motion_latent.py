"""Gate 1 — motion-latent / URMA contract.

The motion latent z must be a separate actor input that (a) changes valid
actions, (b) never unmasks padded actions, (c) never reaches the critic,
(d) is keyed canonically so every family sees the same z for the same
(motion, timestamp), and (e) leaves latent-free checkpoints byte-compatible.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omegaconf import OmegaConf

from loco_mujoco.algorithms import TrajectoryLatentBuffer
from loco_mujoco.trajectory import TrajState

from scaling.parallel_env import ParallelMorphState, ParallelMorphVecEnv
from scaling.urma_networks import URMAActorCritic, URMAAgentConf

LATENT_DIM = 6


def _network(variant, actor_latent_dim=LATENT_DIM):
    return URMAActorCritic(
        action_dim=3,
        base_observation_dim=8,
        joint_feature_start=12,
        num_joints=3,
        joint_description_dim=4,
        joint_state_dim=3,
        joint_feature_dim=8,
        general_indices=(0, 1, 8, 9, 10, 11),
        variant=variant,
        core_hidden=(16, 8),
        latent_slots=8,
        joint_value_dim=2,
        learnable_std=True,
        actor_latent_dim=actor_latent_dim,
    )


def _observations(batch_size=4, mark_last_joint_invalid=False):
    observations = np.zeros((batch_size, 36), dtype=np.float32)
    rng = np.random.default_rng(7)
    observations[:] = rng.normal(size=observations.shape)
    features = observations[:, 12:].reshape(batch_size, 3, 8)
    features[..., -1] = 1.0
    if mark_last_joint_invalid:
        features[:, -1, -1] = 0.0
    return jnp.asarray(observations)


def _latents(batch_size=4, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    return jnp.asarray(
        rng.uniform(-scale, scale, size=(batch_size, LATENT_DIM)).astype(
            np.float32
        )
    )


@pytest.mark.parametrize("variant", ["urma", "urmav2"])
def test_latent_changes_valid_actions_but_not_critic(variant):
    network = _network(variant)
    observations = _observations(mark_last_joint_invalid=True)
    z_a = _latents(seed=1)
    z_b = _latents(seed=2)
    variables = network.init(jax.random.PRNGKey(0), observations, z_a)

    (pi_a, value_a), _ = network.apply(
        variables, observations, z_a, mutable=["run_stats"]
    )
    (pi_b, value_b), _ = network.apply(
        variables, observations, z_b, mutable=["run_stats"]
    )

    mean_a = np.asarray(pi_a.mean())
    mean_b = np.asarray(pi_b.mean())
    # (a) the command changes what the valid joints do,
    assert not np.allclose(mean_a[:, :2], mean_b[:, :2])
    # (b) padded joints stay masked under every command,
    np.testing.assert_allclose(mean_a[:, -1], 0.0, atol=1e-7)
    np.testing.assert_allclose(mean_b[:, -1], 0.0, atol=1e-7)
    np.testing.assert_allclose(np.asarray(pi_a.stddev())[:, -1], 1e-3, rtol=1e-5)
    # (c) the critic never sees z.
    np.testing.assert_allclose(
        np.asarray(value_a), np.asarray(value_b), atol=0.0, rtol=0.0
    )


@pytest.mark.parametrize("variant", ["urma", "urmav2"])
def test_padded_joint_perturbation_never_changes_valid_actions(variant):
    network = _network(variant)
    base = np.array(_observations(mark_last_joint_invalid=True), copy=True)
    z = _latents(seed=3)
    variables = network.init(jax.random.PRNGKey(1), jnp.asarray(base), z)
    (pi_base, _), _ = network.apply(
        variables, jnp.asarray(base), z, mutable=["run_stats"]
    )

    perturbed = np.array(base, copy=True)
    # Scramble the padded joint's description AND state, keep its valid bit 0.
    perturbed[:, 12:].reshape(4, 3, 8)[:, -1, :-1] += 25.0
    (pi_perturbed, _), _ = network.apply(
        variables, jnp.asarray(perturbed), z, mutable=["run_stats"]
    )

    np.testing.assert_allclose(
        np.asarray(pi_base.mean())[:, :2],
        np.asarray(pi_perturbed.mean())[:, :2],
        atol=1e-6,
    )


@pytest.mark.parametrize("variant", ["urma", "urmav2"])
def test_zero_latent_dim_keeps_pre_latent_param_tree(variant):
    """Old no-latent checkpoints must remain loadable: at actor_latent_dim=0
    the module must create exactly the parameters it created before the
    latent existed (the projection Dense must not appear and Flax
    auto-numbering must not shift)."""
    no_latent = _network(variant, actor_latent_dim=0)
    observations = _observations()
    variables = no_latent.init(jax.random.PRNGKey(2), observations)
    no_latent_paths = {
        jax.tree_util.keystr(path)
        for path, _ in jax.tree_util.tree_flatten_with_path(variables["params"])[0]
    }

    with_latent = _network(variant)
    latent_variables = with_latent.init(
        jax.random.PRNGKey(2), observations, _latents()
    )
    latent_paths = {
        jax.tree_util.keystr(path)
        for path, _ in jax.tree_util.tree_flatten_with_path(
            latent_variables["params"]
        )[0]
    }

    assert no_latent_paths.issubset(latent_paths) or not (
        no_latent_paths - latent_paths
    ), (
        "latent_dim=0 must not rename or drop pre-latent parameters; "
        f"missing: {sorted(no_latent_paths - latent_paths)[:5]}"
    )
    assert latent_paths - no_latent_paths, (
        "latent_dim>0 must add the projection parameters"
    )
    # apply without a latent argument still works at latent_dim=0
    (pi, value), _ = no_latent.apply(
        variables, observations, mutable=["run_stats"]
    )
    assert np.isfinite(np.asarray(pi.mean())).all()

    # and a supplied latent at latent_dim=0 is a hard error, not a silent drop
    with pytest.raises(ValueError, match="refusing to silently drop"):
        no_latent.apply(variables, observations, _latents(), mutable=["run_stats"])


@pytest.mark.parametrize("variant", ["urma", "urmav2"])
def test_privileged_critic_sees_extra_dims_actor_does_not(variant):
    """Privileged critic: dims listed in critic_extra_indices must change the
    value and must NOT change the policy (actor stays reference-blind)."""
    network = URMAActorCritic(
        action_dim=3,
        base_observation_dim=8,
        joint_feature_start=12,
        num_joints=3,
        joint_description_dim=4,
        joint_state_dim=3,
        joint_feature_dim=8,
        general_indices=(0, 1, 8, 9, 10, 11),
        variant=variant,
        core_hidden=(16, 8),
        latent_slots=8,
        joint_value_dim=2,
        learnable_std=True,
        actor_latent_dim=LATENT_DIM,
        critic_extra_indices=(36, 37),  # two dims appended after the joint block
    )
    base = np.zeros((4, 38), dtype=np.float32)
    rng = np.random.default_rng(7)
    base[:] = rng.normal(size=base.shape)
    base[:, 12:36].reshape(4, 3, 8)[..., -1] = 1.0
    z = _latents(seed=4)
    variables = network.init(jax.random.PRNGKey(3), jnp.asarray(base), z)
    (pi_a, value_a), _ = network.apply(
        variables, jnp.asarray(base), z, mutable=["run_stats"]
    )

    perturbed = np.array(base, copy=True)
    perturbed[:, 36:38] += 9.0
    (pi_b, value_b), _ = network.apply(
        variables, jnp.asarray(perturbed), z, mutable=["run_stats"]
    )

    assert not np.allclose(np.asarray(value_a), np.asarray(value_b))
    np.testing.assert_allclose(
        np.asarray(pi_a.mean()), np.asarray(pi_b.mean()), atol=1e-6
    )


def test_group_concatenated_cursor_selects_identical_z_across_families():
    """Two families at the same canonical (motion, timestamp) must receive
    the exact same latent row through the grouped environment's carry."""
    buffer = TrajectoryLatentBuffer(
        values=np.arange(10 * LATENT_DIM, dtype=np.float32).reshape(
            10, LATENT_DIM
        ),
        split_points=np.asarray([0, 6, 10]),
    )

    def group_state(traj_no, steps):
        return SimpleNamespace(
            additional_carry=SimpleNamespace(
                traj_state=TrajState(
                    traj_no=jnp.asarray(traj_no),
                    subtraj_step_no=jnp.asarray(steps),
                    subtraj_step_no_init=jnp.zeros(len(steps), dtype=jnp.int32),
                )
            )
        )

    # family A (2 envs) and family B (2 envs) share cursors (0,4) and (1,2)
    state = ParallelMorphState(
        group_states=(
            group_state([0, 1], [4, 2]),
            group_state([0, 1], [4, 2]),
        )
    )
    rows = np.asarray(buffer.get_for_state(state))
    assert rows.shape == (4, LATENT_DIM)
    np.testing.assert_array_equal(rows[0], rows[2])
    np.testing.assert_array_equal(rows[1], rows[3])
    np.testing.assert_array_equal(rows[0], np.asarray(buffer.values[4]))
    np.testing.assert_array_equal(rows[1], np.asarray(buffer.values[6 + 2]))


def test_parallel_env_th_asserts_canonical_alignment():
    def fake_env(split_points, frequency):
        return SimpleNamespace(
            th=SimpleNamespace(
                traj=SimpleNamespace(
                    data=SimpleNamespace(
                        split_points=np.asarray(split_points, dtype=np.int32)
                    ),
                    info=SimpleNamespace(frequency=frequency),
                )
            )
        )

    aligned = SimpleNamespace(
        _raw_envs=(fake_env([0, 5, 9], 30.0), fake_env([0, 5, 9], 30.0)),
        names=("h1", "g1"),
    )
    handler = ParallelMorphVecEnv.th.fget(aligned)
    assert float(handler.traj.info.frequency) == 30.0

    misaligned = SimpleNamespace(
        _raw_envs=(fake_env([0, 5, 9], 30.0), fake_env([0, 4, 9], 30.0)),
        names=("h1", "g1"),
    )
    with pytest.raises(ValueError, match="not canonically aligned"):
        ParallelMorphVecEnv.th.fget(misaligned)

    wrong_frequency = SimpleNamespace(
        _raw_envs=(fake_env([0, 5, 9], 30.0), fake_env([0, 5, 9], 25.0)),
        names=("h1", "g1"),
    )
    with pytest.raises(ValueError, match="frequency"):
        ParallelMorphVecEnv.th.fget(wrong_frequency)


def test_urma_agent_conf_round_trips_latent_buffer():
    buffer = TrajectoryLatentBuffer(
        values=np.random.default_rng(0)
        .normal(size=(7, LATENT_DIM))
        .astype(np.float32),
        split_points=np.asarray([0, 3, 7]),
    )
    config = OmegaConf.create(
        {
            "experiment": {
                "lr": 1e-4,
                "anneal_lr": False,
                "max_grad_norm": 0.5,
                "weight_decay": 0.0,
            }
        }
    )
    conf = URMAAgentConf(
        config=config,
        network=_network("urmav2"),
        tx=None,
        actor_latent_buffer=buffer,
    )
    loaded = URMAAgentConf.from_dict(conf.serialize())
    assert loaded.network.actor_latent_dim == LATENT_DIM
    np.testing.assert_array_equal(
        np.asarray(loaded.actor_latent_buffer.values), np.asarray(buffer.values)
    )
    np.testing.assert_array_equal(
        np.asarray(loaded.actor_latent_buffer.split_points),
        np.asarray(buffer.split_points),
    )

    # A latent-free conf keeps the old serialized shape (None slot only).
    bare = URMAAgentConf(
        config=config, network=_network("urma", actor_latent_dim=0), tx=None
    )
    assert bare.serialize()["actor_latent_buffer"] is None
    assert URMAAgentConf.from_dict(bare.serialize()).actor_latent_buffer is None
