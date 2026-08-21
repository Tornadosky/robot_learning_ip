"""Verification gates for the SONIC-style fake actor-latent pipeline."""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omegaconf import OmegaConf

from loco_mujoco.algorithms import ActorCritic, PPOJax, TrajectoryLatentBuffer
from loco_mujoco.algorithms.common.dataclasses import Transition
from loco_mujoco.algorithms.ppo_jax import PPOAgentConf
from loco_mujoco.core.wrappers.mjx import Metrics
from loco_mujoco.trajectory import TrajState


def _trajectory_data(split_points=(0, 3, 7)):
    return SimpleNamespace(
        split_points=np.asarray(split_points, dtype=np.int32)
    )


def _ppo_config(latent_dim=4):
    return OmegaConf.create(
        {
            "experiment": {
                "hidden_layers": [8],
                "lr": 1e-3,
                "num_envs": 2,
                "num_steps": 2,
                "total_timesteps": 8,
                "num_minibatches": 1,
                "update_epochs": 1,
                "validation": {"num": 1},
                "activation": "tanh",
                "init_std": 0.2,
                "learnable_std": True,
                "anneal_lr": False,
                "max_grad_norm": 1.0,
                "weight_decay": 0.0,
                "actor_latent_dim": latent_dim,
                "fake_latent_seed": 7,
                "fake_latent_scale": 0.5,
            }
        }
    )


def _stub_env(trajectory_data=None):
    shape = lambda *dims: SimpleNamespace(shape=dims)
    return SimpleNamespace(
        info=SimpleNamespace(
            observation_space=shape(5), action_space=shape(2)
        ),
        mdp_info=SimpleNamespace(observation_space=shape(5)),
        # Deliberately no usable observation-group API: latent setup must not
        # need or mutate it.
        obs_container=SimpleNamespace(),
        th=SimpleNamespace(
            traj=SimpleNamespace(
                data=(
                    _trajectory_data()
                    if trajectory_data is None
                    else trajectory_data
                )
            )
        ),
    )


def test_latent_buffer_has_one_row_per_reference_timestamp():
    trajectory_data = _trajectory_data((0, 3, 7))
    values = np.arange(7 * 3, dtype=np.float32).reshape(7, 3)
    buffer = TrajectoryLatentBuffer.from_trajectory_data(
        trajectory_data, values
    )

    assert buffer.values.shape == (7, 3)
    assert buffer.num_trajectories == 2
    np.testing.assert_array_equal(
        np.asarray(
            buffer.global_index(
                jnp.asarray([0, 1]), jnp.asarray([2, 3])
            )
        ),
        [2, 6],
    )
    np.testing.assert_array_equal(
        np.asarray(buffer.get(jnp.asarray([0, 1]), jnp.asarray([2, 3]))),
        values[[2, 6]],
    )


def test_latent_buffer_rejects_timestamp_or_split_mismatch():
    with pytest.raises(ValueError, match="timestamps must match"):
        TrajectoryLatentBuffer(
            values=np.zeros((6, 4), dtype=np.float32),
            split_points=np.asarray([0, 3, 7]),
        )

    buffer = TrajectoryLatentBuffer(
        values=np.zeros((7, 4), dtype=np.float32),
        split_points=np.asarray([0, 3, 7]),
    )
    with pytest.raises(ValueError, match="split_points do not match"):
        buffer.validate_trajectory_data(_trajectory_data((0, 2, 7)))


def test_fake_latents_are_deterministic_but_not_constant():
    trajectory_data = _trajectory_data((0, 100, 1000))
    first = TrajectoryLatentBuffer.fake_from_trajectory_data(
        trajectory_data, 64, seed=11
    )
    same = TrajectoryLatentBuffer.fake_from_trajectory_data(
        trajectory_data, 64, seed=11
    )
    different = TrajectoryLatentBuffer.fake_from_trajectory_data(
        trajectory_data, 64, seed=12
    )

    assert first.values.shape == (1000, 64)
    np.testing.assert_array_equal(first.values, same.values)
    assert not np.array_equal(first.values[0], first.values[1])
    assert not np.array_equal(first.values, different.values)


def test_actor_uses_latent_while_critic_and_obs_stats_do_not():
    network = ActorCritic(
        action_dim=2,
        hidden_layer_dims=(16, 8),
        actor_obs_ind=jnp.asarray([0, 1, 2]),
        critic_obs_ind=jnp.asarray([2, 3, 4]),
        actor_latent_dim=4,
    )
    observations = jnp.asarray(
        np.random.default_rng(3).normal(size=(4, 5)), dtype=jnp.float32
    )
    zero_latent = jnp.zeros((4, 4))
    one_latent = jnp.ones((4, 4))
    variables = network.init(jax.random.key(0), observations, zero_latent)

    (zero_policy, zero_value), _ = network.apply(
        variables, observations, zero_latent, mutable=["run_stats"]
    )
    (one_policy, one_value), _ = network.apply(
        variables, observations, one_latent, mutable=["run_stats"]
    )

    assert not np.allclose(zero_policy.mean(), one_policy.mean())
    np.testing.assert_allclose(zero_value, one_value, atol=0.0, rtol=0.0)
    run_stats = next(iter(variables["run_stats"].values()))
    assert run_stats["mean"].shape == (5,)


def test_single_vector_env_latent_shape_is_supported():
    network = ActorCritic(
        action_dim=2, hidden_layer_dims=(8,), actor_latent_dim=4
    )
    observations = jnp.ones((1, 5))
    latents = jnp.ones((1, 4))
    variables = network.init(jax.random.key(1), observations, latents)
    (policy, value), _ = network.apply(
        variables, observations, latents, mutable=["run_stats"]
    )

    assert policy.mean().shape == (2,)
    assert value.shape == ()


def test_agent_setup_preserves_observation_space_and_round_trips_buffer():
    env = _stub_env()
    observation_shape_before = env.info.observation_space.shape
    agent_conf = PPOJax.init_agent_conf(env, _ppo_config())

    assert env.info.observation_space.shape == observation_shape_before
    assert env.mdp_info.observation_space.shape == observation_shape_before
    assert agent_conf.network.actor_latent_dim == 4
    assert agent_conf.actor_latent_buffer.values.shape == (7, 4)

    loaded = PPOAgentConf.from_dict(agent_conf.serialize())
    assert loaded.network.actor_latent_dim == 4
    np.testing.assert_array_equal(
        loaded.actor_latent_buffer.values,
        agent_conf.actor_latent_buffer.values,
    )
    np.testing.assert_array_equal(
        loaded.actor_latent_buffer.split_points,
        agent_conf.actor_latent_buffer.split_points,
    )


def test_done_boundary_uses_reset_observation_with_reset_cursor_latent():
    buffer = TrajectoryLatentBuffer(
        values=np.arange(5 * 2, dtype=np.float32).reshape(5, 2),
        split_points=np.asarray([0, 5]),
    )
    returned_terminal_obs = jnp.asarray([[90.0, 91.0], [20.0, 21.0]])
    reset_state_obs = jnp.asarray([[10.0, 11.0], [12.0, 13.0]])
    env_state = SimpleNamespace(
        observation=reset_state_obs,
        additional_carry=SimpleNamespace(
            traj_state=TrajState(
                traj_no=jnp.asarray([0, 0]),
                subtraj_step_no=jnp.asarray([0, 3]),
                subtraj_step_no_init=jnp.asarray([0, 0]),
            )
        ),
    )
    done = jnp.asarray([True, False])

    next_actor_obs = PPOJax._next_actor_observation(
        buffer, returned_terminal_obs, env_state, done
    )
    next_actor_latent = PPOJax._actor_latent_for_state(buffer, env_state)

    np.testing.assert_array_equal(next_actor_obs[0], reset_state_obs[0])
    np.testing.assert_array_equal(next_actor_obs[1], returned_terminal_obs[1])
    np.testing.assert_array_equal(next_actor_latent[0], buffer.values[0])
    np.testing.assert_array_equal(next_actor_latent[1], buffer.values[3])


def test_ppo_minibatch_shuffle_keeps_exact_pre_step_latent_and_cursor():
    steps, envs, latent_dim = 2, 3, 4
    local_cursor = jnp.arange(steps * envs).reshape(steps, envs)
    actor_latent = jnp.repeat(
        local_cursor[..., None], latent_dim, axis=-1
    ).astype(jnp.float32)
    scalar = jnp.zeros((steps, envs), dtype=jnp.float32)
    boolean = jnp.zeros((steps, envs), dtype=bool)
    transition = Transition(
        done=boolean,
        absorbing=boolean,
        action=jnp.zeros((steps, envs, 2)),
        value=scalar,
        reward=scalar,
        log_prob=scalar,
        obs=jnp.zeros((steps, envs, 5)),
        info=jnp.zeros((steps, envs, 1)),
        traj_state=TrajState(
            traj_no=jnp.zeros((steps, envs), dtype=jnp.int32),
            subtraj_step_no=local_cursor,
            subtraj_step_no_init=jnp.zeros(
                (steps, envs), dtype=jnp.int32
            ),
        ),
        metrics=Metrics(
            episode_returns=scalar,
            episode_lengths=local_cursor,
            returned_episode_returns=scalar,
            returned_episode_lengths=local_cursor,
            timestep=local_cursor,
            done=boolean,
        ),
        actor_latent=actor_latent,
    )

    batch_size = steps * envs
    flattened = jax.tree.map(
        lambda x: x.reshape((batch_size,) + x.shape[2:]), transition
    )
    permutation = jnp.asarray([4, 0, 5, 1, 3, 2])
    shuffled = jax.tree.map(
        lambda x: jnp.take(x, permutation, axis=0), flattened
    )

    np.testing.assert_array_equal(
        shuffled.actor_latent[:, 0],
        shuffled.traj_state.subtraj_step_no,
    )
