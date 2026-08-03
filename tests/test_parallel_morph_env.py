from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from loco_mujoco.core.utils.env import Box
from loco_mujoco.core.wrappers import LogEnvState

from scaling.parallel_env import (
    ParallelMorphVecEnv,
    ParallelNormalizeReward,
    balanced_group_sizes,
)


@struct.dataclass
class DummyState:
    value: jax.Array


class DummyEnv:
    def __init__(self, morphology_value: float):
        self.morphology_value = float(morphology_value)
        observation_space = Box(np.full((2,), -np.inf), np.full((2,), np.inf))
        action_space = Box(np.full((1,), -1.0), np.full((1,), 1.0))
        self.info = SimpleNamespace(
            observation_space=observation_space,
            action_space=action_space,
        )
        self.mdp_info = self.info
        self.obs_container = SimpleNamespace()

    def reset(self, key):
        del key
        value = jnp.asarray(self.morphology_value, dtype=jnp.float32)
        return jnp.array([value, 0.0]), DummyState(value)

    def step(self, state, action):
        value = state.value + action[0]
        obs = jnp.array([self.morphology_value, value])
        reward = jnp.asarray(self.morphology_value)
        done = value > 10.0
        return obs, reward, done, done, {"value": value}, DummyState(value)


class VariableShapeDummyEnv(DummyEnv):
    def __init__(self, morphology_value: float, obs_dim: int, action_dim: int):
        super().__init__(morphology_value)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.info = SimpleNamespace(
            observation_space=Box(
                np.full((self.obs_dim,), -np.inf),
                np.full((self.obs_dim,), np.inf),
            ),
            action_space=Box(
                np.full((self.action_dim,), -1.0),
                np.full((self.action_dim,), 1.0),
            ),
        )
        self.mdp_info = self.info

    def reset(self, key):
        del key
        value = jnp.asarray(self.morphology_value, dtype=jnp.float32)
        return jnp.zeros((self.obs_dim,)).at[0].set(value), DummyState(value)

    def step(self, state, action):
        value = state.value + jnp.sum(action)
        obs = jnp.zeros((self.obs_dim,)).at[0].set(self.morphology_value)
        obs = obs.at[1].set(value)
        done = value > 10.0
        return (
            obs,
            jnp.asarray(self.morphology_value),
            done,
            done,
            {"action_sum": jnp.sum(action)},
            DummyState(value),
        )


def test_balanced_group_sizes():
    assert balanced_group_sizes(10, 3) == (4, 3, 3)


def test_parallel_groups_are_ordered_and_jittable():
    env = ParallelMorphVecEnv(
        [DummyEnv(1.0), DummyEnv(3.0)],
        [2, 3],
        names=["a", "b"],
    )
    keys = jax.random.split(jax.random.PRNGKey(0), 5)
    obs, state = jax.jit(env.reset)(keys)
    np.testing.assert_allclose(obs[:, 0], [1, 1, 3, 3, 3])

    action = jnp.zeros((5, 1), dtype=jnp.float32)
    next_obs, reward, _, _, info, next_state = jax.jit(env.step)(state, action)
    np.testing.assert_allclose(next_obs[:, 0], [1, 1, 3, 3, 3])
    np.testing.assert_allclose(reward, [1, 1, 3, 3, 3])
    np.testing.assert_allclose(info["value"], [1, 1, 3, 3, 3])
    assert next_state.find(LogEnvState).metrics.done.shape == (5,)


def test_reward_normalization_uses_each_groups_updated_variance():
    grouped = ParallelMorphVecEnv(
        [DummyEnv(1.0), DummyEnv(3.0)],
        [2, 3],
        names=["a", "b"],
    )
    env = ParallelNormalizeReward(grouped, gamma=0.99)
    keys = jax.random.split(jax.random.PRNGKey(0), 5)
    _, state = jax.jit(env.reset)(keys)
    action = jnp.zeros((5, 1), dtype=jnp.float32)
    _, reward, _, _, _, next_state = jax.jit(env.step)(state, action)

    expected = np.concatenate(
        [
            np.full(2, 1.0 / np.sqrt(float(next_state.var[0]) + 1e-8)),
            np.full(3, 3.0 / np.sqrt(float(next_state.var[1]) + 1e-8)),
        ]
    )
    np.testing.assert_allclose(reward, expected, rtol=1e-5)


def test_cross_topology_padding_and_action_slicing_are_jittable():
    env = ParallelMorphVecEnv(
        [
            VariableShapeDummyEnv(1.0, obs_dim=2, action_dim=1),
            VariableShapeDummyEnv(3.0, obs_dim=3, action_dim=2),
        ],
        [1, 1],
        names=["small", "large"],
        pad_to_max_shapes=True,
        append_group_one_hot=True,
        append_action_mask=True,
    )
    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    obs, state = jax.jit(env.reset)(keys)

    assert obs.shape == (2, 7)
    np.testing.assert_allclose(obs[0], [1, 0, 0, 1, 0, 1, 0])
    np.testing.assert_allclose(obs[1], [3, 0, 0, 0, 1, 1, 1])
    np.testing.assert_allclose(env.action_mask, [[1, 0], [1, 1]])

    action = jnp.asarray([[1.0, 9.0], [2.0, 4.0]])
    _, _, _, _, info, _ = jax.jit(env.step)(state, action)
    # The first group's padded action is discarded; the second receives both.
    np.testing.assert_allclose(info["action_sum"], [1.0, 6.0])
