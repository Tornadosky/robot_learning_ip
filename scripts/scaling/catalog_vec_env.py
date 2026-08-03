"""Vector environment that assigns catalog bodies to environment slots.

LocoMuJoCo's ``VecEnv`` vmaps ``reset`` over reset keys only, so an environment
cannot know its own index and therefore cannot be given a *balanced* share of a
fixed catalog.  ``CatalogVecEnv`` resets through ``mjx_reset_with_slot`` instead,
supplying ``slot = 0 .. num_envs - 1``.  Everything after the reset - stepping,
episode logging, reward normalisation - is unchanged LocoMuJoCo code.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from loco_mujoco.core.wrappers import LogWrapper, NormalizeVecReward, VecEnv
from loco_mujoco.core.wrappers.mjx import BaseWrapper, LogEnvState, Metrics


class CatalogVecEnv(BaseWrapper):
    """``VecEnv(LogWrapper(env))`` with slot-aware, balanced catalog resets."""

    def __init__(self, env):
        if getattr(env, "catalog_mode", "continuous") == "continuous":
            raise ValueError("CatalogVecEnv requires an environment in a catalog mode.")
        super().__init__(VecEnv(LogWrapper(env)))
        self._catalog_env = env
        self._vreset = jax.vmap(env.mjx_reset_with_slot, in_axes=(0, 0))

    def reset(self, rng_key):
        num_envs = rng_key.shape[0]
        slots = jnp.arange(num_envs, dtype=jnp.int32)
        env_state = self._vreset(rng_key, slots)
        metrics = Metrics(
            episode_returns=jnp.zeros((num_envs,), dtype=jnp.float32),
            episode_lengths=jnp.zeros((num_envs,), dtype=jnp.int32),
            returned_episode_returns=jnp.zeros((num_envs,), dtype=jnp.float32),
            returned_episode_lengths=jnp.zeros((num_envs,), dtype=jnp.int32),
            timestep=jnp.zeros((num_envs,), dtype=jnp.int32),
            done=jnp.zeros((num_envs,), dtype=bool),
        )
        return env_state.observation, LogEnvState(env_state, metrics)

    def step(self, state, action):
        return self.env.step(state, action)


_CATALOG_ALGORITHM_CACHE: dict[type, type] = {}


def with_catalog_vec_env(algorithm_cls: type) -> type:
    """Return a PPO variant whose environment wrapper is slot-aware.

    Only ``_wrap_env`` changes; the network, optimiser, update rule and
    checkpoint format are the base algorithm's, so checkpoints stay loadable by
    the unmodified class.
    """
    cached = _CATALOG_ALGORITHM_CACHE.get(algorithm_cls)
    if cached is not None:
        return cached

    class CatalogAlgorithm(algorithm_cls):  # type: ignore[valid-type, misc]
        @staticmethod
        def _wrap_env(env, config):
            wrapped = CatalogVecEnv(env)
            if config.normalize_env:
                wrapped = NormalizeVecReward(wrapped, config.gamma)
            return wrapped

    CatalogAlgorithm.__name__ = f"Catalog{algorithm_cls.__name__}"
    CatalogAlgorithm.__qualname__ = CatalogAlgorithm.__name__
    _CATALOG_ALGORITHM_CACHE[algorithm_cls] = CatalogAlgorithm
    return CatalogAlgorithm
