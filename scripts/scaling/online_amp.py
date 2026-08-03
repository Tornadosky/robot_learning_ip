"""AMP over online H1 embodiments and several motions in one dynamic graph.

The static path builds one MJX branch per body and per body-motion cell.  Here
the body is a dynamic model array, the motion index and phase are already
sampled per reset by LocoMuJoCo's trajectory handler, and the whole
body x motion product lives inside a single H1 graph.

Discriminator leakage is the failure mode this module exists to prevent.  The
environment appends a morphology descriptor to every observation.  Expert
transitions are generated from one canonical body, so a discriminator that sees
the descriptor channel can separate expert from policy without looking at the
motion at all - and the AMP reward becomes "which body am I" rather than "does
this look like the reference".  ``DescriptorBlindNet`` removes those columns
before any normalisation or weights are applied.
"""

from __future__ import annotations

from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp

from loco_mujoco.algorithms import AMPJax
from loco_mujoco.algorithms.common.networks import FullyConnectedNet
from loco_mujoco.core.wrappers import NormalizeVecReward

from scaling.catalog_vec_env import CatalogVecEnv


class DescriptorBlindNet(nn.Module):
    """Discriminator that never sees the morphology descriptor columns.

    ``keep_indices`` is applied first, so the running mean/std statistics are
    computed over the kept columns only and no gradient path touches the
    excluded ones.
    """

    hidden_layer_dims: Sequence[int]
    output_dim: int
    keep_indices: tuple[int, ...]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        x = jnp.take(x, jnp.asarray(self.keep_indices), axis=-1)
        return FullyConnectedNet(
            hidden_layer_dims=list(self.hidden_layer_dims),
            output_dim=self.output_dim,
            activation=self.activation,
            output_activation=None,
            use_running_mean_stand=True,
            squeeze_output=True,
        )(x)


def keep_indices_excluding(
    observation_dim: int, start: int, stop: int
) -> tuple[int, ...]:
    """Observation columns kept by the discriminator."""
    if not 0 <= start <= stop <= observation_dim:
        raise ValueError(
            f"Invalid exclusion slice [{start}, {stop}) for dim {observation_dim}."
        )
    return tuple(i for i in range(observation_dim) if not start <= i < stop)


class OnlineMorphAMP(AMPJax):
    """AMP whose bodies are dynamic and whose discriminator is descriptor-blind.

    ``experiment.disc_exclude_start`` / ``disc_exclude_stop`` name the descriptor
    slice.  Setting them equal disables blinding, which is only useful as the
    negative control in the leakage test.
    """

    @classmethod
    def init_agent_conf(cls, env, config):
        agent_conf = super().init_agent_conf(env, config)
        experiment = agent_conf.config.experiment
        observation_dim = int(env.info.observation_space.shape[0])
        keep = keep_indices_excluding(
            observation_dim,
            int(experiment.disc_exclude_start),
            int(experiment.disc_exclude_stop),
        )
        discriminator = DescriptorBlindNet(
            hidden_layer_dims=list(experiment.hidden_layers),
            output_dim=1,
            keep_indices=keep,
            activation=str(experiment.activation),
        )
        return cls._agent_conf(
            agent_conf.config,
            agent_conf.network,
            discriminator,
            agent_conf.tx,
            agent_conf.disc_tx,
        )

    @staticmethod
    def _wrap_env(env, config):
        if getattr(env, "catalog_mode", "continuous") == "continuous":
            raise TypeError(
                "OnlineMorphAMP expects a catalog-mode online environment; use "
                "AMPJax directly for uncontrolled continuous sampling."
            )
        wrapped = CatalogVecEnv(env)
        if config.normalize_env:
            wrapped = NormalizeVecReward(wrapped, config.gamma)
        return wrapped
