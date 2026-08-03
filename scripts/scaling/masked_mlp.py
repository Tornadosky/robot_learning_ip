"""Action-masked MLP PPO for padded cross-topology robot batches."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Sequence

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal
from omegaconf import DictConfig, ListConfig, OmegaConf

from loco_mujoco.algorithms import RunningMeanStd
from loco_mujoco.algorithms.common.base_algorithm import AgentConfBase
from loco_mujoco.algorithms.ppo_jax import PPOAgentState, PPOJax

from scaling.parallel_env import ParallelMorphPPO


class MaskedActorCritic(nn.Module):
    """MLP actor-critic whose padded action dimensions have fixed tiny variance."""

    action_dim: int
    action_mask_start: int
    hidden_layer_dims: tuple[int, ...]
    activation: str = "tanh"
    init_std: float = 0.2
    learnable_std: bool = True

    @nn.compact
    def __call__(self, observation):
        action_mask = observation[
            ..., self.action_mask_start : self.action_mask_start + self.action_dim
        ]
        x = RunningMeanStd()(observation)
        activation = getattr(nn, self.activation)

        actor = x
        for width in self.hidden_layer_dims:
            actor = nn.Dense(
                width,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(actor)
            actor = activation(actor)
        mean = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(actor)
        mean = mean * action_mask

        log_std = self.param(
            "log_std",
            nn.initializers.constant(jnp.log(self.init_std)),
            (self.action_dim,),
        )
        if not self.learnable_std:
            log_std = jax.lax.stop_gradient(log_std)
        scale = jnp.where(action_mask > 0.0, jnp.exp(log_std), 1e-3)
        policy = distrax.MultivariateNormalDiag(mean, scale)

        critic = x
        for width in self.hidden_layer_dims:
            critic = nn.Dense(
                width,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(critic)
            critic = activation(critic)
        value = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(critic)
        return policy, value[..., 0]


def _network_to_dict(network: MaskedActorCritic):
    return {
        "action_dim": int(network.action_dim),
        "action_mask_start": int(network.action_mask_start),
        "hidden_layer_dims": tuple(int(width) for width in network.hidden_layer_dims),
        "activation": str(network.activation),
        "init_std": float(network.init_std),
        "learnable_std": bool(network.learnable_std),
    }


@dataclass(frozen=True)
class MaskedPPOAgentConf(AgentConfBase):
    config: DictConfig
    network: MaskedActorCritic
    tx: Any

    def serialize(self):
        return {
            "config": OmegaConf.to_container(
                self.config, resolve=True, throw_on_missing=True
            ),
            "network": _network_to_dict(self.network),
        }

    @classmethod
    def from_dict(cls, data):
        config = OmegaConf.create(data["config"])
        return cls(
            config=config,
            network=MaskedActorCritic(**data["network"]),
            tx=MaskedParallelPPO._get_optimizer(config),
        )


class MaskedParallelPPO(ParallelMorphPPO):
    """Grouped PPO with a policy distribution that ignores padded actions."""

    _agent_conf = MaskedPPOAgentConf
    _agent_state = PPOAgentState

    @classmethod
    def init_agent_conf(cls, env, config):
        if not getattr(env, "append_action_mask", False):
            raise TypeError(
                "MaskedParallelPPO requires append_action_mask=True in the "
                "parallel environment."
            )
        base_conf = PPOJax.init_agent_conf(env, config)
        hidden: Sequence[int] | str = base_conf.config.experiment.hidden_layers
        if isinstance(hidden, str):
            hidden = ast.literal_eval(hidden)
        elif isinstance(hidden, ListConfig):
            hidden = list(hidden)
        network = MaskedActorCritic(
            action_dim=int(env.info.action_space.shape[0]),
            action_mask_start=int(env.action_mask_observation_start),
            hidden_layer_dims=tuple(int(width) for width in hidden),
            activation=str(base_conf.config.experiment.activation),
            init_std=float(base_conf.config.experiment.init_std),
            learnable_std=bool(base_conf.config.experiment.learnable_std),
        )
        return cls._agent_conf(base_conf.config, network, base_conf.tx)
