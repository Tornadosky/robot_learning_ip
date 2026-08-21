"""Embodiment-aware PPO backbones for the online morphology experiments.

This is a LocoMuJoCo/Flax adaptation of the authors' public URMA design and
the URMAv2 architecture described in Bohlinger & Peters (2025).  It consumes
the structured suffix emitted by :mod:`scaling.online_h1` while retaining the
flat observation/action API expected by the existing PPO implementation.

The current environment integration is deliberately same-topology H1.  The
network itself accepts a per-joint validity mask, so a future padded
cross-topology vector wrapper can reuse it without changing the policy.
"""

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

from scaling.online_h1 import URMAInputLayout


URMA_VARIANTS = ("urma", "urmav2")
URMA_IMPLEMENTATION_REVISION = "raw_binary_valid_mask_padded_std_2026-08-01"


def _activation(name: str):
    try:
        return getattr(nn, name)
    except AttributeError as exc:
        raise ValueError(f"Unknown Flax activation {name!r}.") from exc


def _dense(
    x,
    features: int,
    *,
    gain: float = np.sqrt(2),
    bias: float = 0.0,
    weight_norm: bool = False,
):
    layer = nn.Dense(
        features,
        kernel_init=orthogonal(gain),
        bias_init=constant(bias),
    )
    if weight_norm:
        layer = nn.WeightNorm(layer)
    return layer(x)


class URMAJointEncoder(nn.Module):
    """Description-keyed aggregation of a variable-size joint set."""

    variant: str
    latent_slots: int
    joint_value_dim: int
    temperature: float = 1.0
    temperature_min: float = 0.015
    stability_epsilon: float = 1e-6

    @nn.compact
    def __call__(self, descriptions, joint_states, valid_mask):
        log_temperature = self.param(
            "log_temperature",
            constant(jnp.log(self.temperature - self.temperature_min)),
            (1,),
        )
        temperature = jnp.exp(log_temperature) + self.temperature_min

        if self.variant == "urma":
            logits = nn.Dense(self.latent_slots)(descriptions)
            logits = nn.LayerNorm()(logits)
            logits = nn.elu(logits)
            logits = nn.Dense(self.latent_slots)(logits)
            logits = jnp.clip(
                nn.tanh(logits),
                -1.0 + self.stability_epsilon,
                1.0 - self.stability_epsilon,
            )
            # This matches the released URMA implementation: each joint routes
            # its state into a distribution over latent slots.
            attention = nn.softmax(logits / temperature, axis=-1)
            attention = attention * valid_mask[..., :, None]
            state_latent = nn.elu(nn.Dense(self.joint_value_dim)(joint_states))
            routed = attention[..., :, :, None] * state_latent[..., :, None, :]
            encoded = jnp.sum(routed, axis=-3)
            encoded = encoded.reshape(
                (*encoded.shape[:-2], self.latent_slots * self.joint_value_dim)
            )
            return encoded, attention, state_latent

        if self.variant != "urmav2":
            raise ValueError(f"Unknown URMA variant {self.variant!r}.")

        # URMAv2 widens the state encoder and normalizes attention across the
        # joint set.  The resulting attention vectors are reused by the action
        # decoder, avoiding a separate decoder MLP for every joint.
        desc_latent = _dense(descriptions, self.latent_slots, weight_norm=True)
        desc_latent = nn.elu(nn.LayerNorm()(desc_latent))
        logits = _dense(desc_latent, self.latent_slots, weight_norm=True)
        masked_logits = jnp.where(
            valid_mask[..., :, None] > 0.0,
            logits / temperature,
            jnp.asarray(-1e9, dtype=logits.dtype),
        )
        attention = nn.softmax(masked_logits, axis=-2)
        attention = attention * valid_mask[..., :, None]

        state_latent = _dense(joint_states, 256, weight_norm=True)
        state_latent = nn.elu(state_latent)
        state_latent = _dense(state_latent, self.latent_slots, weight_norm=True)
        state_latent = nn.elu(state_latent)
        encoded = jnp.sum(attention * state_latent, axis=-2)
        return encoded, attention, desc_latent


class URMAPolicyHead(nn.Module):
    variant: str
    core_hidden: tuple[int, ...]
    latent_slots: int
    joint_value_dim: int
    activation: str
    init_std: float
    learnable_std: bool
    motion_latent_dim: int = 0
    motion_latent_embed_dim: int = 64

    @nn.compact
    def __call__(
        self, descriptions, joint_states, general, valid_mask, motion_latent=None
    ):
        encoded, attention, auxiliary = URMAJointEncoder(
            variant=self.variant,
            latent_slots=self.latent_slots,
            joint_value_dim=self.joint_value_dim,
            name="joint_encoder",
        )(descriptions, joint_states, valid_mask)
        core_inputs = [encoded, general]
        if self.motion_latent_dim > 0:
            # The motion command enters the actor core ONLY here.  The small
            # FSQ-width latent is projected up to the embed width so the code
            # dimensionality stays decoupled from the actor's internal width.
            embedded = _dense(
                motion_latent,
                self.motion_latent_embed_dim,
                weight_norm=self.variant == "urmav2",
            )
            core_inputs.append(nn.elu(embedded))
        x = jnp.concatenate(core_inputs, axis=-1)
        activation = _activation(self.activation)

        if self.variant == "urma":
            for layer_index, width in enumerate(self.core_hidden):
                x = _dense(x, width)
                x = nn.LayerNorm()(x) if layer_index == 0 else x
                x = activation(x)
            action_latent = _dense(x, 128, gain=np.sqrt(2))

            description_latent = nn.Dense(128)(descriptions)
            description_latent = nn.elu(nn.LayerNorm()(description_latent))
            description_latent = nn.Dense(128)(description_latent)
            broadcast_latent = jnp.broadcast_to(
                action_latent[..., None, :],
                (*action_latent.shape[:-1], descriptions.shape[-2], 128),
            )
            decoder_input = jnp.concatenate(
                [
                    broadcast_latent,
                    jax.lax.stop_gradient(auxiliary),
                    description_latent,
                ],
                axis=-1,
            )
            mean = _dense(decoder_input, 128)
            mean = nn.elu(nn.LayerNorm()(mean))
            mean = _dense(mean, 1, gain=0.01)[..., 0]
            std_input = description_latent
        else:
            for width in self.core_hidden:
                x = _dense(x, width, weight_norm=True)
                x = activation(x)
            action_latent = _dense(x, self.latent_slots, gain=0.01, weight_norm=True)
            mean = jnp.sum(action_latent[..., None, :] * attention, axis=-1)
            std_input = auxiliary

        mean = jnp.clip(mean, -5.0, 5.0) * valid_mask
        if self.learnable_std:
            log_std = _dense(
                std_input,
                1,
                gain=0.1,
                bias=float(np.log(self.init_std)),
                weight_norm=self.variant == "urmav2",
            )[..., 0]
            log_std = jnp.clip(log_std, np.log(1e-3), np.log(2.0))
        else:
            log_std = jnp.full_like(mean, jnp.log(self.init_std))
        # Padded actions must not inject appreciable random commands.  Keeping a
        # small positive scale preserves a valid Gaussian/log-probability while
        # removing those dimensions from practical control.
        log_std = jnp.where(valid_mask > 0.0, log_std, jnp.log(1e-3))
        return mean, log_std


class URMAValueHead(nn.Module):
    variant: str
    core_hidden: tuple[int, ...]
    latent_slots: int
    joint_value_dim: int
    activation: str

    @nn.compact
    def __call__(self, descriptions, joint_states, general, valid_mask):
        encoded, _, _ = URMAJointEncoder(
            variant=self.variant,
            latent_slots=self.latent_slots,
            joint_value_dim=self.joint_value_dim,
            name="joint_encoder",
        )(descriptions, joint_states, valid_mask)
        x = jnp.concatenate([encoded, general], axis=-1)
        activation = _activation(self.activation)
        for width in self.core_hidden:
            x = _dense(x, width, weight_norm=self.variant == "urmav2")
            x = activation(x)
        value = _dense(
            x,
            1,
            gain=1.0,
            weight_norm=self.variant == "urmav2",
        )
        return value[..., 0]


class URMAActorCritic(nn.Module):
    """Actor-critic with variable-joint encoding and decoding."""

    action_dim: int
    base_observation_dim: int
    joint_feature_start: int
    num_joints: int
    joint_description_dim: int
    joint_state_dim: int
    joint_feature_dim: int
    general_indices: tuple[int, ...]
    variant: str = "urma"
    activation: str = "elu"
    init_std: float = 0.2
    learnable_std: bool = True
    latent_slots: int = 64
    joint_value_dim: int = 4
    core_hidden: tuple[int, ...] = (512, 256, 128)
    #: Width of the separate motion-command latent (FSQ code width, NOT the
    #: 64-D actor conditioning width).  0 keeps the module byte-compatible
    #: with pre-latent checkpoints.
    actor_latent_dim: int = 0
    motion_latent_embed_dim: int = 64
    #: Observation dims appended for the CRITIC only (privileged reference
    #: while the actor's goal copy is blanked).  Empty = both heads see the
    #: same general block, as before.
    critic_extra_indices: tuple[int, ...] = ()

    @nn.compact
    def __call__(self, observation, actor_latent=None):
        if self.action_dim != self.num_joints:
            raise ValueError(
                "URMA requires one padded action slot per encoded joint; "
                f"got action_dim={self.action_dim}, num_joints={self.num_joints}."
            )
        raw_feature_stop = (
            self.joint_feature_start + self.num_joints * self.joint_feature_dim
        )
        raw_features = observation[
            ..., self.joint_feature_start : raw_feature_stop
        ].reshape((*observation.shape[:-1], self.num_joints, self.joint_feature_dim))

        # Normalize continuous descriptions/state/general observations, but keep
        # the padding mask binary.  Normalizing an all-ones mask drives it toward
        # zero and would silently suppress every action after statistics settle.
        x = RunningMeanStd()(observation)
        features = x[..., self.joint_feature_start : raw_feature_stop]
        features = features.reshape(
            (*features.shape[:-1], self.num_joints, self.joint_feature_dim)
        )
        descriptions = features[..., : self.joint_description_dim]
        state_start = self.joint_description_dim
        state_stop = state_start + self.joint_state_dim
        joint_states = features[..., state_start:state_stop]
        valid_mask = raw_features[..., state_stop]
        general = x[..., jnp.asarray(self.general_indices)]

        if self.actor_latent_dim > 0:
            if actor_latent is None:
                raise ValueError(
                    "actor_latent is required when actor_latent_dim is non-zero."
                )
            actor_latent = jnp.asarray(actor_latent, dtype=general.dtype)
            if actor_latent.shape[-1] != self.actor_latent_dim:
                raise ValueError(
                    f"Expected actor_latent width {self.actor_latent_dim}, got "
                    f"{actor_latent.shape[-1]}."
                )
            # RunningMeanStd squeezes a singleton batch; align the latent with
            # the post-normalisation leading shape so n_envs=1 stays supported.
            actor_latent = jnp.reshape(
                actor_latent, general.shape[:-1] + (self.actor_latent_dim,)
            )
        elif actor_latent is not None:
            raise ValueError(
                "actor_latent supplied but actor_latent_dim is 0; refusing to "
                "silently drop the motion command."
            )

        mean, log_std = URMAPolicyHead(
            variant=self.variant,
            core_hidden=self.core_hidden,
            latent_slots=self.latent_slots,
            joint_value_dim=self.joint_value_dim,
            activation=self.activation,
            init_std=self.init_std,
            learnable_std=self.learnable_std,
            motion_latent_dim=self.actor_latent_dim,
            motion_latent_embed_dim=self.motion_latent_embed_dim,
            name="actor",
        )(descriptions, joint_states, general, valid_mask, actor_latent)
        critic_general = general
        if self.critic_extra_indices:
            critic_general = jnp.concatenate(
                [general, x[..., jnp.asarray(self.critic_extra_indices)]],
                axis=-1,
            )
        value = URMAValueHead(
            variant=self.variant,
            core_hidden=self.core_hidden,
            latent_slots=self.latent_slots,
            joint_value_dim=self.joint_value_dim,
            activation=self.activation,
            name="critic",
        )(descriptions, joint_states, critic_general, valid_mask)
        pi = distrax.MultivariateNormalDiag(mean, jnp.exp(log_std))
        return pi, value


def _network_to_dict(network: URMAActorCritic) -> dict[str, Any]:
    return {
        "action_dim": int(network.action_dim),
        "base_observation_dim": int(network.base_observation_dim),
        "joint_feature_start": int(network.joint_feature_start),
        "num_joints": int(network.num_joints),
        "joint_description_dim": int(network.joint_description_dim),
        "joint_state_dim": int(network.joint_state_dim),
        "joint_feature_dim": int(network.joint_feature_dim),
        "general_indices": tuple(int(i) for i in network.general_indices),
        "variant": str(network.variant),
        "activation": str(network.activation),
        "init_std": float(network.init_std),
        "learnable_std": bool(network.learnable_std),
        "latent_slots": int(network.latent_slots),
        "joint_value_dim": int(network.joint_value_dim),
        "core_hidden": tuple(int(i) for i in network.core_hidden),
        "actor_latent_dim": int(network.actor_latent_dim),
        "motion_latent_embed_dim": int(network.motion_latent_embed_dim),
        "critic_extra_indices": tuple(int(i) for i in network.critic_extra_indices),
    }


@dataclass(frozen=True)
class URMAAgentConf(AgentConfBase):
    config: DictConfig
    network: URMAActorCritic
    tx: Any
    #: Motion-command table (one row per canonical trajectory timestamp).
    #: None keeps the pre-latent behaviour and checkpoint format.
    actor_latent_buffer: Any = None

    def serialize(self):
        return {
            "config": OmegaConf.to_container(
                self.config, resolve=True, throw_on_missing=True
            ),
            "network": _network_to_dict(self.network),
            "actor_latent_buffer": (
                None
                if self.actor_latent_buffer is None
                else self.actor_latent_buffer.to_dict()
            ),
        }

    @classmethod
    def from_dict(cls, data):
        from loco_mujoco.algorithms import TrajectoryLatentBuffer

        config = OmegaConf.create(data["config"])
        network = URMAActorCritic(**data["network"])
        serialized_latents = data.get("actor_latent_buffer")
        return cls(
            config=config,
            network=network,
            tx=URMAPPO._get_optimizer(config),
            actor_latent_buffer=(
                None
                if serialized_latents is None
                else TrajectoryLatentBuffer.from_dict(serialized_latents)
            ),
        )


class URMAPPO(PPOJax):
    """PPO using URMA or URMAv2 in place of the fixed MLP actor-critic."""

    _agent_conf = URMAAgentConf
    _agent_state = PPOAgentState

    @classmethod
    def init_agent_conf(cls, env, config, actor_latent_buffer=None):
        layout: URMAInputLayout | None = getattr(env, "urma_input_layout", None)
        if layout is None:
            raise TypeError(
                "URMAPPO requires append_urma_joint_features=True in the environment."
            )
        base_conf = PPOJax.init_agent_conf(env, config, actor_latent_buffer)
        experiment = base_conf.config.experiment
        variant = str(getattr(experiment, "backbone", "urma")).lower()
        if variant not in URMA_VARIANTS:
            raise ValueError(
                f"backbone must be one of {URMA_VARIANTS}, got {variant!r}."
            )

        configured_hidden: Sequence[int] | str = experiment.hidden_layers
        if isinstance(configured_hidden, (str,)):
            configured_hidden = ast.literal_eval(configured_hidden)
        elif isinstance(configured_hidden, ListConfig):
            configured_hidden = list(configured_hidden)
        core_hidden = tuple(int(width) for width in configured_hidden)
        if variant == "urmav2" and len(core_hidden) < 5:
            # URMAv2's larger core is one of the changes reported in the paper.
            core_hidden = core_hidden + (core_hidden[-1],) * (5 - len(core_hidden))

        network = URMAActorCritic(
            action_dim=int(env.info.action_space.shape[0]),
            base_observation_dim=layout.base_observation_dim,
            joint_feature_start=layout.joint_feature_start,
            num_joints=layout.num_joints,
            joint_description_dim=layout.joint_description_dim,
            joint_state_dim=layout.joint_state_dim,
            joint_feature_dim=layout.joint_feature_dim,
            general_indices=layout.general_indices,
            variant=variant,
            activation=str(getattr(experiment, "urma_activation", "elu")),
            init_std=float(experiment.init_std),
            learnable_std=bool(experiment.learnable_std),
            latent_slots=int(getattr(experiment, "urma_latent_slots", 64)),
            joint_value_dim=int(getattr(experiment, "urma_joint_value_dim", 4)),
            core_hidden=core_hidden,
            actor_latent_dim=int(base_conf.network.actor_latent_dim),
            motion_latent_embed_dim=int(
                getattr(experiment, "motion_latent_embed_dim", 64)
            ),
            critic_extra_indices=tuple(
                getattr(layout, "critic_extra_indices", ()) or ()
            ),
        )
        return cls._agent_conf(
            base_conf.config,
            network,
            base_conf.tx,
            actor_latent_buffer=base_conf.actor_latent_buffer,
        )
