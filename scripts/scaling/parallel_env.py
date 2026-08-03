"""Grouped heterogeneous MJX vectorization.

The stock LocoMuJoCo ``VecEnv`` vmaps one fixed environment/model.  This module
keeps a homogeneous vmap inside each morphology group, then composes all groups
inside one outer JIT.  A shared policy therefore observes transitions from every
morphology in every PPO update without requiring the MJX models themselves to
have identical static PyTree metadata.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from loco_mujoco.algorithms import AMPJax, PPOJax
from loco_mujoco.core.utils.env import Box
from loco_mujoco.core.wrappers import LogEnvState, LogWrapper, NStepWrapper
from loco_mujoco.core.wrappers.mjx import BaseWrapper, BaseWrapperState, Metrics

from scaling.joint_descriptions import (
    GENERIC_JOINT_DESCRIPTION_DIM,
    JOINT_FEATURE_DIM,
    JOINT_STATE_DIM,
    JointBlockSpec,
)
from scaling.online_h1 import URMAInputLayout


@dataclass(frozen=True)
class MorphologyGroup:
    """Static description of one homogeneous morphology batch."""

    name: str
    size: int
    start: int
    stop: int


def _concat_pytrees(pytrees):
    """Concatenate equally structured, already-batched PyTrees."""

    if not pytrees:
        raise ValueError("At least one PyTree is required.")
    first = jax.tree_util.tree_structure(pytrees[0])
    for i, tree in enumerate(pytrees[1:], start=1):
        if jax.tree_util.tree_structure(tree) != first:
            raise ValueError(
                f"Morphology group {i} produced a different state structure. "
                "All grouped environments must have the same DOFs, observations, "
                "actions, and trajectory-state structure."
            )
    return jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *pytrees)


@struct.dataclass
class ParallelLogView:
    """Minimal view returned for PPO's ``state.find(LogEnvState)`` call."""

    metrics: Metrics


@struct.dataclass
class ParallelCarryView:
    """Minimal view exposing the flat trajectory state expected by PPO."""

    traj_state: object


@struct.dataclass
class ParallelMorphState:
    """Heterogeneous group states kept separate inside one JAX carry.

    LocoMuJoCo creates an environment-specific ``ObservationStates`` dataclass,
    so full MJX states from two otherwise shape-compatible environments cannot
    be concatenated.  A tuple retains each exact type while the public views
    concatenate only the homogeneous fields PPO consumes.
    """

    group_states: tuple

    def find(self, cls):
        if cls is ParallelMorphState:
            return self
        if cls is LogEnvState:
            metrics = _concat_pytrees(
                [state.find(LogEnvState).metrics for state in self.group_states]
            )
            return ParallelLogView(metrics)
        raise AttributeError(f"Class {cls!r} is not exposed by ParallelMorphState")

    @property
    def additional_carry(self):
        traj_state = _concat_pytrees(
            [state.additional_carry.traj_state for state in self.group_states]
        )
        return ParallelCarryView(traj_state)


class ParallelMorphVecEnv:
    """One vector environment containing several fixed morphology groups.

    ``reset`` and ``step`` accept/return a flat leading environment axis, so the
    existing PPO implementation can consume this object.  Internally each group
    is vmapped against its own fixed MJX environment and model.  The Python loop
    is evaluated while tracing; runtime execution is one compiled JAX program.
    """

    is_parallel_morph_env = True

    def __init__(
        self,
        envs: Sequence,
        group_sizes: Sequence[int],
        names: Sequence[str] | None = None,
        history_length: int = 1,
        pad_to_max_shapes: bool = False,
        append_group_one_hot: bool = False,
        append_action_mask: bool = False,
        joint_block_specs: Sequence[JointBlockSpec] | None = None,
        reserved_observation_dim: int = 0,
        reserved_action_dim: int = 0,
        reserved_group_slots: int = 0,
    ):
        if not envs:
            raise ValueError("At least one morphology environment is required.")
        if len(envs) != len(group_sizes):
            raise ValueError("envs and group_sizes must have equal length.")
        if any(int(n) <= 0 for n in group_sizes):
            raise ValueError("Every morphology group must contain at least one env.")
        if names is None:
            names = [f"morph_{i}" for i in range(len(envs))]
        if len(names) != len(envs):
            raise ValueError("names and envs must have equal length.")

        obs_shapes = [tuple(env.info.observation_space.shape) for env in envs]
        act_shapes = [tuple(env.info.action_space.shape) for env in envs]
        if any(len(shape) != 1 for shape in (*obs_shapes, *act_shapes)):
            raise ValueError(
                "ParallelMorphVecEnv currently supports flat observations/actions."
            )
        if (
            len(set(obs_shapes)) != 1 or len(set(act_shapes)) != 1
        ) and not pad_to_max_shapes:
            raise ValueError(
                f"Grouped morphologies must share observation/action shapes; "
                f"got obs={obs_shapes}, act={act_shapes}. Set "
                "pad_to_max_shapes=True for cross-topology grouping."
            )

        wrapped = []
        for env in envs:
            if history_length > 1:
                env = NStepWrapper(env, history_length)
            wrapped.append(LogWrapper(env))

        groups = []
        offset = 0
        for name, size in zip(names, group_sizes, strict=True):
            size = int(size)
            groups.append(MorphologyGroup(str(name), size, offset, offset + size))
            offset += size

        self.envs = tuple(wrapped)
        self.groups = tuple(groups)
        self.group_sizes = tuple(int(n) for n in group_sizes)
        self.names = tuple(str(name) for name in names)
        self.num_envs = offset
        self.num_morphologies = len(self.envs)
        self.history_length = int(history_length)
        self.pad_to_max_shapes = bool(pad_to_max_shapes)
        self.append_group_one_hot = bool(append_group_one_hot)
        self.append_action_mask = bool(append_action_mask)
        self.group_observation_dims = tuple(shape[0] for shape in obs_shapes)
        self.group_action_dims = tuple(shape[0] for shape in act_shapes)
        # Reserved widths let a run keep slots for a robot that is *not* trained
        # on, so a held-out topology can later be fed to the same fixed-width
        # network.  Reserved slots are always zero-filled and masked off.
        self.max_observation_dim = max(
            (*self.group_observation_dims, int(reserved_observation_dim))
        )
        self.max_action_dim = max((*self.group_action_dims, int(reserved_action_dim)))
        self.one_hot_dim = (
            max(self.num_morphologies, int(reserved_group_slots))
            if self.append_group_one_hot
            else 0
        )
        self.group_one_hot_start = self.max_observation_dim
        self.action_mask_observation_start = self.group_one_hot_start + self.one_hot_dim
        self.joint_feature_start = self.action_mask_observation_start + (
            self.max_action_dim if self.append_action_mask else 0
        )
        self.append_joint_features = joint_block_specs is not None
        self.num_joint_slots = self.max_action_dim if self.append_joint_features else 0
        self.output_observation_dim = self.joint_feature_start + (
            self.num_joint_slots * JOINT_FEATURE_DIM
        )
        self._build_joint_blocks(joint_block_specs)

        # PPO/network construction queries these attributes before wrapping.
        self.info = deepcopy(self.envs[0].info)
        self.mdp_info = deepcopy(self.envs[0].mdp_info)
        self.obs_container = self.envs[0].obs_container
        if (
            self.pad_to_max_shapes
            or self.append_group_one_hot
            or self.append_action_mask
            or self.append_joint_features
        ):
            observation_low = np.full(
                (self.output_observation_dim,), -np.inf, dtype=np.float32
            )
            observation_high = np.full(
                (self.output_observation_dim,), np.inf, dtype=np.float32
            )
            if self.append_group_one_hot:
                start = self.group_one_hot_start
                stop = start + self.one_hot_dim
                observation_low[start:stop] = 0.0
                observation_high[start:stop] = 1.0
            if self.append_action_mask:
                start = self.action_mask_observation_start
                stop = start + self.max_action_dim
                observation_low[start:stop] = 0.0
                observation_high[start:stop] = 1.0
            action_low = np.full((self.max_action_dim,), np.inf, dtype=np.float32)
            action_high = np.full((self.max_action_dim,), -np.inf, dtype=np.float32)
            for wrapped_env, action_dim in zip(
                self.envs, self.group_action_dims, strict=True
            ):
                action_low[:action_dim] = np.minimum(
                    action_low[:action_dim],
                    np.asarray(wrapped_env.info.action_space.low),
                )
                action_high[:action_dim] = np.maximum(
                    action_high[:action_dim],
                    np.asarray(wrapped_env.info.action_space.high),
                )
            self.info.observation_space = Box(observation_low, observation_high)
            self.info.action_space = Box(action_low, action_high)
            self.mdp_info.observation_space = deepcopy(self.info.observation_space)
            self.mdp_info.action_space = deepcopy(self.info.action_space)

        self._reset_fns = tuple(jax.vmap(env.reset, in_axes=(0,)) for env in self.envs)
        self._step_fns = tuple(jax.vmap(env.step, in_axes=(0, 0)) for env in self.envs)
        self.morphology_index = jnp.concatenate(
            [jnp.full((g.size,), i, dtype=jnp.int32) for i, g in enumerate(self.groups)]
        )
        self.action_mask = jnp.concatenate(
            [
                jnp.concatenate(
                    [
                        jnp.ones((group.size, action_dim), dtype=jnp.float32),
                        jnp.zeros(
                            (group.size, self.max_action_dim - action_dim),
                            dtype=jnp.float32,
                        ),
                    ],
                    axis=-1,
                )
                for group, action_dim in zip(
                    self.groups, self.group_action_dims, strict=True
                )
            ],
            axis=0,
        )

    def _build_joint_blocks(self, joint_block_specs) -> None:
        """Pad each robot's structural description block to ``num_joint_slots``."""
        self.joint_block_specs = None
        self.urma_input_layout = None
        self._joint_descriptions = None
        if joint_block_specs is None:
            return
        specs = tuple(joint_block_specs)
        if len(specs) != self.num_morphologies:
            raise ValueError(
                f"Expected {self.num_morphologies} joint block specs, got {len(specs)}."
            )
        for spec, action_dim in zip(specs, self.group_action_dims, strict=True):
            if spec.num_joints != action_dim:
                raise ValueError(
                    f"{spec.name}: joint block has {spec.num_joints} joints but the "
                    f"environment has {action_dim} actions; URMA needs one padded "
                    "action slot per encoded joint."
                )
            if spec.num_joints > self.num_joint_slots:
                raise ValueError(
                    f"{spec.name} needs {spec.num_joints} joint slots, but only "
                    f"{self.num_joint_slots} are padded."
                )
        self.joint_block_specs = specs
        self._joint_descriptions = tuple(
            jnp.asarray(
                np.pad(
                    spec.descriptions,
                    ((0, self.num_joint_slots - spec.num_joints), (0, 0)),
                ),
                dtype=jnp.float32,
            )
            for spec in specs
        )
        self._joint_position_indices = tuple(
            jnp.asarray(spec.position_obs_indices, dtype=jnp.int32) for spec in specs
        )
        self._joint_velocity_indices = tuple(
            jnp.asarray(spec.velocity_obs_indices, dtype=jnp.int32) for spec in specs
        )
        one_hot_range = tuple(
            range(self.group_one_hot_start, self.group_one_hot_start + self.one_hot_dim)
        )
        # The action-mask block is deliberately excluded: it is binary bookkeeping
        # for the masked MLP, and the joint block already carries a per-joint
        # validity bit that URMA reads unnormalised.
        self.urma_input_layout = URMAInputLayout(
            base_observation_dim=self.max_observation_dim,
            morphology_start=self.group_one_hot_start,
            morphology_dim=self.one_hot_dim,
            joint_feature_start=self.joint_feature_start,
            num_joints=self.num_joint_slots,
            joint_description_dim=GENERIC_JOINT_DESCRIPTION_DIM,
            joint_state_dim=JOINT_STATE_DIM,
            joint_feature_dim=JOINT_FEATURE_DIM,
            joint_position_indices=(),
            joint_velocity_indices=(),
            general_indices=tuple(range(self.max_observation_dim)) + one_hot_range,
        )

    def _joint_block(self, observation, group_index: int, last_action):
        """``(n, num_joint_slots * JOINT_FEATURE_DIM)`` from a raw group observation."""
        n = observation.shape[0]
        slots = self.num_joint_slots
        num_joints = self.group_action_dims[group_index]
        dtype = observation.dtype

        descriptions = jnp.broadcast_to(
            self._joint_descriptions[group_index].astype(dtype)[None, :, :],
            (n, slots, GENERIC_JOINT_DESCRIPTION_DIM),
        )
        position = observation[:, self._joint_position_indices[group_index]]
        velocity = observation[:, self._joint_velocity_indices[group_index]]
        if last_action is None:
            last_action = jnp.zeros((n, num_joints), dtype=dtype)
        state = jnp.stack(
            [position, velocity, last_action.astype(dtype)], axis=-1
        )  # (n, num_joints, 3)
        pad = slots - num_joints
        if pad:
            state = jnp.pad(state, ((0, 0), (0, pad), (0, 0)))
        valid = (jnp.arange(slots) < num_joints).astype(dtype)
        valid = jnp.broadcast_to(valid[None, :, None], (n, slots, 1))
        block = jnp.concatenate([descriptions, state, valid], axis=-1)
        return block.reshape((n, slots * JOINT_FEATURE_DIM))

    def _output_observation(self, observation, group_index: int, last_action=None):
        joint_block = (
            self._joint_block(observation, group_index, last_action)
            if self.append_joint_features
            else None
        )
        pad = self.max_observation_dim - observation.shape[-1]
        if pad:
            observation = jnp.pad(observation, ((0, 0), (0, pad)))
        if self.append_group_one_hot:
            group_id = jax.nn.one_hot(
                group_index, self.one_hot_dim, dtype=observation.dtype
            )
            group_id = jnp.broadcast_to(
                group_id[None, :], (observation.shape[0], self.one_hot_dim)
            )
            observation = jnp.concatenate([observation, group_id], axis=-1)
        if self.append_action_mask:
            action_dim = self.group_action_dims[group_index]
            mask = jnp.arange(self.max_action_dim) < action_dim
            mask = jnp.broadcast_to(
                mask[None, :], (observation.shape[0], self.max_action_dim)
            ).astype(observation.dtype)
            observation = jnp.concatenate([observation, mask], axis=-1)
        if joint_block is not None:
            observation = jnp.concatenate([observation, joint_block], axis=-1)
        return observation

    def reset(self, rng_keys):
        if rng_keys.shape[0] != self.num_envs:
            raise ValueError(
                f"Expected {self.num_envs} reset keys, got {rng_keys.shape[0]}."
            )
        outputs = [
            fn(rng_keys[group.start : group.stop])
            for fn, group in zip(self._reset_fns, self.groups, strict=True)
        ]
        observations = jnp.concatenate(
            [self._output_observation(out[0], i) for i, out in enumerate(outputs)],
            axis=0,
        )
        states = ParallelMorphState(tuple(out[1] for out in outputs))
        return observations, states

    def step(self, state, action):
        if action.shape[0] != self.num_envs:
            raise ValueError(
                f"Expected {self.num_envs} actions, got {action.shape[0]}."
            )
        outputs = []
        group_actions = []
        for fn, group, group_state, action_dim in zip(
            self._step_fns,
            self.groups,
            state.group_states,
            self.group_action_dims,
            strict=True,
        ):
            group_action = jax.lax.dynamic_slice_in_dim(
                action, group.start, group.size, axis=0
            )
            group_action = group_action[:, :action_dim]
            group_actions.append(group_action)
            outputs.append(fn(group_state, group_action))

        observations = jnp.concatenate(
            [
                self._output_observation(out[0], i, group_actions[i])
                for i, out in enumerate(outputs)
            ],
            axis=0,
        )
        rewards = jnp.concatenate([out[1] for out in outputs], axis=0)
        absorbing = jnp.concatenate([out[2] for out in outputs], axis=0)
        done = jnp.concatenate([out[3] for out in outputs], axis=0)
        info = _concat_pytrees([out[4] for out in outputs])
        next_state = ParallelMorphState(tuple(out[5] for out in outputs))
        return observations, rewards, absorbing, done, info, next_state


@struct.dataclass
class ParallelNormalizeRewardState(BaseWrapperState):
    env_state: object
    mean: jax.Array
    var: jax.Array
    count: jax.Array
    return_val: jax.Array


class ParallelNormalizeReward(BaseWrapper):
    """Return normalization with independent statistics per morphology group."""

    def __init__(self, env: ParallelMorphVecEnv, gamma: float):
        super().__init__(env)
        self.gamma = float(gamma)
        self.groups = env.groups
        self.num_morphologies = env.num_morphologies
        self.num_envs = env.num_envs

    def reset(self, key):
        obs, env_state = self.env.reset(key)
        state = ParallelNormalizeRewardState(
            env_state=env_state,
            mean=jnp.zeros((self.num_morphologies,), dtype=jnp.float32),
            var=jnp.ones((self.num_morphologies,), dtype=jnp.float32),
            count=jnp.full((self.num_morphologies,), 1e-4, dtype=jnp.float32),
            return_val=jnp.zeros((self.num_envs,), dtype=jnp.float32),
        )
        return obs, state

    def step(self, state, action):
        obs, reward, absorbing, done, info, env_state = self.env.step(
            state.env_state, action
        )
        return_val = state.return_val * self.gamma * (1 - done) + reward
        new_mean = state.mean
        new_var = state.var
        new_count = state.count
        normalized_reward = jnp.zeros_like(reward)

        for i, group in enumerate(self.groups):
            values = jax.lax.dynamic_slice_in_dim(
                return_val, group.start, group.size, axis=0
            )
            batch_mean = jnp.mean(values)
            batch_var = jnp.var(values)
            batch_count = jnp.asarray(group.size, dtype=state.count.dtype)
            delta = batch_mean - state.mean[i]
            total_count = state.count[i] + batch_count
            mean_i = state.mean[i] + delta * batch_count / total_count
            m_a = state.var[i] * state.count[i]
            m_b = batch_var * batch_count
            m2 = m_a + m_b + delta**2 * state.count[i] * batch_count / total_count
            var_i = m2 / total_count

            new_mean = new_mean.at[i].set(mean_i)
            new_var = new_var.at[i].set(var_i)
            new_count = new_count.at[i].set(total_count)
            reward_i = jax.lax.dynamic_slice_in_dim(
                reward, group.start, group.size, axis=0
            ) / jnp.sqrt(var_i + 1e-8)
            normalized_reward = jax.lax.dynamic_update_slice_in_dim(
                normalized_reward, reward_i, group.start, axis=0
            )

        next_state = ParallelNormalizeRewardState(
            env_state=env_state,
            mean=new_mean,
            var=new_var,
            count=new_count,
            return_val=return_val,
        )
        return obs, normalized_reward, absorbing, done, info, next_state


class ParallelMorphPPO(PPOJax):
    """PPOJax variant that consumes an already-vectorized morphology batch."""

    @staticmethod
    def _wrap_env(env, config):
        if not getattr(env, "is_parallel_morph_env", False):
            raise TypeError("ParallelMorphPPO requires ParallelMorphVecEnv.")
        requested_history = int(getattr(config, "len_obs_history", 1))
        if requested_history != env.history_length:
            raise ValueError(
                "Observation history must be applied inside each morphology group; "
                f"config requested {requested_history}, env has {env.history_length}."
            )
        if config.normalize_env:
            env = ParallelNormalizeReward(env, config.gamma)
        return env


class ParallelMorphAMP(AMPJax):
    """AMP using the same grouped-static vectorization as ParallelMorphPPO."""

    @staticmethod
    def _wrap_env(env, config):
        if not isinstance(env, ParallelMorphVecEnv):
            raise TypeError(
                "ParallelMorphAMP expects ParallelMorphVecEnv; received "
                f"{type(env).__name__}."
            )
        if "len_obs_history" in config and config.len_obs_history > 1:
            raise NotImplementedError(
                "History stacking is not implemented for grouped morphology AMP."
            )
        if config.normalize_env:
            env = ParallelNormalizeReward(env, config.gamma)
        return env


def balanced_group_sizes(total_envs: int, num_morphologies: int) -> tuple[int, ...]:
    """Split a total environment budget as evenly as possible."""

    if total_envs < num_morphologies:
        raise ValueError("total_envs must be >= num_morphologies")
    base, remainder = divmod(int(total_envs), int(num_morphologies))
    return tuple(base + (i < remainder) for i in range(num_morphologies))


def describe_layout(names: Sequence[str], sizes: Sequence[int]) -> str:
    return ", ".join(
        f"{name}:{int(size)}" for name, size in zip(names, sizes, strict=True)
    )
