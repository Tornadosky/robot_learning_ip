"""Single-graph, per-environment H1 embodiment randomization.

The grouped-static trainer is useful for a finite set of XML models, but every
model becomes another traced MJX branch.  This module instead keeps the H1
topology fixed and makes the small model arrays that define its proportions
depend on a morphology vector carried by each environment.

Version 1 intentionally changes kinematics and inertial properties, but not the
143k-vertex visual/collision meshes.  Replicating or transforming those vertices
per environment would erase the memory/compile advantage this path is meant to
provide.  Mesh-aware reference retargeting is a separate, offline layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Sequence

import jax
import jax.numpy as jnp
from flax import struct
import mujoco
from mujoco import mjx
import numpy as np

from loco_mujoco.core import MjxState
from loco_mujoco.core.utils.env import Box
from loco_mujoco.environments import LocoEnv
from loco_mujoco.environments.base import LocoCarry
from loco_mujoco.environments.humanoids import MjxUnitreeH1


MORPHOLOGY_NAMES = (
    "leg_length_scale",
    "arm_length_scale",
    "shoulder_width_scale",
    "torso_mass_scale",
)

#: ``continuous``       - draw a fresh uniform body at every reset (the original path).
#: ``fixed_balanced``   - each environment slot owns one catalog body for the whole run.
#: ``catalog_resample`` - each slot walks the catalog on a balanced deterministic schedule.
CATALOG_MODES = ("continuous", "fixed_balanced", "catalog_resample")


@dataclass(frozen=True)
class MorphologyBounds:
    low: tuple[float, float, float, float] = (0.85, 0.85, 0.85, 0.70)
    high: tuple[float, float, float, float] = (1.20, 1.20, 1.20, 1.50)

    def validate(self) -> None:
        if len(self.low) != len(MORPHOLOGY_NAMES) or len(self.high) != len(
            MORPHOLOGY_NAMES
        ):
            raise ValueError(
                f"Morphology bounds must have {len(MORPHOLOGY_NAMES)} values."
            )
        if np.any(np.asarray(self.low) <= 0.0):
            raise ValueError("Morphology scales must be positive.")
        if np.any(np.asarray(self.high) <= np.asarray(self.low)):
            raise ValueError(
                "Every morphology upper bound must exceed its lower bound."
            )


@dataclass(frozen=True)
class URMAInputLayout:
    """Slices for the structured, flat observation consumed by URMA.

    LocoMuJoCo exposes flat observations to PPO.  The optional joint-feature
    suffix keeps that interface while giving an embodiment-aware network an
    unambiguous ``[joint, description/state/mask]`` view.
    """

    base_observation_dim: int
    morphology_start: int
    morphology_dim: int
    joint_feature_start: int
    num_joints: int
    joint_description_dim: int
    joint_state_dim: int
    joint_feature_dim: int
    joint_position_indices: tuple[int, ...]
    joint_velocity_indices: tuple[int, ...]
    general_indices: tuple[int, ...]

    @property
    def joint_feature_stop(self) -> int:
        return self.joint_feature_start + self.num_joints * self.joint_feature_dim

    @property
    def joint_mask_offset(self) -> int:
        return self.joint_description_dim + self.joint_state_dim


@struct.dataclass
class OnlineH1Carry(LocoCarry):
    """LocoMuJoCo carry extended with one persistent embodiment per env.

    ``body_slot`` is the environment's position in the vector batch and never
    changes.  ``body_index`` is the row of the active catalog entry, or -1 in
    continuous mode where no catalog exists.
    """

    morphology: jax.Array
    morphology_generation: jax.Array
    body_slot: jax.Array
    body_index: jax.Array


class OnlineMorphMjxUnitreeH1(MjxUnitreeH1):
    """H1 whose same-DOF morphology is a dynamic JAX value.

    A morphology is sampled at the outer reset.  It can either remain fixed for
    that vector environment or be resampled after every episode.  The reset path
    returns an observation generated from the new body, so asynchronous resets do
    not create an action/body mismatch.
    """

    descriptor_names = MORPHOLOGY_NAMES

    def __init__(
        self,
        *args,
        morphology_low: Sequence[float] = MorphologyBounds().low,
        morphology_high: Sequence[float] = MorphologyBounds().high,
        append_morphology_to_observation: bool = True,
        append_urma_joint_features: bool = False,
        resample_morphology_on_episode_reset: bool = False,
        catalog_descriptors: np.ndarray | None = None,
        catalog_mode: str = "continuous",
        catalog_stride: int = 1,
        **kwargs,
    ):
        bounds = MorphologyBounds(tuple(morphology_low), tuple(morphology_high))
        bounds.validate()
        self._morphology_low_np = np.asarray(bounds.low, dtype=np.float32)
        self._morphology_high_np = np.asarray(bounds.high, dtype=np.float32)
        self._append_morphology = bool(append_morphology_to_observation)
        self._append_urma_joint_features = bool(append_urma_joint_features)
        self._resample_morphology = bool(resample_morphology_on_episode_reset)
        if self._append_urma_joint_features and not self._append_morphology:
            raise ValueError(
                "URMA joint features require the global morphology descriptor."
            )

        if catalog_mode not in CATALOG_MODES:
            raise ValueError(
                f"catalog_mode must be one of {CATALOG_MODES}; got {catalog_mode!r}."
            )
        self._catalog_mode = str(catalog_mode)
        self._catalog_stride = int(catalog_stride)
        if self._catalog_mode == "continuous":
            if catalog_descriptors is not None:
                raise ValueError(
                    "Continuous mode samples uniformly and takes no catalog."
                )
            self._catalog_np = None
            self._num_catalog_bodies = 0
        else:
            if catalog_descriptors is None:
                raise ValueError(f"{self._catalog_mode} requires catalog_descriptors.")
            catalog = np.asarray(catalog_descriptors, dtype=np.float32)
            if catalog.ndim != 2 or catalog.shape[1] != len(MORPHOLOGY_NAMES):
                raise ValueError(
                    f"Catalog descriptors must be (N, {len(MORPHOLOGY_NAMES)}); "
                    f"got {catalog.shape}."
                )
            if not np.all(np.isfinite(catalog)) or np.any(catalog <= 0.0):
                raise ValueError("Catalog descriptors must be finite and positive.")
            if self._catalog_stride < 1:
                raise ValueError("catalog_stride must be at least 1.")
            self._catalog_np = catalog
            self._num_catalog_bodies = int(catalog.shape[0])

        super().__init__(*args, **kwargs)

        self._catalog = (
            None if self._catalog_np is None else jnp.asarray(self._catalog_np)
        )

        self._morphology_low = jnp.asarray(self._morphology_low_np)
        self._morphology_high = jnp.asarray(self._morphology_high_np)
        self._cache_morphology_indices()
        self._cache_urma_constants()

        # These are closed-over constants shared by every vmapped environment.
        # Only the compact transformed arrays below become per-env values.
        self._base_body_pos = self.sys.body_pos
        self._base_body_ipos = self.sys.body_ipos
        self._base_body_mass = self.sys.body_mass
        self._base_body_inertia = self.sys.body_inertia
        self._base_site_pos = self.sys.site_pos

        self._base_observation_dim = int(self.info.observation_space.shape[0])
        extra_low = []
        extra_high = []
        if self._append_morphology:
            extra_low.append(-np.ones(len(MORPHOLOGY_NAMES), dtype=np.float32))
            extra_high.append(np.ones(len(MORPHOLOGY_NAMES), dtype=np.float32))
        if self._append_urma_joint_features:
            n_features = self._num_urma_joints * self._urma_joint_feature_dim
            extra_low.append(np.full(n_features, -np.inf, dtype=np.float32))
            extra_high.append(np.full(n_features, np.inf, dtype=np.float32))
        if extra_low:
            low = np.concatenate(
                [np.asarray(self.info.observation_space.low), *extra_low]
            )
            high = np.concatenate(
                [np.asarray(self.info.observation_space.high), *extra_high]
            )
            self._mdp_info.observation_space = Box(low, high)

        self.urma_input_layout = self._make_urma_input_layout()

    def _body_id(self, name: str) -> int:
        return mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, name)

    def _site_id(self, name: str) -> int:
        return mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, name)

    def _cache_morphology_indices(self) -> None:
        self._leg_inertial_ids = np.asarray(
            [
                self._body_id(f"{side}_{link}_link")
                for side in ("left", "right")
                for link in ("hip_pitch", "knee")
            ],
            dtype=np.int32,
        )
        self._leg_position_ids = np.asarray(
            [
                self._body_id(f"{side}_{link}_link")
                for side in ("left", "right")
                for link in ("knee", "ankle")
            ],
            dtype=np.int32,
        )
        self._upper_arm_ids = np.asarray(
            [self._body_id(f"{side}_shoulder_yaw_link") for side in ("left", "right")],
            dtype=np.int32,
        )
        self._forearm_ids = np.asarray(
            [self._body_id(f"{side}_elbow_link") for side in ("left", "right")],
            dtype=np.int32,
        )
        self._shoulder_ids = np.asarray(
            [
                self._body_id(f"{side}_shoulder_pitch_link")
                for side in ("left", "right")
            ],
            dtype=np.int32,
        )
        self._hand_site_ids = np.asarray(
            [self._site_id(f"{side}_hand_mimic") for side in ("left", "right")],
            dtype=np.int32,
        )
        self._torso_id = self._body_id("torso_link")

    @staticmethod
    def _finite_scaled(values, scale: float) -> jax.Array:
        values = np.asarray(values, dtype=np.float32) / float(scale)
        return jnp.asarray(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0))

    def _cache_urma_constants(self) -> None:
        """Cache action-ordered joint properties used in description vectors."""

        model = self._model
        action_dim = int(self.info.action_space.shape[0])
        joint_ids = np.asarray(model.actuator_trnid[:action_dim, 0], dtype=np.int32)
        if len(np.unique(joint_ids)) != action_dim:
            raise ValueError("URMA currently requires one actuator per H1 joint.")
        body_ids = np.asarray(model.jnt_bodyid[joint_ids], dtype=np.int32)
        dof_ids = np.asarray(model.jnt_dofadr[joint_ids], dtype=np.int32)
        qpos_ids = np.asarray(model.jnt_qposadr[joint_ids], dtype=np.int32)

        qpos_obs = np.asarray(self._obs_indices.JointPos, dtype=np.int32)
        qvel_obs = np.asarray(self._obs_indices.JointVel, dtype=np.int32)
        if len(qpos_obs) != action_dim or len(qvel_obs) != action_dim:
            raise ValueError(
                "URMA H1 expects action-ordered scalar JointPos/JointVel observations."
            )

        child_counts = np.asarray(
            [
                np.count_nonzero(model.body_parentid[body_ids] == body_id)
                for body_id in body_ids
            ],
            dtype=np.float32,
        )[:, None]
        force_ranges = np.asarray(model.actuator_forcerange[:action_dim])
        force_limits = np.max(np.abs(force_ranges), axis=-1, keepdims=True)

        self._urma_joint_ids = jnp.asarray(joint_ids)
        self._urma_body_ids = jnp.asarray(body_ids)
        self._urma_qpos_obs_indices = tuple(int(i) for i in qpos_obs)
        self._urma_qvel_obs_indices = tuple(int(i) for i in qvel_obs)
        self._urma_child_counts = self._finite_scaled(child_counts, 3.0)
        self._urma_nominal_qpos = self._finite_scaled(
            np.asarray(model.qpos0)[qpos_ids, None], np.pi
        )
        self._urma_force_limits = self._finite_scaled(force_limits, 1000.0)
        self._urma_damping = self._finite_scaled(
            np.asarray(model.dof_damping)[dof_ids, None], 10.0
        )
        self._urma_armature = self._finite_scaled(
            np.asarray(model.dof_armature)[dof_ids, None], 0.2
        )
        self._urma_stiffness = self._finite_scaled(
            np.asarray(model.jnt_stiffness)[joint_ids, None], 30.0
        )
        self._urma_frictionloss = self._finite_scaled(
            np.asarray(model.dof_frictionloss)[dof_ids, None], 1.2
        )
        self._urma_joint_range = self._finite_scaled(
            np.asarray(model.jnt_range)[joint_ids], np.pi
        )
        self._urma_ctrl_range = self._finite_scaled(
            np.asarray(model.actuator_ctrlrange)[:action_dim], np.pi
        )
        self._num_urma_joints = action_dim
        # position(3), axis(3), child count, nominal, force, four passive
        # parameters, range(2), control range(2), total/body mass, inertia(3),
        # and the four explicit online morphology coordinates.
        self._urma_joint_description_dim = 26
        self._urma_joint_state_dim = 3  # position, velocity, previous action
        self._urma_joint_feature_dim = (
            self._urma_joint_description_dim + self._urma_joint_state_dim + 1
        )  # final scalar is a valid-joint mask

    def _make_urma_input_layout(self) -> URMAInputLayout | None:
        if not self._append_urma_joint_features:
            return None
        morphology_start = self._base_observation_dim
        feature_start = morphology_start + len(MORPHOLOGY_NAMES)
        excluded = set(self._urma_qpos_obs_indices + self._urma_qvel_obs_indices)
        general = tuple(
            i for i in range(self._base_observation_dim) if i not in excluded
        ) + tuple(range(morphology_start, feature_start))
        return URMAInputLayout(
            base_observation_dim=self._base_observation_dim,
            morphology_start=morphology_start,
            morphology_dim=len(MORPHOLOGY_NAMES),
            joint_feature_start=feature_start,
            num_joints=self._num_urma_joints,
            joint_description_dim=self._urma_joint_description_dim,
            joint_state_dim=self._urma_joint_state_dim,
            joint_feature_dim=self._urma_joint_feature_dim,
            joint_position_indices=self._urma_qpos_obs_indices,
            joint_velocity_indices=self._urma_qvel_obs_indices,
            general_indices=general,
        )

    def _normalized_morphology(self, morphology: jax.Array) -> jax.Array:
        midpoint = 0.5 * (self._morphology_low + self._morphology_high)
        half_range = 0.5 * (self._morphology_high - self._morphology_low)
        return (morphology - midpoint) / half_range

    @property
    def catalog_mode(self) -> str:
        return self._catalog_mode

    @property
    def num_catalog_bodies(self) -> int:
        return self._num_catalog_bodies

    def _body_index_for(self, slot: jax.Array, generation: jax.Array) -> jax.Array:
        """Deterministic catalog row for one environment slot and reset count."""
        slot = jnp.asarray(slot, dtype=jnp.int32)
        generation = jnp.asarray(generation, dtype=jnp.int32)
        if self._catalog_mode == "fixed_balanced":
            index = slot % self._num_catalog_bodies
        else:
            index = (
                slot + generation * self._catalog_stride
            ) % self._num_catalog_bodies
        return index.astype(jnp.int32)

    def _init_additional_carry(
        self,
        key: jax.Array,
        model,
        data,
        backend: ModuleType,
    ) -> OnlineH1Carry:
        return self._init_carry_with_slot(key, model, data, backend, slot=None)

    def _init_carry_with_slot(
        self,
        key: jax.Array,
        model,
        data,
        backend: ModuleType,
        slot: jax.Array | None,
    ) -> OnlineH1Carry:
        carry = super()._init_additional_carry(key, model, data, backend)
        key, body_key = jax.random.split(carry.key)
        if self._catalog_mode == "continuous":
            return OnlineH1Carry(
                morphology=self._sample_morphology(body_key),
                morphology_generation=jnp.asarray(
                    0 if self._resample_morphology else 1, dtype=jnp.int32
                ),
                body_slot=jnp.asarray(-1, dtype=jnp.int32),
                body_index=jnp.asarray(-1, dtype=jnp.int32),
                **vars(carry.replace(key=key)),
            )

        # Catalog modes.  A plain reset has no environment index, so an
        # unforced slot is drawn from the reset key; the slot-aware vector
        # wrapper instead supplies the exact balanced assignment.
        if slot is None:
            slot = jax.random.randint(
                body_key, (), 0, self._num_catalog_bodies, dtype=jnp.int32
            )
        slot = jnp.asarray(slot, dtype=jnp.int32)
        index = self._body_index_for(slot, jnp.asarray(0, dtype=jnp.int32))
        return OnlineH1Carry(
            morphology=self._catalog[index],
            # _mjx_reset_carry advances this to 0 on the very first reset, so
            # the outer reset uses generation 0 rather than skipping a body.
            morphology_generation=jnp.asarray(-1, dtype=jnp.int32),
            body_slot=slot,
            body_index=index,
            **vars(carry.replace(key=key)),
        )

    def mjx_reset_with_slot(self, key: jax.Array, slot: jax.Array):
        """Reset with an externally assigned catalog slot.

        This mirrors ``Mjx.mjx_reset`` exactly; only the carry initialisation
        differs, so ``mjx_reset_with_slot(key, s)`` and ``mjx_reset(key)`` agree
        whenever the key would have drawn slot ``s``.
        """
        if self._catalog_mode == "continuous":
            raise ValueError("Slot assignment requires a catalog mode.")
        key, subkey = jax.random.split(key)
        data = self._first_data
        carry = self._init_carry_with_slot(key, self._model, data, jnp, slot=slot)
        data, carry = self._mjx_reset_carry(self.sys, data, carry)
        data, carry = self.obs_container.reset_state(
            self, self._model, data, carry, jnp
        )
        obs, carry = self._mjx_create_observation(self._model, data, carry)
        info = self._mjx_reset_info_dictionary(obs, data, subkey)
        return MjxState(
            data=data,
            observation=obs,
            reward=0.0,
            absorbing=jnp.array(False, dtype=bool),
            done=jnp.array(False, dtype=bool),
            info=info,
            additional_carry=carry,
        )

    def _sample_morphology(self, key: jax.Array) -> jax.Array:
        return jax.random.uniform(
            key,
            shape=(len(MORPHOLOGY_NAMES),),
            minval=self._morphology_low,
            maxval=self._morphology_high,
        )

    def _apply_morphology(self, model, morphology: jax.Array):
        leg_scale, arm_scale, shoulder_scale, torso_mass_scale = morphology

        body_pos = self._base_body_pos
        leg_pos_scale = jnp.asarray([1.0, 1.0, leg_scale])
        body_pos = body_pos.at[self._leg_position_ids].set(
            self._base_body_pos[self._leg_position_ids] * leg_pos_scale
        )
        body_pos = body_pos.at[self._upper_arm_ids].set(
            self._base_body_pos[self._upper_arm_ids]
            * jnp.asarray([1.0, 1.0, arm_scale])
        )
        body_pos = body_pos.at[self._forearm_ids].set(
            self._base_body_pos[self._forearm_ids] * arm_scale
        )
        body_pos = body_pos.at[self._shoulder_ids].set(
            self._base_body_pos[self._shoulder_ids]
            * jnp.asarray([1.0, shoulder_scale, 1.0])
        )

        # Match the existing XML generator's simple volume/inertia scaling.
        body_scale = jnp.ones_like(self._base_body_ipos)
        body_scale = body_scale.at[self._leg_inertial_ids, 2].set(leg_scale)
        body_scale = body_scale.at[self._upper_arm_ids, 2].set(arm_scale)
        body_scale = body_scale.at[self._forearm_ids, 0].set(arm_scale)
        volume_scale = jnp.prod(body_scale, axis=1)
        inertia_scale = volume_scale * jnp.square(jnp.mean(body_scale, axis=1))
        body_ipos = self._base_body_ipos * body_scale
        body_mass = self._base_body_mass * volume_scale
        body_inertia = self._base_body_inertia * inertia_scale[:, None]
        body_mass = body_mass.at[self._torso_id].multiply(torso_mass_scale)
        body_inertia = body_inertia.at[self._torso_id].multiply(torso_mass_scale)

        site_pos = self._base_site_pos
        site_pos = site_pos.at[self._hand_site_ids].set(
            self._base_site_pos[self._hand_site_ids]
            * jnp.asarray([arm_scale, 1.0, 1.0])
        )

        return model.replace(
            body_pos=body_pos,
            body_ipos=body_ipos,
            body_mass=body_mass,
            body_inertia=body_inertia,
            site_pos=site_pos,
        )

    def _urma_joint_descriptions(self, morphology: jax.Array) -> jax.Array:
        """Return one explicit, online-updated description per actuated joint."""

        model = self._apply_morphology(self.sys, morphology)
        body_ids = self._urma_body_ids
        joint_ids = self._urma_joint_ids
        morphology_desc = self._normalized_morphology(morphology)
        morphology_desc = jnp.broadcast_to(
            morphology_desc[None, :], (self._num_urma_joints, len(MORPHOLOGY_NAMES))
        )
        total_mass = jnp.broadcast_to(
            (jnp.sum(model.body_mass) / 100.0)[None, None],
            (self._num_urma_joints, 1),
        )
        descriptions = jnp.concatenate(
            [
                model.body_pos[body_ids] / 1.0,
                model.jnt_axis[joint_ids],
                self._urma_child_counts,
                self._urma_nominal_qpos,
                self._urma_force_limits,
                self._urma_damping,
                self._urma_armature,
                self._urma_stiffness,
                self._urma_frictionloss,
                self._urma_joint_range,
                self._urma_ctrl_range,
                total_mass,
                model.body_mass[body_ids, None] / 20.0,
                model.body_inertia[body_ids] / 5.0,
                morphology_desc,
            ],
            axis=-1,
        )
        if descriptions.shape[-1] != self._urma_joint_description_dim:
            raise AssertionError(
                f"Unexpected URMA description width {descriptions.shape[-1]}."
            )
        return descriptions

    #: Nominal H1 leg length: two 0.4 m segments from hip pitch to ankle.
    NOMINAL_LEG_LENGTH_M = 0.8

    def root_height_offset(self, morphology: jax.Array) -> jax.Array:
        """How much higher than the reference this body's root should stand.

        The reset raises the root by this amount so a taller body starts on the
        floor rather than inside it.  The morphology-aware terminal handler
        subtracts the same amount before comparing against the reference height
        window, so the two can never drift apart.
        """
        return self.NOMINAL_LEG_LENGTH_M * (morphology[0] - 1.0)

    def _mjx_reset_carry(self, model, data, carry):
        data, carry = super()._mjx_reset_carry(model, data, carry)
        if self._catalog_mode != "continuous":
            generation = carry.morphology_generation + 1
            index = self._body_index_for(carry.body_slot, generation)
            carry = carry.replace(
                morphology=self._catalog[index],
                body_index=index,
                morphology_generation=generation,
            )
        elif self._resample_morphology:
            key, morphology_key = jax.random.split(carry.key)
            carry = carry.replace(
                key=key,
                morphology=self._sample_morphology(morphology_key),
                morphology_generation=carry.morphology_generation + 1,
            )
        # The nominal reference is floor-grounded; a longer leg lifts the root.
        qpos = data.qpos.at[2].add(self.root_height_offset(carry.morphology))
        data = data.replace(qpos=qpos)
        data = mjx.forward(self._apply_morphology(model, carry.morphology), data)
        return data, carry

    @property
    def _body_changes_on_reset(self) -> bool:
        if self._catalog_mode == "catalog_resample":
            return True
        if self._catalog_mode == "fixed_balanced":
            return False
        return self._resample_morphology

    def _mjx_reset_in_step(self, state):
        state = super()._mjx_reset_in_step(state)
        if self._body_changes_on_reset:
            # LocoMjxWrapper normally exposes the terminal observation on a
            # done step.  PPO then uses that observation to choose an action
            # for the already-reset simulator state.  With a newly sampled
            # body this would also expose the old embodiment descriptor.  For
            # this environment, route the reset observation to the runner;
            # PPO already stores the pre-step observation in its transition.
            carry = state.additional_carry.replace(final_observation=state.observation)
            state = state.replace(additional_carry=carry)
        return state

    def _mjx_simulation_pre_step(self, model, data, carry):
        model = self._apply_morphology(model, carry.morphology)
        return super()._mjx_simulation_pre_step(model, data, carry)

    def _mjx_create_observation(self, model, data, carry):
        observation, carry = super()._mjx_create_observation(model, data, carry)
        chunks = [observation]
        if self._append_morphology:
            chunks.append(self._normalized_morphology(carry.morphology))
        if self._append_urma_joint_features:
            descriptions = self._urma_joint_descriptions(carry.morphology)
            joint_state = jnp.stack(
                [
                    observation[jnp.asarray(self._urma_qpos_obs_indices)],
                    observation[jnp.asarray(self._urma_qvel_obs_indices)],
                    carry.last_action,
                ],
                axis=-1,
            )
            valid = jnp.ones((self._num_urma_joints, 1), dtype=observation.dtype)
            chunks.append(
                jnp.concatenate([descriptions, joint_state, valid], axis=-1).reshape(-1)
            )
        return jnp.concatenate(chunks), carry


def register_online_h1_env(env_name: str, xml_path: Path) -> type:
    """Register a non-destructive online-morphology H1 class for one run."""
    if env_name in LocoEnv.registered_envs:
        return LocoEnv.registered_envs[env_name]

    xml_path = Path(xml_path).resolve()

    def get_default_xml_file_path(cls) -> str:
        return str(xml_path)

    env_cls = type(
        env_name,
        (OnlineMorphMjxUnitreeH1,),
        {"get_default_xml_file_path": classmethod(get_default_xml_file_path)},
    )
    LocoEnv.registered_envs[env_name] = env_cls
    return env_cls
