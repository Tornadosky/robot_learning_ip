"""Online within-family morphology randomization for any LocoMuJoCo humanoid.

Generalizes the proven ``online_h1.OnlineMorphMjxUnitreeH1`` pattern (dynamic
JAX morphology, no recompile) to a small family-adapter: each family names its
knee/ankle attachment bodies and torso body; everything else is derived from
the model.  Four shared coordinates per family (identical layout, family-
specific safe bounds are possible but tonight all use the same conservative
box):

    leg_length_scale   — GEOMETRIC: z-offsets of knee/ankle attachments (thigh
                         and shin lengths), with volume-consistent mass and
                         inertia scaling of the parent segments;
    torso_mass_scale   — torso mass + inertia;
    damping_scale      — all dof damping;
    strength_scale     — actuator gain AND force range (so the per-joint
                         force-limit descriptor reflects the sampled actuator).

The sampled model is what the policy must be told about: use
``dynamic_joint_descriptions`` (22 structural dims recomputed from the mutated
model + the 4 normalized morphology coordinates = 26) as the grouped
environment's per-joint description source.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from flax import struct
from mujoco import mjx

from loco_mujoco.environments.base import LocoCarry

from scaling.joint_descriptions import (
    GENERIC_JOINT_DESCRIPTION_DIM,
    _SCALES,
    actuated_joint_ids,
    generic_joint_descriptions,
)
from scaling.morphology_admission import (
    NUM_REASONS as NUM_REJECTION_REASONS,
)
from scaling.morphology_admission import (
    AdmissionConfig,
    AdmissionStats,
    admit_morphology,
    record_state_rejections,
    state_rejection_flags,
    update_stats,
    zero_stats,
)

FAMILY_MORPHOLOGY_SPEC = (
    ("leg_length_scale", 0.90, 1.12),
    ("torso_mass_scale", 0.70, 1.50),
    ("damping_scale", 0.50, 2.00),
    ("strength_scale", 0.70, 1.30),
)
FAMILY_MORPHOLOGY_NAMES = tuple(name for name, _, _ in FAMILY_MORPHOLOGY_SPEC)
FAMILY_MORPHOLOGY_DIM = len(FAMILY_MORPHOLOGY_SPEC)
FAMILY_MORPHOLOGY_LOW = np.asarray(
    [low for _, low, _ in FAMILY_MORPHOLOGY_SPEC], dtype=np.float32
)
FAMILY_MORPHOLOGY_HIGH = np.asarray(
    [high for _, _, high in FAMILY_MORPHOLOGY_SPEC], dtype=np.float32
)

#: 22 structural dims + the normalized family morphology coordinates.
FAMILY_JOINT_DESCRIPTION_DIM = GENERIC_JOINT_DESCRIPTION_DIM + FAMILY_MORPHOLOGY_DIM


@dataclass(frozen=True)
class FamilyBodies:
    """Bodies that define the family's geometric leg coordinate."""

    knee_bodies: tuple[str, str]
    ankle_bodies: tuple[str, str]
    torso_body: str


FAMILY_BODIES = {
    "h1": FamilyBodies(
        knee_bodies=("left_knee_link", "right_knee_link"),
        ankle_bodies=("left_ankle_link", "right_ankle_link"),
        torso_body="torso_link",
    ),
    "g1": FamilyBodies(
        knee_bodies=("left_knee_link", "right_knee_link"),
        ankle_bodies=("left_ankle_pitch_link", "right_ankle_pitch_link"),
        torso_body="torso_link",
    ),
    "atlas": FamilyBodies(
        knee_bodies=("l_lleg", "r_lleg"),
        ankle_bodies=("l_talus", "r_talus"),
        torso_body="utorso",
    ),
    # Added 2026-08-15 for the 5+ topology smoke.  Same convention as above:
    # knee/ankle attachment bodies carry the z-offsets that define thigh and
    # shin length; the torso body carries the mass coordinate.  ToddlerBot is
    # deliberately absent: its knees are closed-loop four-bar rods, so scaling
    # the calf attachment would break the loop constraints.
    "talos": FamilyBodies(
        knee_bodies=("leg_left_4_link", "leg_right_4_link"),
        ankle_bodies=("leg_left_5_link", "leg_right_5_link"),
        torso_body="torso_2_link",
    ),
    "booster_t1": FamilyBodies(
        knee_bodies=("Shank_Left", "Shank_Right"),
        ankle_bodies=("Ankle_Cross_Left", "Ankle_Cross_Right"),
        torso_body="Trunk",
    ),
    "h1v2": FamilyBodies(
        knee_bodies=("left_knee_link", "right_knee_link"),
        ankle_bodies=("left_ankle_pitch_link", "right_ankle_pitch_link"),
        torso_body="torso_link",
    ),
}


@struct.dataclass
class FamilyMorphCarry(LocoCarry):
    morphology: jax.Array
    morphology_generation: jax.Array
    #: Bounded-admission accounting for the bodies this environment drew.
    #: Present unconditionally so the carry pytree structure does not depend on
    #: a flag -- a structure that changes between the AOT compile and the call
    #: is how several previous multi-body arms died.
    admission: AdmissionStats
    #: Raised when a reset consumed its whole resample budget without finding an
    #: admissible body, or when a post-reset check condemned the one it used.
    #: The episode is made absorbing rather than quietly falling back to a
    #: nominal body.
    admission_failed: jax.Array


class FamilyMorphMixin:
    """Mixin over a ``Mjx*`` humanoid environment class.

    Keyword args (consumed here, rest forwarded):
        family_key: name in FAMILY_BODIES.
        resample_morphology_on_episode_reset: online DR when True.
        morphology_catalog: optional (N, 4) fixed bodies; each environment
            keeps one row (chosen from its reset key) for the whole run —
            the deterministic multi-morphology correctness case.
        admission: AdmissionConfig for bounded rejection sampling. The default
            enables the checks and the counters; ``AdmissionConfig(enabled=False)``
            reproduces the old use-every-draw sampler while still counting.
    """

    def __init__(
        self,
        *args,
        family_key: str,
        resample_morphology_on_episode_reset: bool = True,
        morphology_catalog: np.ndarray | None = None,
        admission: AdmissionConfig | None = None,
        **kwargs,
    ):
        self._family_key = str(family_key)
        self._admission = admission if admission is not None else AdmissionConfig()
        self._bodies = FAMILY_BODIES[self._family_key]
        self._resample_morphology = bool(resample_morphology_on_episode_reset)
        if morphology_catalog is not None:
            catalog = np.asarray(morphology_catalog, dtype=np.float32)
            if catalog.ndim != 2 or catalog.shape[1] != FAMILY_MORPHOLOGY_DIM:
                raise ValueError(
                    f"morphology_catalog must be (N, {FAMILY_MORPHOLOGY_DIM})."
                )
            if self._resample_morphology:
                raise ValueError(
                    "Catalog bodies are fixed per environment; disable "
                    "resample_morphology_on_episode_reset."
                )
            self._catalog_np = catalog
        else:
            self._catalog_np = None

        super().__init__(*args, **kwargs)

        model = self._model
        self._catalog = (
            None if self._catalog_np is None else jnp.asarray(self._catalog_np)
        )
        self._morphology_low = jnp.asarray(FAMILY_MORPHOLOGY_LOW)
        self._morphology_high = jnp.asarray(FAMILY_MORPHOLOGY_HIGH)

        def body_id(name):
            index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if index < 0:
                raise ValueError(f"{self._family_key}: no body named {name!r}")
            return index

        knee_ids = [body_id(n) for n in self._bodies.knee_bodies]
        ankle_ids = [body_id(n) for n in self._bodies.ankle_bodies]
        self._leg_position_ids = np.asarray(knee_ids + ankle_ids, dtype=np.int32)
        parents = np.asarray(model.body_parentid)
        # thigh/shin segments = parents of the moved attachments
        self._leg_inertial_ids = np.asarray(
            sorted({int(parents[i]) for i in self._leg_position_ids}),
            dtype=np.int32,
        )
        self._torso_id = body_id(self._bodies.torso_body)

        base = self.sys
        self._base_body_pos = base.body_pos
        self._base_body_ipos = base.body_ipos
        self._base_body_mass = base.body_mass
        self._base_body_inertia = base.body_inertia
        self._base_dof_damping = base.dof_damping
        self._base_actuator_gainprm = base.actuator_gainprm
        self._base_actuator_forcerange = base.actuator_forcerange

        # nominal leg length from the model itself: mean over both sides of
        # |z(knee offset)| + |z(ankle offset)|
        knee_z = np.abs(np.asarray(base.body_pos)[knee_ids, 2])
        ankle_z = np.abs(np.asarray(base.body_pos)[ankle_ids, 2])
        self.nominal_leg_length_m = float(np.mean(knee_z) + np.mean(ankle_z))

        self._cache_description_constants()
        self._cache_admission_constants()

    # -------------------------------------------------------------- admission
    def _cache_admission_constants(self) -> None:
        """Static pieces of the reference screen: the sub-trajectory lengths."""
        split_points = None
        handler = getattr(self, "th", None)
        if handler is not None:
            split_points = np.asarray(handler.traj.data.split_points)
        self._admission_split_points = (
            None if split_points is None else jnp.asarray(split_points)
        )
        self._admission_fractions = jnp.asarray(
            self._admission.reference_screen_fractions, dtype=jnp.float32
        )

    def _screen_steps(self, carry):
        """``(traj_no, steps)`` the reference is screened at for this reset.

        ``steps[0]`` is the phase the episode actually starts on -- that is the
        one ``reference_limit_violation`` judges. The rest are fractions of the
        same sub-trajectory, so a body that can hit the start pose but not the
        middle of the clip is still rejected (``reference_screen_failed``).
        """
        traj_no = carry.traj_state.traj_no
        start = carry.traj_state.subtraj_step_no
        if self._admission_split_points is None:
            return traj_no, jnp.asarray([start], dtype=jnp.int32)
        split = self._admission_split_points
        length = split[traj_no + 1] - split[traj_no]
        sampled = (self._admission_fractions * (length - 1)).astype(jnp.int32)
        return traj_no, jnp.concatenate(
            [jnp.asarray(start, dtype=jnp.int32).reshape(1), sampled]
        )

    def _admission_reject_flags(self, body_model, screen_steps):
        """Model + reference checks, without drawing a new body.

        Used by the fixed-catalog path, where the body must not change between
        episodes: it is checked and counted, and a rejection condemns the
        episode rather than silently swapping in another body.
        """
        from scaling.morphology_admission import (
            model_rejection_flags,
            reference_rejection_flags,
        )

        return jnp.logical_or(
            model_rejection_flags(body_model, self._admission),
            reference_rejection_flags(
                self, body_model, screen_steps, self._admission
            ),
        )

    # ------------------------------------------------------------------ model
    def _apply_morphology(self, model, morphology: jax.Array):
        leg_scale, torso_mass_scale, damping_scale, strength_scale = morphology

        one = jnp.ones_like(leg_scale)
        leg_pos_scale = jnp.stack([one, one, leg_scale])
        body_pos = self._base_body_pos.at[self._leg_position_ids].set(
            self._base_body_pos[self._leg_position_ids] * leg_pos_scale[None, :]
        )

        # volume/inertia scaling of the lengthened segments (matches the
        # online_h1 convention: z-stretch => volume ~ s, inertia ~ s * s^2)
        body_scale = jnp.ones_like(self._base_body_ipos)
        body_scale = body_scale.at[self._leg_inertial_ids, 2].set(leg_scale)
        volume_scale = jnp.prod(body_scale, axis=1)
        inertia_scale = volume_scale * jnp.square(jnp.mean(body_scale, axis=1))
        body_ipos = self._base_body_ipos * body_scale
        body_mass = self._base_body_mass * volume_scale
        body_inertia = self._base_body_inertia * inertia_scale[:, None]
        body_mass = body_mass.at[self._torso_id].multiply(torso_mass_scale)
        body_inertia = body_inertia.at[self._torso_id].multiply(torso_mass_scale)

        return model.replace(
            body_pos=body_pos,
            body_ipos=body_ipos,
            body_mass=body_mass,
            body_inertia=body_inertia,
            dof_damping=self._base_dof_damping * damping_scale,
            actuator_gainprm=self._base_actuator_gainprm.at[:, 0].multiply(
                strength_scale
            ),
            actuator_forcerange=self._base_actuator_forcerange * strength_scale,
        )

    def _normalized_morphology(self, morphology: jax.Array) -> jax.Array:
        midpoint = 0.5 * (self._morphology_low + self._morphology_high)
        half_range = 0.5 * (self._morphology_high - self._morphology_low)
        return (morphology - midpoint) / half_range

    def root_height_offset(self, morphology: jax.Array) -> jax.Array:
        return self.nominal_leg_length_m * (morphology[0] - 1.0)

    #: consumed by ParallelMorphVecEnv to size the shared joint block
    dynamic_joint_description_dim = FAMILY_JOINT_DESCRIPTION_DIM

    # ------------------------------------------------------------ description
    def _cache_description_constants(self) -> None:
        model = self._model
        action_dim = int(self.info.action_space.shape[0])
        joint_ids = actuated_joint_ids(model, action_dim)
        self._desc_joint_ids = jnp.asarray(joint_ids, dtype=jnp.int32)
        self._desc_body_ids = jnp.asarray(
            np.asarray(model.jnt_bodyid[joint_ids]), dtype=jnp.int32
        )
        self._desc_dof_ids = jnp.asarray(
            np.asarray(model.jnt_dofadr[joint_ids]), dtype=jnp.int32
        )
        # static columns never touched by the four coordinates, computed once
        # with the same scaling as the cross-topology static block
        static = generic_joint_descriptions(model, action_dim)
        self._desc_static = jnp.asarray(static)
        # column layout of generic_joint_descriptions:
        # [0:3 body_pos, 3:6 jnt_axis, 6 child, 7 qpos0, 8 force_limit,
        #  9 damping, 10 armature, 11 stiffness, 12 frictionloss,
        #  13:15 jnt_range, 15:17 ctrl_range, 17 total_mass, 18 body_mass,
        #  19:22 body_inertia]

    def dynamic_joint_descriptions(self, morphology: jax.Array) -> jax.Array:
        """(action_dim, 26) descriptions of the SAMPLED model, in-graph."""
        model = self._apply_morphology(self.sys, morphology)
        body_ids = self._desc_body_ids
        dof_ids = self._desc_dof_ids

        descriptions = self._desc_static
        descriptions = descriptions.at[:, 0:3].set(model.body_pos[body_ids] / 1.0)
        # strength scale multiplies the force range symmetrically
        force_limit = jnp.max(
            jnp.abs(model.actuator_forcerange[: descriptions.shape[0]]), axis=-1
        )
        descriptions = descriptions.at[:, 8].set(
            force_limit / _SCALES["force_limit"]
        )
        descriptions = descriptions.at[:, 9].set(
            model.dof_damping[dof_ids] / _SCALES["damping"]
        )
        descriptions = descriptions.at[:, 17].set(
            jnp.sum(model.body_mass) / _SCALES["total_mass"]
        )
        descriptions = descriptions.at[:, 18].set(
            model.body_mass[body_ids] / _SCALES["body_mass"]
        )
        descriptions = descriptions.at[:, 19:22].set(
            model.body_inertia[body_ids] / _SCALES["body_inertia"]
        )
        morphology_block = jnp.broadcast_to(
            self._normalized_morphology(morphology)[None, :],
            (descriptions.shape[0], FAMILY_MORPHOLOGY_DIM),
        )
        return jnp.concatenate([descriptions, morphology_block], axis=-1)

    # ------------------------------------------------------------------ carry
    def _sample_morphology(self, key: jax.Array) -> jax.Array:
        if self._catalog is not None:
            index = jax.random.randint(key, (), 0, self._catalog.shape[0])
            return self._catalog[index]
        return jax.random.uniform(
            key,
            shape=(FAMILY_MORPHOLOGY_DIM,),
            minval=self._morphology_low,
            maxval=self._morphology_high,
        )

    def _init_additional_carry(
        self, key: jax.Array, model, data, backend: ModuleType
    ) -> FamilyMorphCarry:
        carry = super()._init_additional_carry(key, model, data, backend)
        key, body_key = jax.random.split(carry.key)
        # The draw here is provisional: with online DR `_mjx_reset_carry` runs
        # the admitted draw immediately afterwards and discards this one, so it
        # is deliberately NOT counted. In fixed-catalog mode the same body is
        # then checked (not resampled) at every reset.
        return FamilyMorphCarry(
            morphology=self._sample_morphology(body_key),
            morphology_generation=jnp.asarray(
                0 if self._resample_morphology else 1, dtype=jnp.int32
            ),
            admission=zero_stats(FAMILY_MORPHOLOGY_DIM),
            admission_failed=jnp.asarray(False),
            **vars(carry.replace(key=key)),
        )

    def _mjx_reset_carry(self, model, data, carry):
        data, carry = super()._mjx_reset_carry(model, data, carry)

        # The trajectory cursor is now the post-reset one, which is what the
        # reference screen has to judge -- screening the previous episode's
        # phase would admit bodies that cannot start the motion they are about
        # to be reset into.
        screen_steps = self._screen_steps(carry)

        if self._resample_morphology:
            key, morphology_key = jax.random.split(carry.key)
            morphology, flags, draws, exhausted = admit_morphology(
                self, morphology_key, self._sample_morphology,
                screen_steps, self._admission,
            )
            carry = carry.replace(
                key=key,
                morphology=morphology,
                morphology_generation=carry.morphology_generation + 1,
            )
        else:
            # Catalog bodies are fixed per environment for the whole run, so a
            # rejection must not silently swap the body: it is counted and the
            # episode is condemned instead.
            morphology = carry.morphology
            if self._admission.enabled:
                flags = self._admission_reject_flags(
                    self._apply_morphology(self.sys, morphology), screen_steps
                )
            else:
                flags = jnp.zeros((NUM_REJECTION_REASONS,), dtype=bool)
            draws = jnp.asarray(1, dtype=jnp.int32)
            exhausted = jnp.any(flags)

        stats = update_stats(carry.admission, morphology, flags, draws, exhausted)

        qpos = data.qpos.at[2].add(self.root_height_offset(morphology))
        data = data.replace(qpos=qpos)
        body_model = self._apply_morphology(model, morphology)
        data = mjx.forward(body_model, data)

        # Penetration and initial-absorbing need the forwarded state, so they
        # cannot drive a resample; they condemn the episode instead.
        if self._admission.enabled:
            state_flags = state_rejection_flags(
                self, body_model, data, carry, self._admission
            )
            stats, condemned = record_state_rejections(
                stats, state_flags, jnp.logical_not(jnp.any(flags))
            )
        else:
            condemned = jnp.asarray(False)

        carry = carry.replace(
            admission=stats,
            admission_failed=jnp.logical_or(exhausted, condemned),
        )
        return data, carry

    def _mjx_is_absorbing(self, obs, info, data, carry):
        """Fail closed: an inadmissible body ends its episode immediately.

        Done here rather than in a terminal-state handler so the guarantee holds
        for whichever handler the run selects. The alternative -- substituting a
        nominal body -- would turn a sampler bug into a quietly narrower
        training distribution that no metric reports.
        """
        absorbing, carry = super()._mjx_is_absorbing(obs, info, data, carry)
        return jnp.logical_or(absorbing, carry.admission_failed), carry

    def _mjx_reset_in_step(self, state):
        state = super()._mjx_reset_in_step(state)
        if self._resample_morphology:
            # Route the reset observation on body change so PPO never pairs
            # the OLD embodiment descriptor with the NEW body (online_h1
            # convention).
            carry = state.additional_carry.replace(
                final_observation=state.observation
            )
            state = state.replace(additional_carry=carry)
        return state

    def _mjx_simulation_pre_step(self, model, data, carry):
        model = self._apply_morphology(model, carry.morphology)
        return super()._mjx_simulation_pre_step(model, data, carry)


def make_family_morph_env_class(mjx_cls: type, family_key: str) -> type:
    """Concrete online-morph environment class for one family."""
    name = f"FamilyMorph{mjx_cls.__name__}"
    return type(name, (FamilyMorphMixin, mjx_cls), {})
