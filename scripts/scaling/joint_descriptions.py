"""Family-agnostic per-joint descriptions for cross-topology URMA.

:mod:`scaling.online_h1` builds a 26-dim per-joint description that is specific
to the online-morphology H1: it appends the four H1 morphology coordinates to
every joint.  That block cannot transfer to another robot family, because the
coordinates only exist for H1.

This module builds the purely *structural* 22-dim subset from any LocoMuJoCo MJX
environment's MuJoCo model, in actuator order.  Family-level descriptors (a
robot one-hot, or online morphology coordinates) belong in the global part of
the observation instead, so the per-joint block stays comparable across
families.

The scaling constants are the ones already used by ``online_h1._finite_scaled``,
so a joint description produced here has the same magnitudes URMA already
trains on.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

#: body_pos(3), jnt_axis(3), child_count(1), nominal_qpos(1), force_limit(1),
#: damping(1), armature(1), stiffness(1), frictionloss(1), jnt_range(2),
#: ctrl_range(2), total_mass(1), body_mass(1), body_inertia(3).
GENERIC_JOINT_DESCRIPTION_DIM = 22

#: joint position, joint velocity, previous action.
JOINT_STATE_DIM = 3

#: description + state + one binary valid-joint bit.
JOINT_FEATURE_DIM = GENERIC_JOINT_DESCRIPTION_DIM + JOINT_STATE_DIM + 1

#: Same constants as ``OnlineMorphMjxUnitreeH1._cache_urma_constants``.
_SCALES = {
    "child_count": 3.0,
    "nominal_qpos": np.pi,
    "force_limit": 1000.0,
    "damping": 10.0,
    "armature": 0.2,
    "stiffness": 30.0,
    "frictionloss": 1.2,
    "joint_range": np.pi,
    "ctrl_range": np.pi,
    "total_mass": 100.0,
    "body_mass": 20.0,
    "body_inertia": 5.0,
}


@dataclass(frozen=True)
class JointBlockSpec:
    """Everything needed to emit one robot's joint block into a padded vector."""

    name: str
    num_joints: int
    descriptions: np.ndarray
    position_obs_indices: tuple[int, ...]
    velocity_obs_indices: tuple[int, ...]
    joint_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.descriptions.shape != (
            self.num_joints,
            GENERIC_JOINT_DESCRIPTION_DIM,
        ):
            raise ValueError(
                f"{self.name}: descriptions must be "
                f"({self.num_joints}, {GENERIC_JOINT_DESCRIPTION_DIM}); "
                f"got {self.descriptions.shape}."
            )
        if not np.all(np.isfinite(self.descriptions)):
            raise ValueError(f"{self.name}: joint descriptions must be finite.")
        for label, indices in (
            ("position", self.position_obs_indices),
            ("velocity", self.velocity_obs_indices),
        ):
            if len(indices) != self.num_joints:
                raise ValueError(
                    f"{self.name}: expected {self.num_joints} {label} observation "
                    f"indices, got {len(indices)}."
                )


def _finite_scaled(values, scale: float) -> np.ndarray:
    """Scale to O(1) and replace non-finite entries with 0, as URMA already does."""
    values = np.asarray(values, dtype=np.float32) / float(scale)
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def actuated_joint_ids(model, action_dim: int) -> np.ndarray:
    """Joint ids driven by the first ``action_dim`` actuators, in actuator order."""
    trntype = np.asarray(model.actuator_trntype[:action_dim])
    if not np.all(trntype == int(mujoco.mjtTrn.mjTRN_JOINT)):
        raise ValueError(
            "Cross-topology joint descriptions require joint-transmission "
            f"actuators; got transmission types {sorted(set(trntype.tolist()))}."
        )
    joint_ids = np.asarray(model.actuator_trnid[:action_dim, 0], dtype=np.int32)
    if len(np.unique(joint_ids)) != action_dim:
        raise ValueError("Each actuator must drive a distinct joint.")
    return joint_ids


def generic_joint_descriptions(model, action_dim: int) -> np.ndarray:
    """Build the ``(action_dim, 22)`` structural description block for one model."""
    joint_ids = actuated_joint_ids(model, action_dim)
    body_ids = np.asarray(model.jnt_bodyid[joint_ids], dtype=np.int32)
    dof_ids = np.asarray(model.jnt_dofadr[joint_ids], dtype=np.int32)
    qpos_ids = np.asarray(model.jnt_qposadr[joint_ids], dtype=np.int32)

    # Structural fan-out in the full body tree.  online_h1 counted children only
    # among the actuated bodies; over a whole family that undercounts, and the
    # true tree degree is what actually transfers between robots.
    parents = np.asarray(model.body_parentid)
    child_counts = np.asarray(
        [np.count_nonzero(parents == int(body_id)) for body_id in body_ids],
        dtype=np.float32,
    )[:, None]

    force_ranges = np.asarray(model.actuator_forcerange[:action_dim])
    force_limits = np.max(np.abs(force_ranges), axis=-1, keepdims=True)
    total_mass = np.full(
        (action_dim, 1), float(np.sum(np.asarray(model.body_mass))), dtype=np.float32
    )

    blocks = [
        _finite_scaled(np.asarray(model.body_pos)[body_ids], 1.0),
        _finite_scaled(np.asarray(model.jnt_axis)[joint_ids], 1.0),
        _finite_scaled(child_counts, _SCALES["child_count"]),
        _finite_scaled(
            np.asarray(model.qpos0)[qpos_ids, None], _SCALES["nominal_qpos"]
        ),
        _finite_scaled(force_limits, _SCALES["force_limit"]),
        _finite_scaled(
            np.asarray(model.dof_damping)[dof_ids, None], _SCALES["damping"]
        ),
        _finite_scaled(
            np.asarray(model.dof_armature)[dof_ids, None], _SCALES["armature"]
        ),
        _finite_scaled(
            np.asarray(model.jnt_stiffness)[joint_ids, None], _SCALES["stiffness"]
        ),
        _finite_scaled(
            np.asarray(model.dof_frictionloss)[dof_ids, None], _SCALES["frictionloss"]
        ),
        _finite_scaled(np.asarray(model.jnt_range)[joint_ids], _SCALES["joint_range"]),
        _finite_scaled(
            np.asarray(model.actuator_ctrlrange[:action_dim]), _SCALES["ctrl_range"]
        ),
        _finite_scaled(total_mass, _SCALES["total_mass"]),
        _finite_scaled(
            np.asarray(model.body_mass)[body_ids, None], _SCALES["body_mass"]
        ),
        _finite_scaled(
            np.asarray(model.body_inertia)[body_ids], _SCALES["body_inertia"]
        ),
    ]
    descriptions = np.concatenate(blocks, axis=-1).astype(np.float32)
    if descriptions.shape != (action_dim, GENERIC_JOINT_DESCRIPTION_DIM):
        raise AssertionError(
            f"Unexpected generic description shape {descriptions.shape}."
        )
    return descriptions


def joint_observation_indices(env) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Observation indices of each actuated joint's position and velocity.

    LocoMuJoCo emits one scalar ``JointPos``/``JointVel`` observation per model
    joint, but a robot can carry joints that no actuator drives (ToddlerBot has
    38 observed joints for 30 actuators).  Matching by the observation's
    ``xml_name`` therefore beats matching by position.
    """
    model = env._model
    action_dim = int(env.info.action_space.shape[0])
    joint_ids = actuated_joint_ids(model, action_dim)
    joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(joint_id))
        for joint_id in joint_ids
    ]

    resolved = {}
    for kind in ("JointPos", "JointVel"):
        flat = np.asarray(getattr(env._obs_indices, kind), dtype=np.int64)
        ordered_names = [
            str(entry.xml_name)
            for entry in env.obs_container.values()
            if type(entry).__name__ == kind
        ]
        if len(ordered_names) != len(flat):
            raise ValueError(
                f"{kind}: {len(ordered_names)} container entries but {len(flat)} "
                "observation indices; only scalar joint observations are supported."
            )
        lookup = dict(zip(ordered_names, (int(i) for i in flat), strict=True))
        missing = [name for name in joint_names if name not in lookup]
        if missing:
            raise ValueError(f"{kind}: no observation for actuated joints {missing}.")
        resolved[kind] = tuple(lookup[name] for name in joint_names)
    return resolved["JointPos"], resolved["JointVel"]


def build_joint_block_spec(env, name: str) -> JointBlockSpec:
    """Assemble one robot's structural joint block from a built MJX environment."""
    model = env._model
    action_dim = int(env.info.action_space.shape[0])
    position_indices, velocity_indices = joint_observation_indices(env)
    joint_ids = actuated_joint_ids(model, action_dim)
    return JointBlockSpec(
        name=str(name),
        num_joints=action_dim,
        descriptions=generic_joint_descriptions(model, action_dim),
        position_obs_indices=position_indices,
        velocity_obs_indices=velocity_indices,
        joint_names=tuple(
            str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(joint_id)))
            for joint_id in joint_ids
        ),
    )
