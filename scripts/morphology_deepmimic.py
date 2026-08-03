"""Shared backbone for the DeepMimic-on-morphology experiments (H1 & G1).

Used by:
  - retarget_dance_to_variant.py   (build a per-variant reference motion)
  - train_deepmimic_morphology.py  (segmented PPO training with checkpoints)
  - run_morphology_matrix.py       (drive the full robot x variant x clip matrix)
  - render_morphology_deepmimic.py (per-variant timeline + final grid videos)

The whole DeepMimic matrix is built on the *same* Mimic reward / goal / mimic-site
set so every cell (robot x morphology x clip) is directly comparable. Keeping
that config in one place is the point of this module.

Only loco-mujoco + numpy + (cpu) jax.numpy are imported here, so the module stays
importable on the Windows venv for the rendering / bookkeeping side of the work.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import jax.numpy as jnp
import mujoco
import numpy as np

from loco_mujoco.environments import LocoEnv, UnitreeG1, UnitreeH1
from loco_mujoco.environments.humanoids import MjxUnitreeG1, MjxUnitreeH1
from loco_mujoco.task_factories import CustomDatasetConf, ImitationFactory
from loco_mujoco.trajectory import Trajectory, TrajectoryData

WORKSPACE = Path(__file__).resolve().parents[1]
# All matrix artifacts (per-variant references, checkpoints, manifests) live here.
DEEPMIMIC_ROOT = WORKSPACE / "external_data" / "deepmimic_morphology"
# The per-robot morphology preset modules live next to this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))


# --------------------------------------------------------------------------- #
# Shared DeepMimic task configuration (identical across every cell)
# --------------------------------------------------------------------------- #
# These five sites exist with the same names on both the H1 and the G1 XML
# (verified against the stock models), so the reward is portable across robots
# and across every generated morphology variant.
MIMIC_SITES = [
    "upper_body_mimic",
    "left_hand_mimic",
    "left_foot_mimic",
    "right_hand_mimic",
    "right_foot_mimic",
]

MIMIC_REWARD_PARAMS = dict(
    qpos_w_sum=0.4,
    qvel_w_sum=0.2,
    rpos_w_sum=0.5,
    rquat_w_sum=0.3,
    rvel_w_sum=0.1,
    sites_for_mimic=MIMIC_SITES,
)

# visualize_goal draws reference arrows via mjv_initGeom, which crashes with the
# mujoco 3.9 bindings, so it stays off everywhere.
GOAL_PARAMS = dict(visualize_goal=False)


# --------------------------------------------------------------------------- #
# Control configuration (torque vs PD position control)
# --------------------------------------------------------------------------- #
# DefaultControl applies the action as direct torque -- the loco-mujoco DeepMimic
# example uses it for H1. It is very hard to learn on the smaller, 23-DOF G1, so
# we offer PD position control (policy outputs target joint angles, a stiff servo
# supplies stabilising torque) -- the standard DeepMimic choice. Gains follow the
# actuator order in each robot's XML.
# G1 actuator order: L hip pitch/roll/yaw, L knee, L ankle pitch/roll, R (same),
# waist_yaw, L shoulder pitch/roll/yaw, L elbow, L wrist_roll, R (same).
_G1_LEG = [(100.0, 2.0), (100.0, 2.0), (100.0, 2.0), (150.0, 4.0), (40.0, 2.0), (40.0, 2.0)]
_G1_ARM = [(60.0, 2.0), (60.0, 2.0), (60.0, 2.0), (60.0, 2.0), (40.0, 1.0)]
_G1_PAIRS = _G1_LEG + _G1_LEG + [(150.0, 4.0)] + _G1_ARM + _G1_ARM  # 23 actuators
PD_GAINS = {
    "g1": dict(p_gain=[p for p, _ in _G1_PAIRS], d_gain=[d for _, d in _G1_PAIRS]),
}

# Actuator-index groups in the G1 _G1_PAIRS order, so PD gains can be stiffened
# per joint group rather than uniformly. A high-CoM body (e.g. tall&light) needs
# ankle/knee authority for balance, which uniform scaling can't give without also
# overshooting the hips/arms.
G1_JOINT_GROUPS = {
    "hip": [0, 1, 2, 6, 7, 8],
    "knee": [3, 9],
    "ankle": [4, 5, 10, 11],
    "waist": [12],
    "arm": list(range(13, 23)),
}


def apply_group_gain_scales(control_params: dict, group_scales: dict) -> None:
    """Multiply p_gain/d_gain in place for the named G1 joint groups (scale != 1)."""
    for group, scale in group_scales.items():
        if scale == 1.0:
            continue
        for idx in G1_JOINT_GROUPS[group]:
            control_params["p_gain"][idx] *= scale
            control_params["d_gain"][idx] *= scale


def control_config(robot_key: str, mode: str) -> dict:
    """Return env_params for the requested control mode ({} = use the env default torque)."""
    if mode == "pd":
        if robot_key not in PD_GAINS:
            raise KeyError(f"No PD gains defined for robot '{robot_key}'.")
        return dict(control_type="PDControl", control_params=dict(PD_GAINS[robot_key]))
    return {}


@dataclass(frozen=True)
class RobotSpec:
    key: str                      # short id used on the CLI / in paths ("h1", "g1")
    cpu_env_name: str             # registered CPU env ("UnitreeH1")
    mjx_env_name: str             # registered Mjx env ("MjxUnitreeH1")
    retarget_conf_name: str       # SMPL robot conf yaml name
    base_cpu_cls: type            # CPU class to subclass for a variant
    base_mjx_cls: type            # Mjx class to subclass for a variant
    variant_dir_prefix: str       # generated_variants/<prefix><preset>/

    def presets(self) -> dict:
        """Lazy import of the per-robot preset table (avoids cv2 import at module load)."""
        if self.key == "h1":
            from h1_morphology_variants import PRESETS
        else:
            from g1_morphology_variants import PRESETS
        return PRESETS

    def create_variant_xml(self, preset) -> Path:
        if self.key == "h1":
            from h1_morphology_variants import create_h1_variant_xml
            return create_h1_variant_xml(preset)
        from g1_morphology_variants import create_g1_variant_xml
        return create_g1_variant_xml(preset)


ROBOTS: dict[str, RobotSpec] = {
    "h1": RobotSpec(
        key="h1",
        cpu_env_name="UnitreeH1",
        mjx_env_name="MjxUnitreeH1",
        retarget_conf_name="UnitreeH1",
        base_cpu_cls=UnitreeH1,
        base_mjx_cls=MjxUnitreeH1,
        variant_dir_prefix="h1_morphology_",
    ),
    "g1": RobotSpec(
        key="g1",
        cpu_env_name="UnitreeG1",
        mjx_env_name="MjxUnitreeG1",
        retarget_conf_name="UnitreeG1",
        base_cpu_cls=UnitreeG1,
        base_mjx_cls=MjxUnitreeG1,
        variant_dir_prefix="g1_morphology_",
    ),
}


def get_robot(key: str) -> RobotSpec:
    if key not in ROBOTS:
        raise KeyError(f"Unknown robot '{key}'. Choices: {sorted(ROBOTS)}")
    return ROBOTS[key]


# --------------------------------------------------------------------------- #
# Env registration for generated morphology variants
# --------------------------------------------------------------------------- #
def register_variant_env(env_name: str, xml_path: Path, base_cls: type) -> type:
    """Register a LocoEnv subclass of ``base_cls`` that loads ``xml_path``.

    The retargeting path needs a CPU subclass (base = UnitreeH1/UnitreeG1); the
    training path needs an Mjx subclass (base = MjxUnitree*). Same pattern, just a
    different base, so this is parametrised on ``base_cls``.
    """
    if env_name in LocoEnv.registered_envs:
        return LocoEnv.registered_envs[env_name]

    def get_default_xml_file_path(cls) -> str:
        return str(xml_path)

    variant_cls = type(
        env_name,
        (base_cls,),
        {"get_default_xml_file_path": classmethod(get_default_xml_file_path)},
    )
    LocoEnv.registered_envs[env_name] = variant_cls
    return variant_cls


def variant_env_names(robot: RobotSpec, preset_name: str, cache_tag: str) -> tuple[str, str]:
    """Distinct registered names for the CPU (retarget) and Mjx (train) variant envs."""
    cpu = f"{robot.key.upper()}Var_{preset_name}_{cache_tag}_cpu"
    mjx = f"Mjx{robot.key.upper()}Var_{preset_name}_{cache_tag}"
    return cpu, mjx


def prepare_variant(robot: RobotSpec, preset_name: str, cache_tag: str) -> dict:
    """Create the variant XML (if needed) and register both CPU and Mjx envs.

    Returns a dict with the preset, xml path, and the two registered env names.
    """
    preset = robot.presets()[preset_name]
    xml_path = robot.create_variant_xml(preset)
    cpu_name, mjx_name = variant_env_names(robot, preset_name, cache_tag)
    register_variant_env(cpu_name, xml_path, robot.base_cpu_cls)
    register_variant_env(mjx_name, xml_path, robot.base_mjx_cls)
    return {
        "preset": preset,
        "xml_path": xml_path,
        "cpu_env_name": cpu_name,
        "mjx_env_name": mjx_name,
    }


# --------------------------------------------------------------------------- #
# Robot-agnostic floor handling
# --------------------------------------------------------------------------- #
# The H1-specific helpers in retarget_randomized_h1.py look up geoms named
# "left_foot"/"right_foot"; the G1 has no such geoms (its feet are
# left_foot_1_col .. _4_col). These helpers instead use the lowest world-z of any
# non-floor geom, which is correct for every robot and every morphology variant.
def _floor_geom_id(model: mujoco.MjModel) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")


def _lowest_geom_z(model: mujoco.MjModel, data: mujoco.MjData, floor_id: int) -> float:
    mujoco.mj_forward(model, data)
    return min(
        float(data.geom_xpos[g][2]) for g in range(model.ngeom) if g != floor_id
    )


def min_floor_distance(xml_path: Path, traj: Trajectory) -> float:
    """Smallest lowest-geom height (world-z) reached over the whole trajectory."""
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    floor_id = _floor_geom_id(model)
    qpos = np.asarray(traj.data.qpos)
    qvel = np.asarray(traj.data.qvel)
    worst = np.inf
    for frame in range(len(qpos)):
        data.qpos[:] = qpos[frame]
        data.qvel[:] = qvel[frame]
        worst = min(worst, _lowest_geom_z(model, data, floor_id))
    return float(worst)


def clamp_trajectory_to_floor(
    xml_path: Path, traj: Trajectory, clearance: float = 0.002
) -> tuple[Trajectory, float]:
    """Lift each frame so the lowest geom sits ~clearance above the floor.

    Generic across morphologies (no named foot geoms). Root vertical velocity is
    re-derived from the corrected root height so the reference stays consistent.
    """
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    floor_id = _floor_geom_id(model)
    qpos = np.asarray(traj.data.qpos).copy()
    qvel = np.asarray(traj.data.qvel).copy()
    lifts = np.zeros(len(qpos))
    for frame in range(len(qpos)):
        data.qpos[:] = qpos[frame]
        data.qvel[:] = qvel[frame]
        lowest = _lowest_geom_z(model, data, floor_id)
        lifts[frame] = clearance - lowest
    qpos[:, 2] += lifts
    qvel[:, 2] = np.gradient(qpos[:, 2], 1.0 / float(traj.info.frequency))
    corrected = traj.__class__(
        traj.info, traj.data.replace(qpos=jnp.asarray(qpos), qvel=jnp.asarray(qvel))
    )
    return corrected, float(np.abs(lifts).max())


def ground_trajectory_constant(
    xml_path: Path, traj: Trajectory, clearance: float = 0.002
) -> tuple[Trajectory, float]:
    """Lift the WHOLE clip by a single constant so its lowest point clears the floor.

    Unlike the per-frame clamp, this preserves the reference's motion and
    velocities exactly (only a constant is added to root z), so it never injects
    vertical jitter -- the clean way to ground a (possibly taller/shorter) variant
    body onto an otherwise-untouched reference. Returns the corrected trajectory
    and the constant lift applied.
    """
    worst = min_floor_distance(xml_path, traj)  # lowest geom-z over the whole clip
    offset = clearance - worst
    qpos = np.asarray(traj.data.qpos).copy()
    qpos[:, 2] += offset
    corrected = traj.__class__(traj.info, traj.data.replace(qpos=jnp.asarray(qpos)))
    return corrected, float(offset)


# --------------------------------------------------------------------------- #
# Reference-window selection (shared by retarget + train)
# --------------------------------------------------------------------------- #
def pick_highlight_window_frames(traj: Trajectory, duration_seconds: float) -> tuple[int, int]:
    """(start_frame, n_frames) of the most dynamic window (max joint-velocity energy)."""
    frequency = float(traj.info.frequency)
    qvel = np.asarray(traj.data.qvel)
    energy = np.linalg.norm(qvel[:, 6:], axis=1)
    window = int(round(duration_seconds * frequency))
    if window >= len(energy):
        return 0, len(energy)
    cumulative = np.concatenate(([0.0], np.cumsum(energy)))
    sums = cumulative[window:] - cumulative[:-window]
    return int(np.argmax(sums)), window


def resolve_window(traj: Trajectory, duration: float, start_frame: int | None) -> tuple[int, int]:
    """Resolve (start_frame, n_frames): auto-pick the highlight unless start_frame is given."""
    frequency = float(traj.info.frequency)
    if start_frame is None:
        return pick_highlight_window_frames(traj, duration)
    n_frames = min(int(round(duration * frequency)), int(traj.data.n_samples) - start_frame)
    return start_frame, n_frames


def crop_trajectory(traj: Trajectory, start_frame: int, n_frames: int) -> Trajectory:
    """Slice [start_frame, start_frame + n_frames) out of a single-trajectory handler."""
    data = TrajectoryData.dynamic_slice_in_dim(traj.data, 0, start_frame, n_frames, backend=jnp)
    return Trajectory(info=traj.info, data=data)


# --------------------------------------------------------------------------- #
# Cache layout (single source of truth for every script)
# --------------------------------------------------------------------------- #
def window_tag(start_frame: int, n_frames: int) -> str:
    return f"start{start_frame}_{n_frames}f"


def fitted_source_path(robot_key: str, clip: str, start_frame: int, n_frames: int) -> Path:
    """Cached SMPL fit of the *source* motion (shared across all variants of a clip)."""
    return DEEPMIMIC_ROOT / robot_key / "refs" / f"{clip}_{window_tag(start_frame, n_frames)}_smpl.npz"


def reference_path(robot_key: str, clip: str, preset_name: str, start_frame: int, n_frames: int) -> Path:
    """Retargeted + floor-clamped + extended reference motion for one variant cell."""
    return (
        DEEPMIMIC_ROOT / robot_key / "refs"
        / f"{preset_name}_{clip}_{window_tag(start_frame, n_frames)}.npz"
    )


def cell_dir(robot_key: str, clip: str, preset_name: str) -> Path:
    """Output directory for one training cell (checkpoints + manifest live here)."""
    return DEEPMIMIC_ROOT / robot_key / clip / preset_name


# --------------------------------------------------------------------------- #
# Imitation env builder (shared mimic config)
# --------------------------------------------------------------------------- #
def make_mimic_env(env_name: str, traj: Trajectory, **overrides):
    """ImitationFactory.make wired with the shared Mimic reward/goal + a custom traj.

    ``overrides`` are passed straight through (e.g. use_mjwarp / nconmax / headless
    for training, or record / headless for rendering rollouts).
    """
    params = dict(
        custom_dataset_conf=CustomDatasetConf(traj),
        horizon=1000,
        goal_type="GoalTrajMimic",
        goal_params=GOAL_PARAMS,
        reward_type="MimicReward",
        reward_params=MIMIC_REWARD_PARAMS,
    )
    params.update(overrides)
    return ImitationFactory.make(env_name, **params)
