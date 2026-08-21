"""urma2's tracking-reward core, transplanted into the loco-mujoco trainer (T1).

The overnight 18-08 partition proved the TRAINER is the blocker (X1 vs X2) and
bounded every config-expressible convention at <=2x of the ~20x sample gap.
The reward implementation is the largest untested cluster. This class rebuilds
loco_mjx/urma2's `TrackingReward` term structure on our stack so the two
trainers can be compared under (as near as possible) the same reward:

  urma2 term                              transplanted as
  ------------------------------------------------------------------
  coeff*dt * exp(-joint_mse / 0.25)       same; joints = limited hinges only,
                                          ROOT EXCLUDED (heading-free: urma2's
                                          DANCEPROOF fix; our MimicReward
                                          instead scores the absolute root quat
                                          -- a heading anchor)
  0.5 ratio, qvel mse NORMALIZED by the   same; normalizer precomputed from the
  reference's own mean-square velocity    trajectory at init (urma2 does the
                                          same in its command function)
  1.25 ratio, all-body root-relative      our site-based equivalent: the
  positions, FK on the SAMPLED body,      body-correct FK bundle MorphMimic
  NORMALIZED by body size                 already provides, but normalized by a
                                          per-family length scale
  0.75 ratio, rotation-matrix diff        our relative site angles, temp 0.1
  root-height term (1.0*dt, temp 0.01)    same, height error / length scale
  alive floor 0.1*dt                      same
  ~25 gait regularizers + curriculum      DELIBERATELY OMITTED -- this is the
                                          "tracking core"; if the core alone
                                          closes the gap the regularizers are
                                          exonerated, if not they are next

All coefficients are urma2's defaults scaled by env.dt exactly as urma2 scales
them, so the per-step magnitude matches at any control rate.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
from types import ModuleType
from typing import Any, Dict

import numpy as np

_SCRIPTS = str(_Path(__file__).resolve().parents[1])
if _SCRIPTS not in _sys.path:
    _sys.path.insert(0, _SCRIPTS)

_H1MD = str(_Path(__file__).resolve().parents[1] / "h1md")
if _H1MD not in _sys.path:
    _sys.path.insert(0, _H1MD)

from loco_mujoco.core.utils.math import calculate_relative_site_quatities  # noqa: E402

from morph_mimic_reward import MorphMimicReward  # noqa: E402
from scaling.body_correct_reference import clampable_qpos_index  # noqa: E402


class Urma2CoreReward(MorphMimicReward):
    """MorphMimicReward's FK machinery under urma2's reward shape."""

    JOINT_COEFF = 30.0        # mmtrain TRACK_COEFF
    JOINT_TEMP = 0.25
    QVEL_RATIO = 0.5
    QVEL_TEMP = 0.5
    RPOS_RATIO = 1.25
    RPOS_TEMP = 0.01
    RQUAT_RATIO = 0.75
    RQUAT_TEMP = 0.1
    HEIGHT_COEFF = 1.0
    HEIGHT_TEMP = 0.01
    ALIVE_COEFF = 0.1         # alive_clipped + alive_unclipped at curriculum 1
    ACTION_RATE_COEFF = 0.03  # urma2 action_rate 3.0 * dt at 100 Hz ~ 0.03

    def __init__(self, env, root_frame: str = "absolute", **kwargs):
        super().__init__(env, root_frame=root_frame, **kwargs)
        self._u2_dt = float(env.dt)

        # At reward construction the env holds the CPU MjModel as _model
        # (env.sys only exists on the MJX side, later).
        sys_model = getattr(env, "sys", None) or env._model
        qpos_idx, joint_ids = clampable_qpos_index(sys_model)
        self._u2_joint_qpos_idx = np.asarray(qpos_idx, dtype=np.int32)
        self._u2_joint_qvel_idx = np.asarray(
            [int(sys_model.jnt_dofadr[j]) for j in joint_ids], dtype=np.int32)

        # Normalizers are computed lazily from the trajectory: env.th is not
        # attached yet when the reward is constructed. The lazy branch runs at
        # trace time on CONCRETE trajectory arrays, so it never sees a tracer.
        self._u2_qvel_scale = None
        self._u2_length_scale = None

    def _ensure_normalizers(self, env):
        if self._u2_qvel_scale is not None:
            return
        traj = env.th.traj.data
        qvel_np = np.asarray(traj.qvel)[:, self._u2_joint_qvel_idx]
        self._u2_qvel_scale = float(max(np.mean(np.square(qvel_np)), 1e-6))
        # Per-family length scale ~ urma2's robot_dimensions_mean: mean distance
        # of the mimic sites from the root over the clip. Nominal-body constant;
        # the +-30% morphology spread perturbs it second-order (documented).
        site_xpos = np.asarray(traj.site_xpos)
        root_xpos = np.asarray(traj.qpos)[:, :3]
        d = np.linalg.norm(site_xpos - root_xpos[:, None, :], axis=-1)
        self._u2_length_scale = float(max(np.mean(d), 1e-3))

    def __call__(self, state, action, next_state, absorbing: bool,
                 info: Dict[str, Any], env: Any, model, data, carry,
                 backend: ModuleType):
        reward_state = carry.reward_state
        dt = self._u2_dt
        self._ensure_normalizers(env)

        bundle = self.reference_bundle(
            env, model, data, carry, backend, include_site_velocity=False)
        ref_qpos_full = bundle.reference_qpos_clamped

        # (1) joint pose, actuated hinges only, root excluded (heading-free)
        j_idx = self._u2_joint_qpos_idx
        joint_err = data.qpos[j_idx] - ref_qpos_full[j_idx]
        joint_mse = backend.mean(backend.square(joint_err))
        joint_reward = self.JOINT_COEFF * dt * backend.exp(
            -joint_mse / self.JOINT_TEMP)

        # (2) joint velocity, normalized by the reference's own motion scale
        v_idx = self._u2_joint_qvel_idx
        qvel_err = data.qvel[v_idx] - bundle.reference_qvel[v_idx]
        qvel_mse = backend.mean(backend.square(qvel_err)) / self._u2_qvel_scale
        qvel_reward = self.QVEL_RATIO * self.JOINT_COEFF * dt * backend.exp(
            -qvel_mse / self.QVEL_TEMP)

        # (3)+(4) body-correct FK site targets, root-relative, length-normalized
        rpos_reward = 0.0
        rquat_reward = 0.0
        if len(self._rel_site_ids) > 1:
            site_rpos_traj = bundle.relative_site_position
            site_rangles_traj = bundle.relative_site_orientation
            site_rpos, site_rangles, _ = calculate_relative_site_quatities(
                data, self._rel_site_ids, self._rel_body_ids,
                model.body_rootid, backend)
            rpos_dist = backend.mean(backend.square(
                (site_rpos - site_rpos_traj) / self._u2_length_scale))
            rquat_dist = backend.mean(backend.square(
                site_rangles - site_rangles_traj))
            rpos_reward = self.RPOS_RATIO * self.JOINT_COEFF * dt * backend.exp(
                -rpos_dist / self.RPOS_TEMP)
            rquat_reward = self.RQUAT_RATIO * self.JOINT_COEFF * dt * backend.exp(
                -rquat_dist / self.RQUAT_TEMP)

        # (5) root height, length-normalized
        height_err = (data.qpos[2] - ref_qpos_full[2]) / self._u2_length_scale
        height_reward = self.HEIGHT_COEFF * dt * backend.exp(
            -backend.square(height_err) / self.HEIGHT_TEMP)

        # (6) action-rate penalty + alive floor, urma2's positivity convention
        action_rate_pen = self.ACTION_RATE_COEFF * backend.sum(
            backend.square(action - reward_state.last_action))
        alive = self.ALIVE_COEFF * dt

        total = (joint_reward + qvel_reward + rpos_reward + rquat_reward
                 + height_reward - action_rate_pen)
        total = backend.maximum(total, 0.0) + alive
        total = backend.nan_to_num(total, nan=0.0)

        carry = carry.replace(reward_state=reward_state.replace(
            last_qvel=data.qvel, last_action=action))
        return total, carry


Urma2CoreReward.register()
