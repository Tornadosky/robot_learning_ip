"""MimicReward whose site targets are retargeted to the body being controlled.

The multi-body trainer feeds every environment one shared canonical trajectory.
`MimicReward` reads its spatial targets from `traj_data.site_xpos`, which was
computed once, at load time, on the **nominal** model. So a randomized body with
1.18x arms is scored against a nominal robot's hand positions — up to 21 cm
wrong for the bodies actually sampled during training.

This subclass recomputes the reference site quantities by running forward
kinematics of the reference joint angles **on the model currently being
simulated**. In the online-morphology env that model carries this environment's
own `body_pos / body_ipos / site_pos`, so the targets come out correct for the
body, per environment, per step, with no dataset and no per-body retargeting
pass. It is the `fk` reference construction evaluated in-graph.

Only the trajectory-side spatial terms change. Joint-angle and joint-velocity
targets are body-independent and are still read from the trajectory unchanged,
and every weight, exponent and penalty is inherited from upstream.

Since the no-FSQ production fix this class no longer *owns* that computation:
it delegates to ``scaling.body_correct_reference``, the one provider the goal,
the evaluator and the renderer also call. Keeping a second copy here is exactly
how the goal and the reward came to command different spatial targets.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
from types import ModuleType
from typing import Any, Dict

import numpy as np

from loco_mujoco.core.reward.trajectory_based import MimicReward
from loco_mujoco.core.reward.utils import out_of_bounds_action_cost
from loco_mujoco.core.utils.math import (
    calculate_relative_site_quatities,
    quaternion_angular_distance,
)
from loco_mujoco.core.utils.mujoco import mj_jntid2qposid  # noqa: F401  (parity with base)

_SCRIPTS = str(_Path(__file__).resolve().parents[1])
if _SCRIPTS not in _sys.path:
    _sys.path.insert(0, _SCRIPTS)

from scaling.body_correct_reference import (  # noqa: E402
    body_correct_reference,
    clamp_reference_qpos,
    clampable_qpos_index,
    sampled_body_model,
)


class MorphMimicReward(MimicReward):
    """`MimicReward` with body-correct spatial targets.

    ``root_frame`` selects the convention the reference root XY is expressed in;
    see :func:`scaling.body_correct_reference.body_correct_reference`. The
    default reproduces upstream (absolute); ``episode_start`` makes the reward
    agree with the terminal criterion and with the reset pose.
    """

    def __init__(self, env, root_frame: str = "absolute", **kwargs):
        super().__init__(env, **kwargs)
        self._root_frame = str(root_frame)

    # ---- reference clamping to the CURRENT body's joint ranges ------------
    #
    # Once `joint_range_shift` is active (online_h1.MORPHOLOGY_SPEC dim 12) a
    # sampled body may be physically unable to reach part of the shared
    # reference's joint trajectory. Scoring it against an unreachable target
    # asks for a pose the simulator will never produce, so the reference is
    # clipped to THIS body's ranges before it is used -- for the qpos term and
    # for the forward-kinematics site-target pass alike, from one clipped
    # array, so the two can never disagree.
    #
    # When the dim is pinned (every arm before pipeline_v3) the ranges are the
    # nominal ones, and conditioned references are already clamped into them
    # with a 0.5 deg margin, so the clip is a no-op -- verified bit-for-bit by
    # p3_clamp_probe.py rather than assumed.

    def _clampable_qpos_index(self, sys):
        """(qpos indices, joint ids) of limited single-dof joints. Static.

        Thin forwarder kept for callers (probes, older experiment scripts) that
        reached into the reward for this; the implementation lives in the shared
        provider.
        """
        return clampable_qpos_index(sys)

    def _clamp_reference_qpos(self, qpos_traj_full, jnt_range, sys, backend: ModuleType):
        return clamp_reference_qpos(qpos_traj_full, jnt_range, sys, backend)

    def reference_bundle(self, env, model, data, carry, backend: ModuleType,
                         body_model=None, include_site_velocity: bool = False):
        """The shared body-correct bundle, keyed on THIS reward's site set.

        The goal adapter calls the same provider with the same cursor and the
        same body, so the target the actor is commanded toward and the target
        this reward scores are the same numbers by construction, not by
        coincidence.
        """
        if body_model is None:
            body_model = sampled_body_model(
                env, getattr(carry, "morphology", None), backend)
        return body_correct_reference(
            env,
            model,
            data,
            carry,
            backend,
            rel_site_ids=self._rel_site_ids,
            rel_body_ids=self._rel_body_ids,
            root_frame=getattr(self, "_root_frame", "absolute"),
            # morphology never changes the kinematic tree, so body_rootid is
            # topology-static and the two models agree; env.sys is used on the
            # sampled path only because `model` may be the CPU MjModel there.
            body_rootid=(
                env.sys.body_rootid if body_model is not None else model.body_rootid
            ),
            body_model=body_model,
            include_site_velocity=include_site_velocity,
        )

    def _traj_site_quantities(self, env, model, data, qpos_traj_full, carry, backend: ModuleType,
                              body_model=None):
        """Reference site pos/orientation/velocity, evaluated on the CURRENT body.

        Retained as the verifier's entry point (``verify_fk_targets.py`` probes
        an arbitrary ``qpos_traj_full``, not necessarily the one at the cursor),
        so it takes the qpos explicitly and does the site call directly.  The
        production step path goes through :meth:`reference_bundle` instead.
        """
        import mujoco

        if body_model is None:
            body_model = sampled_body_model(
                env, getattr(carry, "morphology", None), backend
            )

        if body_model is not None:
            from mujoco import mjx

            ref = mjx.kinematics(body_model, data.replace(qpos=qpos_traj_full))
            rootid = env.sys.body_rootid
        elif isinstance(model, mujoco.MjModel) and backend is np:
            d = mujoco.MjData(model)
            d.qpos[:] = np.asarray(qpos_traj_full)
            mujoco.mj_forward(model, d)
            ref, rootid = d, model.body_rootid
        else:
            # No per-env morphology (a plain single-body env): the trajectory's
            # own site data is already correct for this body.
            traj_single = env.th.traj.data.get(carry.traj_state.traj_no,
                                               carry.traj_state.subtraj_step_no, backend)
            return calculate_relative_site_quatities(
                traj_single, self._rel_site_ids, self._rel_body_ids,
                model.body_rootid, backend)

        return calculate_relative_site_quatities(
            ref, self._rel_site_ids, self._rel_body_ids, rootid, backend)

    def __call__(self, state, action, next_state, absorbing: bool, info: Dict[str, Any],
                 env: Any, model, data, carry, backend: ModuleType):
        reward_state = carry.reward_state

        # --- ONE call to the shared provider: the body currently being
        # simulated, the reference CLAMPED to its joint ranges, and the site
        # targets recomputed by FK on that same body.  One clipped array feeds
        # both the qpos term and the FK site-target pass, so the two views of
        # "the reference" cannot drift apart; and the goal adapter reads the
        # identical bundle, so the commanded target cannot drift from the scored
        # one either.  With joint_range_shift pinned the clamp is a no-op (see
        # the note on _clamp_reference_qpos).
        #
        # include_site_velocity=False: this reward scores positions and
        # orientations only (rvel_w_sum is not consumed here, matching the
        # behaviour every previous arm was trained under), so the com_pos/
        # com_vel passes the velocity term would need are skipped.
        bundle = self.reference_bundle(
            env, model, data, carry, backend, include_site_velocity=False,
        )
        ref_qpos_full = bundle.reference_qpos_clamped

        # --- joint targets: body-independent apart from the range clamp
        qpos_traj = ref_qpos_full[self._qpos_ind]
        qvel_traj = bundle.reference_qvel[self._qvel_ind]
        qpos_quat_traj = qpos_traj[self._quat_in_qpos].reshape(-1, 4)

        qpos, qvel = data.qpos[self._qpos_ind], data.qvel[self._qvel_ind]
        qpos_quat = qpos[self._quat_in_qpos].reshape(-1, 4)

        qpos_dist = backend.mean(backend.square(
            qpos[~self._quat_in_qpos] - qpos_traj[~self._quat_in_qpos]))
        qpos_dist += backend.mean(quaternion_angular_distance(qpos_quat, qpos_quat_traj, backend))
        qvel_dist = backend.mean(backend.square(qvel - qvel_traj))

        qpos_reward = backend.exp(-self._qpos_w_exp * qpos_dist)
        qvel_reward = backend.exp(-self._qvel_w_exp * qvel_dist)
        total_reward = self._qpos_w_sum * qpos_reward + self._qvel_w_sum * qvel_reward

        # --- spatial targets: recomputed on THIS body
        if len(self._rel_site_ids) > 1:
            site_rpos_traj = bundle.relative_site_position
            site_rangles_traj = bundle.relative_site_orientation
            # site velocities are deliberately not scored here (see the
            # include_site_velocity note above), so they are not computed
            site_rpos, site_rangles, _ = calculate_relative_site_quatities(
                data, self._rel_site_ids, self._rel_body_ids, model.body_rootid, backend)

            rpos_dist = backend.mean(backend.square(site_rpos - site_rpos_traj))
            rangles_dist = backend.mean(backend.square(site_rangles - site_rangles_traj))
            total_reward = (total_reward
                            + self._rpos_w_sum * backend.exp(-self._rpos_w_exp * rpos_dist)
                            + self._rquat_w_sum * backend.exp(-self._rquat_w_exp * rangles_dist))

        # --- penalties, identical to upstream
        penalties = 0.0
        if self._action_out_of_bounds_coeff > 0.0:
            penalties += self._action_out_of_bounds_coeff * -out_of_bounds_action_cost(
                action, lower_bound=env.mdp_info.action_space.low,
                upper_bound=env.mdp_info.action_space.high, backend=backend)
        if self._joint_acc_coeff > 0.0:
            last = reward_state.last_qvel[~self._free_joint_qvel_mask]
            now = data.qvel[~self._free_joint_qvel_mask]
            penalties += self._joint_acc_coeff * -backend.sum(backend.square(now - last) / env.dt)
        if self._action_rate_coeff > 0.0:
            penalties += self._action_rate_coeff * -backend.sum(
                backend.square(action - reward_state.last_action))
        penalties = backend.maximum(penalties, -1.0)

        total_reward = backend.maximum(total_reward + penalties, 0.0)
        total_reward = backend.nan_to_num(total_reward, nan=0.0)

        carry = carry.replace(reward_state=reward_state.replace(
            last_qvel=data.qvel, last_action=action))
        return total_reward, carry


MorphMimicReward.register()
