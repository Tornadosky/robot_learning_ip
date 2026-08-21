"""A `GoalTrajMimic` that also tells the policy where it is relative to the reference.

Finding 41: `MimicReward`'s `qpos` term scores world-frame root position, while
both the robot's own observation (`FreeJointPosNoXY`) and `GoalTrajMimic`'s
reference qpos (`mj_jntid2qposid(root)[2:]`) strip root XY, and every site term
is relative to `upper_body_mimic`. The policy is charged for a displacement it
cannot perceive, so the gradient with respect to it is noise. That is why no
configuration in this audit tracked the root better than a statue.

This is the minimum change that makes the quantity observable: append the root
position **error** — reference root minus current root, rotated into the robot's
own heading frame — to the goal observation. Three numbers.

It is a custom adapter, which the goal document permits ("Adapter for
body-indexed reference data"); the upstream reward, trajectory handling and RSI
are untouched. Rotating into the robot's frame keeps the observation
heading-equivariant, so the policy learns "the target is 0.4 m ahead and 0.1 m
left" rather than a world-frame coordinate it would have to memorise.
"""

from __future__ import annotations

from typing import Any, Dict, List, Union

import numpy as np

from loco_mujoco.core.observations.goals import GoalTrajMimic


class GoalTrajMimicRootErr(GoalTrajMimic):
    """`GoalTrajMimic` plus the local-frame root position error (3 extra dims)."""

    def __init__(self, info_props: Dict, rel_body_names: List[str] = None,
                 include_z: bool = True, **kwargs):
        super().__init__(info_props, rel_body_names=rel_body_names, **kwargs)
        self._include_z = include_z
        self._n_root_err = 3 if include_z else 2

    def _init_from_mj(self, env: Any, model, data, current_obs_size: int):
        super()._init_from_mj(env, model, data, current_obs_size)
        # Widen the declared dimension, re-issue the observation indices, AND
        # re-issue min/max. The base class sizes min/max from `dim` inside its
        # own `_init_from_mj`, and `_get_obs_limits` builds the environment's
        # observation space by concatenating them -- so widening `_dim` alone
        # leaves the env advertising 434 while emitting 437, which the policy
        # network would then be built against.
        self._dim = self._dim + self._n_root_err
        self.obs_ind = np.array([j for j in range(current_obs_size, current_obs_size + self.dim)])
        self.min = [-np.inf] * self.dim
        self.max = [np.inf] * self.dim

    def get_obs_and_update_state(self, env, model, data, carry, backend):
        goal, carry = super().get_obs_and_update_state(env, model, data, carry, backend)

        if backend == np:
            from scipy.spatial.transform import Rotation as R
        else:
            from jax.scipy.spatial.transform import Rotation as R

        traj_data = env.th.traj.data
        traj_state = carry.traj_state
        traj_single = traj_data.get(traj_state.traj_no, traj_state.subtraj_step_no, backend)

        # world-frame error, then rotated into the robot's own frame so the
        # observation is heading-equivariant rather than a world coordinate
        err_world = traj_single.qpos[:3] - data.qpos[:3]
        quat = data.qpos[3:7]
        rot = R.from_quat(backend.concatenate([quat[1:], quat[:1]])).as_matrix()
        err_local = rot.T @ err_world

        if not self._include_z:
            err_local = err_local[:2]

        return backend.concatenate([goal, err_local]), carry


def register(name: str = "GoalTrajMimicRootErr") -> None:
    """Make the goal selectable by string.

    `MuJoCoBase._setup_goal` resolves `goal_type` through `Goal.registered`, so
    that is the dict to add to.
    """
    from loco_mujoco.core.observations.goals import Goal
    Goal.registered[name] = GoalTrajMimicRootErr
