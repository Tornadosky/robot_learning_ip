"""Terminal-state handler that knows how tall the current body should be.

``RootPoseTrajTerminalStateHandler`` derives its root-height window from the
reference trajectory alone::

    root_height_range = (traj_min - margin, traj_max + margin)

That window is morphology-independent, so a legitimately taller robot is declared
absorbing simply for standing higher than the reference human-derived motion.
With the default 0.3 m margin and H1's 0.8 m nominal leg, this silently caps leg
scale at roughly [0.625, 1.375] *regardless of how well the policy controls the
body* - the named `extreme_tall_light` variant (leg 1.5) terminates on step 1
even under zero actions.

This subclass compares the root height in the *reference frame* by subtracting
the same grounding offset the reset applies, so every body is judged against its
own expected standing height. Nothing else about the criterion changes: the root
position deviation and root rotation checks are untouched.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

from loco_mujoco.core.terminal_state_handler.traj import (
    RootPoseTrajTerminalStateHandler,
)


class MorphologyAwareRootPoseTrajTerminalStateHandler(RootPoseTrajTerminalStateHandler):
    """Root-pose termination judged against each body's own standing height."""

    def _is_absorbing_compat(
        self,
        env: Any,
        obs,
        info,
        data,
        carry: Any,
        backend: ModuleType,
    ):
        """Judge the root height against this body's own standing height.

        Both the CPU and MJX entry points funnel through here, so overriding
        this one method covers both without duplicating the criterion.
        """
        morphology = getattr(carry, "morphology", None)
        if morphology is None:
            # Not an online-morphology environment; behave exactly as the base.
            return super()._is_absorbing_compat(env, obs, info, data, carry, backend)

        offset = env.root_height_offset(morphology)
        qpos = data.qpos
        if backend is not None and getattr(backend, "__name__", "") == "numpy":
            qpos = qpos.copy()
            qpos[self.root_height_ind] -= offset
        else:
            qpos = qpos.at[self.root_height_ind].add(-offset)
        return super()._is_absorbing_compat(
            env, obs, info, data.replace(qpos=qpos), carry, backend
        )


class GravityRootPoseTrajTerminalStateHandler(
    MorphologyAwareRootPoseTrajTerminalStateHandler
):
    """Rotation terminal measured from GRAVITY, not the clip's quat centroid.

    The base class allows a fixed angular distance from the centroid of the
    clip's root quaternions -- on dance2_subject4 that centroid is ~90 deg from
    upright, so heading and balance share one ~47 deg budget: a robot can be
    nearly lying down and pass, or upright and fail for turning. This subclass
    keeps the height and root-position checks and replaces the rotation check
    with the tilt of the root frame's z-axis from world-up, which is what
    "fallen" actually means. Heading (yaw) is unconstrained by construction.
    """

    def __init__(self, env, max_tilt_degrees: float = 60.0, **kwargs):
        super().__init__(env, **kwargs)
        import numpy as _np

        self._cos_max_tilt = float(_np.cos(_np.radians(max_tilt_degrees)))

    def _is_absorbing_compat(self, env, obs, info, data, carry, backend):
        saved = self._valid_threshold
        # Disable the centroid rotation check inside the base implementation.
        # Python-level mutation is safe: this executes at trace time only.
        self._valid_threshold = 1e9
        try:
            base_absorbing, carry = super()._is_absorbing_compat(
                env, obs, info, data, carry, backend
            )
        finally:
            self._valid_threshold = saved
        quat = data.qpos[self.root_quat_ind]  # w-first (MuJoCo)
        quat = quat / backend.linalg.norm(quat)
        x, y = quat[1], quat[2]
        # world-z component of the body z-axis: R[2,2] = 1 - 2(x^2 + y^2)
        cos_tilt = 1.0 - 2.0 * (x * x + y * y)
        tilt_cond = backend.less(cos_tilt, self._cos_max_tilt)
        return backend.logical_or(base_absorbing, tilt_cond), carry


MorphologyAwareRootPoseTrajTerminalStateHandler.register()
GravityRootPoseTrajTerminalStateHandler.register()
