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


MorphologyAwareRootPoseTrajTerminalStateHandler.register()
