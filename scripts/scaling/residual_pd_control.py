"""A PD action space with the scale urma2 actually uses.

The diff that matters
---------------------
``loco_mujoco.core.control_functions.pd.PDControl`` builds its action space from
the joint's **entire** travel::

    high_pos_target = jnt_range_hi - nominal
    low_pos_target  = jnt_range_lo - nominal
    norm_act_mean   = (high + low) / 2        # == range midpoint - nominal
    norm_act_delta  = (high - low) / 2        # == half the FULL joint range
    target = clip(nominal + action*delta + mean + offsets, jnt_lo, jnt_hi)

So an action of 1.0 commands the joint to its limit, and — because of
``norm_act_mean`` — an action of **0.0 commands the midpoint of the joint's
range**, not the nominal pose.

``loco_mjx`` (urma2), which trains one policy over ~50 topologies with
randomisation, uses instead::

    center = nominal + reference_action_bias * (reference_position - nominal)
    target = center + action * scaling_factor + offsets

with a per-actuator ``scaling_factor`` of **0.25-0.75 rad** across all 50 robots
(H1 0.75, G1 0.5), and ``tracking_reference_action_bias`` centring the action on
the *reference* pose for tracking runs — its config comment says plainly that
zero "preserves the standard nominal-centered controller".

Measured on this repo's own models, the loco-mujoco half-range is **1.92 rad**
median on G1 (max 2.88) against urma2's 0.5: a **3.8x coarser** action space, up
to 5.8x on the widest joints. The traced G1 policy emits |action| ~0.70 mean and
1.47 max, i.e. it commands 70-100%+ of full joint travel every step, and the
applied torque sits pinned at the actuator limit.

This class changes the two things that need no new plumbing: the scale, and
centring on the nominal pose rather than the range midpoint. Reference centring
(``reference_action_bias``) needs the per-step reference inside the control
function and is left for the follow-up; ``--pd-action-scale`` alone moves the
exploration scale by ~4x, which is the factor the traces indict.
"""

from __future__ import annotations

import numpy as np

from loco_mujoco.core.control_functions.pd import PDControl


class ResidualPDControl(PDControl):
    """``PDControl`` with a fixed per-actuator action scale in radians.

    ``action_scale`` is the urma2 ``scaling_factor``: the joint displacement, in
    radians, that an action of 1.0 commands away from the nominal pose. Zero
    action holds the nominal pose exactly, because ``norm_act_mean`` is dropped.
    """

    def __init__(self, env, action_scale: float = 0.5, **kwargs):
        kwargs.setdefault("scale_action_to_jnt_limits", True)
        super().__init__(env, **kwargs)
        self._action_scale = float(action_scale)
        # Replace the full-range mapping with a residual of `action_scale` rad.
        # Keeping the arrays' shape and dtype means every downstream consumer
        # (domain randomisation offsets, action-space bounds) is unaffected.
        self.norm_act_delta = np.full_like(
            np.asarray(self.norm_act_delta, dtype=np.float64), self._action_scale)
        self.norm_act_mean = np.zeros_like(
            np.asarray(self.norm_act_mean, dtype=np.float64))

    @property
    def action_scale(self) -> float:
        return self._action_scale


def register(name: str = "ResidualPDControl") -> None:
    """Make the class visible to loco-mujoco's control-function factory."""
    from loco_mujoco.core.control_functions.base import ControlFunction

    ControlFunction.registered.setdefault(name, ResidualPDControl)
