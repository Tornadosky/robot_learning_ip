"""A foot that can actually stand on the floor.

The defect
----------
Upstream ``MjxUnitreeH1._modify_spec_for_mjx`` deletes H1's foot meshes and
replaces each foot with **two thin capsules on the foot centre line**::

    back  capsule  pos=(-0.030, 0, -0.050)  r=0.015  half-len 0.025  axis +x
    front capsule  pos=( 0.150, 0, -0.054)  r=0.020  half-len 0.025  axis +y

Both sit at ``y = 0``. Their lowest points are 6.5 cm and 7.4 cm below the
ankle, so **the heel capsule hangs 9 mm above the toe capsule** and never
touches the floor in a neutral pose. What is left is a single lateral roller
under the toe: the support polygon collapses to a line across ``y`` with *zero
fore-aft extent*, and the robot has no passive resistance to pitching at all.

Measured consequence (``metrics/t02_stand_test.json``): H1 topples in
1.0-1.7 s from its own ``qpos0`` under PD hold at 1x, 3x, 10x and 30x gains --
45 of 45 cases. It is not a control-tuning failure, it is balancing a 51 kg
inverted pendulum on a 4 cm roller. G1 is unaffected: ``MjxUnitreeG1`` keeps
four spheres per foot in a 17 x 6 cm rectangle, and G1 does hold its pose at
10x gains.

The fix
-------
Replace the two capsules with one thin box per foot, sized to the real foot
(0.24 x 0.09 m, the span the two capsules already imply) and with its underside
at the same height the toe capsule had, so standing height and every reference
height statistic are unchanged. This is exactly what upstream itself does for
H1v2 (``mjGEOM_BOX``), so it stays inside the modelling conventions of the
framework rather than reintroducing the expensive mesh contact MJX cannot
afford.

Nothing here edits vendored code: the override is a mixin composed into the
environment class at registration time, and it is off unless the trainer is
given ``--foot-model box``.
"""

from __future__ import annotations

import mujoco
from mujoco import MjSpec

#: Per family: (foot body names, box pos, box half-size, capsule names to drop).
#: Geometry is derived from the capsules being replaced -- see the module
#: docstring for the arithmetic -- so the contact surface a policy stands on
#: keeps its height and gains the fore-aft extent it never had.
BOX_FOOT = {
    "h1": dict(
        bodies=("left_ankle_link", "right_ankle_link"),
        pos=[0.05, 0.0, -0.0665],
        size=[0.12, 0.045, 0.0075],
        drop=("left_foot", "right_foot"),
    ),
}


def _add_box_feet(spec: MjSpec, family: str) -> list[str]:
    cfg = BOX_FOOT[family]
    for g in list(spec.geoms):
        if g.name in cfg["drop"]:
            spec.delete(g)
    for g in spec.geoms:
        g.contype = 0
        g.conaffinity = 0
    attr = dict(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=cfg["pos"],
        size=cfg["size"],
        rgba=[1.0, 1.0, 1.0, 0.2],
        contype=0,
        conaffinity=0,
    )
    names = []
    for body_name in cfg["bodies"]:
        side = "left" if body_name.startswith("left") else "right"
        geom_name = f"{side}_foot1"
        body = [b for b in spec.bodies if b.name == body_name][0]
        body.add_geom(name=geom_name, **attr)
        names.append(geom_name)
    for name in names:
        spec.add_pair(geomname1="floor", geomname2=name)
    spec.add_pair(geomname1=names[0], geomname2=names[1])
    return names


def make_box_foot_mixin(family: str) -> type:
    """Mixin overriding ``_modify_spec_for_mjx`` with a flat box foot."""
    if family not in BOX_FOOT:
        raise KeyError(
            f"no box-foot geometry defined for {family!r}; families with an "
            f"already-flat foot (g1) do not need one")

    class BoxFootMixin:
        #: read back by the trainer for the manifest
        foot_model = "box"

        def _modify_spec_for_mjx(self, spec: MjSpec):
            # Deliberately not calling super(): the upstream implementation is
            # what installs the degenerate capsules, and deleting geoms that
            # its pair table already references leaves dangling pairs.
            _add_box_feet(spec, family)
            return spec

    BoxFootMixin.__name__ = f"BoxFoot{family.upper()}Mixin"
    return BoxFootMixin
