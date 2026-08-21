"""What does this humanoid actually stand on?

G1 dies by root-height termination in ~0.31 s and is immune to PD gain scaling,
while inverse dynamics says its dance clip needs only 0.22x of its torque limit.
"Too weak" does not explain that; "cannot balance" does.  Balance is decided by
three model facts that no training metric reports:

* which foot geoms can collide with the floor at all,
* the area of the polygon they span on the ground (the support polygon),
* whether the ankle actuator can generate m*g*d at the polygon's edge.

The important subtlety is *which model to audit*.  The environment the policy
trains in is not the shipped XML: LocoMuJoCo's MJX preparation zeroes every
``contype``/``conaffinity`` and replaces broad-phase collision with a short
explicit ``<pair>`` table, and on H1 it also swaps the foot mesh for two
capsules.  So the audit runs on the ENV model, and reports the raw XML beside it
so the substitution is visible.

Outputs, per robot:

1. Every geom on a foot/ankle body with its collision parameters, and an
   explicit flag for the ones that cannot collide with the floor.
2. Support polygon area per foot and for double stance, from the world-space
   lowest points of the floor-collidable foot geoms in the nominal pose.  Two
   numbers: ``bearing`` (points within a tolerance of the foot's lowest point,
   i.e. what carries load on flat ground) and ``extent`` (all candidate points,
   i.e. what the foot would span if it were perfectly flat).  Their difference
   is the foot's built-in rock.
3. A 3 s drop test from the nominal pose, with zero control and (as the
   informative control) an ideal computed-torque joint hold clipped to the real
   actuator limits.
4. Ankle pitch torque available vs m*g*d demanded at the front edge of the
   support polygon.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

WORKSPACE = Path(__file__).resolve().parents[2]
for _p in (str(WORKSPACE / "scripts"), str(WORKSPACE / "scripts" / "h1md")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

GEOM_TYPES = {0: "plane", 1: "hfield", 2: "sphere", 3: "capsule",
              4: "ellipsoid", 5: "cylinder", 6: "box", 7: "mesh", 8: "sdf"}

#: points this close to the foot's lowest point are treated as load bearing
BEARING_TOL_M = 0.002


# --------------------------------------------------------------------- model
def floor_geom(model) -> int:
    for name in ("floor", "ground", "plane"):
        index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if index >= 0:
            return index
    world = np.nonzero(np.asarray(model.geom_bodyid) == 0)[0]
    if world.size == 0:
        raise ValueError("no floor geom")
    return int(world[0])


def floor_collidable_geoms(model, floor: int) -> list[int]:
    """Robot geoms that can actually produce a contact with the floor.

    Explicit ``<pair>`` entries win: MuJoCo generates those regardless of
    contype/conaffinity, and the MJX-prepared LocoMuJoCo models rely on them
    exclusively (every contype is 0, so a contype-based filter reports that
    NOTHING can touch the ground).
    """
    paired = set()
    for p in range(model.npair):
        g1, g2 = int(model.pair_geom1[p]), int(model.pair_geom2[p])
        if g1 == floor:
            paired.add(g2)
        elif g2 == floor:
            paired.add(g1)
    if paired:
        return sorted(paired)
    return sorted(
        g for g in range(model.ngeom)
        if g != floor and int(model.geom_bodyid[g]) != 0
        and ((int(model.geom_contype[g]) & int(model.geom_conaffinity[floor]))
             or (int(model.geom_conaffinity[g]) & int(model.geom_contype[floor])))
    )


def subtree_bodies(model, root_names) -> list[int]:
    """Body ids of the named bodies and everything below them."""
    parents = np.asarray(model.body_parentid)
    out: set[int] = set()
    for name in root_names:
        root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if root < 0:
            continue
        out.add(root)
    changed = True
    while changed:
        changed = False
        for b in range(model.nbody):
            if b not in out and int(parents[b]) in out and b != 0:
                out.add(b)
                changed = True
    return sorted(out)


def geom_name(model, g: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"<unnamed:{g}>"


# ------------------------------------------------------------ contact points
def geom_lowest_points(model, data, g: int) -> np.ndarray:
    """World points of the geom's surface that can touch a flat floor.

    One point for a sphere, two for a capsule (its endcaps), the corners for a
    box, the vertices for a mesh. Each is pushed down by the geom's radius where
    the primitive has one, so the returned z is the true contact height.
    """
    kind = int(model.geom_type[g])
    pos = np.asarray(data.geom_xpos[g], dtype=np.float64)
    mat = np.asarray(data.geom_xmat[g], dtype=np.float64).reshape(3, 3)
    size = np.asarray(model.geom_size[g], dtype=np.float64)

    if kind == 2:  # sphere
        return (pos + np.array([0.0, 0.0, -size[0]]))[None, :]
    if kind in (3, 5):  # capsule, cylinder
        axis = mat[:, 2] * size[1]
        ends = np.stack([pos + axis, pos - axis])
        drop = size[0] if kind == 3 else 0.0
        return ends - np.array([0.0, 0.0, drop])
    if kind == 4:  # ellipsoid -- lowest point only
        return (pos + np.array([0.0, 0.0, -float(size.max())]))[None, :]
    if kind == 6:  # box
        signs = np.array([[sx, sy, sz]
                          for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
                         dtype=np.float64)
        return pos + (signs * size[:3]) @ mat.T
    if kind == 7:  # mesh
        did = int(model.geom_dataid[g])
        adr = int(model.mesh_vertadr[did])
        num = int(model.mesh_vertnum[did])
        verts = np.asarray(model.mesh_vert[adr:adr + num], dtype=np.float64)
        return pos + verts @ mat.T
    # plane / hfield / sdf: not a foot primitive
    return np.zeros((0, 3))


def hull_area(points: np.ndarray) -> tuple[float, np.ndarray]:
    """2D convex hull area (monotone chain) of the xy projection."""
    pts = np.unique(np.round(points[:, :2], 9), axis=0)
    if pts.shape[0] < 3:
        return 0.0, pts
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[order]

    def half(seq):
        stack: list[np.ndarray] = []
        for p in seq:
            while len(stack) >= 2:
                a, b = stack[-2], stack[-1]
                if np.cross(b - a, p - a) <= 0:
                    stack.pop()
                else:
                    break
            stack.append(p)
        return stack

    hull = np.array(half(pts)[:-1] + half(pts[::-1])[:-1])
    if hull.shape[0] < 3:
        return 0.0, hull
    x, y = hull[:, 0], hull[:, 1]
    area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    return area, hull


def polygon_margin(hull: np.ndarray, point: np.ndarray) -> float:
    """Signed distance from ``point`` to the hull boundary; + means inside."""
    if hull.shape[0] < 3:
        return float("-inf")
    area2 = float(np.dot(hull[:, 0], np.roll(hull[:, 1], -1))
                  - np.dot(hull[:, 1], np.roll(hull[:, 0], -1)))
    poly = hull if area2 > 0 else hull[::-1]
    inside = True
    best = np.inf
    for i in range(poly.shape[0]):
        a, b = poly[i], poly[(i + 1) % poly.shape[0]]
        edge = b - a
        length = float(np.linalg.norm(edge))
        if length == 0:
            continue
        cross = float(np.cross(edge, point - a)) / length
        if cross < 0:
            inside = False
        t = float(np.clip(np.dot(point - a, edge) / (length ** 2), 0.0, 1.0))
        best = min(best, float(np.linalg.norm(point - (a + t * edge))))
    return best if inside else -best


# ------------------------------------------------------------- standing pose
def standing_qpos(model, collidable: list[int],
                  joint_overrides: dict | None = None) -> np.ndarray:
    """``qpos0`` with the root raised so the lowest collidable geom is at z=0."""
    data = mujoco.MjData(model)
    qpos = np.array(model.qpos0, dtype=np.float64)
    for joint, value in (joint_overrides or {}).items():
        qpos[int(model.jnt_qposadr[int(joint)])] = float(value)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    lowest = min(float(geom_lowest_points(model, data, g)[:, 2].min())
                 for g in collidable)
    qpos[2] -= lowest
    return qpos


def contact_support(model, qpos: np.ndarray, sink: float = 0.001) -> dict:
    """Support polygon from MuJoCo's OWN contact points, at a small sink.

    This is the load-bearing definition, and it differs from the geometric one:
    both robots' feet are rockers, so at exactly zero penetration the solver
    reports contact on the front points or the rear points but never both. A
    fraction of a millimetre of sink -- less than these contacts see under body
    weight with ``solref=[0.02, 1]`` -- is what a standing robot actually has.
    """
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.qpos[2] -= sink
    mujoco.mj_forward(model, data)
    pts = np.array([data.contact.pos[c] for c in range(data.ncon)]) \
        if data.ncon else np.zeros((0, 3))
    sides: dict = {"left": [], "right": []}
    for c in range(data.ncon):
        g = int(data.contact.geom[c][1])
        body = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                  int(model.geom_bodyid[g])) or "").lower()
        sides["left" if "left" in body or body.startswith("l_")
              else "right"].append(data.contact.pos[c])
    com = np.asarray(data.subtree_com[0], dtype=np.float64)
    area, hull = hull_area(pts) if pts.shape[0] else (0.0, np.zeros((0, 2)))
    out = {
        "sink_m": sink,
        "ncon": int(data.ncon),
        "contact_geoms": sorted({
            geom_name(model, int(data.contact.geom[c][1]))
            for c in range(data.ncon)}),
        "double_stance_area_m2": area,
        "com_xy_m": [float(com[0]), float(com[1])],
        "com_inside_polygon": bool(polygon_margin(hull, com[:2]) > 0),
        "com_margin_to_edge_m": float(polygon_margin(hull, com[:2])),
    }
    if pts.shape[0]:
        out["contact_x_range_m"] = [float(pts[:, 0].min()), float(pts[:, 0].max())]
        out["contact_y_range_m"] = [float(pts[:, 1].min()), float(pts[:, 1].max())]
        out["com_ahead_of_rear_edge_m"] = float(com[0] - pts[:, 0].min())
        out["com_behind_front_edge_m"] = float(pts[:, 0].max() - com[0])
    for side, chunk in sides.items():
        if chunk:
            a, _ = hull_area(np.array(chunk))
            out[f"{side}_foot_area_m2"] = a
            out[f"{side}_foot_points"] = len(chunk)
        else:
            out[f"{side}_foot_area_m2"] = 0.0
            out[f"{side}_foot_points"] = 0
    return out


def ankle_pitch_joints(model, family_key: str) -> dict:
    """``{'left': jid, 'right': jid}`` for the ankle PITCH joints.

    Resolved through ``FAMILY_BODIES.ankle_bodies`` -- which by construction are
    the ankle-pitch attachment bodies -- and not by joint name: LocoMuJoCo
    renames H1's joints to an OpenSim-style convention (``ankle_angle_l``), so a
    name filter looking for "left" silently finds nothing on H1.
    """
    from scaling.family_morphology import FAMILY_BODIES

    out = {}
    for name in FAMILY_BODIES[family_key].ankle_bodies:
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body < 0:
            continue
        side = "left" if "left" in name.lower() or name.lower().startswith("l_") \
            else "right"
        joints = [j for j in range(model.njnt)
                  if int(model.jnt_bodyid[j]) == body
                  and int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_HINGE)]
        if joints:
            out[side] = joints[0]
    return out


def _ankle_pitch_scan(model, collidable, mass: float, leg_length: float,
                      family_key: str, sink: float = 0.001,
                      samples: int = 481) -> dict:
    """Best real contact polygon reachable by pitching BOTH ankles together.

    Scored on MuJoCo's contact set at a 1 mm sink, so a foot that is a rocker
    scores its rocking, not the tolerance the analyst chose.
    """
    joints = ankle_pitch_joints(model, family_key)
    if not joints:
        return {}
    j0 = next(iter(joints.values()))
    low, high = float(model.jnt_range[j0, 0]), float(model.jnt_range[j0, 1])
    if not model.jnt_limited[j0] or high <= low:
        low, high = -0.6, 0.6
    best = {"double_stance_area_m2": -1.0}
    for angle in np.linspace(low, high, samples):
        overrides = {j: angle for j in joints.values()}
        qpos = standing_qpos(model, collidable, overrides)
        entry = contact_support(model, qpos, sink)
        if entry["double_stance_area_m2"] > best["double_stance_area_m2"]:
            best = dict(entry)
            best["ankle_pitch_rad"] = float(angle)
            best["ankle_pitch_deg"] = float(np.degrees(angle))
    best["area_per_mass_leg_m2_per_kgm"] = (
        best["double_stance_area_m2"] / (mass * leg_length))
    best["area_over_leg_length_sq"] = (
        best["double_stance_area_m2"] / (leg_length ** 2))
    best["joints"] = {s: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
                      for s, j in joints.items()}
    best["scan_range_rad"] = [low, high]
    best["samples"] = samples
    return best


# ------------------------------------------------------------------ measures
def foot_report(model, robot: str, family_key: str) -> dict:
    from scaling.family_morphology import FAMILY_BODIES

    bodies = FAMILY_BODIES[family_key]
    floor = floor_geom(model)
    collidable = floor_collidable_geoms(model, floor)
    foot_bodies = subtree_bodies(model, bodies.ankle_bodies)
    foot_body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
                       for b in foot_bodies]
    foot_geoms = [g for g in range(model.ngeom)
                  if int(model.geom_bodyid[g]) in foot_bodies]

    mass = float(mujoco.mj_getTotalmass(model))
    gravity = float(abs(model.opt.gravity[2]))
    knee_z = np.abs(np.asarray(model.body_pos)[
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
         for n in bodies.knee_bodies], 2])
    ankle_z = np.abs(np.asarray(model.body_pos)[
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
         for n in bodies.ankle_bodies], 2])
    leg_length = float(knee_z.mean() + ankle_z.mean())

    inventory = []
    for g in foot_geoms:
        contype, conaff = int(model.geom_contype[g]), int(model.geom_conaffinity[g])
        inventory.append({
            "geom": geom_name(model, g),
            "geom_id": g,
            "body": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                      int(model.geom_bodyid[g])),
            "type": GEOM_TYPES.get(int(model.geom_type[g]), "?"),
            "size": [float(v) for v in model.geom_size[g]],
            "pos": [float(v) for v in model.geom_pos[g]],
            "contype": contype,
            "conaffinity": conaff,
            "condim": int(model.geom_condim[g]),
            "friction": [float(v) for v in model.geom_friction[g]],
            "solref": [float(v) for v in model.geom_solref[g]],
            "solimp": [float(v) for v in model.geom_solimp[g]],
            "collides_with_floor": g in collidable,
            "cannot_collide_at_all": contype == 0 and conaff == 0 and g not in collidable,
        })

    # ---- support polygon in the nominal standing pose
    qpos = standing_qpos(model, collidable)
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)

    sides = {"left": [], "right": []}
    for g in collidable:
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                 int(model.geom_bodyid[g])) or ""
        side = "left" if body.lower().startswith(("left", "l_")) or "left" in body.lower() \
            else "right"
        sides[side].append(geom_lowest_points(model, data, g))

    polygons = {}
    all_bearing = []
    for side, chunks in sides.items():
        if not chunks:
            continue
        pts = np.concatenate(chunks)
        zmin = float(pts[:, 2].min())
        bearing = pts[pts[:, 2] <= zmin + BEARING_TOL_M]
        all_bearing.append(bearing)
        area_b, hull_b = hull_area(bearing)
        area_e, _ = hull_area(pts)
        polygons[side] = {
            "candidate_points": int(pts.shape[0]),
            "bearing_points": int(bearing.shape[0]),
            "lowest_z_m": zmin,
            "contact_height_spread_m": float(pts[:, 2].max() - zmin),
            "bearing_area_m2": area_b,
            "extent_area_m2": area_e,
            "bearing_length_x_m": float(bearing[:, 0].max() - bearing[:, 0].min()),
            "bearing_width_y_m": float(bearing[:, 1].max() - bearing[:, 1].min()),
            "extent_length_x_m": float(pts[:, 0].max() - pts[:, 0].min()),
            "extent_width_y_m": float(pts[:, 1].max() - pts[:, 1].min()),
            "bearing_area_per_mass_leg_m2_per_kgm": area_b / (mass * leg_length),
            "bearing_area_over_leg_length_sq": area_b / (leg_length ** 2),
            "extent_area_per_mass_leg_m2_per_kgm": area_e / (mass * leg_length),
        }
    double_area, double_hull = 0.0, np.zeros((0, 2))
    if all_bearing:
        double_area, double_hull = hull_area(np.concatenate(all_bearing))
    com = np.asarray(data.subtree_com[0], dtype=np.float64)
    polygons["double_stance"] = {
        "bearing_area_m2": double_area,
        "bearing_area_per_mass_leg_m2_per_kgm": double_area / (mass * leg_length),
        "bearing_area_over_leg_length_sq": double_area / (leg_length ** 2),
        "com_xy_m": [float(com[0]), float(com[1])],
        "com_inside_polygon": bool(polygon_margin(double_hull, com[:2]) > 0),
        "com_margin_to_edge_m": float(polygon_margin(double_hull, com[:2])),
    }

    # The geometry above is tolerance-dependent. These are MuJoCo's own contact
    # points: at the nominal pose, at a sweep of sink depths (how deep does the
    # foot have to press before the whole sole engages?), and at the best ankle
    # pitch. The 9 mm height offset between H1's two foot capsules means its
    # nominal pose contacts the toe capsule ONLY, whatever tolerance is used.
    polygons["contacts_at_nominal_pose"] = {
        f"sink_{int(s * 1000 * 10)}e-4m": contact_support(model, qpos, s)
        for s in (0.0, 0.0005, 0.001, 0.002, 0.005)
    }
    polygons["best_over_ankle_pitch"] = _ankle_pitch_scan(
        model, collidable, mass, leg_length, family_key)

    return {
        "robot": robot,
        "total_mass_kg": mass,
        "nominal_leg_length_m": leg_length,
        "floor_geom": geom_name(model, floor),
        "n_contact_pairs": int(model.npair),
        "collision_filtering": (
            "explicit <pair> table only (every contype/conaffinity is 0)"
            if int(model.npair) and not int(np.asarray(model.geom_contype).max())
            else "contype/conaffinity"),
        "floor_collidable_geoms": [geom_name(model, g) for g in collidable],
        "foot_bodies": foot_body_names,
        "foot_geoms": inventory,
        "foot_geoms_that_cannot_collide": [
            e["geom"] for e in inventory if not e["collides_with_floor"]],
        "standing_root_height_m": float(qpos[2]),
        "support_polygon": polygons,
    }


def ankle_torque_report(model, family_key: str, report: dict,
                        strength_scale: float = 1.0) -> dict:
    """Ankle pitch authority vs the m*g*d the support polygon demands."""
    mass = float(mujoco.mj_getTotalmass(model))
    weight = mass * float(abs(model.opt.gravity[2]))

    collidable = floor_collidable_geoms(model, floor_geom(model))
    qpos = standing_qpos(model, collidable)
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)

    entries = []
    pitch_joints = {j: side for side, j in
                    ankle_pitch_joints(model, family_key).items()}
    for a in range(model.nu):
        joint = int(model.actuator_trnid[a, 0])
        if joint not in pitch_joints:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint) or ""
        side = pitch_joints[joint]
        # These models leave actuator_forcerange at [0,0]/forcelimited=False, so
        # the real authority is gear * gain * ctrlrange (reference_feasibility.py).
        force = max(abs(model.actuator_forcerange[a]))
        source = "forcerange"
        if not model.actuator_forcelimited[a] or force <= 0:
            force = (abs(model.actuator_gear[a, 0])
                     * abs(model.actuator_gainprm[a, 0])
                     * max(abs(model.actuator_ctrlrange[a])))
            source = "gear*gain*ctrlrange"
        anchor = np.asarray(data.xanchor[joint], dtype=np.float64)
        poly = report["support_polygon"].get(side)
        if poly is None:
            continue
        # forward/backward lever arms from the ankle pitch axis to the polygon
        pts_x = [poly["bearing_length_x_m"], poly["extent_length_x_m"]]
        entries.append({
            "actuator": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a),
            "joint": name,
            "side": side,
            "torque_limit_Nm": float(force) * strength_scale,
            "limit_source": source,
            "ankle_anchor_xyz": [float(v) for v in anchor],
            "bearing_length_x_m": pts_x[0],
            "extent_length_x_m": pts_x[1],
        })

    # lever arms measured against the ankle anchors, per side
    per_side = {}
    for side in ("left", "right"):
        poly = report["support_polygon"].get(side)
        anchors = [e["ankle_anchor_xyz"][0] for e in entries if e["side"] == side]
        if poly is None or not anchors:
            continue
        anchor_x = float(np.mean(anchors))
        # rebuild the polygon points to get signed distances
        pts = []
        for g in collidable:
            body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                     int(model.geom_bodyid[g])) or ""
            if side not in body.lower():
                continue
            pts.append(geom_lowest_points(model, data, g))
        pts = np.concatenate(pts)
        zmin = float(pts[:, 2].min())
        bearing = pts[pts[:, 2] <= zmin + BEARING_TOL_M]
        d_front = float(bearing[:, 0].max() - anchor_x)
        d_back = float(anchor_x - bearing[:, 0].min())
        d_front_ext = float(pts[:, 0].max() - anchor_x)
        limits = [e["torque_limit_Nm"] for e in entries if e["side"] == side]
        limit = float(np.mean(limits))
        # single support: the whole weight over one ankle
        need_single = weight * max(d_front, 1e-9)
        need_double = 0.5 * weight * max(d_front, 1e-9)
        per_side[side] = {
            "ankle_pitch_torque_limit_Nm": limit,
            "lever_arm_front_bearing_m": d_front,
            "lever_arm_back_bearing_m": d_back,
            "lever_arm_front_extent_m": d_front_ext,
            "torque_needed_single_support_Nm": need_single,
            "torque_needed_double_support_Nm": need_double,
            "available_over_needed_single": limit / need_single if need_single else None,
            "available_over_needed_double": limit / need_double if need_double else None,
            "can_hold_com_at_front_edge_single_support": bool(limit >= need_single),
            "max_com_offset_ankle_can_hold_m": limit / weight,
            "max_com_offset_over_front_lever": (
                (limit / weight) / d_front if d_front > 0 else None),
        }
    return {"actuators": entries, "per_side": per_side,
            "body_weight_N": weight, "strength_scale": strength_scale}


# ----------------------------------------------------------------- drop test
def drop_test(model, seconds: float = 3.0, mode: str = "zero_ctrl",
              sample_hz: float = 20.0, joint_overrides: dict | None = None) -> dict:
    collidable = floor_collidable_geoms(model, floor_geom(model))
    qpos0 = standing_qpos(model, collidable, joint_overrides)
    if mode.startswith("rigid_body"):
        # Freeze every joint by giving its dof enormous rotor inertia. Unlike
        # writing qpos back after each step (which injects energy and launches
        # the robot), this is a legal MuJoCo model: the joints simply cannot
        # accelerate, so the humanoid behaves as one rigid body standing on its
        # feet. Toppling here indicts the support polygon, nothing else.
        import copy
        model = copy.deepcopy(model)
        model.dof_armature[6:] = 1.0e5
    data = mujoco.MjData(model)
    data.qpos[:] = qpos0
    mujoco.mj_forward(model, data)

    dofs, ranges = [], []
    for a in range(model.nu):
        joint = int(model.actuator_trnid[a, 0])
        dofs.append(int(model.jnt_dofadr[joint]))
        force = max(abs(model.actuator_forcerange[a]))
        if not model.actuator_forcelimited[a] or force <= 0:
            force = (abs(model.actuator_gear[a, 0])
                     * abs(model.actuator_gainprm[a, 0])
                     * max(abs(model.actuator_ctrlrange[a])))
        ranges.append(force)
    dofs = np.asarray(dofs)
    ranges = np.asarray(ranges)
    qadr = np.asarray([int(model.jnt_qposadr[int(model.actuator_trnid[a, 0])])
                       for a in range(model.nu)])
    q_target = qpos0[qadr].copy()

    steps = int(round(seconds / model.opt.timestep))
    every = max(1, int(round(1.0 / (sample_hz * model.opt.timestep))))
    times, heights, ncons, tilts = [], [], [], []
    up = np.array([0.0, 0.0, 1.0])
    for s in range(steps):
        if mode == "computed_torque":
            # ideal joint tracker: gravity/Coriolis compensation + stiff PD,
            # clipped to the actuators' own torque limits. If the robot still
            # topples, the limit is the contact geometry, not the controller.
            data.qacc[:] = 0.0
            mujoco.mj_inverse(model, data)
            gravcomp = np.array(data.qfrc_inverse[dofs])
            error = q_target - data.qpos[qadr]
            tau = gravcomp + 400.0 * error - 20.0 * data.qvel[dofs]
            data.qfrc_applied[:] = 0.0
            data.qfrc_applied[dofs] = np.clip(tau, -ranges, ranges)
        data.ctrl[:] = 0.0
        mujoco.mj_step(model, data)
        if s % every == 0:
            times.append(float(s * model.opt.timestep))
            heights.append(float(data.qpos[2]))
            ncons.append(int(data.ncon))
            quat = np.array(data.qpos[3:7])
            mat = np.zeros(9)
            mujoco.mju_quat2Mat(mat, quat)
            tilts.append(float(np.degrees(np.arccos(
                np.clip(mat.reshape(3, 3)[:, 2] @ up, -1.0, 1.0)))))

    joint_error = float(np.abs(np.asarray(data.qpos[qadr]) - q_target).max())
    heights = np.asarray(heights)
    tilts = np.asarray(tilts)
    ncons_a = np.asarray(ncons)
    last_second = heights[int(len(heights) * 2 / 3):]
    toppled = bool(tilts[-1] > 45.0 or heights[-1] < 0.6 * heights[0])
    settled = bool(not toppled
                   and float(last_second.max() - last_second.min()) < 0.01)
    return {
        "mode": mode,
        "seconds": seconds,
        "initial_root_height_m": float(heights[0]),
        "final_root_height_m": float(heights[-1]),
        "min_root_height_m": float(heights.min()),
        "final_tilt_deg": float(tilts[-1]),
        "max_tilt_deg": float(tilts.max()),
        "ncon_initial": int(ncons_a[0]),
        "ncon_median": float(np.median(ncons_a)),
        "ncon_max": int(ncons_a.max()),
        "pct_samples_no_contact": float(100.0 * np.mean(ncons_a == 0)),
        "max_joint_tracking_error_rad": joint_error,
        "toppled": toppled,
        "settled": settled,
        "series": {"t_s": times, "root_height_m": [float(v) for v in heights],
                   "ncon": ncons, "tilt_deg": [float(v) for v in tilts]},
    }


# --------------------------------------------------------------------- main
def _env_model(robot: str, reference_root: Path, clip_window: str):
    from scaling.parallel_cross_humanoid_train import (
        _build_robot_env,
        _ensure_latent_defaults,
    )
    args = _ensure_latent_defaults(SimpleNamespace(
        source="h1", reference_mode="direct", reference_root=reference_root,
        clip=None, start_frame=None, frames=None, clip_windows=[clip_window],
        morphology="continuous", use_mjwarp=False,
        reward_type="MorphMimicReward", goal_type="GoalTrajMimic",
    ))
    env = _build_robot_env(args, robot)[0]
    return env._model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robots", nargs="+", default=["h1", "g1"])
    parser.add_argument("--clip-window", default="dance2_subject4:19482:800")
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument(
        "--reference-root", type=Path,
        default=WORKSPACE / "external_data" / "cross_humanoid")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from scaling.body_correct_reference import cpu_morphology_model
    from scaling.cross_humanoid_retarget import HUMANOIDS
    from scaling.family_morphology import (
        FAMILY_MORPHOLOGY_HIGH,
        FAMILY_MORPHOLOGY_LOW,
    )

    out: dict = {"clip_window": args.clip_window, "robots": {}}
    for robot in args.robots:
        model = _env_model(robot, args.reference_root, args.clip_window)
        raw = mujoco.MjModel.from_xml_path(str(HUMANOIDS[robot].xml_path))

        report = foot_report(model, robot, robot)
        report["raw_xml"] = {
            "path": str(HUMANOIDS[robot].xml_path),
            **{k: v for k, v in foot_report(raw, robot, robot).items()
               if k in ("floor_collidable_geoms", "n_contact_pairs",
                        "collision_filtering", "support_polygon")},
        }
        report["drop_tests"] = {
            mode: drop_test(model, args.seconds, mode)
            for mode in ("zero_ctrl", "computed_torque", "rigid_body")
        }
        # A rigid-body arm at the ankle angle that makes the foot flat. On H1
        # the default pose bears on the toe capsule alone, so this separates
        # "the foot geometry cannot support the robot" from "the default pose
        # happens to stand on the toe".
        flat = report["support_polygon"]["best_over_ankle_pitch"]
        joints = ankle_pitch_joints(model, robot)
        overrides = ({j: flat["ankle_pitch_rad"] for j in joints.values()}
                     if "ankle_pitch_rad" in flat else {})
        if overrides:
            report["drop_tests"]["rigid_body_flat_foot"] = drop_test(
                model, args.seconds, "rigid_body", joint_overrides=overrides)
            report["drop_tests"]["rigid_body_flat_foot"]["ankle_pitch_deg"] = \
                flat["ankle_pitch_deg"]

        # morphology randomization: geoms are untouched, but mass and actuator
        # force range are not -- so the ankle margin moves with the body.
        report["ankle_torque"] = {}
        for label, morph in (("nominal", np.ones(4, dtype=np.float32)),
                             ("low_corner", FAMILY_MORPHOLOGY_LOW),
                             ("high_corner", FAMILY_MORPHOLOGY_HIGH)):
            body_model = (model if label == "nominal"
                          else cpu_morphology_model(model, robot, morph))
            body_report = (report if label == "nominal"
                           else foot_report(body_model, robot, robot))
            report["ankle_torque"][label] = ankle_torque_report(
                body_model, robot, body_report)
            if label != "nominal":
                report["ankle_torque"][label]["support_polygon"] = \
                    body_report["support_polygon"]
                report["ankle_torque"][label]["morphology"] = [
                    float(v) for v in morph]

        out["robots"][robot] = report
        poly = report["support_polygon"]
        print(f"[foot] {robot}: {len(report['floor_collidable_geoms'])} "
              f"floor-collidable geoms; per-foot bearing area "
              f"{poly['left']['bearing_area_m2'] * 1e4:.1f} cm2 "
              f"(extent {poly['left']['extent_area_m2'] * 1e4:.1f} cm2, "
              f"height spread {poly['left']['contact_height_spread_m'] * 1000:.1f} mm); "
              f"double stance {poly['double_stance']['bearing_area_m2'] * 1e4:.1f} cm2",
              flush=True)
        for key, c in poly["contacts_at_nominal_pose"].items():
            print(f"       contacts@{c['sink_m'] * 1000:.1f}mm sink: ncon "
                  f"{c['ncon']:2d} area {c['double_stance_area_m2'] * 1e4:6.1f} cm2 "
                  f"CoM inside={c['com_inside_polygon']} margin "
                  f"{c['com_margin_to_edge_m'] * 100:+.1f} cm  {c['contact_geoms']}",
                  flush=True)
        best = poly["best_over_ankle_pitch"]
        print(f"       best contact polygon over ankle pitch: "
              f"{best.get('double_stance_area_m2', 0) * 1e4:.1f} cm2 at "
              f"{best.get('ankle_pitch_deg', 0):.2f} deg, ncon "
              f"{best.get('ncon', 0)}, CoM inside="
              f"{best.get('com_inside_polygon')} margin "
              f"{best.get('com_margin_to_edge_m', 0) * 100:+.1f} cm", flush=True)
        for mode, dt in report["drop_tests"].items():
            print(f"       drop[{mode}]: root {dt['initial_root_height_m']:.3f} -> "
                  f"{dt['final_root_height_m']:.3f} m, tilt "
                  f"{dt['final_tilt_deg']:.0f} deg, ncon med {dt['ncon_median']:.0f}, "
                  f"joint err {dt['max_joint_tracking_error_rad']:.3f} rad, "
                  f"toppled={dt['toppled']} settled={dt['settled']}", flush=True)
        for side, e in report["ankle_torque"]["nominal"]["per_side"].items():
            print(f"       ankle[{side}]: {e['ankle_pitch_torque_limit_Nm']:.1f} Nm vs "
                  f"{e['torque_needed_single_support_Nm']:.1f} Nm needed "
                  f"(d_front {e['lever_arm_front_bearing_m'] * 100:.1f} cm) -> "
                  f"{e['available_over_needed_single']:.2f}x; max CoM offset "
                  f"{e['max_com_offset_ankle_can_hold_m'] * 100:.1f} cm", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[foot] -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
