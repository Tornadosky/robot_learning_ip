"""Derive a family's CLIP_SIGNS table from the clip itself, with no loco-mujoco.

The retargets in ``external_data/amass_converted/LAFAN1`` are produced on
loco-mujoco's CPU models and consumed by loco_mjx's MJX models. The two keep the
same links and the same joint names but reverse a great many joint axes -- 13 of
H1's 19, 15 of G1's 23, 14 of booster_t1's 23. A family added without its sign
table silently defaults every joint to +1.0 and the policy then chases a
part-mirrored reference. That has happened three times (G1 fixed 08-09, t1 fixed
08-21), each time costing a campaign of numbers.

The existing generator (``experiments/pipeline_hg_20260809/stage3_check_signs.py``)
compares ``dot(axis_locomujoco, axis_urma2)`` and therefore needs BOTH packages
importable. loco-mujoco's submodule now points at the group repo's integration
branch, which ships no ``loco_mujoco/trajectory`` at all, so that path does not
run locally any more.

This script needs only mujoco and the clip. The clip records, per frame, both the
joint angles ``qpos`` AND the world poses ``xpos`` that those angles produced in
the source model. So the signs are recoverable by asking which sign vector makes
loco_mjx's own forward kinematics reproduce the recorded world poses: set the
free joint from the clip, set each hinge to ``s_j * clip_angle``, run FK, and
compare root-relative body positions against the clip's own ``xpos``.

Validate before trusting: run with ``--check`` on h1, g1 and booster_t1, which
have known-correct tables, and confirm the derivation reproduces them exactly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

WORKSPACE = Path(__file__).resolve().parents[2]
ROBOT_DIR = WORKSPACE / "loco_mjx" / "loco_mjx" / "environments" / "robots"
CLIP_DIR = WORKSPACE / "external_data" / "amass_converted" / "LAFAN1"

# robot key -> (loco_mjx model directory, clip subdirectory)
FAMILIES = {
    "h1": ("unitree_h1", "UnitreeH1"),
    "g1": ("unitree_g1", "UnitreeG1"),
    "booster_t1": ("booster_t1", "BoosterT1"),
    "atlas": ("atlas", "Atlas"),
    "talos": ("talos", "Talos"),
    "toddlerbot": ("toddlerbot", "ToddlerBot"),
    "h1v2": ("unitree_h1v2", "UnitreeH1v2"),
    "apollo": ("apptronik_apollo", "Apollo"),
    "gr1t2": ("fourier_gr1t2", "FourierGR1T2"),
}


def load_clip(clip_path: Path):
    d = np.load(clip_path, allow_pickle=True)
    return {
        "qpos": np.asarray(d["qpos"], dtype=np.float64),
        "xpos": np.asarray(d["xpos"], dtype=np.float64),
        "xquat": np.asarray(d["xquat"], dtype=np.float64),
        "joint_names": [str(x) for x in d["joint_names"]],
        "body_names": [str(x) for x in d["body_names"]],
        "jnt_type": np.asarray(d["jnt_type"]),
    }


def pick_frames(qpos_hinges: np.ndarray, n: int) -> np.ndarray:
    """Frames where the joints are far from zero.

    A frame near the home pose constrains nothing -- every sign reproduces it --
    so scoring on such frames would make the objective flat and the descent
    arbitrary. Rank by total absolute joint excursion and spread the picks over
    the whole clip so one pose cannot dominate.
    """
    energy = np.abs(qpos_hinges).sum(axis=1)
    order = np.argsort(energy)[::-1][: max(n * 8, n)]
    order = np.sort(order)
    if len(order) <= n:
        return order
    step = len(order) / n
    return order[(np.arange(n) * step).astype(int)]


def fk_poses(model, data, clip, frames, col_of, adr_of, signs, body_ids):
    """Root-relative body positions AND world orientations under a sign vector.

    Orientation is not optional. A joint at the end of its chain whose child body
    origin sits on the joint axis -- every ankle roll, both elbows, the head
    pitch -- moves no body ORIGIN when its sign flips, so a position-only
    objective is exactly flat in those coordinates and the descent keeps whatever
    it started with. Scored on positions alone this derivation reproduces the
    clip to 0.0000 cm and still disagrees with the known-good tables on 2/19
    (H1), 3/23 (G1) and 4/23 (t1) joints -- and every one of those is such a
    terminal joint. Orientation sees them.
    """
    pos = np.empty((len(frames), len(body_ids), 3))
    quat = np.empty((len(frames), len(body_ids), 4))
    for i, t in enumerate(frames):
        data.qpos[:] = 0.0
        data.qpos[0:7] = clip["qpos"][t, 0:7]
        for j, adr in enumerate(adr_of):
            data.qpos[adr] = signs[j] * clip["qpos"][t, col_of[j]]
        mujoco.mj_forward(model, data)
        p = data.xpos[body_ids]
        pos[i] = p - p[0]
        quat[i] = data.xquat[body_ids]
    return pos, quat


def derive(robot: str, clip_name: str, n_frames: int, verbose: bool = True):
    model_dir, clip_sub = FAMILIES[robot]
    xml = ROBOT_DIR / model_dir / "data" / "plane.xml"
    clip_path = CLIP_DIR / clip_sub / clip_name
    if not xml.exists():
        raise FileNotFoundError(xml)
    if not clip_path.exists():
        raise FileNotFoundError(clip_path)

    clip = load_clip(clip_path)
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)

    # hinge columns of the clip, in clip order, that also exist in the mjx model
    free_nq = 7 if int(clip["jnt_type"][0]) == mujoco.mjtJoint.mjJNT_FREE else 0
    names, col_of, adr_of = [], [], []
    col = free_nq
    for jn, jt in zip(clip["joint_names"], clip["jnt_type"]):
        if int(jt) == mujoco.mjtJoint.mjJNT_FREE:
            continue
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid >= 0:
            names.append(jn)
            col_of.append(col)
            adr_of.append(model.jnt_qposadr[jid])
        col += 1

    # bodies the two models share, root first
    shared, body_ids, clip_body_idx = [], [], []
    for bi, bn in enumerate(clip["body_names"]):
        if bn == "world":
            continue
        mid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bn)
        if mid >= 0:
            shared.append(bn)
            body_ids.append(mid)
            clip_body_idx.append(bi)
    body_ids = np.asarray(body_ids)
    clip_body_idx = np.asarray(clip_body_idx)

    hinge_cols = np.asarray(col_of)
    frames = pick_frames(clip["qpos"][:, hinge_cols], n_frames)

    ref_pos = clip["xpos"][np.ix_(frames, clip_body_idx)]
    ref_pos = ref_pos - ref_pos[:, :1, :]
    ref_quat = clip["xquat"][np.ix_(frames, clip_body_idx)]

    def cost(signs):
        pos, quat = fk_poses(model, data, clip, frames, col_of, adr_of, signs,
                             body_ids)
        e_pos = float(np.linalg.norm(pos - ref_pos, axis=-1).mean())
        # geodesic angle between the two orientations; |dot| so q and -q agree
        dot = np.abs((quat * ref_quat).sum(axis=-1)).clip(0.0, 1.0)
        e_ang = float((2.0 * np.arccos(dot)).mean())
        # metres and radians summed directly: on a ~1 m robot a radian of link
        # error is the same order as a metre of it, and the position term
        # already dominates wherever it is not degenerate.
        return e_pos + e_ang

    signs = np.ones(len(names))
    best = cost(signs)
    if verbose:
        print(f"  start (all +1): cost {best:.6f}")
    for sweep in range(6):
        improved = False
        for j in range(len(names)):
            trial = signs.copy()
            trial[j] = -trial[j]
            c = cost(trial)
            if c < best - 1e-9:
                signs, best, improved = trial, c, True
        if verbose:
            print(f"  sweep {sweep + 1}: cost {best:.6f}  "
                  f"({int((signs < 0).sum())} negated)")
        if not improved:
            break
    return names, signs, best, len(frames)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--robots", nargs="+", default=["h1", "g1", "booster_t1"])
    p.add_argument("--clip", default="dance2_subject4.npz")
    p.add_argument("--clip-dir", type=Path, default=None, help="clip root (default external_data/amass_converted/LAFAN1)")
    p.add_argument("--frames", type=int, default=40)
    p.add_argument("--check", action="store_true",
                   help="compare against the tables in clip_reference.py")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    if args.clip_dir is not None:
        global CLIP_DIR
        CLIP_DIR = args.clip_dir

    known = {}
    if args.check:
        import ast
        src = (WORKSPACE / "loco_mjx" / "loco_mjx" / "environments" / "locomotion"
               / "urma2" / "mjx" / "clip_reference.py").read_text()
        for node in ast.walk(ast.parse(src)):
            if (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id.endswith("_CLIP_SIGNS")):
                known.update(ast.literal_eval(node.value))

    result, failures = {}, 0
    for robot in args.robots:
        print(f"\n=== {robot}")
        try:
            names, signs, resid, nf = derive(robot, args.clip, args.frames)
        except FileNotFoundError as exc:
            print(f"  SKIP (missing {exc})")
            continue
        table = {n: float(s) for n, s in zip(names, signs)}
        # A family whose clip the mjx model can reproduce differs from the source
        # ONLY in axis directions, which is the assumption a sign table encodes.
        # The three validated families land at 1.5-2.9e-4; anything far above
        # that differs in geometry too (different link lengths, joint offsets, or
        # closed chains) and no sign table can fix it. Measured: atlas 0.043,
        # talos 0.83, toddlerbot 2.40 -- toddlerbot barely improves on all-+1 at
        # all, which is what a 4-bar knee looks like to a hinge-by-hinge model.
        verdict = "USABLE" if resid < 0.01 else "NOT REPRODUCIBLE BY SIGNS ALONE"
        result[robot] = {"signs": table, "residual": resid, "frames": nf,
                         "verdict": verdict}
        if resid >= 0.01:
            failures += 1
            print(f"  VERDICT: {verdict} (residual {resid:.4f}, "
                  f"{500 * resid:.0f}x the validated families)")
        else:
            print(f"  VERDICT: {verdict}")
        print(f"  {len(names)} joints, {int((signs < 0).sum())} negated, "
              f"residual {resid:.6f} (m+rad) over {nf} frames")
        if args.check:
            overlap = [n for n in names if n in known]
            bad = [n for n in overlap if known[n] != table[n]]
            if not overlap:
                print("  no published table to check against")
            elif bad:
                failures += 1
                print(f"  MISMATCH on {len(bad)}/{len(overlap)}: {bad}")
            else:
                print(f"  MATCHES the published table on all {len(overlap)} joints")
    if args.out:
        args.out.write_text(json.dumps(result, indent=2))
        print(f"\nwrote {args.out}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
