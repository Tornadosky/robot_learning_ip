"""C15 -- per-body diagnostic video with the reward's own target sites drawn.

The goal document requires, for every evaluated body, a video in which the robot
and the exact targets the reward scores are visible together, and it is explicit
that the marker coordinates must come from the **same reference provider and
frame index as the reward** -- not from a separately reconstructed approximation.

C5 established that both `GoalTrajMimic` and `GoalTrajMimicv2` crash at render on
MuJoCo 3.9 (`mjv_initGeom` TypeError), so the upstream arrow path is unavailable.
This uses the mocap-sphere fallback the document permits: spheres are added to
the model as mocap bodies (no collision, no dynamics) and their positions are set
each frame from `traj.data.site_xpos[rel_site_ids]` -- literally the array
`MimicReward` reads.

Two modes per body:
  reference  kinematic playback of the body's own reference
  policy     a trained checkpoint rolled out deterministically

Runs on CPU with EGL; safe to run while a trainer holds the GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE / "scripts"))
sys.path.insert(0, str(WORKSPACE / "scripts" / "h1md"))

from c6_reward_discrimination import build_model, catalog, reground  # noqa: E402
from c8_reference_feasibility import reference_qvel  # noqa: E402
from c9_shared_policy import MIMIC_SITES, W_DANCE, make_complete_trajectory, register_variant  # noqa: E402

# distinct, consistent colours across bodies (goal document requirement)
SITE_COLOURS = {
    "upper_body_mimic": (0.95, 0.85, 0.20, 0.85),
    "left_hand_mimic": (0.20, 0.75, 0.95, 0.85),
    "right_hand_mimic": (0.10, 0.45, 0.90, 0.85),
    "left_foot_mimic": (0.95, 0.45, 0.30, 0.85),
    "right_foot_mimic": (0.85, 0.20, 0.25, 0.85),
}
SPHERE_R = 0.045


def add_target_spheres(xml_path: Path) -> Path:
    """Add one mocap sphere per mimic site. Visual only: no collision, no mass.

    Written next to the source XML on purpose: the model's mesh paths are
    relative (`assets/...`), so a copy in any other directory fails to load.
    """
    out_xml = xml_path.parent / "h1_with_targets.xml"
    spec = mujoco.MjSpec.from_file(str(xml_path))
    world = spec.worldbody
    for name in MIMIC_SITES:
        body = world.add_body()
        body.name = f"target_{name}"
        body.mocap = True
        body.pos = [0.0, 0.0, -5.0]  # parked below the floor until positioned
        geom = body.add_geom()
        geom.name = f"targetgeom_{name}"
        geom.type = mujoco.mjtGeom.mjGEOM_SPHERE
        geom.size = [SPHERE_R, 0.0, 0.0]
        geom.rgba = list(SITE_COLOURS[name])
        geom.contype = 0
        geom.conaffinity = 0
        geom.group = 2
        geom.mass = 0.0
    out_xml.write_text(spec.to_xml(), encoding="utf-8")
    mujoco.MjModel.from_xml_path(str(out_xml))
    return out_xml


def render_clip(model, qpos_seq, target_seq, out_path: Path, fps: int,
                width: int, height: int, label: str) -> dict:
    data = mujoco.MjData(model)
    mocap_ids = [model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"target_{n}")]
                 for n in MIMIC_SITES]

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance, cam.azimuth, cam.elevation = 3.6, 135, -12

    # cv2 rather than imageio: it is what this venv actually has, and what the
    # repo's existing render scripts use.
    import cv2

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cv2 could not open a writer for {out_path}")
    frames = 0
    try:
        with mujoco.Renderer(model, height=height, width=width) as renderer:
            for i, q in enumerate(qpos_seq):
                data.qpos[:] = q if len(q) == model.nq else np.pad(q, (0, model.nq - len(q)))
                for j, mid in enumerate(mocap_ids):
                    data.mocap_pos[mid] = target_seq[i, j]
                mujoco.mj_forward(model, data)
                cam.lookat[:] = [data.qpos[0], data.qpos[1], 0.9]
                renderer.update_scene(data, camera=cam)
                writer.write(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
                frames += 1
    finally:
        writer.release()
    return {"path": str(out_path), "frames": frames, "fps": fps, "label": label}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--video-dir", type=Path,
                    default=WORKSPACE / "experiments" / "h1_morphology_deepmimic_20260808" / "videos")
    ap.add_argument("--bodies", nargs="+", default=None, help="default: whole catalog")
    ap.add_argument("--start", type=int, default=19482)
    ap.add_argument("--frames", type=int, default=800)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--xml-root", type=Path,
                    default=WORKSPACE / "experiments" / "h1_morphology_deepmimic_20260808" / "body_catalog" / "xml")
    args = ap.parse_args()

    from loco_mujoco.task_factories import ImitationFactory, LAFAN1DatasetConf

    src = ImitationFactory.make("UnitreeH1", lafan1_dataset_conf=LAFAN1DatasetConf(["dance2_subject4"]))
    th = src.th.traj
    freq = float(th.info.frequency)
    qpos = np.asarray(th.data.qpos)[args.start:args.start + args.frames].astype(np.float64)
    qvel = np.asarray(th.data.qvel)[args.start:args.start + args.frames].astype(np.float64)
    dt = 1.0 / freq

    bodies = dict(catalog())
    names = args.bodies or list(bodies)
    records = []

    for name in names:
        model_plain = build_model(name, bodies[name], args.xml_root)
        xml_path = args.xml_root / f"h1_morphology_c2_{name}" / "h1.xml"
        cpu_name = f"C15Cpu_{name}"
        register_variant(cpu_name, xml_path, mjx=False)

        # the body's own reference, and the COMPLETE trajectory the reward reads
        q_ref, _ = reground(model_plain, qpos)
        v_ref = reference_qvel(qvel, q_ref, dt)
        traj = make_complete_trajectory(cpu_name, q_ref, v_ref, freq, th)

        # the exact array MimicReward indexes: traj site_xpos at the reward's site ids
        rel_site_ids = np.array([mujoco.mj_name2id(model_plain, mujoco.mjtObj.mjOBJ_SITE, n)
                                 for n in MIMIC_SITES])
        traj_site_xpos = np.asarray(traj.data.site_xpos)
        targets = traj_site_xpos[:, rel_site_ids, :]
        traj_qpos = np.asarray(traj.data.qpos)

        viz_xml = add_target_spheres(xml_path)
        viz_model = mujoco.MjModel.from_xml_path(str(viz_xml))

        sl = slice(None, None, args.stride)
        rec = render_clip(
            viz_model, traj_qpos[sl], targets[sl],
            args.video_dir / f"{name}_reference_with_targets.mp4",
            fps=max(1, int(round(freq / args.stride))),
            width=args.width, height=args.height,
            label=f"{name} reference + reward targets")
        rec.update({
            "body_id": name,
            "morphology": bodies[name],
            "mode": "reference",
            "target_source": "traj.data.site_xpos[rel_site_ids] -- the array MimicReward indexes",
            "reference_frequency_hz": float(traj.info.frequency),
            "reference_samples": int(traj.data.n_samples),
            "site_order": list(MIMIC_SITES),
        })
        records.append(rec)
        print(f"{name}: {rec['frames']} frames -> {rec['path']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "component": "C15_target_overlay_videos",
        "why_custom": "GoalTrajMimic and GoalTrajMimicv2 both crash at render on MuJoCo 3.9 (C5, Finding 21)",
        "overlay": "mocap spheres, visual only (contype=0, conaffinity=0, mass=0)",
        "colours": {k: list(v) for k, v in SITE_COLOURS.items()},
        "sphere_radius_m": SPHERE_R,
        "videos": records,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
