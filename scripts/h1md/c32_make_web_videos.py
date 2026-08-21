"""C32 -- render the analysis video set as small, web-embeddable animated WebP.

Every clip shows the robot together with the **exact target sites the reward
scores**, drawn as mocap spheres whose positions are read from
`traj.data.site_xpos[rel_site_ids]` -- the same array `MimicReward` indexes, at
the same frame. Nothing is reconstructed for display.

Camera framing differs from the earlier renders on purpose: it tracks the
midpoint between the robot and its targets and widens to keep both in shot, so
that when the policy drifts away from the reference you can see the gap rather
than losing the spheres off-frame.

Colours (consistent across every clip):
    yellow  upper_body_mimic      cyan  left_hand   blue  right_hand
    orange  left_foot             red   right_foot
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

from c6_reward_discrimination import build_model, catalog  # noqa: E402
from c8_reference_feasibility import reference_qvel  # noqa: E402
from c9_shared_policy import MIMIC_SITES, W_DANCE, make_complete_trajectory, register_variant  # noqa: E402
from c15_render_targets import add_target_spheres  # noqa: E402
from c15b_policy_videos import rollout_qpos  # noqa: E402
from refbuild import build_reference  # noqa: E402

XML_ROOT = WORKSPACE / "experiments" / "h1_morphology_deepmimic_20260808" / "body_catalog" / "xml"


def render_webp(model, qpos_seq, target_seq, out_path: Path, fps: int,
                width: int, height: int, quality: int = 42) -> dict:
    from PIL import Image

    data = mujoco.MjData(model)
    mocap_ids = [model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"target_{n}")]
                 for n in MIMIC_SITES]
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.azimuth, cam.elevation = 135, -14

    frames = []
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        for i, q in enumerate(qpos_seq):
            data.qpos[:] = q if len(q) == model.nq else np.pad(q, (0, model.nq - len(q)))
            tgt = target_seq[i] if target_seq is not None else None
            if tgt is not None:
                for j, mid in enumerate(mocap_ids):
                    data.mocap_pos[mid] = tgt[j]
            mujoco.mj_forward(model, data)

            # frame robot AND targets: look at their midpoint, widen with the gap
            robot_xy = np.array([data.qpos[0], data.qpos[1]])
            if tgt is not None:
                tgt_xy = tgt[:, :2].mean(axis=0)
                mid = 0.5 * (robot_xy + tgt_xy)
                gap = float(np.linalg.norm(robot_xy - tgt_xy))
            else:
                mid, gap = robot_xy, 0.0
            cam.lookat[:] = [mid[0], mid[1], 0.85]
            cam.distance = 3.4 + 1.25 * gap
            renderer.update_scene(data, camera=cam)
            frames.append(Image.fromarray(renderer.render()))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=int(round(1000 / fps)), loop=0, quality=quality, method=4)
    return {"path": str(out_path), "frames": len(frames), "fps": fps,
            "kb": out_path.stat().st_size // 1024}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--video-dir", type=Path,
                    default=WORKSPACE / "experiments" / "h1_morphology_deepmimic_20260808" / "videos" / "web")
    ap.add_argument("--width", type=int, default=440)
    ap.add_argument("--height", type=int, default=330)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--frames", type=int, default=800)
    args = ap.parse_args()

    from loco_mujoco.algorithms import PPOJax
    from loco_mujoco.task_factories import CustomDatasetConf, ImitationFactory, LAFAN1DatasetConf

    bodies = dict(catalog())
    CKPT = WORKSPACE / "experiments" / "h1_morphology_deepmimic_20260808" / "checkpoints"

    # (label, clip, window start, body, arm, mode)
    jobs = [
        ("ref_nominal_dance", "dance2_subject4", 19482, "body00_nominal", "fk", "reference"),
        ("ref_body04_dance", "dance2_subject4", 19482, "body04_seed1004", "fk", "reference"),
        ("policy_nominal_dance", "dance2_subject4", 19482, "body00_nominal", "ik_scaled_DANCE", "policy"),
        ("policy_body04_dance", "dance2_subject4", 19482, "body04_seed1004", "fk", "policy"),
        ("ref_nominal_walk", "walk1_subject1", 10521, "body00_nominal", "fk", "reference"),
        ("policy_nominal_walk", "walk1_subject1", 10521, "body00_nominal", "fk", "policy"),
    ]

    records = []

    # --- locomotion baseline: a different env family (RLFactory, velocity
    # command, no reference) so it renders on its own path and draws no spheres.
    loco_ckpt = CKPT / "c31_locomotion" / "PPOJax_saved.pkl"
    if loco_ckpt.exists():
        from loco_mujoco.task_factories import RLFactory

        lenv = RLFactory.make("UnitreeH1", headless=True, horizon=1000,
                              goal_type="GoalRandomRootVelocity",
                              reward_type="LocomotionReward")
        agent_conf, agent_state = PPOJax.load_agent(loco_ckpt)
        seq = rollout_qpos(lenv, agent_conf, agent_state, 400)
        nominal_xml = XML_ROOT / "h1_morphology_c2_body00_nominal" / "h1.xml"
        viz = mujoco.MjModel.from_xml_path(str(add_target_spheres(nominal_xml)))
        rec = render_webp(viz, seq[::2], None,
                          args.video_dir / "locomotion_baseline.webp",
                          fps=max(1, int(round(1.0 / lenv.dt / 2))),
                          width=args.width, height=args.height)
        travelled = float(np.linalg.norm(seq[-1, :2] - seq[0, :2]))
        rec.update({"label": "locomotion_baseline", "clip": "velocity command",
                    "body": "body00_nominal", "arm": "none", "mode": "control",
                    "note": "plain locomotion policy, no imitation",
                    "steps": int(len(seq)), "reference_travel_m": None,
                    "mean_root_error_m": None, "net_travel_m": travelled})
        records.append(rec)
        print(f"{'locomotion_baseline':24s} {rec['frames']:3d} frames  {rec['kb']:5d} KB  "
              f"travelled {travelled:.2f} m in {len(seq)} steps")
    else:
        print("locomotion_baseline: no checkpoint, skipped")

    for label, clip, start, body, arm, mode in jobs:
        src = ImitationFactory.make("UnitreeH1", lafan1_dataset_conf=LAFAN1DatasetConf([clip]))
        th = src.th.traj
        freq = float(th.info.frequency)
        dt = 1.0 / freq
        qpos = np.asarray(th.data.qpos)[start:start + args.frames].astype(np.float64)
        qvel = np.asarray(th.data.qvel)[start:start + args.frames].astype(np.float64)

        model_plain = build_model(body, bodies[body], XML_ROOT)
        xml_path = XML_ROOT / f"h1_morphology_c2_{body}" / "h1.xml"
        env_name = f"C32_{label}"
        register_variant(env_name, xml_path, mjx=False)

        # `_DANCE` is only a checkpoint-directory suffix, not a reference arm
        q_ref = build_reference(body, arm.replace("_DANCE", ""), qpos, XML_ROOT)
        v_ref = reference_qvel(qvel, q_ref, dt)
        traj = make_complete_trajectory(env_name, q_ref, v_ref, freq, th)

        rel_site_ids = np.array([mujoco.mj_name2id(model_plain, mujoco.mjtObj.mjOBJ_SITE, n)
                                 for n in MIMIC_SITES])
        targets = np.asarray(traj.data.site_xpos)[:, rel_site_ids, :]

        if mode == "reference":
            seq = np.asarray(traj.data.qpos)
            note = "kinematic playback of the reference"
        else:
            ckpt = None
            for cand in (CKPT / f"{body}__{arm}" / "PPOJax_saved.pkl",
                         CKPT / f"{body}__{arm}" / "checkpoint_final" / "PPOJax_saved.pkl"):
                if cand.exists():
                    ckpt = cand
                    break
            if ckpt is None:
                print(f"{label}: no checkpoint, skipped")
                continue
            env = ImitationFactory.make(
                env_name, custom_dataset_conf=CustomDatasetConf(traj),
                headless=True, horizon=1000,
                th_params=dict(random_start=False, fixed_start_conf=(0, 0)),
                goal_type="GoalTrajMimic", goal_params=dict(visualize_goal=False),
                reward_type="MimicReward",
                reward_params=dict(**W_DANCE, sites_for_mimic=MIMIC_SITES))
            agent_conf, agent_state = PPOJax.load_agent(ckpt)
            seq = rollout_qpos(env, agent_conf, agent_state, min(args.frames, len(targets)))
            note = f"deterministic rollout of the trained policy ({len(seq)} steps)"

        viz = mujoco.MjModel.from_xml_path(str(add_target_spheres(xml_path)))
        sl = slice(None, None, args.stride)
        rec = render_webp(viz, seq[sl], targets[: len(seq)][sl],
                          args.video_dir / f"{label}.webp",
                          fps=max(1, int(round(freq / args.stride))),
                          width=args.width, height=args.height)
        # honest scale: how far did the reference actually travel?
        ref_travel = float(np.linalg.norm(np.asarray(traj.data.qpos)[len(seq) - 1, :2]
                                          - np.asarray(traj.data.qpos)[0, :2]))
        root_err = float(np.linalg.norm(seq[:, :2] - np.asarray(traj.data.qpos)[:len(seq), :2],
                                        axis=1).mean())
        rec.update({"label": label, "clip": clip, "body": body, "arm": arm, "mode": mode,
                    "note": note, "steps": int(len(seq)),
                    "reference_travel_m": ref_travel, "mean_root_error_m": root_err})
        records.append(rec)
        print(f"{label:24s} {rec['frames']:3d} frames  {rec['kb']:5d} KB  "
              f"root err {root_err:.2f} m vs travel {ref_travel:.2f} m")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"component": "C32_web_videos", "videos": records},
                                   indent=2), encoding="utf-8")
    total = sum(r["kb"] for r in records)
    print(f"\ntotal {total} KB across {len(records)} clips -> {args.out}")


if __name__ == "__main__":
    main()
