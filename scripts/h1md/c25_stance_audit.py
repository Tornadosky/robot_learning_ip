"""C25 -- is the missing stance phase a retargeting defect or a property of the clip?

Finding 33: the frozen dance window has a genuine stance foot (low AND slow) in
only 10.9% of frames, and as retargeted never brings a foot within 5 mm of the
floor at all. That is the binding constraint on the whole pipeline -- a robot
cannot translate without a stance foot -- so it matters a great deal whether it
is a property of *this dance* or of *the retargeting*.

A walking clip is the discriminator. Walking is ~100% single- or double-support:
if walk1_subject1 also shows ~11% stance and floats, the retargeting pipeline is
producing contact-inconsistent references for everything, and that is the first
thing to fix. If walk looks healthy, the dance clip is simply airborne a lot and
its root trajectory may be genuinely hard rather than impossible.

Reported per clip: ground clearance as retargeted, stance fraction before and
after per-frame re-grounding, and the speed of the "contacting" foot.

CPU only.
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

LOW_M = 0.005
SLOW_MS = 0.10


def stance_stats(model, qpos, dt) -> dict:
    data = mujoco.MjData(model)
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    feet = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("left_foot", "right_foot")]
    sites = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n)
             for n in ("left_foot_mimic", "right_foot_mimic")]
    H = np.zeros((len(qpos), 2))
    XY = np.zeros((len(qpos), 2, 2))
    for i, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        for j, f in enumerate(feet):
            H[i, j] = mujoco.mj_geomDistance(model, data, floor, f, 10.0, None)
        XY[i] = data.site_xpos[sites][:, :2]
    V = np.zeros_like(H)
    V[1:] = np.linalg.norm(np.diff(XY, axis=0), axis=-1) / dt

    low = H < LOW_M
    stance = low & (V < SLOW_MS)
    return {
        "min_foot_floor_m": float(H.min()),
        "median_foot_floor_m": float(np.median(H.min(axis=1))),
        "frac_foot_low": float(low.mean()),
        "frac_stance": float(stance.mean()),
        "frac_frames_with_any_stance": float(stance.any(axis=1).mean()),
        "median_speed_while_low_ms": float(np.median(V[low])) if low.any() else None,
        "root_travel_m": float(np.linalg.norm(qpos[-1, :2] - qpos[0, :2])),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--clips", nargs="+",
                    default=["dance2_subject4", "walk1_subject1", "dance2_subject1"])
    ap.add_argument("--frames", type=int, default=800)
    ap.add_argument("--xml-root", type=Path,
                    default=WORKSPACE / "experiments" / "h1_morphology_deepmimic_20260808" / "body_catalog" / "xml")
    args = ap.parse_args()

    from loco_mujoco.task_factories import ImitationFactory, LAFAN1DatasetConf

    bodies = dict(catalog())
    model = build_model("body00_nominal", bodies["body00_nominal"], args.xml_root)

    results = {}
    for clip in args.clips:
        env = ImitationFactory.make("UnitreeH1", lafan1_dataset_conf=LAFAN1DatasetConf([clip]))
        th = env.th.traj
        freq = float(th.info.frequency)
        dt = 1.0 / freq
        full = np.asarray(th.data.qpos).astype(np.float64)
        # the frozen window for the dance, an equivalent-length window elsewhere
        start = 19482 if clip == "dance2_subject4" else max(0, len(full) // 2 - args.frames // 2)
        win = full[start:start + args.frames]

        raw = stance_stats(model, win, dt)
        grounded = stance_stats(model, reground(model, win)[0], dt)
        whole = stance_stats(model, full[:: max(1, len(full) // 2000)], dt * max(1, len(full) // 2000))

        results[clip] = {
            "frequency_hz": freq, "window_start": start, "window_frames": int(len(win)),
            "raw_window": raw, "regrounded_window": grounded, "whole_clip_subsampled": whole,
        }
        print(f"{clip}")
        print(f"   raw window     : min floor {raw['min_foot_floor_m']:+.4f} m, "
              f"stance {raw['frac_frames_with_any_stance']:.3f}")
        print(f"   regrounded win : stance {grounded['frac_frames_with_any_stance']:.3f}, "
              f"median speed while low {grounded['median_speed_while_low_ms']:.2f} m/s")
        print(f"   whole clip raw : min floor {whole['min_foot_floor_m']:+.4f} m, "
              f"median clearance {whole['median_foot_floor_m']:+.4f} m, "
              f"stance {whole['frac_frames_with_any_stance']:.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "component": "C25_stance_audit",
        "question": "is the missing stance phase a retargeting defect or a property of the dance?",
        "definitions": {"low_m": LOW_M, "slow_ms": SLOW_MS},
        "clips": results,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
