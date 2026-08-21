"""C27 -- stance fraction as a clip-selection criterion, and its sensitivity to morphology.

Finding 34 showed the pipeline tracks world-space motion only as well as the
reference's contact schedule allows: walking re-grounds to 91.6% stance, the
frozen dance to 10.9%. That makes stance fraction a **screening metric** — a
property you can measure in seconds, before spending any GPU, that predicts
whether a clip can support a spatial-tracking claim at all.

Two questions here:

1. **Which of the available clips are suitable?** Screen every LAFAN1 clip in the
   local cache and rank them.
2. **Does morphology change the answer?** If a longer-legged body has a
   materially different contact schedule on the same clip, then contact timing
   is body-specific and per-body references need to account for it — a real
   scaling consideration. If not, one screening pass per clip serves every body.

CPU only, seconds per clip.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE / "scripts"))
sys.path.insert(0, str(WORKSPACE / "scripts" / "h1md"))

from c6_reward_discrimination import build_model, catalog, reground  # noqa: E402
from c25_stance_audit import stance_stats  # noqa: E402

XML_ROOT = WORKSPACE / "experiments" / "h1_morphology_deepmimic_20260808" / "body_catalog" / "xml"


def available_clips() -> list[str]:
    import loco_mujoco
    v = yaml.safe_load(open(loco_mujoco.PATH_TO_VARIABLES))
    d = Path(v["LOCOMUJOCO_CONVERTED_LAFAN1_PATH"]) / "UnitreeH1"
    return sorted(p.stem for p in d.glob("*.npz"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frames", type=int, default=800)
    ap.add_argument("--morph-clip", default="dance2_subject4",
                    help="clip used for the across-bodies sensitivity check")
    args = ap.parse_args()

    from loco_mujoco.task_factories import ImitationFactory, LAFAN1DatasetConf

    bodies = dict(catalog())
    nominal = build_model("body00_nominal", bodies["body00_nominal"], XML_ROOT)

    clips = available_clips()
    print(f"screening {len(clips)} clips\n")
    per_clip = {}
    for clip in clips:
        try:
            env = ImitationFactory.make("UnitreeH1", lafan1_dataset_conf=LAFAN1DatasetConf([clip]))
        except Exception as exc:
            print(f"{clip:26s} SKIP ({type(exc).__name__})")
            continue
        th = env.th.traj
        freq = float(th.info.frequency)
        full = np.asarray(th.data.qpos).astype(np.float64)
        n = min(args.frames, len(full))
        start = 19482 if clip == "dance2_subject4" and len(full) > 20282 else max(0, len(full) // 2 - n // 2)
        win = full[start:start + n]
        s = stance_stats(nominal, reground(nominal, win)[0], 1.0 / freq)
        per_clip[clip] = {"frequency_hz": freq, "n_samples": int(len(full)),
                          "window_start": int(start), "window_frames": int(len(win)), **s}
        print(f"{clip:26s} stance {s['frac_frames_with_any_stance']:.3f} | "
              f"contact-foot speed {s['median_speed_while_low_ms']:.2f} m/s | "
              f"root travel {s['root_travel_m']:.2f} m")

    ranked = sorted(per_clip.items(), key=lambda kv: -kv[1]["frac_frames_with_any_stance"])
    print("\nranked by stance fraction (suitability for a spatial-tracking claim):")
    for name, s in ranked:
        verdict = ("suitable" if s["frac_frames_with_any_stance"] >= 0.5
                   else "marginal" if s["frac_frames_with_any_stance"] >= 0.25 else "UNSUITABLE")
        print(f"  {s['frac_frames_with_any_stance']:.3f}  {name:26s} {verdict}")

    # sensitivity of the contact schedule to morphology
    print(f"\nmorphology sensitivity on {args.morph_clip}:")
    env = ImitationFactory.make("UnitreeH1", lafan1_dataset_conf=LAFAN1DatasetConf([args.morph_clip]))
    th = env.th.traj
    freq = float(th.info.frequency)
    full = np.asarray(th.data.qpos).astype(np.float64)
    start = 19482 if len(full) > 20282 else 0
    win = full[start:start + args.frames]
    per_body = {}
    for name, morph in catalog():
        model = build_model(name, morph, XML_ROOT)
        s = stance_stats(model, reground(model, win)[0], 1.0 / freq)
        per_body[name] = {"morphology": morph, **s}
        print(f"  {name:22s} stance {s['frac_frames_with_any_stance']:.3f} | "
              f"speed {s['median_speed_while_low_ms']:.2f} m/s")
    vals = [v["frac_frames_with_any_stance"] for v in per_body.values()]
    spread = max(vals) - min(vals)
    print(f"  spread across bodies: {spread:.3f} "
          f"({'body-specific' if spread > 0.10 else 'morphology-insensitive'})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "component": "C27_clip_selection",
        "screening_metric": "fraction of frames with a foot within 5 mm AND moving < 10 cm/s, "
                            "after per-frame re-grounding",
        "thresholds": {"suitable": 0.5, "marginal": 0.25},
        "per_clip": per_clip,
        "morphology_sensitivity": {"clip": args.morph_clip, "per_body": per_body,
                                   "stance_spread": spread},
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
