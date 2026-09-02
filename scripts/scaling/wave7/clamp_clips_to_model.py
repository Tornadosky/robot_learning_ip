"""Clamp clip joint angles to the loco_mjx MODEL's joint ranges, in the model's
sign convention (loader alias + sign tables applied, then inverted on write).

Why: the feasibility re-issue (reissue_clips.py) enforces limits through soft
joint-limit constraints in loco-mujoco's model; on Apollo the shoulder internal
rotation still overshoots the +-0.47 rad range by up to 0.2 rad on ~25 % of
frames, Atlas ankles/shoulders by <0.1 rad on ~6 %. An unreachable target is a
constant tracking error the policy cannot remove, so clamp. qvel is left as is.
Sidecars (_zq/_win) must be re-emitted afterwards (emit_sidecars_any.py).

    python scripts/scaling/wave7/clamp_clips_to_model.py --clip-dir experiments/fsq_khaendler/clips_5r --robots atlas apptronik_apollo
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(R / "scripts/scaling/wave7"))
from emit_sidecars_any import ROBOT_DIR, TABLES, actuator_joint_names  # noqa: E402
from verify_clips_5r import model_ranges  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-dir", default=str(R / "experiments/fsq_khaendler/clips_5r"))
    ap.add_argument("--robots", nargs="+", required=True)
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    cd = Path(a.clip_dir)
    for robot in a.robots:
        sub = ROBOT_DIR[robot]
        tab = json.load(open(TABLES / f"{sub}.json")) if (TABLES / f"{sub}.json").exists() else {}
        aliases, signs = tab.get("joint_aliases", {}), dict(tab.get("signs", {}))
        if not signs:
            sys.path.insert(0, str(R / "loco_mjx/loco_mjx/environments/locomotion/urma2/mjx"))
            import clip_reference as cr
            signs = {**cr.H1_CLIP_SIGNS, **cr.G1_CLIP_SIGNS, **cr.T1_CLIP_SIGNS}
        jn = actuator_joint_names(robot)
        rng = model_ranges(robot)
        for c in sorted(p for p in (cd / sub).glob("*.npz") if not p.stem.endswith(("_zq", "_win"))):
            d = dict(np.load(c, allow_pickle=True))
            cj = [str(n) for n in d["joint_names"]]
            clip_joints = [aliases.get(n, n) for n in cj[1:]]
            qpos = np.array(d["qpos"], dtype=np.float64)
            changed = 0; worst = 0.0; per = {}
            for n in jn:
                if n not in clip_joints or n not in rng:
                    continue
                col = 7 + clip_joints.index(n)
                sg = float(signs.get(n, 1.0))
                lo, hi = rng[n]
                v = qpos[:, col] * sg
                w = np.clip(v, lo + a.margin, hi - a.margin)
                dv = np.abs(w - v)
                if dv.max() > 1e-9:
                    changed += int((dv > 1e-3).sum()); worst = max(worst, float(dv.max())); per[n] = float((dv > 1e-3).mean())
                    qpos[:, col] = w * sg
            T = qpos.shape[0]
            top = sorted(per.items(), key=lambda kv: -kv[1])[:3]
            print(f"{sub}/{c.name}: T={T} clamped samples={changed} worst={worst:.3f} rad top={[(k, round(f, 3)) for k, f in top]}")
            if not a.dry and changed:
                d["qpos"] = qpos.astype(d["qpos"].dtype)
                tmp = c.with_suffix(".tmp.npz")
                np.savez(tmp, **d)
                tmp.replace(c)


if __name__ == "__main__":
    main()
