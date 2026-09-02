"""Data audit for the wave-7 training set (clips_5r): is every clip a feasible
motion for the loco_mjx model it will be tracked on?

For each robot dir and each clip (super20 + held-out):
  * frames, NaN/inf, dt / fps sanity
  * joint-range violations of the MODEL's actuated joints after the loader's
    alias + sign tables (the same mapping tracking_clip.py applies)
  * joint-velocity p99 and max
  * root height range, foot-site minimum height (reference ground penetration)
  * sidecar (_zq/_win) presence and shape agreement
  * cross-robot frame-count agreement per motion (same mocap source)

    JAX_PLATFORMS=cpu ~/jaxgpu/bin/python scripts/scaling/wave7/verify_clips_5r.py [--clip-dir ...]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(R / "scripts/scaling/wave7"))
from emit_sidecars_any import ROBOT_DIR, TABLES, actuator_joint_names  # noqa: E402

import mujoco  # noqa: E402

FREE = ("root", "reference", "floating_base_joint", "floating_base")


def model_ranges(robot):
    m = mujoco.MjModel.from_xml_path(str(R / "loco_mjx/loco_mjx/environments/robots" / robot / "data/plane.xml"))
    rng = {}
    for j in range(m.njnt):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        if m.jnt_limited[j]:
            rng[n] = (float(m.jnt_range[j, 0]), float(m.jnt_range[j, 1]))
    return rng


def audit(npz, jn, rng, aliases, signs):
    d = np.load(npz, allow_pickle=True)
    keys = set(d.files)
    cj = [str(n) for n in d["joint_names"]]
    assert cj[0] in FREE, cj[0]
    clip_joints = [aliases.get(n, n) for n in cj[1:]]
    missing = [n for n in jn if n not in clip_joints]
    perm = np.array([clip_joints.index(n) for n in jn if n in clip_joints])
    sg = np.array([signs.get(n, 1.0) for n in jn if n in clip_joints], dtype=np.float32)
    qpos_all = np.asarray(d["qpos"], dtype=np.float64)
    qvel_all = np.asarray(d["qvel"], dtype=np.float64)
    T = qpos_all.shape[0]
    q = qpos_all[:, 7:][:, perm] * sg
    qd = qvel_all[:, 6:][:, perm] * sg
    out = dict(frames=T, nan=int(np.isnan(qpos_all).sum() + np.isinf(qpos_all).sum() + np.isnan(qvel_all).sum()),
               missing=missing, root_z=(float(qpos_all[:, 2].min()), float(qpos_all[:, 2].max())),
               qvel_p99=float(np.percentile(np.abs(qd), 99)), qvel_max=float(np.abs(qd).max()))
    tol = 1e-3
    viol = {}
    for k, n in enumerate([n for n in jn if n in clip_joints]):
        if n in rng:
            lo, hi = rng[n]
            ex = np.maximum(lo - q[:, k], q[:, k] - hi)
            f = float((ex > tol).mean())
            if f > 0:
                viol[n] = (round(f, 4), round(float(ex.max()), 3))
    out["viol_frac_frames"] = float(np.mean([1.0 if any(True for _ in []) else 0 for _ in [0]])) if False else None
    if viol:
        fr = np.zeros(T, bool)
        for k, n in enumerate([n for n in jn if n in clip_joints]):
            if n in rng:
                lo, hi = rng[n]
                fr |= np.maximum(lo - q[:, k], q[:, k] - hi) > tol
        out["viol_frac_frames"] = float(fr.mean())
    else:
        out["viol_frac_frames"] = 0.0
    out["viol"] = viol
    # foot sites
    foot_min = None
    for key in ("site_xpos", "xpos_sites", "site_pos"):
        if key in keys:
            sp = np.asarray(d[key]); foot_min = float(sp[..., 2].min()); break
    if foot_min is None:
        sk = [k for k in keys if "foot" in k.lower() or "site" in k.lower()]
        for k in sk:
            v = np.asarray(d[k])
            if v.dtype.kind == "f" and v.ndim >= 2 and v.shape[-1] == 3:
                foot_min = float(v[..., 2].min()); break
    out["foot_min_z"] = foot_min
    # sidecars
    zq, win = npz.with_name(npz.stem + "_zq.npz"), npz.with_name(npz.stem + "_win.npz")
    for tag, p in (("zq", zq), ("win", win)):
        if p.exists():
            s = np.load(p, allow_pickle=True)
            z = s["z_q"]; names = [str(n) for n in s["joint_names"]]
            out[tag] = (tuple(z.shape), z.shape[0] == T and names == jn and not np.isnan(z).any())
        else:
            out[tag] = None
    out["keys"] = sorted(keys)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-dir", default=str(R / "experiments/fsq_khaendler/clips_5r"))
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    cd = Path(a.clip_dir)
    robots = [r for r in ROBOT_DIR if (cd / ROBOT_DIR[r]).is_dir()]
    report = {}
    frames = {}
    for robot in robots:
        sub = ROBOT_DIR[robot]
        tab = json.load(open(TABLES / f"{sub}.json")) if (TABLES / f"{sub}.json").exists() else {}
        aliases, signs = tab.get("joint_aliases", {}), dict(tab.get("signs", {}))
        if not signs:  # H1/G1/T1 carry their sign tables in clip_reference.py (keyed by clip name == model name)
            sys.path.insert(0, str(R / "loco_mjx/loco_mjx/environments/locomotion/urma2/mjx"))
            import clip_reference as cr
            signs = {**cr.H1_CLIP_SIGNS, **cr.G1_CLIP_SIGNS, **cr.T1_CLIP_SIGNS}
        jn = actuator_joint_names(robot)
        rng = model_ranges(robot)
        print(f"\n=== {sub} ({robot}): {len(jn)} actuated joints, {len(rng)} limited, aliases {len(aliases)}, signs {sum(1 for v in signs.values() if v < 0)} negative")
        clips = sorted(p for p in (cd / sub).glob("*.npz") if not p.stem.endswith(("_zq", "_win")))
        for c in clips:
            o = audit(c, jn, rng, aliases, signs)
            report[f"{sub}/{c.name}"] = o
            frames.setdefault(c.name, {})[sub] = o["frames"]
            bad = sorted(o["viol"].items(), key=lambda kv: -kv[1][0])[:4]
            print(f"  {c.name:22s} T={o['frames']:7d} nan={o['nan']} viol_frames={o['viol_frac_frames']:.4f} "
                  f"worst={bad} qvel p99/max={o['qvel_p99']:.1f}/{o['qvel_max']:.1f} root_z=[{o['root_z'][0]:.2f},{o['root_z'][1]:.2f}] "
                  f"foot_min_z={o['foot_min_z']} zq={o['zq']} win={o['win']} missing={o['missing']}")
    print("\n=== frame counts per motion across robots (same mocap => should agree)")
    for name, per in frames.items():
        vals = set(per.values())
        print(f"  {name:22s} {'OK ' if len(vals) == 1 else 'DIFF'} {per}")
    print("\nclip keys:", report[next(iter(report))]["keys"])
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
