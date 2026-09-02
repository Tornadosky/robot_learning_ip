#!/usr/bin/env python3
"""A3 window-ablation sidecars via TIME-SHIFT (no tokenizer retraining).

The production tokenizer encodes a FUTURE-only window [t .. t+10] (lookahead=10
at 40 fps ~ 250 ms). A token stream whose window is centered or past-only is the
same stream re-indexed:

    past-only[t]  = future[t-10]   (window covers [t-10 .. t])
    centered[t]   = future[t-5]    (window covers [t-5 .. t+5])

Frames before the shift is available repeat the first token (affects <=10 of
~9000 frames). Emits design-B dirs (ORIGINAL npz + shifted z_q) from the local
mc mirror, ready for upload to Viper.

Run: .venv/Scripts/python.exe scripts/scaling/make_a3_shifted_sidecars.py
"""
import shutil
from pathlib import Path

import numpy as np

SRC = Path("experiments/fsq_khaendler/tokentest_local_mc")
OUTS = {  # name -> shift k: out[t] = src[max(t-k, 0)]
    "tokentest_mc_cent": 5,
    "tokentest_mc_past": 10,
}
CLIP = "dance2_subject4"

for name, k in OUTS.items():
    out = SRC.parent / name
    for robot_dir in sorted(SRC.iterdir()):
        if not robot_dir.is_dir():
            continue
        od = out / robot_dir.name
        od.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(robot_dir / f"{CLIP}.npz", od / f"{CLIP}.npz")
        d = np.load(robot_dir / f"{CLIP}_zq.npz")
        z = d["z_q"]
        idx = np.maximum(np.arange(z.shape[0]) - k, 0)
        np.savez(od / f"{CLIP}_zq.npz", z_q=z[idx], joint_names=d["joint_names"])
        print(f"{name}/{robot_dir.name}: z_q {z.shape} shifted by {k}")
    # Sanity: shifted stream is the original delayed by k.
    a = np.load(SRC / "UnitreeH1" / f"{CLIP}_zq.npz")["z_q"]
    b = np.load(out / "UnitreeH1" / f"{CLIP}_zq.npz")["z_q"]
    assert np.array_equal(b[k:], a[:-k]) and np.array_equal(b[0], a[0])
    print(f"{name}: verified shift={k}")
print("DONE")
