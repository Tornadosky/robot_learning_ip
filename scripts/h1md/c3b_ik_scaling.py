"""C3b -- Phase 6 scalability curve for the winning reference method.

C3 showed `ik_scaled` is the only IK construction that produces a trackable
reference. This measures what it costs to produce that reference for many
bodies: warm throughput versus body count, plus the projected wall time for the
1 000-body target the goal document names.

The number that matters is not "seconds per body" in isolation but whether
reference production can hide behind training. C4 measured the trainer at
73 655 steps/s, so this reports the ratio directly.

Run under WSL dance_env with the GPU visible.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE / "scripts"))
sys.path.insert(0, str(WORKSPACE / "scripts" / "h1md"))

from c3_reference_methods import IKKernel, LIMB_SCALE_INDEX  # noqa: E402

TRAINER_STEPS_PER_S = 73_655.0  # measured in C4 on this GPU


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--start", type=int, default=19482)
    ap.add_argument("--frames", type=int, default=800)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--body-counts", type=int, nargs="+", default=[1, 4, 8, 32, 128, 512])
    ap.add_argument("--target-bodies", type=int, default=1000)
    ap.add_argument("--target-frames", type=int, default=3000)
    ap.add_argument("--xml-root", type=Path,
                    default=WORKSPACE / "experiments" / "h1_morphology_deepmimic_20260808" / "body_catalog" / "xml")
    args = ap.parse_args()

    from loco_mujoco.task_factories import ImitationFactory, LAFAN1DatasetConf

    env = ImitationFactory.make("UnitreeH1", lafan1_dataset_conf=LAFAN1DatasetConf(["dance2_subject4"]))
    th = env.th.traj
    qpos = np.asarray(th.data.qpos)[args.start:args.start + args.frames][:: args.stride].astype(np.float32)
    T = len(qpos)
    Q = jnp.asarray(qpos)

    kern = IKKernel(args.xml_root / "h1_morphology_c2_body00_nominal" / "h1.xml")
    nom_sites = np.asarray(kern.fk_bt(jnp.ones((1, 3), jnp.float32), Q))[0]
    nom_targets = jnp.asarray(nom_sites[:, kern.TARGET_SITES, :])
    root_nom = jnp.asarray(qpos[:, 0:3])

    def scaled_targets(morph):
        rel = nom_targets - root_nom[:, None, :]
        s = morph[LIMB_SCALE_INDEX][None, :, None]
        root = root_nom.at[:, 2].multiply(morph[0])
        return root[:, None, :] + rel * s

    rows = []
    for B in args.body_counts:
        morphs = jnp.asarray(
            np.random.default_rng(B).uniform([0.85] * 3, [1.20] * 3, size=(B, 3)).astype(np.float32))
        tgt = jax.vmap(scaled_targets)(morphs)
        try:
            t0 = time.perf_counter()
            xs, rs = kern.ik(morphs, Q, tgt)
            xs.block_until_ready()
            cold = time.perf_counter() - t0

            t0 = time.perf_counter()
            xs, rs = kern.ik(morphs, Q, tgt)
            xs.block_until_ready()
            warm = time.perf_counter() - t0
        except Exception as exc:
            rows.append({"bodies": B, "frames": T, "error": f"{type(exc).__name__}: {exc}"})
            print(f"B={B}: FAILED {type(exc).__name__}")
            continue

        err = np.linalg.norm(np.asarray(rs).reshape(B, T, 4, 3), axis=-1)
        cells = B * T
        rows.append({
            "bodies": B, "frames": T, "cells": cells,
            "cold_s": cold, "warm_s": warm,
            "cells_per_second": cells / warm,
            "seconds_per_body": warm / B,
            "residual_cm_mean": float(err.mean() * 100),
            "residual_cm_p99": float(np.percentile(err, 99) * 100),
        })
        print(f"B={B:5d}  warm {warm:7.3f} s  {cells / warm:10,.0f} cells/s  "
              f"{warm / B * 1000:8.2f} ms/body  residual {err.mean() * 100:.4f} cm")

    ok = [r for r in rows if "cells_per_second" in r]
    best = max(ok, key=lambda r: r["cells_per_second"])
    target_cells = args.target_bodies * args.target_frames
    projected_s = target_cells / best["cells_per_second"]

    # How long does the trainer run for one morphology refresh round, and does
    # the reference production for the next pool fit inside it?
    projection = {
        "best_throughput_cells_per_second": best["cells_per_second"],
        "best_at_body_count": best["bodies"],
        "target": {"bodies": args.target_bodies, "frames": args.target_frames, "cells": target_cells},
        "projected_seconds": projected_s,
        "projected_minutes": projected_s / 60,
        "trainer_steps_per_second_c4": TRAINER_STEPS_PER_S,
        "training_steps_produced_in_that_time": TRAINER_STEPS_PER_S * projected_s,
        "note": (
            "if a morphology refresh round trains for more steps than the value above, "
            "reference production for the next pool is fully hidden behind training"
        ),
    }
    print(f"\nprojected {args.target_bodies} bodies x {args.target_frames} frames: "
          f"{projected_s:.1f} s ({projected_s / 60:.2f} min) at best throughput")
    print(f"the trainer produces {TRAINER_STEPS_PER_S * projected_s:,.0f} steps in that time")

    out = {
        "component": "C3b_ik_scaling",
        "method": "ik_scaled (the only IK construction that passed the C3 validity gates)",
        "backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
        "frames_per_body": T,
        "rows": rows,
        "projection": projection,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
