"""Stage-3 for tall&light G1: train on the SMPL-RETARGETED reference, not the raw clip.

Stages 1-2 (uniform gains, torque, exploration, ankle/knee stiffening -- 16
recipes) all stayed stuck ~110, training on the RAW LAFAN1 clip constant-grounded
onto the body. But the raw clip is the joint trajectory of a NOMINAL-proportion
dancer; on a body with 1.55x legs / 0.55x torso it places the feet and CoM in
near-infeasible spots, so no gain can balance it. The SMPL-retargeted reference
instead re-solves the motion FOR this body's proportions (balance-feasible target,
same dance). This drops --raw-reference so the trainer builds/uses that reference.

Same early-stop protocol (60M probe; extend the first to climb out; stop at the
first crossing).

Run inside WSL conda env locodm (GPU JAX + CPU torch/SMPL for the retarget):
    python run_g1_tall_retarget.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent / "external_data" / "deepmimic_morphology" / "g1" / "dance2_subject4"
PRESET = "extreme_tall_light"

PROBE_STEPS, PROBE_CKPTS = "60e6", "4"
FULL_STEPS, FULL_CKPTS = "120e6", "8"
PROMISING_LEN, CROSS_LEN = 250.0, 330.0

# All use the retargeted reference (NO --raw-reference). Vary only the gains.
CONFIGS = [
    ("retgt_3x",     ["--control", "pd", "--pd-gain-scale", "3.0", "--lr", "3e-4", "--init-std", "0.2"]),
    ("retgt_stock",  ["--control", "pd", "--pd-gain-scale", "1.0", "--lr", "3e-4", "--init-std", "0.2"]),
    ("retgt_ankle3", ["--control", "pd", "--pd-gain-scale", "1.0", "--lr", "3e-4",
                      "--init-std", "0.2", "--ankle-gain-scale", "3.0"]),
]


def best_checkpoint(manifest: dict) -> dict:
    return max(manifest["checkpoints"], key=lambda c: c["mean_episode_return"])


def run_train(tag: str, extra: list[str], steps: str, ckpts: str, suffix: str) -> Path | None:
    # NOTE: no --raw-reference -> trainer retargets (cached after the first run).
    cmd = [
        sys.executable, "train_deepmimic_morphology.py",
        "--robot", "g1", "--preset", PRESET, "--clip", "dance2_subject4",
        "--duration", "30", "--num-envs", "2048",
        "--total-timesteps", steps, "--num-checkpoints", ckpts,
        "--seed", "1", "--out-suffix", suffix,
    ] + extra
    print(f"\n$ {' '.join(cmd)}", flush=True)
    rc = subprocess.run(cmd, cwd=str(SCRIPTS)).returncode
    out_dir = ROOT / f"{PRESET}__{suffix}"
    if rc != 0 or not (out_dir / "manifest.json").exists():
        print(f"[retgt] {tag} ({suffix}) FAILED rc={rc}", flush=True)
        return None
    return out_dir


def main() -> None:
    results = {}
    winner = None
    for tag, extra in CONFIGS:
        suffix = f"rt_{tag}"
        out_dir = run_train(tag, extra, PROBE_STEPS, PROBE_CKPTS, suffix)
        if out_dir is None:
            results[tag] = {"status": "failed"}
            continue
        m = json.loads((out_dir / "manifest.json").read_text())
        ck = best_checkpoint(m)
        lens = [c["mean_episode_length"] for c in m["checkpoints"]]
        best_len = ck["mean_episode_length"]
        print(f"[retgt] PROBE {tag}: best R={ck['mean_episode_return']:.0f} "
              f"len={best_len:.0f} (curve {'/'.join('%.0f' % l for l in lens)})", flush=True)

        if best_len >= CROSS_LEN:
            results[tag] = {"status": "crossed_at_probe", "best_return": ck["mean_episode_return"],
                            "best_length": best_len, "suffix": suffix}
            winner = (tag, suffix)
            print(f"[retgt] {tag} CROSSED at probe (len {best_len:.0f}) -> winner", flush=True)
            break

        if best_len >= PROMISING_LEN:
            print(f"[retgt] {tag} promising (len {best_len:.0f}); extending to {FULL_STEPS} ...",
                  flush=True)
            shutil.rmtree(out_dir, ignore_errors=True)
            full = run_train(tag, extra, FULL_STEPS, FULL_CKPTS, suffix)
            if full is not None:
                mf = json.loads((full / "manifest.json").read_text())
                ckf = best_checkpoint(mf)
                results[tag] = {"status": "extended", "best_return": ckf["mean_episode_return"],
                                "best_length": ckf["mean_episode_length"], "suffix": suffix}
                print(f"[retgt] {tag} EXTENDED: best R={ckf['mean_episode_return']:.0f} "
                      f"len={ckf['mean_episode_length']:.0f}", flush=True)
                if ckf["mean_episode_length"] >= PROMISING_LEN:
                    winner = (tag, suffix)
                    break
            continue

        results[tag] = {"status": "stuck", "best_return": ck["mean_episode_return"],
                        "best_length": best_len}
        print(f"[retgt] {tag} stuck (len {best_len:.0f}); discarding, next.", flush=True)
        shutil.rmtree(out_dir, ignore_errors=True)

    summary = {"winner": winner[0] if winner else None,
               "winner_suffix": winner[1] if winner else None, "configs": results}
    (ROOT / "tall_retarget_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n[retgt] RETARGET_DONE winner=" + (winner[0] if winner else "NONE"))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
