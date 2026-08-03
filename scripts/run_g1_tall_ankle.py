"""Stage-2 search for tall&light G1: per-joint (ankle/knee) gain stiffening.

Uniform PD scaling failed (every 1.5-3x overshoots; sub-unity / torque / extra
exploration all stay stuck ~100). A high-CoM body balances through ANKLE torque
(controls the centre of pressure) and knee support, while stiff hips/arms just add
overshoot. G1's stock ankle gain is only 40 (vs hip 100, knee 150), so the ankle
is the weakest link. This probes recipes that stiffen ankle/knee specifically.

Same early-stop protocol as run_g1_tall_search.py: 60M probe per recipe; extend
the first one that climbs out of the stuck basin, stop at the first that crosses.

Run inside WSL conda env locodm (GPU JAX):
    python run_g1_tall_ankle.py
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

# Ranked per-joint recipes (tag -> extra train args). Base stays pd, lr 3e-4,
# init_std 0.2; ankle/knee stiffened well above the (overshooting) uniform scale.
CONFIGS = [
    ("ankle3",        ["--control", "pd", "--pd-gain-scale", "1.0", "--lr", "3e-4",
                       "--init-std", "0.2", "--ankle-gain-scale", "3.0"]),
    ("ankle4_knee2",  ["--control", "pd", "--pd-gain-scale", "1.0", "--lr", "3e-4",
                       "--init-std", "0.2", "--ankle-gain-scale", "4.0", "--knee-gain-scale", "2.0"]),
    ("ankle5_knee3",  ["--control", "pd", "--pd-gain-scale", "1.0", "--lr", "3e-4",
                       "--init-std", "0.2", "--ankle-gain-scale", "5.0", "--knee-gain-scale", "3.0"]),
    ("base2_ankle2",  ["--control", "pd", "--pd-gain-scale", "2.0", "--lr", "3e-4",
                       "--init-std", "0.2", "--ankle-gain-scale", "2.0"]),
    ("ankle4_armsoft", ["--control", "pd", "--pd-gain-scale", "1.0", "--lr", "3e-4",
                        "--init-std", "0.2", "--ankle-gain-scale", "4.0", "--arm-gain-scale", "0.5"]),
]


def best_checkpoint(manifest: dict) -> dict:
    return max(manifest["checkpoints"], key=lambda c: c["mean_episode_return"])


def run_train(tag: str, extra: list[str], steps: str, ckpts: str, suffix: str) -> Path | None:
    cmd = [
        sys.executable, "train_deepmimic_morphology.py",
        "--robot", "g1", "--preset", PRESET, "--clip", "dance2_subject4",
        "--duration", "30", "--num-envs", "2048",
        "--total-timesteps", steps, "--num-checkpoints", ckpts,
        "--raw-reference", "--seed", "1", "--out-suffix", suffix,
    ] + extra
    print(f"\n$ {' '.join(cmd)}", flush=True)
    rc = subprocess.run(cmd, cwd=str(SCRIPTS)).returncode
    out_dir = ROOT / f"{PRESET}__{suffix}"
    if rc != 0 or not (out_dir / "manifest.json").exists():
        print(f"[ankle] {tag} ({suffix}) FAILED rc={rc}", flush=True)
        return None
    return out_dir


def main() -> None:
    results = {}
    winner = None
    for tag, extra in CONFIGS:
        suffix = f"ank_{tag}"
        out_dir = run_train(tag, extra, PROBE_STEPS, PROBE_CKPTS, suffix)
        if out_dir is None:
            results[tag] = {"status": "failed"}
            continue
        m = json.loads((out_dir / "manifest.json").read_text())
        ck = best_checkpoint(m)
        lens = [c["mean_episode_length"] for c in m["checkpoints"]]
        best_len = ck["mean_episode_length"]
        print(f"[ankle] PROBE {tag}: best R={ck['mean_episode_return']:.0f} "
              f"len={best_len:.0f} (curve {'/'.join('%.0f' % l for l in lens)})", flush=True)

        if best_len >= CROSS_LEN:
            results[tag] = {"status": "crossed_at_probe", "best_return": ck["mean_episode_return"],
                            "best_length": best_len, "suffix": suffix}
            winner = (tag, suffix)
            print(f"[ankle] {tag} CROSSED at probe (len {best_len:.0f}) -> winner", flush=True)
            break

        if best_len >= PROMISING_LEN:
            print(f"[ankle] {tag} promising (len {best_len:.0f}); extending to {FULL_STEPS} ...",
                  flush=True)
            shutil.rmtree(out_dir, ignore_errors=True)
            full = run_train(tag, extra, FULL_STEPS, FULL_CKPTS, suffix)
            if full is not None:
                mf = json.loads((full / "manifest.json").read_text())
                ckf = best_checkpoint(mf)
                results[tag] = {"status": "extended", "best_return": ckf["mean_episode_return"],
                                "best_length": ckf["mean_episode_length"], "suffix": suffix}
                print(f"[ankle] {tag} EXTENDED: best R={ckf['mean_episode_return']:.0f} "
                      f"len={ckf['mean_episode_length']:.0f}", flush=True)
                if ckf["mean_episode_length"] >= PROMISING_LEN:
                    winner = (tag, suffix)
                    break
            continue

        results[tag] = {"status": "stuck", "best_return": ck["mean_episode_return"],
                        "best_length": best_len}
        print(f"[ankle] {tag} stuck (len {best_len:.0f}); discarding, next.", flush=True)
        shutil.rmtree(out_dir, ignore_errors=True)

    summary = {"winner": winner[0] if winner else None,
               "winner_suffix": winner[1] if winner else None, "configs": results}
    (ROOT / "tall_ankle_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n[ankle] ANKLE_DONE winner=" + (winner[0] if winner else "NONE"))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
