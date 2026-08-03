"""Map the G1 'tall & light' controllability cliff.

extreme_tall_light (legs 1.55x) is past G1's control envelope: 19 recipes all
failed to cross. This trains the graduated tall&light family (legs 1.20 / 1.35 /
1.50x, torso lightening in step) to find where learning breaks -- the point
between "balances & tracks the dance" and "can't".

Per severity, an early-stopping gain search: probe 3x -> 2x -> 1.5x PD gains at
60M; the moment one crosses the balance transition, extend it to the full budget
and finalize. A severity where no gain crosses marks the far side of the cliff.
Winners are finalized under {preset}__cliff for the plot/grid.

Run inside WSL conda env locodm (GPU JAX):
    python run_g1_tall_cliff.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent / "external_data" / "deepmimic_morphology" / "g1" / "dance2_subject4"

PRESETS = ["tall_light_leg120", "tall_light_leg135", "tall_light_leg150"]
GAINS = [3.0, 2.0, 1.5]            # crossing window; stiffest first
PROBE_STEPS, PROBE_CKPTS = "60e6", "4"
FULL_STEPS, FULL_CKPTS = "120e6", "8"
PROMISING_LEN, CROSS_LEN = 250.0, 330.0


def best_checkpoint(manifest: dict) -> dict:
    return max(manifest["checkpoints"], key=lambda c: c["mean_episode_return"])


def train(preset: str, scale: float, steps: str, ckpts: str, suffix: str) -> Path | None:
    cmd = [
        sys.executable, "train_deepmimic_morphology.py",
        "--robot", "g1", "--preset", preset, "--clip", "dance2_subject4",
        "--duration", "30", "--num-envs", "2048",
        "--total-timesteps", steps, "--num-checkpoints", ckpts,
        "--raw-reference", "--control", "pd", "--lr", "3e-4", "--init-std", "0.2",
        "--pd-gain-scale", str(scale), "--seed", "1", "--out-suffix", suffix,
    ]
    print(f"\n$ {' '.join(cmd)}", flush=True)
    rc = subprocess.run(cmd, cwd=str(SCRIPTS)).returncode
    out_dir = ROOT / f"{preset}__{suffix}"
    if rc != 0 or not (out_dir / "manifest.json").exists():
        print(f"[cliff] {preset} scale={scale} ({suffix}) FAILED rc={rc}", flush=True)
        return None
    return out_dir


def finalize(out_dir: Path, preset: str) -> None:
    """Promote a winning run to the canonical {preset}__cliff dir for plot/grid."""
    canonical = ROOT / f"{preset}__cliff"
    if canonical.exists():
        shutil.rmtree(canonical)
    old = out_dir.name
    shutil.move(str(out_dir), str(canonical))
    mpath = canonical / "manifest.json"
    mpath.write_text(mpath.read_text().replace(old, canonical.name))


def main() -> None:
    results = {}
    for preset in PRESETS:
        crossed = False
        best_overall = None  # (length, return, scale)
        for scale in GAINS:
            suffix = f"cliff_s{('%g' % scale).replace('.', '')}"
            out_dir = train(preset, scale, PROBE_STEPS, PROBE_CKPTS, suffix)
            if out_dir is None:
                continue
            m = json.loads((out_dir / "manifest.json").read_text())
            ck = best_checkpoint(m)
            lens = [c["mean_episode_length"] for c in m["checkpoints"]]
            bl = ck["mean_episode_length"]
            print(f"[cliff] PROBE {preset} scale={scale}: best R={ck['mean_episode_return']:.0f} "
                  f"len={bl:.0f} (curve {'/'.join('%.0f' % l for l in lens)})", flush=True)
            if best_overall is None or bl > best_overall[0]:
                best_overall = (bl, ck["mean_episode_return"], scale)

            if bl >= PROMISING_LEN:
                print(f"[cliff] {preset} scale={scale} climbing; extending to {FULL_STEPS} ...",
                      flush=True)
                shutil.rmtree(out_dir, ignore_errors=True)
                full = train(preset, scale, FULL_STEPS, FULL_CKPTS, suffix)
                if full is not None:
                    mf = json.loads((full / "manifest.json").read_text())
                    ckf = best_checkpoint(mf)
                    print(f"[cliff] {preset} scale={scale} EXTENDED: "
                          f"R={ckf['mean_episode_return']:.0f} len={ckf['mean_episode_length']:.0f}",
                          flush=True)
                    finalize(full, preset)
                    results[preset] = {"status": "crossed", "scale": scale,
                                       "best_return": ckf["mean_episode_return"],
                                       "best_length": ckf["mean_episode_length"]}
                    crossed = True
                    break
                continue
            shutil.rmtree(out_dir, ignore_errors=True)  # stuck at this gain

        if not crossed:
            results[preset] = {"status": "stuck", "best_length": best_overall[0],
                               "best_return": best_overall[1], "best_scale": best_overall[2]}
            print(f"[cliff] {preset} STUCK at all gains "
                  f"(best len {best_overall[0]:.0f} @ scale {best_overall[2]})", flush=True)

    (ROOT / "tall_cliff_summary.json").write_text(json.dumps(results, indent=2))
    print("\n[cliff] CLIFF_DONE")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
