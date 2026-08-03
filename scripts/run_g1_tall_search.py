"""Early-stopping recipe search to make the tall&light G1 body actually learn.

extreme_tall_light (legs 1.55x, torso 0.55x -> very high CoM) stays stuck at
episode length ~130-155 under every uniform PD-gain scale 1.5-3x, and only crawls
to ~284 at stock(1x)/300M -- it never *crosses* the balance transition the way
short&heavy / nominal do (a sharp rise to len 600+).

This driver tries a ranked list of recipe modifications, each as a short 60M
PROBE. The decision, per the "stop early if it isn't learning" idea:
  - best probe length >= CROSS_LEN  -> already crossed; keep the probe as winner.
  - best probe length >= PROMISING   -> clearly climbing out of the stuck basin;
                                        extend the SAME recipe to the full budget.
  - else                             -> still stuck; discard and try the next.
The search stops at the first recipe that crosses.

Run inside WSL conda env locodm (GPU JAX):
    python run_g1_tall_search.py
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

PROBE_STEPS = "60e6"
PROBE_CKPTS = "4"          # segments at 15/30/45/60M -> a crossing shows by 60M
FULL_STEPS = "120e6"
FULL_CKPTS = "8"
PROMISING_LEN = 250.0      # stuck configs top out ~155; a real rise clears this
CROSS_LEN = 330.0          # already crossed at probe budget -> no extension needed

# Ranked recipe modifications (tag -> extra train args). Stock recipe baseline is
# pd control, lr 3e-4, init_std 0.2; each entry overrides as needed.
CONFIGS = [
    ("pd06",        ["--control", "pd", "--pd-gain-scale", "0.6", "--lr", "3e-4", "--init-std", "0.2"]),
    ("pd04",        ["--control", "pd", "--pd-gain-scale", "0.4", "--lr", "3e-4", "--init-std", "0.2"]),
    ("torque",      ["--control", "torque", "--lr", "1e-4", "--init-std", "0.2"]),
    ("pd1_explore", ["--control", "pd", "--pd-gain-scale", "1.0", "--lr", "3e-4",
                     "--init-std", "0.4", "--learnable-std"]),
    ("pd08_explore", ["--control", "pd", "--pd-gain-scale", "0.8", "--lr", "3e-4",
                      "--init-std", "0.3", "--learnable-std"]),
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
        print(f"[search] {tag} ({suffix}) FAILED rc={rc}", flush=True)
        return None
    return out_dir


def main() -> None:
    results = {}
    winner = None
    for tag, extra in CONFIGS:
        probe_suffix = f"ts_{tag}"
        out_dir = run_train(tag, extra, PROBE_STEPS, PROBE_CKPTS, probe_suffix)
        if out_dir is None:
            results[tag] = {"status": "failed"}
            continue
        m = json.loads((out_dir / "manifest.json").read_text())
        ck = best_checkpoint(m)
        lens = [c["mean_episode_length"] for c in m["checkpoints"]]
        best_len = ck["mean_episode_length"]
        print(f"[search] PROBE {tag}: best R={ck['mean_episode_return']:.0f} "
              f"len={best_len:.0f} (curve {'/'.join('%.0f' % l for l in lens)})", flush=True)

        if best_len >= CROSS_LEN:
            results[tag] = {"status": "crossed_at_probe", "best_return": ck["mean_episode_return"],
                            "best_length": best_len, "suffix": probe_suffix}
            winner = (tag, probe_suffix)
            print(f"[search] {tag} CROSSED at probe (len {best_len:.0f}) -> winner", flush=True)
            break

        if best_len >= PROMISING_LEN:
            print(f"[search] {tag} promising (len {best_len:.0f}); extending to {FULL_STEPS} ...",
                  flush=True)
            shutil.rmtree(out_dir, ignore_errors=True)  # clear probe ckpts before full run
            full = run_train(tag, extra, FULL_STEPS, FULL_CKPTS, probe_suffix)
            if full is not None:
                mf = json.loads((full / "manifest.json").read_text())
                ckf = best_checkpoint(mf)
                results[tag] = {"status": "extended", "best_return": ckf["mean_episode_return"],
                                "best_length": ckf["mean_episode_length"], "suffix": probe_suffix}
                print(f"[search] {tag} EXTENDED: best R={ckf['mean_episode_return']:.0f} "
                      f"len={ckf['mean_episode_length']:.0f}", flush=True)
                if ckf["mean_episode_length"] >= PROMISING_LEN:
                    winner = (tag, probe_suffix)
                    break
            continue

        # Still stuck -> discard and move on.
        results[tag] = {"status": "stuck", "best_return": ck["mean_episode_return"],
                        "best_length": best_len}
        print(f"[search] {tag} stuck (len {best_len:.0f}); discarding, next.", flush=True)
        shutil.rmtree(out_dir, ignore_errors=True)

    summary = {"winner": winner[0] if winner else None,
               "winner_suffix": winner[1] if winner else None, "configs": results}
    (ROOT / "tall_search_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n[search] SEARCH_DONE winner=" + (winner[0] if winner else "NONE"))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
