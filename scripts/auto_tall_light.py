"""Autonomous overnight driver: get a definitive result for whether G1
extreme_tall_light (legs 1.55x) can learn to track dance2_subject4 with more
training, within a ~10h GPU budget.

Background: prior 300M stock-gains run was a *slow monotonic crawl* (episode
length 100->134->167->202->238->284, still rising at 300M), never crossing the
balance transition. 19 other recipes also failed. Open question the user wants
settled: does the crawl eventually cross with far more steps, or asymptote below?

Strategy (resume is broken cross-process, so each run is from-scratch, continuous):
  Phase 1  -- 1B-step probe (stock gains). 10 x 100M checkpoints.
             * crossed (max len >= CROSS)        -> SUCCESS, stop.
             * flat & low (max<340, end-slope<25) -> PLATEAU confirmed, stop (saves budget).
             * still climbing otherwise           -> Phase 2.
  Phase 2  -- commit the *remaining* wall-clock budget to one big fresh run
             (sized to fit 10h), then report whether it crossed or asymptoted.

Run inside WSL conda env locodm (GPU JAX).
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent / "external_data" / "deepmimic_morphology" / "g1" / "dance2_subject4"
PRESET = "extreme_tall_light"

RATE = 5.4e6            # steps/min observed on this RTX 4060 Ti (300M / 55.5min)
BUDGET_MIN = 10 * 60    # hard wall-clock budget
CROSS = 500.0           # episode length (of 1000) that counts as "tracking / crossed"
PLATEAU_MAX = 340.0     # below this, with a flat tail, = confirmed plateau
PLATEAU_SLOPE = 25.0    # len gain over last 3 checkpoints considered "flat"
START = time.time()


def elapsed_min() -> float:
    return (time.time() - START) / 60.0


def remaining_min() -> float:
    return BUDGET_MIN - elapsed_min()


def train(steps: str, ckpts: int, suffix: str) -> Path | None:
    cmd = [
        sys.executable, "train_deepmimic_morphology.py",
        "--robot", "g1", "--preset", PRESET, "--clip", "dance2_subject4",
        "--duration", "30", "--num-envs", "2048",
        "--total-timesteps", str(steps), "--num-checkpoints", str(ckpts),
        "--raw-reference", "--control", "pd", "--seed", "1", "--out-suffix", suffix,
    ]
    print(f"\n[auto] t={elapsed_min():.0f}min  $ {' '.join(cmd)}", flush=True)
    rc = subprocess.run(cmd, cwd=str(SCRIPTS)).returncode
    out = ROOT / f"{PRESET}__{suffix}"
    if rc != 0 or not (out / "manifest.json").exists():
        print(f"[auto] run {suffix} FAILED rc={rc}", flush=True)
        return None
    return out


def curve(out: Path):
    m = json.loads((out / "manifest.json").read_text())
    cks = m["checkpoints"]
    lens = [c["mean_episode_length"] for c in cks]
    rets = [c["mean_episode_return"] for c in cks]
    steps = [c["cumulative_steps"] for c in cks]
    best = max(cks, key=lambda c: c["mean_episode_length"])
    return lens, rets, steps, best


def classify(lens):
    mx = max(lens)
    slope = lens[-1] - lens[-3] if len(lens) >= 3 else lens[-1] - lens[0]
    if mx >= CROSS:
        return "crossed", mx, slope
    if mx < PLATEAU_MAX and slope < PLATEAU_SLOPE:
        return "plateau", mx, slope
    return "climbing", mx, slope


def main() -> None:
    results = {"preset": PRESET, "budget_min": BUDGET_MIN}

    # ---- Phase 1: 1B probe ----
    p = train("1e9", 10, "deep1b")
    if p is None:
        results["verdict"] = "PROBE_FAILED"
        (ROOT / "auto_tall_light_summary.json").write_text(json.dumps(results, indent=2))
        print("[auto] DONE verdict=PROBE_FAILED"); return
    lens, rets, steps, best = curve(p)
    state, mx, slope = classify(lens)
    results["probe"] = {"steps": steps, "lens": lens, "rets": rets,
                        "max_len": mx, "end_slope3": slope, "state": state,
                        "best_ckpt": best["agent_path"],
                        "best_len": best["mean_episode_length"],
                        "best_ret": best["mean_episode_return"]}
    print(f"[auto] PROBE state={state} max_len={mx:.0f} slope3={slope:.0f} "
          f"lens={['%.0f' % x for x in lens]}", flush=True)

    if state == "crossed":
        results["verdict"] = "SUCCESS_PROBE"
    elif state == "plateau":
        results["verdict"] = "PLATEAU_CONFIRMED_1B"
    else:
        # ---- Phase 2: commit remaining budget to one big continuous run ----
        rem = remaining_min() * 0.9
        big = int(min(3.2e9, max(1.2e9, rem * RATE)))
        nck = max(10, big // 100_000_000)
        print(f"[auto] climbing -> BIG run {big:,} steps / {nck} ckpts "
              f"(remaining {remaining_min():.0f}min)", flush=True)
        b = train(str(big), int(nck), "deepbig")
        if b is None:
            results["verdict"] = "BIG_FAILED_after_climbing_probe"
        else:
            lens2, rets2, steps2, best2 = curve(b)
            state2, mx2, slope2 = classify(lens2)
            results["big"] = {"req_steps": big, "steps": steps2, "lens": lens2, "rets": rets2,
                              "max_len": mx2, "end_slope3": slope2, "state": state2,
                              "best_ckpt": best2["agent_path"],
                              "best_len": best2["mean_episode_length"],
                              "best_ret": best2["mean_episode_return"]}
            print(f"[auto] BIG state={state2} max_len={mx2:.0f} slope3={slope2:.0f} "
                  f"lens={['%.0f' % x for x in lens2]}", flush=True)
            results["verdict"] = "SUCCESS_BIG" if state2 == "crossed" else "PLATEAU_CONFIRMED_BIG"

    results["elapsed_min"] = elapsed_min()
    (ROOT / "auto_tall_light_summary.json").write_text(json.dumps(results, indent=2))
    print(f"\n[auto] AUTO_DONE verdict={results['verdict']} elapsed={elapsed_min():.0f}min", flush=True)
    print(json.dumps({k: v for k, v in results.items() if k != "probe" or True}, indent=2))


if __name__ == "__main__":
    main()
