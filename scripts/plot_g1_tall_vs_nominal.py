"""Plot the G1 tall&light (legs 1.20x) DeepMimic return curve against nominal G1.

The question this answers: nominal G1 learns to track the dance in ~120M steps
with the stiff3 recipe (pd control, gain-scale 3.0, lr 3e-4, net [512,256]). The
tall&light family got stuck at ~150/1000 episode length. Here we throw the same
winning recipe -- plus a deeper net [512,512,256] and ~10x the training budget
(~1.2B steps) -- at tall_light_leg120 to see whether more compute crosses the
controllability cliff or whether it is a genuine wall.

Two panels share the comparison:
  left  : return vs absolute environment steps (so nominal's early crossing and
          the tall body's long crawl are on the same physical x-axis).
  right : return vs training progress (%), normalising out the 10x budget gap so
          the *shape* of each learning curve is directly comparable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WORKSPACE = Path(__file__).resolve().parents[1]
ROOT = WORKSPACE / "external_data" / "deepmimic_morphology" / "g1" / "dance2_subject4"

# (cell dir, label, colour). Nominal is the reference success; the deep10x cell is
# the tall&light experiment this script is built to evaluate.
NOMINAL_CELL = "nominal__stiff3_lr3"
TALL_CELL = "tall_light_leg120__deep10x"


def load_curve(cell: str) -> tuple[dict, np.ndarray, np.ndarray]:
    path = ROOT / cell / "manifest.json"
    m = json.loads(path.read_text(encoding="utf-8"))
    curve = np.asarray(m["return_curve_every_update"], dtype=float)
    steps = np.linspace(0.0, float(m["total_timesteps"]), len(curve))
    return m, steps, curve


def best_len(m: dict) -> float:
    return max(c["mean_episode_length"] for c in m["checkpoints"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=WORKSPACE / "images" / "g1_tall_vs_nominal_returns.png")
    parser.add_argument("--tall-cell", default=TALL_CELL)
    parser.add_argument("--nominal-cell", default=NOMINAL_CELL)
    args = parser.parse_args()

    nom_m, nom_steps, nom_curve = load_curve(args.nominal_cell)
    tall_m, tall_steps, tall_curve = load_curve(args.tall_cell)

    nom_hidden = nom_m.get("hidden_layers", [512, 256])
    tall_hidden = tall_m.get("hidden_layers", [512, 256])
    nom_label = (f"nominal G1 (net {nom_hidden}, "
                 f"{nom_m['total_timesteps'] / 1e6:.0f}M steps)")
    tall_label = (f"tall&light 1.20x (net {tall_hidden}, "
                  f"{tall_m['total_timesteps'] / 1e6:.0f}M steps)")

    fig, (ax_abs, ax_pct) = plt.subplots(1, 2, figsize=(14, 5.4))

    # --- Left: return vs absolute env steps ---
    ax_abs.plot(nom_steps / 1e6, nom_curve, color="#2c7", lw=2.0, label=nom_label)
    ax_abs.plot(tall_steps / 1e6, tall_curve, color="#d4691e", lw=2.0, label=tall_label)
    ax_abs.set_title("DeepMimic return vs training steps")
    ax_abs.set_xlabel("Environment steps (millions)")
    ax_abs.set_ylabel("Mean episode return")
    ax_abs.grid(alpha=0.25)
    ax_abs.legend(loc="upper left", fontsize=8)

    # --- Right: return vs normalised progress ---
    ax_pct.plot(np.linspace(0, 100, len(nom_curve)), nom_curve,
                color="#2c7", lw=2.0, label=nom_label)
    ax_pct.plot(np.linspace(0, 100, len(tall_curve)), tall_curve,
                color="#d4691e", lw=2.0, label=tall_label)
    ax_pct.set_title("Same curves, normalised by budget")
    ax_pct.set_xlabel("Training progress (%)")
    ax_pct.set_ylabel("Mean episode return")
    ax_pct.grid(alpha=0.25)
    ax_pct.legend(loc="upper left", fontsize=8)

    fig.suptitle(
        f"Can more compute cross the cliff?  "
        f"nominal best len {best_len(nom_m):.0f}/1000  vs  "
        f"tall&light best len {best_len(tall_m):.0f}/1000",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=130)
    print(f"[plot] saved {args.output}")

    summary = {
        "nominal": {
            "cell": args.nominal_cell,
            "hidden_layers": nom_hidden,
            "total_timesteps": nom_m["total_timesteps"],
            "best_return": float(np.max(nom_curve)),
            "best_length": best_len(nom_m),
            "final_return": float(nom_curve[-1]),
        },
        "tall_light_leg120": {
            "cell": args.tall_cell,
            "hidden_layers": tall_hidden,
            "total_timesteps": tall_m["total_timesteps"],
            "best_return": float(np.max(tall_curve)),
            "best_length": best_len(tall_m),
            "final_return": float(tall_curve[-1]),
        },
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
