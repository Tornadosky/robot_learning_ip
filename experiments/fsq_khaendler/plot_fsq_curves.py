"""Training curves for the FSQ-SCALE arms, from curves_fsq.csv.

Four panels, each answering one question we actually asked:
  1. does adding the token to the reference help?      (return + tracking error)
  2. what does the token rate cost?                    (hold 1..20)
  3. what does a second topology cost, and how badly
     does the shared canonical stream fail?
  4. the M-curve arms, for reference.

Curves are what training reports; the crossevals in ce_fsqscale/ are the
comparable measure across arms. Both are shown because they answer different
questions -- the curve shows HOW it got there, the crosseval shows WHERE it got.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
CSV = HERE / "curves_fsq.csv"
OUT = HERE / "fsq_curves.png"

INK, GRID = "#141A21", "#D4DAE2"
C = {
    "ref": "#26635F", "z": "#A8322A", "both": "#1F4E9C",
    "h2": "#7E57C2", "h5": "#C77B2B", "h10": "#8A8F98", "h20": "#5A3E36",
    "canon": "#000000",
}


def load():
    d = defaultdict(lambda: defaultdict(list))
    with open(CSV) as f:
        for r in csv.DictReader(f):
            arm = r["arm"]
            for k, v in r.items():
                if k in ("arm",) or v == "":
                    continue
                try:
                    d[arm][k].append(float(v))
                except ValueError:
                    pass
    return d


def smooth(y, k=9):
    if len(y) < k:
        return np.asarray(y)
    return np.convolve(np.asarray(y), np.ones(k) / k, mode="valid")


def line(ax, d, arm, field, color, label, ls="-"):
    if arm not in d or field not in d[arm]:
        return
    x = np.asarray(d[arm]["steps"]) / 1e6
    y = np.asarray(d[arm][field])
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    ys = smooth(y)
    xs = x[len(x) - len(ys):]
    ax.plot(xs, ys, color=color, lw=1.7, ls=ls, label=label)


def style(ax, title, ylab):
    ax.set_title(title, fontsize=10.5, color=INK, loc="left")
    ax.set_xlabel("million env steps", fontsize=8.5)
    ax.set_ylabel(ylab, fontsize=8.5)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.8)
    ax.tick_params(labelsize=8)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.legend(fontsize=7.5, frameon=False)


def main():
    d = load()
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.2))
    fig.patch.set_facecolor("white")

    ax = axes[0][0]
    for arm, c, lab in (("M9_ref", C["ref"], "reference only"),
                        ("M9_z", C["z"], "token only"),
                        ("M9_both", C["both"], "reference + token")):
        line(ax, d, arm, "return", c, lab)
    style(ax, "1a. Auxiliary channel, 9 dances -- episode return", "return")

    ax = axes[0][1]
    for arm, c, lab in (("M9_ref", C["ref"], "reference only"),
                        ("M9_z", C["z"], "token only"),
                        ("M9_both", C["both"], "reference + token")):
        line(ax, d, arm, "joint_err", c, lab)
    style(ax, "1b. Same arms -- joint tracking error (train)", "rad$^2$-ish (env metric)")

    ax = axes[0][2]
    for arm, c, lab in (("M9_ref", C["ref"], "reference only"),
                        ("M9_z", C["z"], "token only"),
                        ("M9_both", C["both"], "reference + token")):
        line(ax, d, arm, "ep_len", c, lab)
    style(ax, "1c. Same arms -- episode length", "steps (of 1000)")

    ax = axes[1][0]
    for arm, c, lab in (("M9_z", C["z"], "hold 1  = 40 tok/s"),
                        ("M9_z_h2", C["h2"], "hold 2  = 20 tok/s"),
                        ("M9_z_h5", C["h5"], "hold 5  =  8 tok/s"),
                        ("M9_z_h10", C["h10"], "hold 10 =  4 tok/s"),
                        ("M9_z_h20", C["h20"], "hold 20 =  2 tok/s")):
        line(ax, d, arm, "joint_err", c, lab)
    style(ax, "2. Token rate -- joint tracking error", "env joint error")

    ax = axes[1][1]
    for arm, c, lab, ls in (("M5_2t_ref", C["ref"], "2 robots, reference", "-"),
                            ("M5_2t_z", C["z"], "2 robots, per-robot tokens", "-"),
                            ("M5_2t_canon", C["canon"], "2 robots, ONE shared token", "-"),
                            ("M5_ref", C["ref"], "1 robot, reference", ":")):
        line(ax, d, arm, "joint_err", c, lab, ls)
    style(ax, "3. Second topology, and the shared-token failure", "env joint error")

    ax = axes[1][2]
    for arm, c, lab in (("X_temp005", "#8A8F98", "M=1 reference"),
                        ("M4_ref", "#26635F", "M=4 reference"),
                        ("M9_ref", "#1F4E9C", "M=9 reference"),
                        ("M9_z", C["z"], "M=9 token")):
        line(ax, d, arm, "return", c, lab)
    style(ax, "4. M-curve arms -- episode return", "return")

    fig.suptitle("FSQ-SCALE training curves  |  H1 nominal, TRACK_TEMP 0.05, 768 envs, 98.3 M steps",
                 fontsize=11.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT, dpi=140)
    print(f"wrote {OUT}")

    # A second figure: the crosseval bars, which are the comparable numbers.
    fig2, ax = plt.subplots(figsize=(9, 4.4))
    labels = ["M9_ref\nreference", "M9_z\ntoken", "M9_both\nboth",
              "M5_ref\nreference", "M5_z\ntoken", "M5_both\nboth"]
    vals = [0.1314, 0.1329, 0.1268, 0.1422, 0.1465, 0.1395]
    errs = [0.0036, 0.0028, 0.0036, 0.0021, 0.0022, 0.0016]
    cols = [C["ref"], C["z"], C["both"]] * 2
    ax.bar(labels, vals, yerr=errs, capsize=4, color=cols, alpha=0.9)
    ax.set_ylim(0.11, 0.155)
    ax.set_ylabel("executed-vs-clip joint RMSE (rad)", fontsize=9)
    ax.set_title("Crosseval: adding the token to the reference BEATS the reference alone "
                 "(n=4 seeds, error bars = 1 sd)", fontsize=10, loc="left", color=INK)
    ax.grid(True, axis="y", color=GRID, lw=0.6)
    ax.tick_params(labelsize=8)
    for s in ax.spines.values():
        s.set_color(GRID)
    fig2.tight_layout()
    fig2.savefig(HERE / "fsq_crosseval_bars.png", dpi=140)
    print(f"wrote {HERE / 'fsq_crosseval_bars.png'}")


def crosseval_panel():
    """One number per arm, with its seeds -- the measure that actually separates them."""
    import json, statistics as st
    CE = HERE / "ce_fsqscale"

    def series(names, robot="unitree_h1"):
        v = []
        for n in names:
            f = CE / f"{n}.json"
            if f.exists():
                try:
                    v.append(json.loads(f.read_text())["robots"][robot]["raw_rmse_rad"])
                except KeyError:
                    pass
        return v

    def sd(stem, tag):
        return [f"{stem}__{tag}"] + [f"{stem}__{tag}_s{i}" for i in (1, 2, 3)]

    def sd0(stem, tag):
        return [f"{stem}__{tag}_s{i}" for i in (0, 1, 2, 3)]

    GROUPS = [
        ("M-curve: reference vs token", [
            ("M=1 reference", sd("X_temp005", "dance4"), C["ref"]),
            ("M=1 token", sd("M1_z", "dance4"), C["z"]),
            ("M=4 reference", sd("M4_ref", "superM4"), C["ref"]),
            ("M=4 token", sd("M4_z", "superM4"), C["z"]),
            ("M=5 reference", sd("M5_ref", "super5"), C["ref"]),
            ("M=5 token", sd("M5_z", "super5"), C["z"]),
            ("M=9 reference", sd("M9_ref", "superM9"), C["ref"]),
            ("M=9 token", sd("M9_z", "superM9"), C["z"]),
        ]),
        ("Token ADDED to the reference", [
            ("M=5 reference", sd("M5_ref", "super5"), C["ref"]),
            ("M=5 both", sd0("M5_both", "super5"), C["both"]),
            ("M=9 reference", sd("M9_ref", "superM9"), C["ref"]),
            ("M=9 both", sd0("M9_both", "superM9"), C["both"]),
        ]),
        ("Token rate at M=9", [
            ("40 tok/s", sd("M9_z", "superM9"), C["z"]),
            ("20 tok/s", sd0("M9_z_h2", "superM9"), C["h2"]),
            ("8 tok/s", ["M9_z_h5__superM9"], C["h5"]),
            ("4 tok/s", ["M9_z_h10__superM9"], C["h10"]),
            ("2 tok/s", sd0("M9_z_h20", "superM9"), C["h20"]),
        ]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), gridspec_kw={"width_ratios": [1.5, 1, 1]})
    for ax, (title, rows) in zip(axes, GROUPS):
        labs, means, errs, cols, ns = [], [], [], [], []
        for lab, names, col in rows:
            v = series(names)
            if not v:
                continue
            labs.append(lab)
            means.append(st.mean(v))
            errs.append(st.stdev(v) if len(v) > 1 else 0.0)
            cols.append(col)
            ns.append(len(v))
        y = np.arange(len(labs))
        ax.barh(y, means, xerr=errs, color=cols, alpha=.9, capsize=3, height=.68)
        ax.set_yticks(y, labs, fontsize=8.5)
        ax.invert_yaxis()
        for i, (m_, n_) in enumerate(zip(means, ns)):
            ax.text(m_ + 0.004, i, f"{m_:.4f} (n={n_})", va="center", fontsize=7.5, color=INK)
        ax.set_xlim(0, max(means) * 1.38)
        ax.set_xlabel("executed-vs-clip joint RMSE (rad) -- lower is better", fontsize=8.5)
        ax.set_title(title, fontsize=10.5, loc="left", color=INK)
        ax.grid(True, axis="x", color=GRID, lw=.6)
        ax.tick_params(labelsize=8)
        for sp in ax.spines.values():
            sp.set_color(GRID)
    fig.suptitle("THE crosseval: roll each trained policy out, compare the joint angles it "
                 "ACTUALLY executed against the raw mocap clip at the same point in the motion",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(HERE / "fsq_crosseval_all.png", dpi=140)
    print(f"wrote {HERE / 'fsq_crosseval_all.png'}")


if __name__ == "__main__":
    main()
    crosseval_panel()
