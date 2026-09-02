"""Every figure for the WAVE 2 dashboard, plain white matplotlib.

Training curves come from curves_all.csv (scraped from the trainer's console
tables on Viper, 300 arms). Crosseval numbers are typed in from
REPORT_FSQ_WAVE2.md, because they are pooled means over >= 4 rollout seeds and
the pooling rules (filter on the clip, never mix conditions) are easier to audit
written out than re-derived here.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "plots_wave2"
OUT.mkdir(exist_ok=True)

BLUE, AMBER, PLUM = "#1B62A5", "#C67B10", "#8E3C86"
GREY, LGREY = "#5A6B73", "#C6D2D7"
RED = "#A8382C"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.edgecolor": "#4A5A62", "axes.linewidth": 0.9,
    "axes.grid": True, "grid.color": "#E2E9EC", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.titlesize": 11.5, "axes.titleweight": "bold",
    "axes.labelsize": 10, "legend.frameon": False, "legend.fontsize": 9,
    "xtick.color": "#4A5A62", "ytick.color": "#4A5A62",
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
})


def tidy(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def smooth(y, k=21):
    if len(y) < k:
        return y
    return pd.Series(y).rolling(k, center=True, min_periods=1).mean().to_numpy()


def load():
    df = pd.read_csv(HERE / "curves_all.csv")
    # one arm can appear under several job ids (resubmissions); keep the longest run
    keep = df.groupby(["arm", "job"]).steps.max().reset_index()
    best = keep.sort_values("steps").groupby("arm").tail(1)[["arm", "job"]]
    return df.merge(best, on=["arm", "job"])


def series(df, arm, col):
    d = df[df.arm == arm].sort_values("steps")
    d = d.dropna(subset=[col])
    return d.steps.to_numpy() / 1e6, d[col].to_numpy()


# ----------------------------------------------------------------- figure 1
def fig_heading(df):
    """The night's most consequential curve: heading is UNLEARNED."""
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    arms = [("M9_ref", BLUE, "reference only"),
            ("M9_both", AMBER, "reference + token"),
            ("M9_head05", PLUM, "heading term on, weight 0.5"),
            ("M9_head20", RED, "heading term on, weight 2.0")]
    for arm, c, lab in arms:
        x, y = series(df, arm, "heading_err")
        if not len(x):
            continue
        ax.plot(x, np.degrees(smooth(y)), color=c, lw=1.9, label=lab)
    ax.axhline(5.4, color=GREY, ls="--", lw=1.4)
    ax.text(2, 8.0, "zero-action floor  5.4°",
            ha="left", va="bottom", fontsize=8.6, color=GREY)
    ax.set_xlabel("training steps (millions)")
    ax.set_ylabel("mean root-heading error (degrees)")
    ax.set_title("The policy does not fail to learn heading — it unlearns it")
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.42))
    ax.set_ylim(0, 95)
    tidy(ax)
    fig.text(0.005, -0.045,
             "Every arm starts near the reference's own motion (~11 deg) and reaches ~84 deg within "
             "10M steps. Turning the heading reward on does not prevent it: weight 0.5 is "
             "indistinguishable, and weight 2.0 ends only ~8 deg lower while costing 18 % of joint accuracy.",
             fontsize=8.4, color="#4A5A62")
    fig.savefig(OUT / "f1_heading_unlearned.png")
    plt.close(fig)


# ----------------------------------------------------------------- figure 2
def fig_blind(df):
    """Why the crosseval exists: the training metrics cannot see the effect."""
    cols = [("ep_len", "episode length (steps)", None),
            ("return", "episode return", None),
            ("joint_err", "joint tracking error (rad)", None)]
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.5))
    for ax, (col, lab, _) in zip(axes, cols):
        for arm, c, l in [("M9_ref", BLUE, "reference"), ("M9_z", AMBER, "token"),
                          ("M9_both", PLUM, "both")]:
            x, y = series(df, arm, col)
            if len(x):
                ax.plot(x, smooth(y), color=c, lw=1.7, label=l)
        ax.set_xlabel("training steps (millions)")
        ax.set_ylabel(lab)
        tidy(ax)
    axes[0].legend(loc="lower right")
    axes[0].set_title("Episode length")
    axes[1].set_title("Return")
    axes[2].set_title("Joint tracking error")
    fig.suptitle("The training curves cannot separate the three channels — only the crosseval can",
                 fontsize=11.5, fontweight="bold", y=1.04)
    fig.savefig(OUT / "f2_training_is_blind.png")
    plt.close(fig)


# ----------------------------------------------------------------- figure 3
def fig_degrade():
    K = np.array([1, 5, 10, 20])
    ref = np.array([0.1315, 0.2065, 0.2847, 0.3549])
    ref_sd = np.array([0.0029, 0.0068, 0.0132, 0.0121])
    fresh = np.array([0.1263, 0.1484, 0.1740, 0.2126])
    fresh_sd = np.array([0.0044, 0.0053, 0.0042, 0.0051])
    match = np.array([0.1263, 0.1776, 0.2516, 0.3376])
    match_sd = np.array([0.0044, 0.0074, 0.0088, 0.0085])
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for y, sd, c, lab, ls in [
            (ref, ref_sd, BLUE, "reference only", "-"),
            (fresh, fresh_sd, AMBER, "+ token (token stays fresh)", "-"),
            (match, match_sd, PLUM, "+ token (token held at K too)", "-")]:
        ax.errorbar(K, y, yerr=sd, color=c, lw=1.9, ls=ls, marker="o", ms=5.5,
                    capsize=3, label=lab)
    ax.axhline(0.3991, color=GREY, ls="--", lw=1.4)
    ax.text(20, 0.404, "zero-action floor", ha="right", fontsize=8.6, color=GREY)
    ax.set_xlabel("observed reference frozen every K clip frames  (K=1 is fresh, 40 fps)")
    ax.set_ylabel("executed-vs-clip shape RMSE (rad)")
    ax.set_title("Take the reference away and the token carries more of the load")
    ax.set_xticks(K)
    ax.legend(loc="upper left")
    tidy(ax)
    fig.text(0.005, -0.05,
             "n = 4 rollout seeds per point. The middle line flatters the token — it keeps a per-frame "
             "channel the reference lost.\nThe purple line is the fair comparison, both channels equally stale, "
             "and the token still wins by more than it does when both are fresh.",
             fontsize=8.4, color="#4A5A62")
    fig.savefig(OUT / "f3_reference_degradation.png")
    plt.close(fig)


# ----------------------------------------------------------------- figure 4
def fig_rate():
    hold = np.array([1, 2, 5, 10, 20])
    val = np.array([0.1320, 0.1280, 0.1422, 0.1620, 0.2228])
    sd = np.array([0.0024, 0.0031, 0.0037, 0.0047, 0.0067])
    rate = 40 / hold
    fig, ax = plt.subplots(figsize=(7.0, 4.1))
    ax.errorbar(rate, val, yerr=sd, color=BLUE, lw=1.9, marker="o", ms=6, capsize=3)
    ax.scatter([20], [0.1280], s=130, facecolor=BLUE, edgecolor="white", zorder=5,
               linewidth=1.6)
    ax.annotate("20 tokens/s beats 40", (20, 0.1280), textcoords="offset points",
                xytext=(14, -20), fontsize=9, color=BLUE)
    ax.set_xscale("log")
    ax.set_xticks(rate)
    ax.set_xticklabels([f"{r:.0f}" for r in rate])
    ax.set_xlabel("tokens issued per second (clip runs at 40 fps)")
    ax.set_ylabel("executed-vs-clip shape RMSE (rad)")
    ax.set_title("The token rate curve, and its knee")
    tidy(ax)
    fig.text(0.005, -0.05,
             "H1, nine dances, token replacing the reference. The knee sits between 20 and 8 tokens/s; "
             "holding a token one extra\nframe is very slightly better than issuing one every frame.",
             fontsize=8.4, color="#4A5A62")
    fig.savefig(OUT / "f4_rate_curve.png")
    plt.close(fig)


# ----------------------------------------------------------------- figure 5
def fig_msweep():
    labels = ["M=1\n1 dance", "M=4\n4 dances", "M=5\n5 dances", "M=9\n9 dances"]
    ref = np.array([0.1343, 0.1363, 0.1424, 0.1315])
    ref_sd = np.array([0.0025, 0.0044, 0.0023, 0.0029])
    both = np.array([0.1267, 0.1244, 0.1395, 0.1263])
    both_sd = np.array([0.0024, 0.0032, 0.0016, 0.0044])
    x = np.arange(4)
    w = 0.36
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.1),
                                  gridspec_kw={"width_ratios": [1.25, 1]})
    ax.bar(x - w/2, ref, w, yerr=ref_sd, color=BLUE, capsize=3, label="reference only")
    ax.bar(x + w/2, both, w, yerr=both_sd, color=AMBER, capsize=3, label="reference + token")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("shape RMSE (rad)"); ax.set_ylim(0, 0.175)
    ax.set_title("The token helps at every number of motions")
    ax.legend(loc="upper right")
    tidy(ax)
    gain = 100 * (both / ref - 1)
    ax2.bar(x, gain, 0.5, color=[AMBER if g < 0 else RED for g in gain])
    for xi, g in zip(x, gain):
        ax2.text(xi, g - 0.45, f"{g:.1f}%", ha="center", va="top", fontsize=9.5,
                 color="#20303A", fontweight="bold")
    ax2.axhline(0, color="#4A5A62", lw=1)
    ax2.set_xticks(x); ax2.set_xticklabels([l.split("\n")[0] for l in labels])
    ax2.set_ylabel("token effect (%)"); ax2.set_ylim(-11, 2)
    ax2.set_title("Always negative, never monotone in M")
    tidy(ax2)
    fig.savefig(OUT / "f5_motion_sweep.png")
    plt.close(fig)


# ----------------------------------------------------------------- figure 6
def fig_2x2():
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    groups = ["1 topology\nnominal bodies", "1 topology\nrandomized bodies",
              "2 topologies\nnominal bodies", "2 topologies\nrandomized bodies"]
    ref = [0.1315, 0.1376, 0.1557, 0.1585]
    both = [0.1263, 0.1321, 0.1573, 0.1577]
    x = np.arange(4); w = 0.36
    ax.bar(x - w/2, ref, w, color=BLUE, label="reference only")
    ax.bar(x + w/2, both, w, color=AMBER, label="reference + token")
    for xi, (r, b) in enumerate(zip(ref, both)):
        g = 100 * (b / r - 1)
        ax.text(xi, max(r, b) + 0.004, f"{g:+.1f}%", ha="center", fontsize=9.5,
                fontweight="bold", color=AMBER if g < -1 else GREY)
    ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=9)
    ax.set_ylabel("shape RMSE (rad), H1"); ax.set_ylim(0, 0.20)
    ax.set_title("Randomizing the body costs the token nothing. A second robot costs it everything")
    ax.legend(loc="upper left")
    tidy(ax)
    fig.text(0.005, -0.055,
             "H1 column only. Randomization moves both arms by the same +4.6 %, leaving the token effect at "
             "−4.0 % in both\ncolumns. Adding G1 erases it. The tokenizer is already fitted jointly on both "
             "bodies, so that is NOT the cause; a single-topology G1 pair is running to\nseparate "
             "'sharing one policy' from 'G1's token is simply weaker'.",
             fontsize=8.4, color="#4A5A62")
    fig.savefig(OUT / "f6_two_by_two.png")
    plt.close(fig)


# ----------------------------------------------------------------- figure 7
def fig_reconstruction():
    fig, ax = plt.subplots(figsize=(7.0, 4.1))
    width = np.array([1, 2, 4])
    h1 = np.array([0.2486, 0.2178, 0.1774])
    g1 = np.array([0.2434, 0.2151, 0.1787])
    ax.plot(width, h1, color=BLUE, marker="o", ms=6, lw=1.9, label="H1 (same body)")
    ax.plot(width, g1, color=AMBER, marker="s", ms=6, lw=1.9,
            label="G1 (decoded onto the other body)")
    ax.axhline(0.10, color=RED, ls="--", lw=1.5)
    ax.text(4, 0.107, "gate 0.10 rad", ha="right", color=RED, fontsize=9)
    ax.set_xscale("log", base=2); ax.set_xticks(width); ax.set_xticklabels(["1", "2", "4"])
    ax.set_xlabel("decoder width multiplier"); ax.set_ylabel("reconstruction error (rad)")
    ax.set_ylim(0.05, 0.28)
    ax.set_title("The body-independent token: the right suspect, and still not enough")
    ax.legend(loc="upper right")
    tidy(ax)
    fig.text(0.005, -0.05,
             "Two doublings buy 29 %, monotone and unsaturated — against 6 % for a 32× bigger codebook and "
             "0 % for 14 encoder\nsites instead of 4. Reaching the gate would need roughly three more doublings.",
             fontsize=8.4, color="#4A5A62")
    fig.savefig(OUT / "f7_decoder_width.png")
    plt.close(fig)


# ----------------------------------------------------------------- figure 8
def fig_heading_bar():
    rows = [("zero action, H1", 5.39, True), ("zero action, G1", 16.24, True),
            ("scale_2t, G1", 69.64, False), ("M5_2t_ref_morph, G1", 76.84, False),
            ("M9_head20 (term on, w=2.0)", 72.4, False),
            ("M9_head05 (term on, w=0.5)", 79.9, False),
            ("scale_2t, H1", 81.82, False), ("M9_both, H1", 82.39, False),
            ("M9_ref, H1", 82.40, False), ("M5_2t_both_morph, G1", 82.35, False),
            ("M1_both, H1", 83.01, False), ("M5_2t_both_morph, H1", 83.78, False),
            ("M9_both_scram, H1", 84.34, False), ("M5_2t_ref_morph, H1", 84.40, False),
            ("M4_both, H1", 85.68, False), ("M5_both_scram, H1", 86.91, False)]
    rows.sort(key=lambda r: r[1])
    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    cols = [AMBER if r[2] else BLUE for r in rows]
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    y = np.arange(len(rows))
    ax.barh(y, vals, color=cols, height=0.68)
    for yi, v in zip(y, vals):
        ax.text(v + 1.2, yi, f"{v:.1f}°", va="center", fontsize=8.6, color="#4A5A62")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8.8)
    ax.invert_yaxis()
    ax.set_xlabel("mean root-heading error (degrees)")
    ax.set_xlim(0, 100)
    ax.set_title("Sixteen arms, one control, and nothing in between")
    ax.grid(axis="y", visible=False)
    tidy(ax)
    fig.text(0.005, -0.035,
             "Amber is the zero-action control — a policy that emits no torque. Everything trained lands "
             "between 70° and 87°.",
             fontsize=8.4, color="#4A5A62")
    fig.savefig(OUT / "f8_heading_across_arms.png")
    plt.close(fig)


# ----------------------------------------------------------------- figure 9
def fig_ladder(df):
    """Earlier campaign: the difficulty rungs, on three motions."""
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.6), sharey=True)
    motions = [("walk", "walk1_subject1"), ("dance4", "dance2_subject4"),
               ("dance3", "dance2_subject3")]
    for ax, (suffix, full) in zip(axes, motions):
        for rung, c, lab in [("L0", BLUE, "L0  nominal body"),
                             ("L1", AMBER, "L1  randomized bodies"),
                             ("L2", PLUM, "L2  + second topology")]:
            arm = f"{rung}_{suffix}"
            x, y = series(df, arm, "joint_err")
            if len(x):
                ax.plot(x, smooth(y), color=c, lw=1.7, label=lab)
        ax.set_title(full, fontsize=10.5, fontweight="normal")
        ax.set_xlabel("training steps (millions)")
        tidy(ax)
    axes[0].set_ylabel("joint tracking error (rad)")
    axes[0].legend(loc="upper right")
    fig.suptitle("The LADDER rungs — what each step of generality costs during training",
                 fontsize=11.5, fontweight="bold", y=1.04)
    fig.savefig(OUT / "f9_ladder_rungs.png")
    plt.close(fig)


# ---------------------------------------------------------------- figure 10
def fig_canonical(df):
    """The one failure a training curve CAN see."""
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for arm, c, lab in [("M5_2t_ref", BLUE, "explicit reference, per robot"),
                        ("M5_2t_z", AMBER, "per-joint token, per robot"),
                        ("M5_2t_canon", RED, "one shared body-independent stream")]:
        x, y = series(df, arm, "joint_err")
        if len(x):
            ax.plot(x, smooth(y), color=c, lw=1.9, label=lab)
    ax.set_xlabel("training steps (millions)")
    ax.set_ylabel("joint tracking error (rad)")
    ax.set_title("The exception: a failure large enough for the training curve to see")
    ax.legend(loc="upper right")
    tidy(ax)
    fig.text(0.005, -0.05,
             "The shared canonical stream never converges. Every other comparison in this project is invisible "
             "here and needs the crosseval.",
             fontsize=8.4, color="#4A5A62")
    fig.savefig(OUT / "f10_canonical_fails_visibly.png")
    plt.close(fig)


# ---------------------------------------------------------------- figure 11
def fig_selloff():
    """The policy acquires heading and foot-lift, then trades them away."""
    df = pd.read_csv(HERE / "selloff.csv")
    arms = [("M9_ref", GREY, "baseline (both terms off)", 2.4),
            ("fx_foot", BLUE, "foot-height reward on", 1.9),
            ("fx_footz", AMBER, "+ vertical-speed penalty cut 10x", 1.9),
            ("fx_all", PLUM, "+ heading term at 2.0", 1.9)]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9))
    for arm, c, lab, lw in arms:
        d = df[df.arm == arm].sort_values("steps")
        if not len(d):
            continue
        x = d["steps"].to_numpy() / 1e6
        ratio = smooth(d["air"].to_numpy() / np.maximum(d["refair"].to_numpy(), 1e-9), 31)
        axes[0].plot(x, ratio, color=c, lw=lw, label=lab)
        axes[1].plot(x, np.degrees(smooth(d["head"].to_numpy(), 31)), color=c, lw=lw)
        axes[2].plot(x, smooth(d["joint"].to_numpy(), 31), color=c, lw=lw)
    axes[0].axhline(1.0, color="#8A5A00", ls="--", lw=1.3)
    axes[0].text(29.5, 1.04, "lifts as often as the clip asks", ha="right",
                 fontsize=8.4, color="#8A5A00")
    axes[0].set_ylabel("foot-lift ratio  (policy / reference)")
    axes[0].set_ylim(0, 1.65); axes[0].set_title("Feet")
    axes[1].set_ylabel("root-heading error (degrees)")
    axes[1].set_ylim(0, 95); axes[1].set_title("Heading")
    axes[2].set_ylabel("joint tracking error (rad)")
    axes[2].set_ylim(0, 0.32); axes[2].set_title("Joint accuracy — what it is traded for")
    for ax in axes:
        ax.set_xlabel("training steps (millions)")
        ax.set_xlim(0, 30)
        tidy(ax)
    axes[0].legend(loc="upper right", fontsize=8.2)
    fig.suptitle("The recipe does not fail to learn heading and foot-lift — it sells them off",
                 fontsize=11.5, fontweight="bold", y=1.05)
    fig.text(0.005, -0.06,
             "Baseline in grey: foot-lift starts ABOVE 1.0 and heading near 11 deg, then heading goes "
             "between 6M and 10M and the feet between 10M and 20M,\nwhile joint error improves the whole "
             "way. Turning the foot terms on lifts the feet back (0.30 -> 0.54 at 15-20M) and costs 44 % "
             "joint accuracy. Nothing moves heading.",
             fontsize=8.4, color="#4A5A62")
    fig.savefig(OUT / "f11_selloff.png")
    plt.close(fig)


# ---------------------------------------------------------------- figure 12
def fig_3t():
    """One policy, three topologies, one motion -- the first working run."""
    df = pd.read_csv(HERE / "curves_3t.csv")
    x = df["steps"].to_numpy() / 1e6
    robots = [("h1", BLUE, "H1  (19 joints)"),
              ("g1", AMBER, "G1  (23 joints)"),
              ("t1", PLUM, "booster T1  (23 joints)")]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9))
    for r, c, lab in robots:
        axes[0].plot(x, smooth(df[f"eplen_{r}"].to_numpy(), 21), color=c, lw=1.9, label=lab)
        axes[1].plot(x, smooth(df[f"joint_{r}"].to_numpy(), 21), color=c, lw=1.9)
        ratio = df[f"air_{r}"].to_numpy() / np.maximum(df[f"refair_{r}"].to_numpy(), 1e-9)
        axes[2].plot(x, smooth(ratio, 31), color=c, lw=1.9)
    axes[0].axhline(1000, color=GREY, ls=":", lw=1.2)
    axes[0].text(0.3, 940, "episode cap 1000", fontsize=8.4, color=GREY)
    axes[0].set_ylabel("episode length (steps)"); axes[0].set_ylim(0, 1080)
    axes[0].set_title("Survival"); axes[0].legend(loc="lower right", fontsize=8.6)
    # t1 starts at 0.78 rad and only crosses 0.45 near 16M; clipping it would
    # hide the fact that the third robot converges late, which is the point.
    axes[1].set_ylabel("joint tracking error (rad)"); axes[1].set_ylim(0, 0.85)
    axes[1].set_title("Tracking")
    axes[2].axhline(1.0, color="#8A5A00", ls="--", lw=1.3)
    axes[2].set_ylabel("foot-lift ratio  (policy / reference)"); axes[2].set_ylim(0, 2.6)
    axes[2].set_title("Feet")
    for ax in axes:
        ax.set_xlabel("training steps (millions)"); ax.set_xlim(0, 19.2); tidy(ax)
    fig.suptitle("One policy, three topologies, one motion — H1 + G1 + booster T1 on dance2_subject4",
                 fontsize=11.5, fontweight="bold", y=1.05)
    fig.text(0.005, -0.06,
             "19.17M steps, 576 environments per robot, local CUDA (Viper's ROCm cannot compile three "
             "robot graphs). Final episode length 878 of 1000; joint error 0.081 / 0.038 / 0.120 rad for "
             "H1 / G1 / t1. G1 tracks best inside the shared policy, as it does in every two-topology run.",
             fontsize=8.4, color="#4A5A62")
    fig.savefig(OUT / "f12_three_topologies.png")
    plt.close(fig)



if __name__ == "__main__":
    df = load()
    print(f"loaded {len(df)} rows / {df.arm.nunique()} arms")
    fig_heading(df); fig_blind(df); fig_degrade(); fig_rate(); fig_msweep()
    fig_2x2(); fig_reconstruction(); fig_heading_bar(); fig_ladder(df); fig_canonical(df)
    fig_selloff()
    fig_3t()
    for p in sorted(OUT.glob("*.png")):
        print(f"  {p.name}  {p.stat().st_size/1024:.0f} kB")
