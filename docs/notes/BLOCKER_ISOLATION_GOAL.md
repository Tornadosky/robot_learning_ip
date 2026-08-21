# Isolating the blocker — overnight plan, 2026-08-17

## What is already established

* urma2's own pipeline trains one policy that tracks `dance2_subject4` on H1 **and**
  G1: 965/1000 and 950/1000 steps, ~19.3 s of motion, verified on video.
* Our loco-mujoco trainer's best on the same clip and robots is 427 steps at
  100 Hz = **4.3 s**, and its G1 was below its own zero-action baseline.
* **urma2 crossed at ~10-13M steps** (episode length 100 -> 743 between 8M and
  13M). Our 60M and 300M arms are therefore not a budget problem; they are on
  the wrong side of a transition that a working setup reaches in ~11M.

**That last point sets the whole design: 15M steps is the discriminator.** A 15M
arm costs ~8 min locally (1.93M steps/min) or ~13 min on Viper. We can afford
dozens. Nothing in this plan needs a long run.

## Two facts discovered while designing this, which change the comparison

1. **urma2's clips are the same file format as ours.** Both are loco_mujoco
   `Trajectory` npz dumps with identical schema (`qpos`, `qvel`, `xpos`,
   `site_xpos`, ... on the same model). **The reference swap is nearly free in
   both directions** — no converter, only a window/rate alignment.
2. **The two references are not the same motion.**

   | | frames | rate | duration | root z mean | root z min |
   | --- | --- | --- | --- | --- | --- |
   | urma2 `LAFAN1/UnitreeH1/dance2_subject4.npz` | 9025 | 40 Hz | 225.6 s | 1.0245 | **0.2945** |
   | ours `cross_humanoid/.../start19482_800f_direct.npz` | 800 | 100 Hz | 8.0 s | 1.0611 | **0.9219** |

   Ours is an 8-second window resampled to 100 Hz whose root never drops below
   0.92 m; theirs is the entire 225 s clip at the raw 40 Hz, dropping to 0.29 m.
   **Every "urma2 succeeds where we fail" comparison so far has been across two
   different tasks.** Wave 1 must equalise this before anything else is believed.

## Measurement protocol (non-negotiable — this is where the project keeps failing)

* **Never the training log's aggregate `len`.** It mixes families and env counts
  and it has already killed three claims in this project. Per-family only.
* **Every arm scored against a same-body zero-action baseline.** G1 sat below
  its own zero-action for the entire project and no aggregate revealed it.
* **Common currency across pipelines**: seconds of motion survived
  (`episode_length / control_hz`), fraction of horizon, and tracking error **in
  metres**. urma2's `rpos_tracking_error` is a body-length-normalised MEAN
  SQUARED error — convert with `sqrt(x) * body_length_scale`.
* **Crossing criterion at 15M: >= 40% of horizon, per family.** urma2 was at 74%
  by 13.2M; our arms sit at 5-18%. Anything in between is "partial", and gets a
  second seed before it is believed.
* Two seeds for any arm that decides a branch. One seed is a screen, not a result.
* RSI flatters early frames — an episode starts exactly on the reference. Judge
  by survival and by error at fixed late phase, never by a sampled still.

---

## Wave 0 — zero training, < 1 hour, run first

Cheapest possible discrimination, and it may end the investigation outright.

| # | check | decides |
| --- | --- | --- |
| W0.1 | Run `reference_root_residual.py` + `reference_grounding.py` on **urma2's** clip file, same battery we ran on ours (ours: 100% frames with zero ground contact, root residual 0.5-3.4x body weight) | If urma2's reference is grounded and force-feasible while ours is not, the reference is the blocker and Wave 1 confirms it |
| W0.2 | `clip_comparison`-style stats on the **same 8 s window** of both files (ours: frames 0-800 @100 Hz; theirs: frames 7793-8113 @40 Hz — 194.8 s into the clip) | Separates "different motion" from "same motion, different processing" |
| W0.3 | Inverse-dynamics torque feasibility of urma2's clip on H1 and G1 | Ours saturates the arms 57-96% of the time; if theirs does not, the retarget/window is the difference |
| W0.4 | Diff the two `qpos` streams over the aligned window after resampling theirs to 100 Hz | Quantifies how much is window choice vs retargeting method |

**Gate:** if W0.1-W0.3 show urma2's reference is grounded/feasible and ours is
not, promote the reference branch (Wave 1 + Wave 3 group R) and demote the
trainer branch to a background check.

## Wave 1 — the partition: two-sided reference swap (2 arms, ~1 h)

The single most decisive pair. Same format, so both directions are cheap.

| arm | config | where | budget |
| --- | --- | --- | --- |
| **X1** | **our trainer + urma2's clip**, cropped to our 8 s window and resampled to 100 Hz | Viper | 15M |
| **X2** | **urma2 + our clip**, padded/resampled to their loader's convention | local | 15M |
| X0 | urma2 baseline re-run at 15M as the yardstick | local | 15M |

| outcome | conclusion |
| --- | --- |
| X1 crosses, X2 fails | **The reference is the blocker.** Our trainer is fine; fix retargeting/grounding/window and stop porting conventions. |
| X1 fails, X2 crosses | **The trainer is the blocker.** The reference is adequate; Wave 2 names which trainer component. |
| both cross | The blocker is an interaction (most likely window length x horizon); go to Wave 2 + 3 with the window equalised. |
| neither crosses | The alignment itself is wrong — check the resampling before drawing any conclusion. |

## Wave 2 — subtractive ladder on the WORKING system (local GPU, ~8 arms)

Breaking a working system one knob at a time is far more informative than fixing
a broken one, because there is a clear signal to lose. Each arm is urma2 at 15M
with exactly one setting moved to ours. **Whichever arm collapses names the blocker.**

| arm | change from urma2 default | tests |
| --- | --- | --- |
| S1 | `tracking_reference_action_bias=0.0` | action centred on nominal instead of the reference — the divergence I rate most likely |
| S2 | `scaling_factors` x4 (≈ our full-joint-range action) | action scale |
| S3 | `control_frequency_hz=100` | control rate (needs a small plumb; it is a constructor default, not a config key) |
| S4 | `reward.nominal_diff_target=nominal` | urma2's own comment says this "rewards the ignore-the-reference baseline" |
| S5 | our clip (= X2, shared with Wave 1) | reference |
| S6 | `deepmimic_enabled=False` | whether the site/DeepMimic terms carry it |
| S7 | `morphology_coeff=0.3` | **does randomization survive?** — the end state you require |
| S8 | `tracking_clip_fit_per_variant=False` | per-body retargeting — the other half of that requirement |

S7 and S8 are not ablations of a defect; they are the acceptance tests for
"randomization + per-body retargeting still works". Run them regardless of what
S1-S6 say.

## Wave 3 — additive ladder on OUR trainer (Viper, 8 concurrent)

Each arm is our production config at 15M with one thing moved to urma2's
convention. Group R first if Wave 0 promotes the reference branch.

| group | arm | change |
| --- | --- | --- |
| R | A1 | urma2's clip (= X1) |
| R | A2 | our clip, per-frame grounded (`--reference-grounding per_frame`) — retest at 15M, since the 60M result may have been past the transition |
| R | A3 | our clip, window widened to the full motion range (root z down to ~0.3) |
| C | A4 | 50 Hz control (`n_substeps=10`) — needs a passthrough flag |
| C | A5 | horizon 20 s to match urma2 |
| B | A6 | `--pd-action-scale 0.5` on **both** families (H1 currently trains on torque; urma2 uses position actuators for both) |
| B | A7 | reference-centred residual action (needs implementing — see below) |
| E | A8 | rotation terminal measured from **gravity**, not the clip's quaternion centroid |
| E | A9 | no rotation terminal at all (bracket for A8) |
| O | A10 | `--update-epochs 5` (local only; ROCm forbids it on Viper) |
| O | A11 | wider net 512x256 |
| — | A0 | baseline at 15M + zero-action, both families — the yardstick every arm is read against |

## Implementation prerequisites (do these before the runs)

| item | size | needed by |
| --- | --- | --- |
| clip aligner: crop/resample a `Trajectory` npz between 40/100 Hz and window | ~40 lines, reuse `make_control_references.py` | W0.2, X1, X2 |
| `--n-substeps` passthrough in our trainer | ~5 lines | A4 |
| reference-centred residual PD (put the per-step reference into the control function) | moderate — the control function needs the trajectory cursor via carry | A7 |
| gravity-based rotation terminal subclass | ~30 lines | A8 |
| urma2 `control_frequency_hz` as a config key | ~5 lines | S3 |

## Resource schedule (~10 h overnight)

* **Local RTX 4060 Ti** — urma2 arms. ~33 min each at 15M (23 min run + ~10 min
  compile; the XLA cache is warm for the H1+G1 graph). Sequential, under the
  existing `flock`. Order: X0, X2/S5, S1, S4, S7, S8, S2, S3, S6 → ~9 arms in 5 h,
  leaving headroom for follow-ups on whichever arm collapses.
* **Viper, 8 concurrent** — our-trainer arms, ~13 min run each; batch them so the
  XLA cache stays warm (identical graph shapes across arms). Wave 3 is ~12 arms
  x 2 seeds = 24 runs ≈ 3 rounds of 8 ≈ 2-3 h including scheduling.
* Both waves start simultaneously; neither blocks the other.
* Reserve the last 2 h for **second seeds on whichever arms decided something**,
  and for a video of the winning configuration. No claim ships without a video —
  that rule was earned today.

## Decision tree

```
W0.1-W0.3: is urma2's reference grounded + feasible where ours is not?
├─ YES ──> reference branch. X1 should cross, S5 should collapse.
│          Then A2/A3 say whether it is grounding or window choice.
│          Fix = retargeting, not the trainer.
└─ NO ───> trainer branch. X2 should cross, X1 should fail.
           Wave 2 names the component: S1 (action centring), S2 (action scale),
           S3 (control rate), S4 (reward target), S6 (reward composition).
           The first arm to collapse below 40% of horizon at 15M is the blocker.
```

In both branches S7/S8 must still pass, or the result does not meet the
requirement of a working pipeline **with** randomization and per-body retargeting.

## Traps this plan is written to avoid

* Reading the aggregate `len` (killed three claims today).
* Judging by sampled video stills right after an RSI reset (killed one).
* Believing a single seed.
* Comparing across pipelines without converting to seconds and metres.
* Assuming a config default carries over on replay — the renderer's
  `tracking_reference_action_bias` default of 0.0 would silently mis-drive a
  policy trained at 1.0, loading cleanly and looking like a failure.
