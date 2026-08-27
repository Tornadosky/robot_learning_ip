# FSQ — next wave on Viper

Written 2026-08-26 from the overnight results. Everything here is scoped so the
*decision* is made by a cheap measurement before any 2-hour training arm is spent.

---

## STATUS after the night of 2026-08-26/27

Full numbers in `experiments/fsq_khaendler/REPORT_FSQ_WAVE2.md`.

| idea | verdict |
|---|---|
| **E1** reference degradation | **RAN, and it is the night's result.** The scramble control passed decisively (M=5: real token vs scrambled t ≈ 6.6, p < 0.001; the scrambled token is *worse* than no token). Reference degradation shows the token arm losing ~3× less than reference-only at every staleness. Two of E1's three named controls are still owed — see below. |
| **E2** canonical decoder bottleneck | **CLOSED by its own gate.** `gate_canon` measured 0.1774 rad against a 0.10 gate and declined to submit the RL arm. But decoder width was the *right* suspect: 0.2486 → 0.2178 → 0.1774 across two doublings, monotone and unsaturated, against 6 % for a 32× codebook and 0 % for encoder sites. It is a real limiter that is not enough on its own. |
| **E3** rate curve | **DONE at n=4.** Knee between 20 and 8 tokens/s; h=2 is marginally *better* than h=1. |
| **E4** unblock the third robot | **BLOCKED, twice over, and neither reason was the one in this document.** See below. |
| **E5** novel-motion generator | prep ran; clips regenerated at three tolerances. |
| **E6** token as retargeter | inherits E2's gate and is therefore not opened. |

### E1 RESULT: gate passed, and the mechanism is narrowed

Both halves of E1's gate hold. Token helps more as the reference degrades
(4.0 % at 40 Hz -> 14.0 % at 8 Hz -> 11.6 % at 4 Hz, both channels equally
stale), and the scramble does not reproduce it (real vs scrambled t ~ 6.6 at
M=5; the scrambled token is *worse* than no token).

Control 2 (raw lookahead) came back NEGATIVE in its shift form: moving the
observed reference L frames forward is 16.5 % worse at L=5 and 79.4 % worse at
L=10. So the token is not a future sample of the reference. Still owed: the
ADDITIVE form (a future channel alongside the present), which needs an extra
observation channel and a width change to `get_observation_space`.

Control 3 (smoothed reference, no token) was not run.

**New in E1's neighbourhood:** the token's gain is unaffected by morphology
randomization (-4.0 % with and without) but vanishes with a second topology
(-0.5 % H1, +3.5 % worse G1). The next E1-shaped question is why.

NOT because the token is H1-only: `tokenizer_M9/config.json` records
robots ['UnitreeH1','UnitreeG1'] and G1 carries its own 23-joint z_q, so the
tokenizer is ALREADY jointly fitted. That hypothesis was checked and killed
before it cost a slot. What the same config does show is that G1 reconstructs 2x
worse than H1 (0.0130 vs 0.0064), leaving two candidates: (a) one policy shared
across two bodies cannot exploit the extra channel, or (b) G1's token is simply
weaker and the 2t null is G1's null in the average. A single-topology G1 pair
(M5_g1_ref / M5_g1_both, never run before) separates them.

### What actually blocks E4

This document assumed the third robot was blocked by missing retargets. Two
separate walls were found instead, and both are now measured:

1. **The trainer supports exactly two topologies on this ROCm stack.** Every
   one- and two-topology combination trains, including both pairs involving
   booster_t1. Every three-topology combination aborts with
   `ROCM_ERROR_ILLEGAL_ADDRESS` — at 768 and 384 envs per robot, under
   `urma2.mjx` and `urma2.mjx_split`, with and without
   `--xla_gpu_enable_command_buffer=`, and in either robot order. Each abort is a
   *different* kernel, which is the signature of an out-of-bounds write rather
   than one broken kernel. `mjx_split` does **not** rescue it, contrary to the
   standing assumption.
2. **Only three families have usable data.** The BoosterT1 clip was already on
   the cluster and needed only a `tracking_clip_robot_map` entry. But of the four
   further families uploaded, none survives a forward-kinematics screen against
   loco_mjx's own models — atlas 0.043, talos 0.83, toddlerbot 2.40 residual
   against ~1.5e-4 for the three good ones, and h1v2 has no loco_mjx model at
   all. Use `scripts/scaling/derive_clip_signs.py` to screen a family before
   spending an arm on it; it also derives that family's `CLIP_SIGNS` table from
   the clip alone, with no loco-mujoco import (which no longer works locally).

### THE RECIPE DEFECT — bigger than anything FSQ-shaped

Measured 2026-08-27 from the trainer's own logs, which have carried it since the
project began. The policy does not FAIL to learn heading and foot placement, it
SELLS THEM OFF:

  M9_ref band-averaged     0-1M    3-6M    6-10M   10-15M  20-30M  85-100M
  foot-lift ratio          1.37    0.82    0.98    0.42    0.25    0.20
  heading error (deg)      10.9    26.9    65.5    83.4    83.2    85.0
  joint tracking error    0.296   0.077   0.056   0.045   0.034   0.026

Heading goes between 6M and 10M, the feet between 10M and 20M, joint error
improves monotonically. It is a trade against the only heavily weighted term
(joint_tracking_coeff=30 at temperature 0.05). NOTHING before ~20M is readable:
at 3M the baseline sits at ratio 0.93 and looks healthy.

Fix arms (30M steps, read at 15-20M against the baseline's own 15-20M band):
  FOOTH=1.0 alone            ratio 0.30 -> 0.48, clearance 6.2 -> 15.0 mm
  FOOTH=1.0 + FOOTZVEL 10->1 ratio 0.30 -> 0.54, penetration 7.24 -> 4.48 mm
  ...costing +44 % joint error
  heading term at 2.0        85.2 deg vs the baseline's 83.6 -- NO EFFECT, and
                             +109 % joint error

So the foot half is recoverable by re-weighting and the heading half is NOT.
Three independent attempts have now failed to move heading with this term
(M9_head05/M9_head20, fx_all, and the older dp_head_* probes). Next step is to
read what `root_heading_tracking_reward` actually computes rather than sweep the
weight a fourth time.

### New, and not in this document

- **The crosseval never measured heading.** Trained arms track joints ~3× better
  than the zero-action floor and heading **15× worse** (82.4° ± 1.8 vs
  5.4° ± 0.1, n=4). `heading_error_deg_mean`/`_p95` are now emitted. Every FSQ
  comparison ever made here was between arms all failing the same way on an axis
  none of them was scored on.

Reference material: `experiments/fsq_khaendler/REPORT_FSQ_SCALE.md` (numbers),
`fsq_curves.png` (training curves), `fsq_crosseval_bars.png`,
`FSQ_STATUS_FOR_SUPERVISORS.md` (plain-language summary).

---

## What the curves tell us before we design anything

Look at panels 1a/1b/1c of `fsq_curves.png`. Episode return, training joint error
and episode length are **the same for reference-only, token-only and both** —
they converge together and saturate (episode length pins at ~980 of 1000). So:

> **The training metrics cannot see the effect we are studying.** Return and
> episode length are saturated; only the executed-vs-clip crosseval separates the
> arms, and only at n = 4 seeds.

That is a methodological result in its own right, and it should govern the wave:
never judge an FSQ arm by its training curve.

The one place the curve *is* decisive is panel 3: the shared canonical token
never converges at all (0.28 vs 0.03). A failure that large is visible anywhere.

---

## E1 — Degrade the reference, keep the token  *(the most promising)*

**Why.** Adding the token *on top of* the reference beat reference-alone:
−3.5 % at M=9 (t=1.80) and −1.9 % at M=5 (t=2.10). That is the first positive FSQ
result. The mechanism is probably **lookahead**: the reference channel shows the
target at the current instant only, while the token summarises frames t…t+9 —
and the clip runs at 40 fps, so the token carries **250 ms of future** the
reference channel never shows.

If that is right, the token should help *more* as the reference gets worse, and
the natural product claim is: a cheap side-channel that keeps tracking alive when
the primary command is slow, stale or noisy.

**Arms (4).** Reference degraded to 8 Hz and to 4 Hz (hold the reference the way
we hold the token), each with and without the token.

**Controls, and the wave is worthless without them (3 more arms).**
1. **Scrambled token** — same token stream, time-shuffled, so its lookahead is
   invalid. If the gain survives, it is regularisation, not information.
2. **Raw lookahead, no token** — feed the reference at t+5 and t+10 uncompressed.
   If that reproduces the whole gain, the benefit is lookahead and FSQ is merely a
   *compact way to deliver it* (still a good result: 32 numbers vs two extra
   19-dim targets).
3. **Smoothed reference, no token** — because holding the token at 20/s beat 40/s
   by 3.7 %, plain temporal smoothing is a live explanation.

**Gate.** Token helps more at 4 Hz reference than at 40 Hz, AND the scrambled
control does not reproduce it.
**PASSED 2026-08-27 on both halves** — see the STATUS block at the top. The
degradation arms were run as CROSSEVALS on existing checkpoints rather than as
four new training arms, using a new `tracking_clip_reference_hold` knob; that
turned a 4-arm, 12-hour experiment into twenty minutes of 4-minute jobs.

---

## E2 — The canonical decoder bottleneck  *(cheapest, no RL arm)*

### What the bottleneck is

The canonical tokenizer is a chain:

```
task-space features → encoder → FSQ quantiser → code z → decoder(z, joint description) → joint angle
```

We tested three links and eliminated two:

| link | what we changed | result |
|---|---|---|
| codebook size (how many distinct codes exist) | 1 000 → 32 768 | 6 % better |
| encoder input (what it can see) | 4 sites → 14 sites incl. knees & elbows | **0 % better** |
| code width (numbers per code) | 4 → 16 → 32 | **32 % better** (0.366 → 0.249) |

At 32 numbers per frame the code is **nearly unique per frame** — 42 978 distinct
codes for 45 125 frames. Every frame effectively has its own address, and
reconstruction is *still* 0.25 rad against the per-joint design's 0.05.

> If every frame has a unique address and the motion still cannot be reproduced,
> the problem is not the address. It is whatever reads it.

What reads it is the **decoder**: one shared network mapping
(code, 47-dim joint description) → that joint's angle. It is shared across all
joints *and* both robots — a single function that must answer 42 different
questions from the same code, at `network_width_multiplier = 1.0`.

### The tests

1. **Decoder width sweep** — 2x and 4x. One flag, ~1 h each, no RL arm.
2. **Unique-ID upper bound** — replace the code with a one-hot frame index and
   train only the decoder. This is the *best case the decoder can ever achieve*.
   If a perfect address still reconstructs at ~0.25 rad, the decoder or its
   conditioning is proven to be the wall, and no encoder work will help.

**Gate.** Reconstruction below 0.10 rad. Only then does a cross-topology RL arm
become worth its 3 hours. If the unique-ID bound is itself above 0.10 rad, the
canonical direction closes on measurement.

---

## E3 — Finish the rate curve  *(cheap, and it already has a surprise)*

Measured at M=9, executed-vs-clip RMSE relative to 40 tokens/s:

| rate | change | seeds |
|---|---|---|
| 40 tok/s (hold 1) | — | 4 |
| **20 tok/s (hold 2)** | **−3.7 %** | 4 |
| 8 tok/s (hold 5) | +8.9 % | **1** |
| 4 tok/s (hold 10) | +24.2 % | **1** |
| 2 tok/s (hold 20) | +67.6 % | 4 |

**The curve has a minimum at 20 tok/s, not at the highest rate.** Slowing the
token down *helps* before it hurts. Two points are single-seed against a ~2-5 %
noise floor, so the shape is not yet trustworthy.

**Work.** Four seeds at hold 5 and hold 10 (8 short evals, no training). Then one
control arm: a **smoothed reference** at the equivalent rate, to test whether the
20 tok/s gain is the token or just low-pass filtering.

Already settled and not worth repeating: a **longer window does not rescue low
rates**. Lookahead 20 at hold 10 gave 0.1665 vs 0.1651 for lookahead 10 — no
change. The "held code goes stale" explanation is dead.

---

## E4 — Unblock the third robot  *(data work, not research)*

Probes at 3, 4 and 6 topologies all died in under 2 minutes:

```
ValueError: No clip mapping for robot 't1' / 't1'.
  known: ['g1', 'h1', 'unitree_g1', 'unitree_h1']
```

**This is not a memory or compute wall** — earlier notes suspected one, and the
probe shows otherwise. It is a missing offline retarget: the multi-motion clip set
has only `UnitreeH1/` and `UnitreeG1/` subdirectories.

**Work.** One offline retarget of the super clip per family (booster_t1 first,
then talos / toddlerbot), then re-run the 3/4/6-topology probes to get the real
compute-scaling curve.

**Why it matters.** Body-independent tokens are only *interesting* across many
bodies. Right now the embodiment-agnostic claim cannot even be tested beyond two
robots, so this unblocks the entire direction — and it is a day of data work.

**Known cost for planning.** The environment splits a fixed env budget across
robots (`create_env.py:88`), so a second robot halves each robot's experience.
Matching per-robot data costs ~2.7x, not 2x. If the 3-robot probe shows that
worsening, `urma2.mjx_split` (per-robot graphs) is the path.

---

## E6 — Use the token as a RETARGETER, not as a control channel  *(new, highest ceiling)*

This came out of measuring the sidecars today, and it reframes the whole
body-independent direction.

**What we found.** In the shared-token arm, all 19 joints receive the *identical*
4 numbers — max difference between joints in a frame is exactly 0.0, one distinct
value per frame. Kevin's per-joint sidecar has 17 distinct values per frame. So
the canonical arm was never "a token instead of a reference": it was a **global**
signal with no per-joint addressing, and the policy had to solve an inverse
problem the reference channel never poses. That is a large handicap that has
nothing to do with whether body-independent tokens are a good idea.

**The reframe.** Do not feed the shared code to the policy at all. Feed it to the
**decoder**, and give the policy the decoded **per-joint targets** through the
ordinary reference channel. The observation then has exactly the structure the
reference arm has; the only difference is that the information travelled through
a body-independent bottleneck on the way.

Stated that way, the token stops being a control interface and becomes an
**online cross-body retargeter**: encode motion from H1, decode it onto G1's
joints, and you have a G1 reference without running the offline SMPL retarget for
that family. That is a thing this project actually needs — the current plan is
"loco-mujoco once per family, then an online retargeter", and this would be the
online retargeter.

**And it is measurable with no RL at all.** Encode from H1, decode onto G1, and
compare against the properly retargeted G1 clip we already have. One script, one
hour, one number.

**Gate.** Decoded-onto-G1 within ~0.10 rad of the offline-retargeted G1 clip. For
scale: the offline retarget is what every arm has trained on, and Kevin's
same-body reconstruction is 0.05 rad.

**Honest caveat.** This is gated on E2. Canonical currently decodes onto its OWN
body at 0.25 rad, and decoding onto a different body cannot be easier. If the
decoder tests do not move that number, E6 fails for the same reason and both
close together.

## E5 — Retune the novel-motion generator  *(small, fixes my own bad parameter)*

`stitch_novel_clip.py` produced only 46 cuts in 9 000 frames — 99.5 % of the
"novel" sequence was contiguous original footage, so it did not test novelty. The
code-distance tolerance was too tight.

**Work.** Sweep tolerance until the consecutive fraction is ~0.7-0.8, then re-run
the existing evaluation (both policies already exist, no training). Report the
novelty explicitly (cuts, unique source frames, distance from any training
window) so the claim is bounded.

Note the honesty limit: because the stitched motion is made of real frames, an
explicit-reference control **does** exist, so this is not the "no control
possible" test. That one needs text→codes, which we cannot do yet.

---

## Suggested order

| priority | item | cost | decides |
|---|---|---|---|
| 1 | E2 decoder tests | 2 short jobs, no RL | whether body-independent tokens live or die |
| 2 | E3 seeds + smoothing control | 8 evals + 1 arm | whether the 20 tok/s gain is real |
| 3 | E1 degraded reference + 3 controls | 7 arms | whether the auxiliary gain is lookahead |
| 4 | E4 retarget booster_t1 | data work | unblocks everything multi-robot |
| 5 | E5 generator retune | 1 short job | makes the novelty claim testable |

E2 first because it is the cheapest and the most decisive: it can close a whole
direction for the price of two 1-hour jobs.
