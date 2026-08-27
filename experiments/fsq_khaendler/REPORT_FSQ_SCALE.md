# FSQ-SCALE — does the token interface earn its place?

**2026-08-25 22:00 → 08-26. Viper only. Dashboard: `dashboard_fsq_scale.html`
(regenerate with `build_fsqscale_dashboard.py`). Raw crossevals: `ce_fsqscale/`
(63 files) — every table below is generated from those. Shareable page:
https://claude.ai/code/artifact/918d2e26-ebac-4396-ab22-20d54da1418a**

## THE ANSWER

> **The token interface loses ~2 % to the explicit reference on one topology —
> 1.1–3.3 % across M ∈ {1, 4, 5, 9}, at n = 4 rollout seeds per point, against a
> matched control at every M — and ~6 % when one policy covers two topologies.
> That penalty does not shrink as the motion set grows, and it does not grow
> either. The shared embodiment-agnostic stream, the one place a token could not
> lose, tracks worse than a zero action — because the canonical tokenizer
> reconstructs at 0.366 rad against Kevin's 0.051, which was measured and stated
> before its RL arm ran.**

Three gates, three verdicts: **C1 falsified** (no M-dependence), **C2 failed**
(and a 32x bigger codebook proves it failed on the canonical *feature set*, not
on capacity — so it is a design problem, not a tunable), **C3 partially met**
(8 tokens/s costs 5–13 %, 4 tokens/s costs ~24 %; the rate floor is between them).

The one thing tonight found that was not on the list: **rollout seed alone moves
an arm by up to 4.95 %**, which is the size of every FSQ effect ever reported on
this stack, including the −5.8 % this campaign was built to explain.

No video was produced — see the access note.

Live report. Recipe fixed (`TRACK_TEMP=0.05`, H1 nominal, `ANCHOR=absolute`,
`REFROOT_FLOOR=True`, `REFBIAS=0.0`, `QVEL_TEMP=10`, contact dose
`floor/0.25/1000/20/10`, 98.304 M steps, 768 envs). Every number below is from
Viper; Viper and local eval numbers are not interchangeable.

## The axis being measured

The explicit reference is **also** fixed-width (19 numbers on H1) and does not
grow with M, so "fixed vs growing width" was never the axis. Three structural
differences remain, and only they get slots:

1. reference is instantaneous, token encodes a **window** → predicts the token
   penalty **shrinks with M** (C1);
2. canonical token is **embodiment-agnostic**, the reference cannot be → the
   control cannot be constructed (C2);
3. the token can be held at a **low rate** (C3).

## Status — 2026-08-25 22:35 CEST

| job | arm | M | interface | state | updates |
|---|---|---|---|---|---|
| 11019837 | M1_z | 1 | z-only | R | 27810 |
| 11019838 | M1_z_h5 | 1 | z-only, hold 5 | R | 27930 |
| 11019839 | M5_ref | 5 | reference | R | 27570 |
| 11019840 | M5_z | 5 | z-only | R | 27930 |
| 11019841 | M5_z_h5 | 5 | z-only, hold 5 | R | 27480 |
| 11019842 | M5_z_h10 | 5 | z-only, hold 10 | R | 27660 |
| 11019843 | M5_2t_ref | 5 | reference, H1+G1 | R | 17730 |
| 11019844 | M5_2t_z | 5 | z-only, H1+G1 | R | 17400 |
| 11021360 | M4_ref | 4 | reference | R | 5850 |
| 11021361 | M9_ref | 9 | reference | R | 5850 |
| 11022118 | tok_kevin (C2 v2) | — | tokenizer fit | PD | — |
| 11022119 | M4_z | 4 | z-only | PD (afterok C2) | — |
| 11022120 | M9_z | 9 | z-only | PD (afterok C2) | — |

`X_temp005` (already trained, 25-08) is the **M=1 explicit-reference control** at
this exact recipe. So the reference side of the M-curve is M ∈ {1, 4, 5, 9} and
the token side matches it.

## Fixed tonight

**C2 v1 (11021362) died in 13 s and took Tier B with it** (`afterok` →
M4_z/M9_z CANCELLED). Cause: the Viper shim exposed only `tools/lm_algorithms`
as `loco_mujoco.algorithms`, but `autoencoder/__init__` → `train/train_fn.py`
imports `loco_mujoco.core.trajectory.dataclasses` and
`loco_mujoco.environments.mulitenvironment`, and Viper carries **loco_mjx, not
loco-mujoco**. `lm_algorithms` has no `__init__.py`, so it imported as an
implicit namespace package and the failure surfaced one level deeper than the
symlink.

Two fixes, both verified on the login node before resubmitting (`IMPORT_OK`):

1. **Stub the two modules.** `train_fn.train()` is never called from this
   pipeline — `cmd_train` drives `train/step.py` directly, which needs only
   jax/optax/flax. The `nn/` subtree is pure flax.
2. **`--desc-cache`** (new flag on `scripts/scaling/khaendler_fsq_clip.py`). The
   47-dim joint descriptions depend only on the **nominal** model, so they are
   identical across every fit — checked byte-for-byte across the four local
   tokenizers (`tokenizer`, `kevin_tokenizer_mc`, `_denoise`, `_denoise2`);
   H1 (19, 47), G1 (23, 47). The cached `descriptions.npz` therefore replaces
   the `RLFactory` model compile that needed the whole loco-mujoco stack.
   `cmd_reconstruct` already read descriptions from the tokenizer directory, so
   it needed no change.

**A second, silent defect fixed in the same pass.** C2 v1 ran
`reconstruct --out $CL` — into the clip directory itself. `cmd_reconstruct`
writes a **FSQ-reconstructed copy of the clip** next to the sidecar, so that
would have overwritten `superM4.npz`/`superM9.npz` in place: the z arms would
have tracked the tokenizer's reconstruction instead of the true motion, and
M4_ref/M9_ref would have been reading a corrupted reference on any restart.
v2 reconstructs into `clips/clips_Mrec` and copies back **only** the
`*_zq.npz` sidecars, with an assert that sidecar frames match clip frames.

## C2 built — the canonical shared stream (22:50)

Wave 1's two-topology pair is **not** the topology-invariance test, and its own
submit script says so: `M5_2t_z` uses Kevin's **per-robot** tokens, so each robot
gets its own stream. It measures the cost of a second topology, with a control.

C2 proper was not built, so it was built this tick, matched to those two arms on
the identical clip (`super5dance`, 45 125 frames):

- `canonical_bundle_M5` — 22-dim canonical task-space features from **H1's clip
  only** (root height/attitude/velocities + 4 end-effector positions in the root
  frame), with per-robot decoder targets and 47-dim joint descriptions for both
  robots. Built locally (needs the loco-mujoco model compile), uploaded, and the
  `clip_dir` in `meta.json` rewritten to the Viper path — the recorded one was a
  backslashed Windows relative path that would not resolve on Linux.
- `canon_c2` **11022549** → `M5_2t_canon` **11022551** (`afterok`). One shared
  FSQ stream, codebook (8, 5, 5, 5) = 1000 codes, **one code per frame**,
  `latent_dim=4`.
- New `scripts/scaling/canonical_zq_sidecar.py`: `canonical reconstruct` writes
  `z_canonical.npz` (shared), but `tracking_clip.py` reads a per-robot
  `<clip>_zq.npz` of shape (T, J, latent) and maps it by joint **name** (line
  241). The tiler broadcasts the shared code across each robot's joints, so H1
  (19 wide) and G1 (23 wide) receive byte-identical content, and copies the clip
  through **unchanged** so the z-only arm still tracks the true motion. The job
  asserts `z_H1 == z_G1` element-wise before the RL arm is allowed to start.

So the two-topology comparison at M=5 is a **three-way** one, all on one clip:

| arm | channel | control status |
|---|---|---|
| `M5_2t_ref` 11019843 | explicit reference, per robot | the strongest control |
| `M5_2t_z` 11019844 | Kevin tokens, **per robot** | cost of a 2nd topology |
| `M5_2t_canon` 11022551 | canonical, **one shared stream** | no reference control exists |

**Stated before the number arrives:** canonical reconstruction is pinned at
0.310 rad (H1) / 0.284 rad (G1) at M=1, and 2026-08-22 measured that this does
not move with more data, more epochs, or a 1M-code codebook (0.343 / 0.314 at
`richbig`, 0.383 / 0.346 at `rich40`). 22 task-space dims underdetermine elbows
and knees. If `M5_2t_canon` is poor, that number says it is the tokenizer.

## Tokenizer reconstruction — measured BEFORE any RL number rides on it

**Kevin's per-joint FSQ**, fitted on `superM9` (46 770 frames, H1+G1 jointly,
400 epochs, latent 32, `tok_kevin` 11022118, 14 min):
final train 0.0266, eval H1 0.0064 / G1 0.0130.

| clip | H1 qpos RMSE | H1 max | G1 qpos RMSE | G1 max | unique per-joint codes |
|---|---|---|---|---|---|
| superM4 | **0.0529 rad** | 1.045 | **0.0898 rad** | 1.101 | 582 k/686 k H1, 705 k/830 k G1 |
| superM9 | **0.0508 rad** | 1.613 | **0.0896 rad** | 1.294 | 697 k/889 k H1, 837 k/1076 k G1 |

Note the error does **not** grow from M=4 to M=9 — nine motions cost this
tokenizer nothing in fidelity, which is the precondition for the C1 curve to be
about the *interface* rather than about the code running out of capacity. Code
utilisation is 78–85 %, so the book is not saturated either.

**Canonical (shared-stream)**, fitted on the same M=5 super clip (45 125 frames,
150 epochs, book 8x5x5x5 = 1000 codes, one code per frame, `canon_c2` 11023598,
1 h 19):

| | H1 | G1 |
|---|---|---|
| qpos RMSE | **0.3661 rad** | **0.3402 rad** |
| max abs err | 2.530 rad | 2.576 rad |

**~7x worse than Kevin's**, and the reason is now measured rather than assumed:
**all 1000 codes are used** (1000/1000 over 45 125 frames). At M=1 the same
design left the book unsaturated; at M=5 it is completely full, so canonical is
**capacity-limited here, not only architecture-limited**. That is a concrete,
fixable statement — a bigger book is a one-line change — and it means a weak
`M5_2t_canon` is a statement about a 1000-code bottleneck, not about
embodiment-agnostic tokens as an idea.

The shared-stream property itself is verified, not assumed: the job asserts
`z_H1 == z_G1` element-wise before the RL arm starts, and sidecars are
(45 125, 19, 4) for H1 and (45 125, 23, 4) for G1 — the same 4 numbers per frame
tiled across each robot's own actuators.

So any weakness in `M5_2t_canon` is attributable to the tokenizer; weakness in
`M4_z`/`M9_z` is **not** — at 0.05 rad the per-joint code is a faithful channel.

## Two traps closed ahead of the evaluations (23:20)

Both would have produced a wrong number or no number at all when wave 1 lands.

1. **`crosseval_motion.py` hardcoded `tracking_clip_latent_dim = 32`.** That is
   Kevin's per-joint width. The canonical stream is **4** (one code per frame,
   book 8x5x5x5), so `M5_2t_canon` could not have been evaluated at all — the
   observation width would not match the checkpoint. Added `--latent_dim`
   (default 32, so every existing invocation is unchanged) and `CE_LATENT_DIM`.
2. **No zero-action baseline existed in the harness.** The C2 gate is stated
   against one ("clearly better than zero-action"), and an earlier zero-shot
   claim died for want of exactly this control. Added `--zero_action` /
   `CE_ZERO=1`, which rolls out with a zero action of the correct per-robot
   width (`as_shape[0] - missing_nr_of_actions[i]`) instead of the policy.

`canon_c2` v1 (11022549) was also **killed by its own arithmetic** and resubmitted
as **11023598**: 32 s/epoch on 45 125 frames (the URMA decoder runs per robot
over 11 lookahead rows x J joints) means 500 epochs is 4.4 h against a 3 h limit,
`afterok` would have failed, and `M5_2t_canon` would never have started. Now 150
epochs (26 400 optimizer steps — *more* than the 08-22 fits that established the
~0.3 rad floor; its eval loss was already rising by epoch 40: H1 2.044 -> 2.104)
with a 6 h limit. New chain: `canon_c2` 11023598 -> `M5_2t_canon` 11023599.

## Wave 1 landed (23:45) — all six COMPLETED, crossevals in flight

`M1_z` 11019837, `M1_z_h5` 11019838, `M5_ref` 11019839, `M5_z` 11019840,
`M5_z_h5` 11019841, `M5_z_h10` 11019842 — all `COMPLETED 0:0`, ~2 h 10 each.
Seven crossevals submitted (11024580-86), each naming its condition, at
`CE_REFBIAS=0.0` / H1 alone / `REFROOT_FLOOR=True` to match the arms, with
`CE_RAW_DIR=clips_super` for the M=5 arms (they are not LAFAN1) and `CE_HOLD`
reproducing each arm's trained token rate. The seventh is the **zero-action
floor** on the same body and clip, so "the policy tracks" gets a denominator.

Queue refilled with **`M9_z_h5` 11024649 and `M9_z_h10` 11024650**: the rate
sweep sat at M=1 (hold 1, 5) and M=5 (hold 1, 5, 10), and the C3 gate wants rate
at two values of M. M=9 makes it three and asks whether holding the token costs
more or less as the motion set grows — the C1 window argument read on the time
axis. The `superM9` sidecars already exist, so these needed no upstream job.

## The M-curve is capped at M=9, and that is a data limit, not a choice

`external_data/amass/DanceDB` holds 85 raw AMASS npz files, and
`external_data/amass_converted/AMASS/` contains only `shape_optimized.pkl`
SMPL fits — **no DanceDB motion has ever been retargeted on this stack**. The
retargeted corpus is LAFAN1: 5 dances + 5 walks = 10 clips, one of which
(`dance2_subject1`) is the known-bad control and is excluded. So the honest
M-curve tonight is M in {1, 4, 5, 9}, not the {1, 5, 15, 30+} the gate names.
Four points still test monotonicity; they just cannot reach 30. Closing that gap
needs the SMPL retarget + the four screening tools run over ~76 motions, which
is a night of its own and was explicitly kept off the critical path.

## Results — first numbers, 00:20 CEST

All crossevals: H1, 64 envs, 1000 steps, executed joints scored against the raw
clip at the same reference phase. `rmse` is the centred (shape) metric,
`rmse_abs` the absolute one; both are on the same machine and harness, so they
are comparable to each other and to the 25-08 ladder.

| arm | M | channel | hold | rmse (rad) | rmse_abs | alive | foot pen. |
|---|---|---|---|---|---|---|---|
| `X_temp005` | 1 | reference | — | **0.1365** | 0.1457 | 1.000 | 3.2 mm |
| `M1_z` | 1 | token | 1 | **0.1366** | 0.1432 | 1.000 | 3.3 mm |
| `M1_z_h5` | 1 | token | 5 | 0.1546 | 0.1573 | 1.000 | 3.0 mm |
| `M5_ref` | 5 | reference | — | **0.1436** | 0.1586 | 1.000 | 3.8 mm |
| `M5_z` | 5 | token | 1 | **0.1446** | 0.1538 | 1.000 | 3.5 mm |
| `M5_z_h5` | 5 | token | 5 | 0.1532 | 0.1618 | 1.000 | 3.2 mm |
| `M5_z_h10` | 5 | token | 10 | 0.1815 | 0.1900 | 1.000 | 3.3 mm |
| `M5_ref` **zero action** | 5 | — | — | **0.4107** | 0.4918 | 0.931 | 13.7 mm |

### C1 so far — the token MATCHES the reference, it does not lose

- **delta(1) = +0.0001 rad** (0.1366 vs 0.1365) — a dead tie.
- **delta(5) = +0.0010 rad** (0.1446 vs 0.1436) — **+0.7 %**.

Both are ~2.9x better than the same-body zero-action floor (0.4107), which is
what makes "it tracks" a claim rather than an assertion. The floor also loses
6.9 % of its episodes and penetrates the ground 13.7 mm against 3–4 mm for every
trained arm.

Caveat, stated rather than buried: the M=1 pair is **not perfectly matched**.
`M1_z` trained on `clips_kevin_mc` — the FSQ-*reconstructed* clip — so its
reward target sits 0.0524 rad from raw, against 0.0445 for `X_temp005`. That
handicaps the token arm, and it tied anyway. The M=5 pair **is** perfectly
matched: both train on the identical original `super5dance`, and only the
observation channel differs.

### The 25-08 penalty was mostly the recipe, not the interface

Same clip, same harness, same machine, `TRACK_TEMP` the only difference:

| recipe | reference | token | delta |
|---|---|---|---|
| 25-08, `TRACK_TEMP=0.25` (`Fz_multi_*`) | 0.1768 | 0.1859 | **+0.0091 (+5.1 %)** |
| tonight, `TRACK_TEMP=0.05` (`M5_*`) | 0.1436 | 0.1446 | **+0.0010 (+0.7 %)** |

The penalty shrank **9x** when the shared baseline got better, and the token arm
improved more in absolute terms than the reference arm did (−0.0413 vs −0.0332).
So "the token interface loses ~5 % at multi-motion", measured four campaigns
running, was substantially an artefact of a weaker baseline. Under the best
recipe we know how to build, it is a tie.

### C3 — the cost of holding the token SHRINKS with M

| M | 40 tok/s (hold 1) | 8 tok/s (hold 5) | 4 tok/s (hold 10) |
|---|---|---|---|
| 1 | 0.1366 | 0.1546 (**+13.2 %**) | — |
| 5 | 0.1446 | 0.1532 (**+5.9 %**) | 0.1815 (+25.5 %) |

Holding the token to 8 per second costs 13.2 % at one motion but only 5.9 % at
five — the *same direction* the window argument predicts for C1, read on the
time axis instead of the motion axis. `M9_z_h5` / `M9_z_h10` (11024649/50) will
say whether it continues to M=9. At 4 tokens/s the interface clearly breaks
down (+25.5 %), so the usable floor is ~8 tokens/s, not 4.

## The reference side of the M-curve — more motions do not cost accuracy

| M | clip | reference rmse |
|---|---|---|
| 1 | dance2_subject4 | 0.1365 |
| 4 | superM4 (4 dances) | 0.1430 |
| 5 | super5dance (5 dances, incl. the bad one) | 0.1436 |
| 9 | superM9 (4 dances + 5 walks) | **0.1341** |

M=9 tracks *better* than M=1. The multi-motion plateau that shaped three earlier
campaigns is not visible at this recipe: nine motions in one policy cost nothing
against one motion, and the M=5 clip is the worst of the four precisely because
it is the one that still contains `dance2_subject1`. The token arms at M=4 and
M=9 are in flight (11027926/27).

## The two-topology pair at M=5 — n = 4, and this penalty IS real

| robot | reference | Kevin tokens (per-robot) | delta | t |
|---|---|---|---|---|
| H1 | 0.1555 ± 0.0030 | 0.1644 ± 0.0049 | **+5.74 %** | 3.09 |
| G1 | 0.1308 ± 0.0019 | 0.1399 ± 0.0007 | **+6.95 %** | 8.98 |
| — zero-action floor — | H1 0.4126 / G1 0.3948 | | | |

Unlike the M-curve deltas, this one survives four seeds comfortably — t = 3.1 on
H1 and t = 9.0 on G1. **The token penalty is about three times larger when one
policy covers two topologies (+6 %) than when it covers one (+2.2 %)**, on the
same clip, the same recipe and the same number of steps.

That is the opposite of what the token argument wants. The hope was that a code
would generalise across bodies better than a body-specific reference; measured,
sharing a policy across H1 and G1 makes the token channel *more* expensive, not
less.

Two more facts from the same table:

1. **A second topology costs H1 9.4 % on the reference channel alone**
   (0.1422 -> 0.1555), so most of the two-topology cost is not about tokens.
2. **G1 outperforms H1 inside the shared policy** (0.1308 vs 0.1555) — the same
   ordering recorded in the 25-08 ladder, now with error bars.

### The two-topology penalty vs M — no consistent effect there either

| robot | M=5 | M=9 |
|---|---|---|
| H1 | +5.74 % (t = 3.09) | +2.93 % (t = 1.17) |
| G1 | +6.95 % (t = 8.98) | **+8.02 % (t = 7.33)** |

At M=9: reference H1 0.1521 ± 0.0058 / G1 0.1241 ± 0.0011, token H1 0.1565 ±
0.0049 / G1 0.1340 ± 0.0025, zero-action floor H1 0.4000 / G1 0.3887.

The penalty falls on H1 and **rises on G1**. Averaged over both robots it moves
from +6.3 % to +5.5 %, which is nothing. So the window argument fails on the
topology axis as well as the motion axis: **there is no value of M, on either
robot, at which the token channel stops costing what it costs.**

## C1 — ANSWERED, at n = 4 per point. The gate is falsified; the penalty is real but M-independent.

Four rollout seeds per arm, H1, executed joints against the raw clip.

| M | reference (mean ± sd) | token (mean ± sd) | delta | delta % | t = delta/SE |
|---|---|---|---|---|---|
| 1 | 0.1322 ± 0.0038 | 0.1366 ± 0.0021 | +0.0044 | **+3.31 %** | 2.01 |
| 4 | 0.1366 ± 0.0050 | 0.1386 ± 0.0037 | +0.0020 | **+1.44 %** | 0.63 |
| 5 | 0.1422 ± 0.0021 | 0.1465 ± 0.0022 | +0.0042 | **+2.97 %** | 2.73 |
| 9 | 0.1314 ± 0.0036 | 0.1329 ± 0.0028 | +0.0015 | **+1.13 %** | 0.65 |

**The gate — "delta shrinks monotonically with M" — is FALSIFIED.** The sequence
is 3.31 → 1.44 → 2.97 → 1.13 %: no trend, in either direction, across a 9x range
in M. The window argument predicted that an instantaneous per-joint target would
become progressively more ambiguous as motions accumulate. It does not.

**What the four points DO show is a small, consistent, M-independent penalty.**
Every delta is positive (sign test on four points, p = 0.0625) and they average
**+2.2 %**. So the token interface is neither at parity nor at the −5.8 % of
25-08: it pays about two percent, and it pays the same two percent whether the
policy covers one motion or nine.

Two single-seed readings from earlier tonight are superseded by this table: the
+0.02 % "dead tie" at M=1 (actually +3.31 %) and the "delta grows with M" shape
(actually no shape at all). Both were n = 1.

### The FSQ sentence

> **The token interface loses ~2 % to the explicit reference — 1.1–3.3 % across
> M ∈ {1, 4, 5, 9}, at n = 4 rollout seeds per point, against a matched control
> at every M — and that penalty does not shrink as the motion set grows. It also
> does not grow. The −5.8 % of 25-08 was measured at a worse recipe and at n = 1.**

## The noise floor — and it swallows every FSQ effect ever reported here

Two rollout seeds on the **same checkpoint**, same clip, same harness:

| arm | seed 0 | seed 1 | spread |
|---|---|---|---|
| `M9_ref` | 0.1341 | 0.1279 | **4.61 %** |
| `M9_z` | 0.1359 | 0.1292 | **4.95 %** |
| `M5_ref` | 0.1436 | 0.1409 | 1.83 % |
| `M5_z` | 0.1446 | 0.1451 | 0.29 % |

And the same delta recomputed at the second seed:

| M | delta @ seed 0 | delta @ seed 1 |
|---|---|---|
| 5 | +0.74 % | +2.92 % |
| 9 | +1.33 % | +0.96 % |

**Every delta on the M-curve is smaller than the spread of a single arm across
rollout seeds.** So:

- "delta shrinks with M" — unsupported.
- "delta grows with M" — equally unsupported. The apparent +0.02 → +1.33 % trend
  is not resolvable at n = 1.
- "the token matches the reference" — this one **survives**, and is in fact
  strengthened: every delta is far inside the noise, at four values of M.

It also puts a question mark on the numbers this campaign was built to explain.
**The −5.8 % multi-motion and −5.1 % zero-shot penalties of 25-08 were
single-seed differences of the same magnitude as this spread.** They were never
separated from the measurement. That is not a claim that they were wrong — it is
a statement that the experiment could not have told the difference, and neither
could tonight's, until now.

### n = 4, and the picture sharpens rather than dissolving

| arm | mean | sd | sd % | range |
|---|---|---|---|---|
| `M9_ref` | 0.1314 | 0.0036 | 2.76 % | 0.1279–0.1350 |
| `M9_z` | 0.1329 | 0.0028 | 2.11 % | 0.1292–0.1359 |
| `M5_ref` | 0.1422 | 0.0021 | 1.50 % | 0.1400–0.1445 |
| `M5_z` | 0.1465 | 0.0022 | 1.54 % | 0.1446–0.1496 |

| M | delta (mean) | SE | \|delta\|/SE |
|---|---|---|---|
| 5 | **+2.97 %** | 0.0015 | **2.73** — resolvable |
| 9 | **+1.13 %** | 0.0023 | **0.65** — not resolvable |

So the single-seed numbers were *both* misleading, in opposite directions:
delta(5) is bigger than it looked (+0.74 % → +2.97 %) and delta(9) is a
non-result rather than the largest point on the curve. With four rollouts each,
the token loses a resolvable ~3 % at M=5 and is **indistinguishable from the
reference at M=9** — which is the direction the window argument predicts, from
the two points that now have error bars.

That is a claim about two points, and two points are not a curve. M=1 and M=4
are still n = 1, so three more seeds each are running (11031199-210) before any
monotonicity statement is made.

## C2 — ANSWERED, and it fails cleanly. The canonical tokenizer cannot carry it.

`M5_2t_canon`: one policy over H1 and G1, z-only, driven by ONE shared canonical
stream computed from **H1's clip alone**. Four rollout seeds.

| | H1 | G1 |
|---|---|---|
| explicit reference (per robot) | 0.1555 ± 0.0030 | 0.1308 ± 0.0019 |
| Kevin tokens (per robot) | 0.1644 ± 0.0049 | 0.1399 ± 0.0007 |
| **canonical, one shared stream** | **0.4558 ± 0.0084** | **0.4010 ± 0.0064** |
| zero action (same body) | 0.4126 | 0.3948 |

**The shared-stream arm is worse than doing nothing.** +10.5 % above the
zero-action floor on H1, +1.6 % on G1, and +177 / +187 % against the same
policy's per-robot tokens. The gate — "within 25 % of G1-under-G1-tokens and
clearly better than zero-action" — fails on both halves.

It is not a training failure: `alive_fraction` is **1.000** on both robots,
against 0.931 / 0.975 for zero action.

**And the video corrects how that failure looks.** The arm is *not* standing
still — per-joint motion is 0.220 rad sd on H1 (0.69x the reference) and
**0.162 rad on G1, which is 1.51x the reference**. It moves as much as or more
than the target; the motion is simply uncorrelated with it. The failure mode is
a robot that stays upright and dances *something else*, not a robot that freezes.
An earlier draft of this report said "learned to stand and stay standing" — that
was inferred from the alive fraction and is wrong.

**And the tokenizer number predicted exactly this, before the arm ran.** The
canonical reconstruction on this clip is 0.3661 rad (H1) / 0.3402 (G1) against
Kevin's 0.0508 / 0.0896, with **all 1000 codes saturated**. A channel that cannot
reproduce the motion offline cannot deliver it online. That is why the
reconstruction error was measured first, and it is the reason this result is
about **the canonical tokenizer**, not about embodiment-agnostic tokens as an
idea. Stated in advance, in this report, and not revised afterwards.

What C2 does establish: the seam works. `tracking_clip.py` tiled a 4-wide shared
code across H1's 19 actuators and G1's 23 by joint name, byte-identical for both
(asserted element-wise), and the policy trained and ran. The plumbing for an
embodiment-agnostic channel exists; what does not exist is a canonical tokenizer
worth putting through it.

## C3 — the rate curve, complete

Cost of holding the token, relative to each M's own 40 tok/s arm (n = 4 for the
hold-1 arms, n = 1 for the held arms; the noise floor is ~2 %):

| M | 8 tok/s (hold 5) | 4 tok/s (hold 10) |
|---|---|---|
| 1 | 0.1546 (**+13.2 %**) | 0.1584 (**+16.0 %**) |
| 5 | 0.1532 (**+4.6 %**) | 0.1815 (**+23.9 %**) |
| 9 | 0.1447 (**+8.9 %**) | 0.1651 (**+24.2 %**) |

- **8 tokens/s is usable**: 5–13 % over the 40 Hz stream, and at M=5 that is
  barely outside the noise floor. The gate said "within noise of 40 tokens/s";
  it is close but not inside it.
- **4 tokens/s is not**: ~24 % at M=5 and M=9. The interface has a rate floor,
  and it sits between 4 and 8 codes per second.
- The M=1 column behaves differently from the others (13 % at hold 5 but only
  16 % at hold 10, where the multi-motion arms cost ~24 %), which is what a
  single 9-second clip looks like: there is less to lose.

## ANSWERED: C2 failed on FEATURES, not capacity

Access came back at 08:10 and both jobs had completed. Same bundle, same 150
epochs, only the codebook size changed:

| book | codes used | H1 RMSE | G1 RMSE |
|---|---|---|---|
| 1 000 (8·5·5·5) | **1000 / 1000 — saturated** | 0.3661 | 0.3402 |
| 12 800 (8·8·8·5·5) | 9 654 / 45 125 | 0.3501 | 0.3272 |
| 32 768 (8^5) | 16 711 / 45 125 | **0.3442** | **0.3223** |

**32x more codes buys 6 % less error**, and the book stops being saturated — at
32 768 the fit uses only half the slots available, so capacity is no longer
binding and the error still sits at 0.34 rad. The 1000-code saturation was a
symptom, not the cause.

**So the canonical design is what is wrong, and it is wrong at the feature set:**
22 task-space dimensions (root height/attitude/velocities + 4 end-effector
positions) do not determine elbow and knee angles, and no codebook can index
information the encoder never sees. This closes the question the 08-22 1M-code
experiment could not answer, because that one ran at M=1 where the book was not
saturated and so could not separate the two explanations.

**Consequence for C2:** the cross-topology claim is not blocked by a tunable. An
embodiment-agnostic token stream needs a *different feature set* — one that
carries joint-level information while staying body-independent — and that is a
design task, not a hyperparameter. Rerunning `M5_2t_canon` with a bigger book
would be a wasted arm.

## ACCESS LOST 05:45 — self-inflicted, and what it does and does not cost

`ssh viper11 date` began hanging. The ControlMaster showed "Master running", so
the connection looked healthy while being dead. I ran `ssh -O exit gate` to force
a clean re-establish. **That was the mistake.** The gate master had been
authenticated interactively at 22:22 with a Kerberos ticket / password that this
session does not hold; killing it destroyed the only credential in play. There is
no ssh-agent, no key, and `klist` is not even installed. Viper cannot be reached
again until a human authenticates.

The goal document's own instruction — "never let an ad-hoc command own the
ControlMaster" — was about *creating* masters. The symmetric rule, now learned
the expensive way: **never destroy one either.** A hung master is still a
credential; the correct move was to open a second connection and leave the master
alone.

**What this does NOT cost.** Everything on the critical path is already local:
all 63 crosseval JSONs (`ce_fsqscale/`), the report, and the dashboard. Both
`canon_book` jobs run to completion on Viper by themselves; their result is a
printed reconstruction number waiting in a log.

**What it costs.** Two things, both named rather than quietly dropped:
1. The capacity-vs-features answer for C2 (`canon_b12800` 11034720,
   `canon_b32768` 11034721) — the numbers exist in
   `/ptmp/akalenik/urma/logs/canon_b*.out` and in
   `clips/clips_canon_b*rec/reconstruction_report.json`, unread.
2. **The videos.** No render was produced tonight. The path had been identified
   (`--dump_qpos` on `crosseval_motion.py`, whose env construction is already
   correct for every arm, plus a minimal qpos renderer) but not built, because
   the existing `rf_render_dance.py` needs an npz only `rf_eval_dance.py`
   produces and that script has no `--latent` / `--anchor` / `--refroot` flags,
   so it cannot reproduce a token arm's observation layout.

To resume: authenticate once (`ssh gate` from a terminal), then
`ssh -MNf viper11`. Nothing needs resubmitting.

## Falsified, eliminated, or superseded tonight

**Claims tested and killed:**

1. **C1's gate — "the token penalty shrinks with M."** 3.31 / 1.44 / 2.97 /
   1.13 % at M = 1 / 4 / 5 / 9, n = 4 each. No trend either way. The window
   argument has no support on the motion axis.
2. **The same argument on the topology axis.** Two-topology penalty H1
   5.74 → 2.93 %, G1 6.95 → 8.02 % from M=5 to M=9. It falls on one robot and
   rises on the other.
3. **C2's gate — "shared canonical tokens drive G1 within 25 % of G1's own
   tokens and clearly beat zero action."** The shared-stream arm is *worse than
   zero action* (+10.5 % H1, +1.6 % G1) and +177/+187 % against per-robot tokens.
4. **"The token loses ~5.8 % at multi-motion" (25-08).** Reproduced at the old
   recipe (+5.1 % here on the same clip and harness) and then shown to be
   (a) mostly a weak-baseline artefact — the same comparison at TRACK_TEMP=0.05
   is +2.97 % — and (b) measured at n = 1 against a noise floor of the same size.
5. **"The multi-motion plateau."** The reference side of the M-curve is 0.1322 /
   0.1366 / 0.1422 / 0.1314 at M = 1 / 4 / 5 / 9. Nine motions in one policy
   track as well as one. M=5 is the worst point because it is the clip that
   still contains `dance2_subject1`.

**Superseded readings from earlier in this same night** (all n = 1, all replaced
by n = 4 numbers above): the "+0.02 % dead tie" at M=1; "delta grows with M";
"the token matches the reference at every M".

**Two harness defects and two job defects fixed, each of which would have
produced a wrong number or none at all:**

- `crosseval_motion.py` hardcoded `tracking_clip_latent_dim = 32`, so no
  canonical (4-wide) checkpoint could be evaluated at all.
- No zero-action baseline existed in the harness, though C2's gate is stated
  against one.
- `tok_kevin` v1 died on a missing `loco_mujoco.core` and took Tier B with it
  via `afterok`.
- The same job would have reconstructed **into the clip directory**, overwriting
  `superM4/M9.npz` with their FSQ reconstructions — the z arms would have
  tracked the tokenizer's output instead of the true motion, silently.

**And one methodological finding that outlives all of the above:** rollout seed
alone moves an arm's score by up to 4.95 %. Every FSQ number ever reported on
this stack was a single-seed difference of that magnitude. Nothing in this
directory should be read at n = 1 again.

## What is still open

1. **Videos.** None were produced. See the access note above for why and for the
   path that was identified but not built.
2. **M beyond 9.** Needs the DanceDB retarget (85 raw AMASS files, none ever
   retargeted on this stack) plus the four screening tools. A night of its own.
3. **Training-seed variance.** Tonight bounded *rollout* noise only. RL-X exposes
   no seed on this path, so a second training seed per arm was not available —
   and it is the larger term. Every delta here should be treated as an upper
   bound on the confidence available.

## Runbook — how to finish this in ~20 minutes

1. **Reconnect.** In a terminal: `wsl -d Ubuntu ssh gate` (password + OTP once),
   then `wsl -d Ubuntu ssh -MNf viper11`. Do **not** `ssh -O exit` anything.
2. **Read the two answers already waiting** (both jobs were submitted at 04:45
   and need nothing further):

   ```
   ssh viper11 'grep -a RMSE /ptmp/akalenik/urma/logs/canon_b12800_11034720.out
                grep -a RMSE /ptmp/akalenik/urma/logs/canon_b32768_11034721.out'
   ```

   Baseline to compare against: **0.3661 rad H1 / 0.3402 G1** at 1000 codes,
   saturated. A large drop means C2 failed on **capacity** and deserves a rerun
   with the bigger book (`chain_canon.sh`, swapping `TAG`/`LEVELS`); little or no
   drop means the 22 task-space **features** are wrong and no book will fix it.
3. **Videos**, if wanted. The path: add `--dump_qpos out.npz` to
   `crosseval_motion.py` (it already builds the exact env for every arm — the
   rollout loop has `qpos` and the blended reference `ref` in hand at
   `crosseval_motion.py:279`), then render with a minimal qpos-only script.
   `rf_render_dance.py` cannot be reused directly: it needs
   `reference_joint_targets` / `reference_root_yaw_delta` / `root_yaw_origin` /
   `heading_error`, which only `rf_eval_dance.py` writes, and that script has no
   `--latent` / `--anchor` / `--refroot` flags, so it cannot reproduce a token
   arm's observation layout.
4. **Regenerate the dashboard** after any new crosseval:
   `python experiments/fsq_khaendler/build_fsqscale_dashboard.py`. It reads
   `ce_fsqscale/` and rewrites `dashboard_fsq_scale.html`; every table in this
   report comes from it.

Nothing needs resubmitting. Every arm trained tonight is on Viper under
`loco_mjx/experiments/runs/urma2_h1g1/`, and all 63 crossevals are already local.

## Videos (produced 08:35, after access returned)

`videos/fsq_scale/`. Left pane = the policy as it actually moved; right pane =
the reference it was scored against, placed at the policy's own position and the
clip's target heading, so the visible difference is exactly what the policy
failed to reproduce.

| file | what it shows |
|---|---|
| `M9_ref_h1.mp4` | explicit reference, 9 dances, H1 — the best arm of the night |
| `M9_z_h1.mp4` | Kevin's tokens, same clip, matched. **The +1.13 % pair — the difference should be invisible, and that is the result.** |
| `M5_2t_canon_h1.mp4` | shared canonical stream, H1 — moves, but uncorrelated with the target |
| `M5_2t_canon_g1.mp4` | shared canonical stream, G1 — **moves 1.5x more than the reference**, in the wrong directions |
| `M5_z_h10_h1.mp4` | 4 tokens/s, where the interface breaks (+24 %) |

**How they were made, since the obvious path does not work.** `rf_render_dance.py`
needs an npz that only `rf_eval_dance.py` writes, and that script has no
`--latent` / `--anchor` / `--refroot` flags, so it cannot reproduce a token arm's
observation layout. `crosseval_motion.py` *can* — it already builds each arm's
exact env — so it gained `--dump_render`, which writes the same field names
(`qpos`, `reference_joint_targets`, `reference_root_yaw_delta`,
`root_yaw_origin`, `heading_error`, `dt`) straight out of the rollout it was
already doing.

**Rendering has to happen locally.** The compute nodes have no OSMesa
(`PyOpenGL` raises `'NoneType' object has no attribute 'glGetError'`) and no
`/dev/dri` for EGL, so `MUJOCO_GL` has no working backend there. The dumps are
~300 kB each; pull them and render on Windows with `MUJOCO_GL=wgl`. Note that
`rf_render_dance.py` does `os.environ.setdefault("MUJOCO_GL", "osmesa")`, so the
variable must be set explicitly or MuJoCo refuses to import on Windows.

**Read the heading with care.** Mean heading error over the rendered span is
53-92 degrees across these arms, so the two panes often face different ways.
That is a separate, known defect of the root/heading term, it is present on the
reference arm as much as the token arms, and it is not what these videos are
about — watch the limbs, not the facing.
