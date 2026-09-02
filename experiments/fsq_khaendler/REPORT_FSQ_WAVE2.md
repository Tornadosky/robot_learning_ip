# FSQ WAVE 2 — where the token pays, and what the night falsified

Window: 2026-08-26 22:35 → 2026-08-27 08:30 Berlin. MPCDF Viper, plus the local
box for verification. Live document; numbers are appended as jobs land.

Read with [`REPORT_FSQ_SCALE.md`](REPORT_FSQ_SCALE.md) (last night) and
[`../../docs/notes/FSQ_NEXT_WAVE.md`](../../docs/notes/FSQ_NEXT_WAVE.md).

---

## THE FSQ SENTENCE, updated

> **The token carries information the reference does not.** A token stream whose
> contents are real beats reference-alone by 2–4 % on a single body; the same
> token with its phase destroyed is *worse* than no token at all. The advantage
> **grows as the reference channel degrades** — 4.0 % at 40 Hz, 14.0 % at 8 Hz,
> 11.6 % at 4 Hz, measured with both channels equally stale. It is **unaffected
> by randomizing the body** (−4.0 % with and without). It **does not survive a
> second topology**.
>
> What the token is *not*: it is not smoothing (the scramble kills it), it is not
> merely a wider observation (the scrambled arm has the same width and is worse),
> and it is not a future sample of the reference (shifting the reference forward
> is 16–79 % *worse*). What fits every control is that it is a better **summary of
> a window** — frames t…t+9, present and future together — than a single-instant
> sample of the same channel at the same rate.

Two directions close on measurement, as the goal document hoped one might:

- **The body-independent token closes.** Its gate measured 0.1774 rad (a figure
  RETRACTED on 2026-08-27 as train-contaminated — see §6) against a
  0.10 bar and declined to spend the RL slot. Decoder capacity was the right
  suspect and remains the only lever that moves — 29 % over two doublings,
  unsaturated — but it is not enough alone.
- **Three topologies close on this stack**, for a reason unrelated to FSQ: an
  out-of-bounds GPU write that survives every configuration tried.

And one finding outranks all of the above for the pipeline:

- **Nothing in the reward scores heading**, and every arm ever measured here
  tracks heading 15× *worse* than emitting no torque. Worse, the training curves
  show heading being **actively given up in the first 10M steps** — and turning
  the term on does not reverse it (§2.5c). This is not an FSQ result; it is the
  recipe every FSQ result was measured on top of.

---

## ARE WE CLOSE TO UNDERSTANDING WHEN FSQ WORKS?

**Partly. We can now predict the sign reliably and the size not at all.**

What is settled enough to design against — four conditions, each measured with
its own control:

| condition | effect on the token's value | evidence |
|---|---|---|
| **The robot is H1** | this is where every positive result lives. **On G1 alone the effect is +0.6 %, t = 0.73** — nothing | §2.6e |
| **On top of the reference, not instead of it** | the only configuration that ever wins. Token-*replacing* is 1–3 % worse than reference | every M, §2.2f |
| **The reference channel is degraded** | value **rises** — 4.0 % → 14.0 % → 11.6 % as the reference drops 40 → 8 → 4 Hz | §2.2d |
| **The body is randomized** | **no effect at all** — −4.0 % with and without | §2.6b |
| ~~A second topology is present~~ | ~~value collapses~~ — **struck**: it was G1's null in the average, not an effect of sharing | §2.6e |

Those four give a usable rule: *the token is worth having when it supplements a
reference that is slow, stale or unreliable, on a body family it was fitted to.*
That is a real engineering claim and it is testable in the products this feeds.

What is **not** understood, and blocks a mechanism:

1. **Why it works at all.** Three explanations are excluded — smoothing
   (scramble), extra observation width (scramble again), and lookahead-as-a-later-
   reference (the lead arms are 16–79 % *worse*). What survives is "a better
   summary of a window than an instant sample", which is a description, not a
   mechanism. The additive-lookahead control that would test it directly was not
   run.
2. **Why the size varies.** −5.7 %, −8.7 %, −2.0 %, −4.0 % at M = 1, 4, 5, 9.
   Not monotone, not explained by the number of motions, and the spread is larger
   than most of the effects it contains.
3. **Why a second robot erases it.** ~~The per-joint code is fitted on H1's clip,
   so G1 reads an address never built for it.~~ **Falsified before it cost a slot
   — see §2.6d.** The tokenizer is already fitted jointly on both bodies and G1
   carries its own 23-joint code. Two candidates remain and a single-topology G1
   pair separates them: the loss is caused by *sharing one policy across two
   bodies*, or G1's token was simply never as good as H1's.

So: the *when* is roughly known, the *why* is not, and the honest reading is that
FSQ is currently a **conditional engineering win with an unexplained mechanism**,
not a understood component. Question 3 is the cheapest next step and would also
tell us whether the whole direction scales past one body family — which is the
question the project actually needs answered.

---

## 0. The state at handoff was not what the handoff said

44 jobs were queued and chained. Within ten minutes of the window opening, three
of the four Tier-0 questions had already failed — silently, because two of the
failures still reported success to Slurm.

| Tier-0 arm | Reported | Actually |
|---|---|---|
| `prep_wave` | COMPLETED | genuinely fine |
| `canon_dw2`, `canon_dw4` | FAILED (49 s) | crashed at model init |
| `gate_canon` | never ran | `DependencyNeverSatisfied` behind the two above |
| `scale_3t_d4` | **COMPLETED** (1 m 43 s) | crashed; wrapper swallowed the exit code |
| its 4 crossevals | FAILED (2 s) | ran against a checkpoint that was never written |
| 8 rate-curve crossevals | COMPLETED | genuinely fine — real data |

Everything below Tier 0 (the scramble control and its descendants) was unaffected
and ran as designed.

---

## 1. Four defects found and fixed

Each of these had already cost at least one arm, and two of them had been
misattributed to physical limits of the cluster.

### 1.1 `network_width_multiplier` never worked in the vendored decoder

`canonical_bundle_*/nn_vendor/decoder.py` scales most of its layers through
`scaled_width(256, multiplier)`, but two sites keep a hardcoded `256`: the
`temporal_latent` output width and the reshape immediately after it. The
per-joint mask, built from `encoder_dense_1`, *does* scale. At multiplier 1.0
both are 256 and the mismatch is invisible; at any other value the final
broadcast fails:

```
TypeError: mul got incompatible shapes for broadcasting:
    (2, 11, 1, 256), (2, 1, 19, 512)
```

So **the decoder-width knob has never been exercised at any width but 1.0** — in
this repo or, since the file is vendored from the colleague's `autoencoder`
branch, upstream. The decoder-capacity hypothesis was not previously testable,
and the gate that was supposed to decide the whole body-independent direction
could not have run.

Fixed by scaling both sites. `canon_dw2` and `canon_dw4` now train.

### 1.2 The 3-topology run: an arithmetic constraint, not a memory wall

Two different explanations were on record for why runs with more than two robot
topologies die — a ROCm memory wall at any env count, and (in the handoff) a
missing clip. Both are wrong. With the clip present the run reaches the trainer
and asserts:

```
AssertionError: Minibatch size must be divisible by the number of train robots
```

`MINIBATCH` defaults to 8192, and `8192 % 3 == 2`. That is the whole blocker.

Notably, the log shows the three-robot environment **building successfully on
Viper's ROCm APU** — `train env spaces check (3 robots) took 57.2 s`, then
construction in 7.4 s — before the assert fires in the algorithm constructor.
The "more than two topologies crashes at any env count" belief does not survive
this: construction is not where it died.

Fix: `MINIBATCH=6144`, which is divisible by 1, 2 and 3 and divides the batch
(768 envs × 64 steps = 49 152) exactly, giving 8 minibatches. A matched
`scale_2t_mb6144` control was submitted alongside, because the existing 1t/2t
arms ran at 8192 and a minibatch change is a confound in a scaling comparison.

That fix is necessary but **not sufficient**, and the first read of it here was
premature. The arm cleared the assertion, compiled for two more minutes, and then
aborted at 4 m 17 s with `rc=134` (SIGABRT). See §1.2b — the second blocker is a
different bug entirely.

### 1.2b The real multi-topology wall: concurrent XLA compiles corrupt the HIP context

Past the minibatch assert, the 3-robot arm dies inside
`multi_environment.py:56`, which compiles each robot's `reset` in a
`ThreadPoolExecutor`:

```
Failed call to hipGetFuncBySymbol: ROCM_ERROR_ILLEGAL_ADDRESS
Non-OK-status: hipCtxSetCurrent(context_) "Failed setting context"
Status: INTERNAL: Failed setting context: ROCM_ERROR_ILLEGAL_ADDRESS
Fatal Python error: Aborted
```

Host RSS peaked at 11.8 GB of the node's 108 GB, so this is not a host OOM, and
the matched 2-topology control at the same minibatch has been running happily
throughout. The difference is the **number of concurrent compilations**: two
worker threads on one ROCm context survive, three abort.

This, not memory, is what every "more than two topologies crashes at any env
count" observation has been measuring. The fix is one flag —
`--algorithm.parallel_compile=False` — which serialises the per-robot compiles
and costs startup time, not capability. Resubmitted as job 11078022.

Both explanations previously on record were wrong, and so was the first one this
report offered: the failure is neither a memory wall nor arithmetic alone, it is
a ROCm threading bug behind an arithmetic bug.

That is again necessary and not sufficient. The arm reached 7 m 58 s and aborted
with a *third* distinct failure — this time not in compilation but in **kernel
execution**:

```
Failed to complete all kernels launched on stream:
    Could not synchronize on ROCM stream: ROCM_ERROR_ILLEGAL_ADDRESS
```

### 1.2c Two hypotheses, both killed by cheap tests

A fact that reframes every previous measurement: **`nr_envs` is per robot, not
total.** `MultiEnvironment` hands each robot `config.environment.nr_envs`, so the
3-robot run at 768 was allocating **2304 envs against the 2-robot run's 1536**.
Every historical "more topologies dies" observation carried that confound.

| test | question | result |
|---|---|---|
| `t1_only` — booster_t1 alone, 768 envs | is it the t1 environment? | **trained 28+ min.** Not the env. |
| `scale_3t_e384` — 3 robots, 384 each = 1152 total | is it env count? | **aborted at 8 m 05 s**, at fewer total envs than the working 2-robot run. Not the count. |

The second failure is a different kernel again:
`Failed to launch ROCm kernel: loop_add_broadcast_fusion with block dimensions:
8x1x1: ROCM_ERROR_ILLEGAL_ADDRESS`.

So three topologies fused into one jitted graph fail on this ROCm stack
regardless of size, and regardless of which robots are in the trio.

### 1.2d `mjx_split` does not rescue it either, and the fault looks like corruption

Two more attempts, both failed:

| arm | change | result |
|---|---|---|
| `split_3t` | `ALGORITHM=urma2.mjx_split` — per-robot graphs, its own AOT cache | aborted 4 m 31 s, `input_scatter_fusion_104` |
| `scale_3t_nocb` | drop `--xla_gpu_enable_command_buffer=` | aborted 8 m 29 s, stream sync failure |

`split_3t` genuinely took the split path (`mjx_split -> JAX persistent cache OFF`
on line 1, and the traceback is in `urma2_split.py:256`), so the standing note
that the topology fault is an artifact of the *fused* `urma2.mjx` and that
`mjx_split` is the many-family path **does not hold on this stack**. Dropping the
command-buffer flag, which the sbatch's own comments name as a previous cause of
an identical abort, does not help either.

The pattern across five attempts is the useful part. Every abort is a
**different** kernel — `hipGetFuncBySymbol`, `loop_add_broadcast_fusion`,
`input_scatter_fusion_104`, and twice a bare stream synchronisation — at
different elapsed times, under different algorithms, at env counts both above and
below a configuration that works. One broken kernel does not move around like
that. An out-of-bounds **write** does: it corrupts GPU memory and the victim is
whatever runs next.

That reframes the search from "which ROCm kernel is broken" to "what indexes out
of range when a third robot is present".

### 1.2e The wall is exactly at three, and it is not any particular robot

| configuration | robots | result |
|---|---|---|
| `t1_only` | booster_t1 | **trains** |
| (many prior runs) | unitree_h1 + unitree_g1 | **trains** |
| `pair_h1t1` | unitree_h1 + booster_t1 | **trains** (2910 updates and counting) |
| `scale_3t_d4` ×2 | h1 + g1 + t1, 768 envs | aborts |
| `scale_3t_e384` | h1 + g1 + t1, 384 envs | aborts |
| `split_3t` | same three, `mjx_split` | aborts |
| `scale_3t_nocb` | same three, no command buffer | aborts |
| `order_3t` | **t1 + g1 + h1** (largest first) | aborts |

Every one- and two-topology combination works, including both pairs involving
booster_t1. Every three-topology combination fails, at two env counts, under two
algorithms, with and without the command-buffer flag, in either robot order.

`order_3t` failing kills the tidiest hypothesis: `MultiEnvironment.__init__` reads
`single_observation_space`, `joint_observation_size` and `joint_description_size`
from `train_envs[0]`, and every working configuration had the smallest robot in
that slot — but putting the largest first changes nothing. Whatever indexes out
of range is not keyed on which robot is first.

So the statement to carry forward is narrow and well-supported: **on this ROCm
stack the fused trainer supports exactly two topologies.** Not "3–4 is a memory
wall", not "t1 is broken", not "mjx_split fixes it". Three fails, at any size, and
the diagnosis needs someone to run it under `AMD_SERIALIZE_KERNEL=3` or a device
sanitiser to find the offending write — which is a debugging session, not an
overnight slot.

### 1.2f RETRACTED — it was never about topology count. It is the node.

**The paragraph above is wrong**, and the thing that exposed it was a run that had
nothing to do with topology count. A **single**-topology G1 arm
(`M5_g1_ref` / `M5_g1_both`, 2026-08-27 09:31) aborted identically:

```
HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION: The agent attempted to access
memory beyond the largest legal address.  code: 0x29
Failed to launch ROCm kernel: input_scatter_fusion_38 ... ROCM_ERROR_ILLEGAL_ADDRESS
```

One robot. So "three topologies" cannot be the discriminator. Tallying every
training arm since 22:00 by the node it landed on:

| | nodes |
|---|---|
| produced aborts | vipa**1018**, **1080** (×2), **1116** (×3), **1137**, **1140**, **1143** |
| trained to completion | 1024, 1057, 1092, 1097, 1133, 1140, 1154, 1168, 1196, 1271, 1274, 1283, 1297, 1298, 1299 |

**Every one of the six three-topology attempts landed on a node from the first
row** — 1018, 1137, 1143, 1116, 1116, 1140 — and not one ever ran on a node from
the second. The "wall" was six unlucky assignments in a row, and the elaborate
structure built on top of it (different kernel each time ⇒ out-of-bounds write ⇒
a bug in our indexing) was explanation fitted to a sampling artefact.

What the evidence actually supports is much duller and much more actionable:
**a subset of the APU nodes is faulty**, and any job that lands on one dies with
an aperture violation in whatever kernel happens to be running. That is also the
honest reading of the "different kernel every time" observation — it was pointing
at the hardware, not at our code, and I read it the other way.

Retested with `--exclude=vipa1018,vipa1080,vipa1116,vipa1137,vipa1140,vipa1143`:
`scale_3t_clean` (three topologies) and the G1-only pair. **§1.2b through §1.2e
should be read as provisional until those land.** The one durable item from that
sequence is the minibatch-divisibility assert in §1.2, which is a real code
constraint and was reproduced deterministically.

### 1.6 The sign-table lookup is only *accidentally* correct, and Atlas breaks it

`clip_reference.py` resolves a joint's axis sign by chaining
`H1_CLIP_SIGNS.get(name, G1_CLIP_SIGNS.get(name, T1_CLIP_SIGNS.get(name, 1.0)))`
— keyed on joint name alone, with no notion of which family is being loaded. The
code comments assert the three name spaces do not collide, and that is true
today (verified: 0 overlap between all three pairs).

It is not a property of the design, only of the three families that happen to be
in it. **Atlas shares 13 joint names with H1's table, and 6 of them would receive
the wrong sign** — `hip_flexion_l`, `hip_adduction_l`, `hip_rotation_l`,
`knee_angle_l`, `hip_flexion_r`, `knee_angle_r`: both knees and the whole left
leg. Adding Atlas to a run would silently reproduce the exact class of defect
that has already cost this project three campaigns.

The lookup should be keyed by family, not name. That change is *not* being made
tonight: `clip_reference.py` is on the construction path of six running arms, and
a typo there would cost more than the latent bug does while no fourth family is
in play. It is the first thing to fix when the cluster is idle.

### 1.7 A family's sign table can now be derived from the clip alone

New: `scripts/scaling/derive_clip_signs.py`.

The existing generator compares `dot(axis_locomujoco, axis_urma2)` and so needs
both packages importable — which no longer works locally, because the
`loco-mujoco` submodule now points at the group repo's `integration` branch and
that branch ships **zero tracked files** under `loco_mujoco/trajectory/`.

The clip already contains what is needed. It records, per frame, both the joint
angles and the world poses those angles produced in the source model. So the
signs are recoverable by asking which sign vector makes loco_mjx's own forward
kinematics reproduce the recorded poses. Coordinate descent over ±1 per joint,
40 high-excursion frames.

**Orientation turned out to be mandatory.** Scored on body *positions* only, the
derivation reproduces the clip to 0.0000 cm and still disagrees with the
known-good tables on 2/19 (H1), 3/23 (G1) and 4/23 (t1) joints — and every single
disagreement is a terminal joint whose child body origin lies on its own axis
(both ankle rolls, the elbows, the head pitch). Flipping those moves no body
origin at all, so the objective is exactly flat there and the descent keeps its
`+1` initialisation. Adding the geodesic orientation error fixes it:

| family | joints | negated | residual (m+rad) | vs published table |
|---|---:|---:|---:|---|
| h1 | 19 | 13 | 1.6e-4 | **matches all 19** |
| g1 | 23 | 15 | 2.9e-4 | **matches all 23** |
| booster_t1 | 23 | 14 | 1.5e-4 | **matches all 23** |

### 1.8 Only three of the seven families are safe to train on

With the tool validated, the four families whose `dance2_subject4` retarget was
uploaded tonight were screened before any GPU time was spent on them:

| family | residual | verdict |
|---|---:|---|
| atlas | 0.043 | **not reproducible by signs alone** (300× the validated three) |
| talos | 0.83 | not reproducible |
| toddlerbot | 2.40 | not reproducible — barely improves on all-`+1` |
| h1v2 | — | no loco_mjx model exists (`unitree_h1v2/data/plane.xml` absent) |

A residual near 1e-4 means the two models differ *only* in axis directions, which
is the entire assumption a sign table encodes. Atlas, Talos and ToddlerBot differ
by more than that — different link geometry, joint offsets, or closed chains.
ToddlerBot barely improving on all-`+1` is what a 4-bar knee looks like to a
hinge-by-hinge model.

So the honest ceiling on *data* is **three families, not seven** — and that limit
is independent of the separate, unresolved question in §1.2c of whether the
trainer can fuse three topologies at all. Two ceilings, two causes: even if
`mjx_split` unlocks the trainer tonight, there is no fourth family to put in it
until someone regenerates those retargets against loco_mjx's own models.

The screen costs about a minute per family and would have caught this before a
3-hour arm; it should be a precondition for adding any family.

### 1.3 The clip robot map had no entry for t1

`resolve_clip_path` maps a robot to its offline-retarget subdirectory, and the
default map knew only `h1`/`g1` (+ `unitree_` aliases). Adding a third family
raised `No clip mapping for robot 't1' / 't1'` — note the robot's own name is
`t1`, not `booster_t1`, so `ROBOTS_LIST=booster_t1` alone was never going to
resolve.

`clips/LAFAN1/BoosterT1/dance2_subject4.npz` was **already on Viper**. The
3-topology question was blocked by a missing dictionary entry.

Map extended with `t1`/`booster_t1` → `BoosterT1`, and pre-emptively with
`atlas`, `talos`, `toddlerbot`, `h1v2`, all of which have a
`dance2_subject4` retarget locally.

**Sign convention checked before spending the slot.** The BoosterT1 clip file
dates from 08-15, before the 08-21 t1 axis-flip fix, which raised the question of
whether it is stale. It is not: the correction lives in the *consumer*
(`T1_CLIP_SIGNS` in `clip_reference.py`), not in the stored file. The lookup
chains H1 → G1 → T1 by joint name, so it is only safe if the three name spaces
are disjoint; verified programmatically — 19/13, 23/15 and 23/14 entries negated,
**zero overlap between any pair**.

### 1.4 `viper_train.sbatch` reported success for runs that died

The script ended on `echo "TRAINING EXITED rc=$?"`, so its own exit status was
`echo`'s — always 0. Every `--dependency=afterok` chain behind a training arm
was therefore unprotected: `scale_3t_d4` died in 1 m 43 s, was recorded
COMPLETED, and released four crossevals that failed two seconds later against a
checkpoint that did not exist. Now captures `rc` and exits with it.

### 1.5 (tooling) Command substitution was silently running on the wrong machine

Not a repo defect, but it cost this session ~20 minutes and would corrupt any
future automation of this shape. Driving Viper as
`wsl bash -lc '... ssh viper11 "…$(cmd)…"'` evaluates `$(cmd)` **locally in
WSL**, not on Viper. The symptom is absurd and easy to misread: `squeue` works
while `sbatch` reports `command not found` even though `/usr/bin/sbatch` is a
present, executable ELF — because only the substituted call ran locally. Proof:
`$(hostname)` returned the Windows machine name inside a script whose plain
`hostname` returned `viper11`.

Every remote command in this session is now written to a file, stripped of CRLF,
and piped in over stdin. Worth adopting permanently.

---

## 2. Results

### 2.1 The token-rate curve is complete, at n = 4

H1, `superM9` clip, executed-vs-clip shape RMSE, 4 rollout seeds each. The clip
is 40 fps, so hold *h* means a new token every *h* frames.

| hold | tokens/s | shape RMSE (rad) | vs h=1 |
|---:|---:|---|---:|
| 1 | 40 | 0.1320 ± 0.0024 *(n=3)* | — |
| 2 | 20 | **0.1280 ± 0.0031** | −3.0 % |
| 5 | 8 | 0.1422 ± 0.0037 | +7.7 % |
| 10 | 4 | 0.1620 ± 0.0047 | +22.7 % |
| 20 | 2 | 0.2228 ± 0.0067 | +68.8 % |

Two things fall out. The curve has a **knee between 20 and 8 tokens/s**, which
sharpens last night's "between 4 and 8" to a tighter bracket. And h=2 is not
merely as good as h=1 but slightly *better* than it — holding a token for one
extra frame costs nothing and may act as mild temporal smoothing. Everything at
or below 8 tokens/s degrades monotonically and steeply.

### 2.2 The auxiliary result stands — small, and marginal at n=4

All six arms are now n=4 under one tag convention (the seed-0 fills landed).
H1, shape RMSE:

| arm | M=5 (`super5`) | M=9 (`superM9`) |
|---|---|---|
| reference only | 0.1424 ± 0.0023 | 0.1311 ± 0.0034 |
| token only | 0.1461 ± 0.0027 | 0.1341 ± 0.0047 |
| **both** | **0.1395 ± 0.0016** | **0.1268 ± 0.0036** |
| both vs reference | −2.0 % (t ≈ 2.1) | −3.3 % (t ≈ 1.7) |

Token-alone is slightly *worse* than reference-alone at both M; token *on top of*
reference is better than either. That is the shape you expect if the token
carries something the reference does not, rather than a better encoding of the
same thing.

Taken alone this is a **2–3 % effect with t ≈ 1.7–2.1 at n=4** — a consistent
sign across two values of M, not a demonstrated effect. What settles it is the
scramble control.

### 2.2b THE SCRAMBLE VERDICT: the token carries information

`M9_both_scram` and `M5_both_scram` train on the *same token stream rolled by
7919 frames*. Identical marginal statistics, identical dimensionality, identical
everything — except that the token no longer describes the frames it accompanies.
If the auxiliary gain were smoothing, regularisation, or a wider observation
being helpful for its own sake, the scramble would keep it.

All n=4, H1, shape RMSE:

| arm | M=5 (`super5`) | M=9 (`superM9`) |
|---|---|---|
| reference only | 0.1424 ± 0.0023 | 0.1311 ± 0.0034 |
| token only | 0.1461 ± 0.0027 | 0.1341 ± 0.0047 |
| **both (real token)** | **0.1395 ± 0.0016** | **0.1268 ± 0.0036** |
| both, **scrambled token** | 0.1480 ± 0.0020 | 0.1332 ± 0.0047 |
| real vs scrambled | **−6.1 %, t ≈ 6.6** | −5.0 %, t ≈ 2.2 |
| scrambled vs reference-only | +3.9 % *worse* | +1.6 % *worse* |

**The scramble loses the gain, and then some.** At M=5 the separation between the
real token and its scrambled twin is t ≈ 6.6 (p < 0.001) — by a wide margin the
most significant FSQ result this project has produced. At M=9 the direction is
identical at t ≈ 2.2.

The second row of the comparison matters as much as the first: the scrambled
token is *worse than no token at all*, at both M. That is what a channel of pure
noise added to an observation should do, and it is the opposite of what a
regulariser does. Both readings that would have closed this direction —
smoothing, and "a wider observation is just easier to optimise" — are excluded by
the same control.

> **The FSQ sentence, updated: the token carries information the reference does
> not.** A token whose contents are real beats reference-alone by 2–3 %; the same
> token with its phase destroyed is worse than reference-alone. The mechanism
> remains most likely lookahead — the clip is 40 fps and the token spans frames
> t…t+9, about 250 ms of future the reference channel never shows — but this
> experiment establishes that the content is used, not that lookahead is why.

### 2.2c Reference degradation: the token substitutes for a stale reference

The env change the goal document anticipated is now in:
`tracking_clip_reference_hold` freezes the **observed** reference to every K-th
clip frame while the reward target stays fresh. Default 1 is byte-identical to
previous behaviour, and the key is registered at setup under a static branch
exactly as `reference_joint_latent` is, so the `internal_state` pytree never
changes shape mid-episode. It is exposed to the crosseval as `--reference_hold`
(`CE_REFHOLD`) and recorded in `eval_condition`, so a held run can never be
mistaken for a fresh one.

That made the experiment askable of checkpoints that already exist, in minutes
rather than three hours per arm. H1, `superM9`, n=4 at every cell:

| observed-reference hold | `M9_ref` | vs fresh | `M9_both` | vs fresh |
|---:|---:|---:|---:|---:|
| 1 | 0.1315 ± 0.0029 | — | 0.1263 ± 0.0044 | — |
| 5 | 0.2065 ± 0.0068 | **+57.1 %** | 0.1484 ± 0.0053 | **+17.5 %** |
| 10 | 0.2847 ± 0.0132 | **+116.6 %** | 0.1740 ± 0.0042 | **+37.8 %** |
| 20 | 0.3549 ± 0.0121 | **+169.9 %** | 0.2126 ± 0.0051 | **+68.4 %** |

The token arm degrades roughly **3× less** at every level (t ≈ 13 at hold=5). By
hold=20 the reference-only arm is at 0.3549, closing on the zero-action floor of
0.3991 — it has nearly stopped being a policy — while the token arm is still at
0.2126.

**What this does and does not show.** It shows the token is a sufficient
substitute for the reference channel: strip the reference of temporal resolution
and the token carries most of what was lost. It does **not** yet show lookahead,
because in this grid the token stayed fresh at 40 Hz while the reference was
held — the token arm simply retained a per-frame channel the other had lost, so
some of its advantage is bought with freshness rather than with content.

### 2.2d Matched staleness: most of it was freshness, but not all of it

Holding **both** channels at the same K. H1, `superM9`, n=4 everywhere:

| K | `ref` (no token) | `both`, token fresh | `both`, token also held at K | token gain at matched K |
|---:|---:|---:|---:|---:|
| 1 | 0.1315 ± 0.0029 | 0.1263 ± 0.0044 | (same) | 4.0 % |
| 5 | 0.2065 ± 0.0068 | 0.1484 ± 0.0053 | **0.1776 ± 0.0074** | **14.0 %**, t ≈ 5.8 |
| 10 | 0.2847 ± 0.0132 | 0.1740 ± 0.0042 | **0.2516 ± 0.0088** | **11.6 %**, t ≈ 4.2 |
| 20 | 0.3549 ± 0.0121 | 0.2126 ± 0.0051 | **0.3376 ± 0.0085** | 4.9 %, t ≈ 2.3 |

Two things, and the first corrects §2.2c's framing. **Most of the token arm's
advantage in the previous grid was freshness**: once the token is held at the
same rate, the gap collapses from 28/39/40 % to 14/12/5 %. A reading that stopped
at §2.2c would have badly overstated the token.

But a residual survives, significantly at K=5 and K=10, and it is **larger than
the fresh-vs-fresh gap of 4.0 %**. At an equal sample rate a 32-number FSQ token
is a better channel than the 19-dim raw reference delta, and the margin grows as
the rate falls: 4.0 % at 40 Hz, 14.0 % at 8 Hz, 11.6 % at 4 Hz.

That is precisely E1's gate — *"token helps more at 4 Hz reference than at 40 Hz,
AND the scrambled control does not reproduce it"* — and **both halves now hold**.

### 2.2e Raw lookahead does not reproduce it — the token is not a future sample

E1's control 2. `tracking_clip_reference_lead` shifts the observed reference L
frames forward while the reward target stays at *t*; same observation width, one
index change on the seam above. Two arms trained from scratch with it, n=4:

| arm | shape RMSE | vs `M9_ref` | heading |
|---|---:|---:|---:|
| `M9_ref` (lead 0) | 0.1315 ± 0.0029 | — | 82.4° |
| **`M9_both`** (token) | **0.1263 ± 0.0044** | **−4.0 %** | 82.4° |
| `M9_lead5` | 0.1532 ± 0.0040 | **+16.5 %** | 82.4° |
| `M9_lead10` | 0.2359 ± 0.0076 | **+79.4 %** | 84.2° |

Raw lookahead does not reproduce the token's gain. It is sharply *worse*, and
worse in proportion to how far ahead it looks.

**Be precise about what was tested.** This shifts the reference — the policy sees
*t+L instead of t*. E1's control 2 as written asks for the future to be *added*
alongside the present, which needs an extra observation channel and therefore a
width change to `get_observation_space` and its index loops; that was judged too
risky to attempt while six arms were mid-flight. So the question answered here is
narrower: **is a future reference sample, on its own, as good as the token?** No,
emphatically.

That is still informative, because it rules out the tidiest version of the
lookahead story. The token is not simply "the reference, but later". Put together
with the other two controls, the picture is consistent and fairly specific:

- the **scramble** says the token's contents matter, not its presence;
- **matched staleness** says that at an equal sample rate the token beats the raw
  reference delta, and by more as the rate falls;
- **lead** says the advantage is not obtained by re-timing the reference.

What fits all three is that the token is a better *summary of a window* — it
spans *t…t+9*, carrying present and future together — than a single-instant
sample of the same channel is. Confirming that specifically still needs the
additive-channel version of control 2.

### 2.3 The canonical (body-independent) token is *worse than doing nothing*

Zero-action floors are now n=4 as well, and they are tight (sd ≈ 0.5 %):

| arm | robot | shape RMSE | vs its own zero-action floor |
|---|---|---|---|
| `M5_2t_canon` | H1 | 0.4558 ± 0.0084 | floor 0.4117 ± 0.0019 → **+10.7 % worse** |
| `M5_2t_canon` | G1 | 0.4010 ± 0.0064 | floor 0.3944 ± 0.0025 → +1.7 %, marginal |
| `M5_2t_ref` | H1 | 0.1557 ± 0.0036 | 2.6× better than floor |
| `M5_2t_ref` | G1 | 0.1311 ± 0.0022 | 3.0× better than floor |

At n=4 on both sides the H1 result is decisive (t ≈ 10): the canonical arm is not
a weak policy, it is an actively harmful one. Note also that the zero-action
policy *survives* — alive 0.93 (H1) / 0.98 (G1) — while the canonical arm has
alive 1.000 and still tracks worse. It stays up and moves wrongly.

---

### 2.4 Heading is badly wrong even where joint tracking is good

The four Viper render jobs failed at 20:05 for the known reason (no OSMesa, no
`/dev/dri` on a compute node) — but only their *second* step failed. The rollout
dumps were written. Pulled and rendered locally (`MUJOCO_GL=wgl`), policy beside
reference, 260 frames each:

| arm | robot | mean heading error | max |
|---|---|---:|---:|
| `M9_ref` | H1 | 59.9° | 126.9° |
| `M9_z` | H1 | 93.4° | 178.8° |
| `M9_both` | H1 | 95.6° | 179.7° |
| `M5_2t_canon` | H1 | 55.5° | 177.9° |
| `M5_2t_canon` | G1 | 66.0° | 93.9° |

These arms track joint angles at ~0.13 rad while facing, on average, 60–96° away
from the heading the clip asks for — and at times fully reversed. Shape RMSE
cannot see this, which is precisely why it looks like the arms agree. This is
consistent with the standing note that the heading term saturates, and it means
**the tracking numbers in this file are joint-space claims only**. One rollout
(env 0) per arm, so the ordering between arms is not readable; the magnitude is.

A renderer fix went in alongside: it enabled geom groups 0 and 2, which are H1's
floor and meshes, but G1's visual meshes are in group 1, so a G1 pane would have
rendered empty. Groups 0, 1 and 2 are now enabled — still excluding 3–5, which
are the collision proxies whose over-drawing the original comment warns about.

**Why nobody saw it: the crosseval never measured heading.** The per-robot report
carries `raw_rmse_rad`, its absolute twin, per-joint versions of both,
`alive_fraction`, `reference_vs_raw_rmse_rad` and ten foot metrics — and nothing
about where the robot is pointing. `heading_error` *was* being computed in
`crosseval_motion.py`, but only inside the `--dump_render` branch, and it was
written to the render npz and then thrown away. Every arm in every table in this
project has been ranked on a metric blind to a degree of freedom the videos show
going badly wrong.

`crosseval_motion.py` now accumulates absolute heading error over all envs and
emits `heading_error_deg_mean` and `heading_error_deg_p95`. The change is purely
additive and wrapped: if a configuration lacks the yaw bookkeeping, the keys come
back `None` (never `0.0`, which would read as perfect) and the joint metrics are
unaffected — the chained crossevals behind six running arms must not be put at
risk by a new metric.

### 2.5 Doing nothing tracks heading 15× better than any trained arm

With the metric in place, run against already-scored checkpoints on `superM9`,
H1 — and, decisively, against the zero-action floor:

| arm | joint RMSE (rad) | heading mean | n |
|---|---:|---:|---:|
| `M9_both` | 0.1268 | **82.39° ± 1.78** | 4 |
| `M9_ref` | 0.1311 | **82.40° ± 3.58** | 4 |
| `M9_z` | 0.1385 | 81.01° | 1 |
| **zero action, H1** | 0.3991 | **5.39° ± 0.13** | 4 |
| **zero action, G1** | 0.3804 | 16.24° ± 0.56 | 4 |

The p95 for the trained arms sits near 169°.

Every trained arm beats the floor on joints by ~3× and loses to it on heading by
~15×. A p95 near 169° means that in the worst 5 % of the rollout the robot is
facing almost exactly backwards. The three arms are indistinguishable from each
other here (81–83°), so this is not something the token does or fails to do — it
is a property of the recipe they all share.

This is the standing "the heading term saturates" note, finally with a number
against a denominator. It also means the entire FSQ comparison — every table in
this file and in `REPORT_FSQ_SCALE.md` — has been conducted between arms that are
all failing the same way on an axis none of them was scored on. It does not
invalidate the joint-space rankings, which are like-for-like. It does mean
"tracks the clip" was never an accurate description of what these policies do.

At n=4 the gap is **15.3×** with error bars of 1.8° and 0.13° — this is not seed
noise, and it is the largest and cleanest effect measured in this project.

It also holds across **every arm now scored**, which is the part that makes it a
property of the recipe rather than of a configuration:

| arm | robot | heading mean | n |
|---|---|---:|---:|
| `M1_both` | H1 | 83.0° ± 1.1 | 4 |
| `M4_both` | H1 | 85.7° ± 1.3 | 4 |
| `M9_both` | H1 | 82.4° ± 1.8 | 4 |
| `M9_ref` | H1 | 82.4° ± 3.6 | 4 |
| `M9_both_scram` | H1 | 84.3° ± 1.4 | 4 |
| `M5_both_scram` | H1 | 86.9° ± 2.8 | 4 |
| `M5_2t_ref_morph` | H1 / G1 | 84.4° ± 2.7 / 76.8° ± 6.2 | 4 |
| `M5_2t_both_morph` | H1 / G1 | 83.8° ± 2.3 / 82.4° ± 1.1 | 4 |
| `scale_2t_mb6144` | H1 / G1 | 81.8° ± 4.1 / 69.6° ± 4.7 | 4 |
| **zero action** | H1 / G1 | **5.4° ± 0.1 / 16.2° ± 0.6** | 4 |

Thirteen arms, one motion count from 1 to 9, one and two topologies, fixed and
randomized bodies, with and without the token, real token and scrambled — every
one of them lands between 70° and 87°, and the thing that does nothing lands at
5°. Nothing in the FSQ design space moves this. It is the reward recipe.

### 2.5b The cause: nothing in the reward scores heading

Not saturation, not a hard control problem — the term is off.

```
default_config.py:457   "root_heading_tracking_weight_ratio": 0.0,
viper_train.sbatch:144  ...weight_ratio="${HEADING_RATIO:-0.0}"
default_config.py:461   "deepmimic_heading_free": True,
```

`HEADING_RATIO` was UNSET in every arm of this campaign, so the explicit heading
term contributed nothing. And `deepmimic_heading_free=True` *removes* the
implicit heading anchor that the DeepMimic pose terms used to carry — its own
comment says "this is where heading is supposed to be scored instead". With the
anchor removed and the replacement term at weight zero, **no reward term is a
function of where the robot is facing.** The policies are not failing to track
heading; they were never asked to.

The temperature compounds it. `viper_train.sbatch:143` records that at
temperature 0.25 the term "saturates dead past ~60 deg" — and these arms sit at
82°, where `exp(-1.43² / 0.25) ≈ 3e-4`. Switching the weight on at the default
temperature would have produced almost no gradient, which is very likely why
earlier heading probes (`dp_head_*`) read as a dead end.

### 2.5c Turning the term on does NOT fix it — and the training curve shows why

Both arms finished. n=4:

| arm | joint RMSE | vs `M9_ref` | heading |
|---|---:|---:|---:|
| `M9_ref` (term off) | 0.1315 | — | 82.4° |
| `M9_head05` (ratio 0.5, temp 2.0) | 0.1337 | +1.7 % | 79.9° |
| `M9_head20` (ratio 2.0, temp 2.0) | 0.1550 | **+17.9 %** | 72.4° |

**The fix fails.** Four times the heading weight buys 10° of heading and costs
18 % of joint accuracy, and at 72° the arm is still 13× worse than the
zero-action floor's 5.4°. So §2.5b identified a true fact — no term scored
heading — but the inference that switching it on would repair the behaviour was
wrong.

The training curves say what is actually happening, and they were available all
along: `env_info/root_heading_error` has been logged every update of every run
in this project.

> Heading error **starts at ~11° and climbs to ~84° within the first 10M steps**,
> then sits there for the remaining 90M. Reference-only, token, both, scrambled,
> one topology, two, nominal bodies, randomized — every arm traces the same
> curve. The heading-term arms climb the same way and merely settle a little
> lower.

The policy is not failing to acquire heading. **It acquires the task by giving
heading up**, early, and no weight tried here reverses that. Something in the
recipe actively rewards rotating away — the pose terms are computed in a
heading-free frame (`deepmimic_heading_free=True`), so yaw is nearly free to the
dominant terms while presumably buying something on velocity-command or contact
terms. Finding what it buys is the next experiment, and it is a reward-shaping
question, not an FSQ one.

*(This also reconciles a number that looked like a contradiction: a live training
log sampled early reads ~0.19 rad, which is the 11° start, not a disagreement
with the 82° crosseval.)*

---

### 2.5d The ankle chatter: the feet are never allowed to leave the floor

Raised from watching `M5_2t_ref_morph` — H1's right ankle visibly jitters. It is
real, it is measurable, and it is not what randomization or the token does.

**Where it is.** Jitter = mean |second difference| of the angle series, on the raw
dump. H1, `M5_2t_ref_morph`:

| joint | executed | reference | ratio |
|---|---:|---:|---:|
| `ankle_angle_r` | **0.0682** | 0.0077 | **8.9×** |
| `ankle_angle_l` | 0.0395 | 0.0065 | 6.0× |
| `knee_angle_r` | 0.0521 | 0.0140 | 3.7× |

**The reference is smooth.** At 0.008 rad the commanded ankle trajectory has
essentially no frame-to-frame content; the policy's is nine times rougher. The
chatter is generated, not followed. Ankles are the worst joint on every arm
measured, morph or not (`M9_ref`, nominal bodies: 0.058 / 0.046) — so
randomization is not the cause either.

**Why.** The foot metrics say it plainly. H1, same arm:

| | policy | reference asks for |
|---|---:|---:|
| foot airborne fraction | **0.019** | **0.215** |
| foot penetration | **7.4 mm** | 0.0 mm |
| foot clearance | 5.5 mm | — |
| foot slip speed² | 0.336 | — |

**The policy keeps its feet on the ground 98 % of the time while the dance asks
for them to be airborne 21 % of the time**, pressed 7.4 mm into a floor the
reference never penetrates, and slipping. A foot that is pinned to the ground
still has to accommodate a moving torso, and the ankle is the last joint in that
chain — so the residual arrives there as rapid corrections. That is the jiggle.

**And the recipe asks for exactly this.** From the resolved flags:

```
--environment.reward.deepmimic_foot_height_weight_ratio=0.0   # foot-height tracking OFF
--environment.reward.foot_z_velocity_coeff=10.0               # vertical foot speed PENALISED
--environment.reward.foot_slip_coeff=20.0
--environment.reward.ground_penetration_coeff=1000
```

Nothing rewards the foot for reaching the height the clip asks for, and moving it
vertically is penalised at 10.0. The policy's solution — never lift, absorb
everything in the ankle — is the rational one under that reward.

This is structurally the same defect as §2.5b/§2.5c: **a shaping term at weight
zero next to a penalty pushing the other way.** Two of the three things a
DeepMimic recipe is supposed to track — where the robot faces, and where its feet
go — are unweighted in every arm this project has run.

*(One caution against a tempting story: the right ankle is worse than the left
here, 0.068 vs 0.039, and H1 has a documented history of right-sagittal-chain
sign bugs. But on `M9_ref` the **left** is the worse of the two, 0.058 vs 0.046,
so the asymmetry does not replicate and is most likely which foot carries stance
in that segment. It is not evidence of a sign error.)*

### 2.5e The recipe is not failing to learn these things — it SELLS THEM OFF

The single most useful measurement of the campaign, and it was available in every
log since the project began. `M9_ref`, band-averaged over training:

| steps (M) | foot-lift ratio | clearance | penetration | heading | joint err | ep_len |
|---:|---:|---:|---:|---:|---:|---:|
| 0–1 | **1.37** | 18.3 mm | 13.1 mm | 10.9° | 0.2959 | 16 |
| 3–6 | 0.82 | 8.6 mm | 9.2 mm | 26.9° | 0.0770 | 58 |
| 6–10 | 0.98 | 8.3 mm | 7.3 mm | **65.5°** | 0.0562 | 301 |
| 10–15 | **0.42** | 6.9 mm | 7.0 mm | 83.4° | 0.0445 | 789 |
| 20–30 | 0.25 | 6.2 mm | 7.3 mm | 83.2° | 0.0335 | 943 |
| 85–100 | 0.20 | — | — | 85.0° | 0.0259 | — |

At one million steps the policy lifts its feet **more** than the clip asks
(ratio 1.37) and faces nearly the right way (10.9°). **Heading is sold off
between 6M and 10M; the feet between 10M and 20M**; joint error improves
monotonically throughout. This is not a failure to acquire either behaviour.

*(An earlier draft of this section attributed the trade to
`joint_tracking_coeff=30` being "the only term carrying real weight". That is
wrong — see §2.5f. Joint tracking is 24 % of the reward; the root-pose terms are
64 %.)*

It also invalidates any comparison made before ~20M: at 3M the baseline sits at
ratio 0.93 and looks healthy.

**Does turning the terms on stop it?** Three arms at 30M, read at 15–20M against
the baseline's own 15–20M band:

| arm | ratio | clearance | penetration | heading | joint err |
|---|---:|---:|---:|---:|---:|
| `M9_ref` baseline | 0.30 | 6.2 mm | 7.24 mm | 83.6° | 0.0382 |
| `fx_foot` (FOOTH 1.0) | 0.48 | 15.0 mm | 4.38 mm | 86.0° | 0.0598 |
| **`fx_footz`** (FOOTH 1.0, FOOTZVEL 10→1) | **0.54** | **15.0 mm** | **4.48 mm** | 85.1° | 0.0550 |
| `fx_all` (+ heading 2.0) | 0.47 | 10.0 mm | 4.87 mm | 85.2° | 0.0798 |

**The feet respond. Heading does not.**

- Foot-lift ratio **0.30 → 0.54** (+80 %), clearance **6.2 → 15.0 mm** (2.4×),
  penetration **7.24 → 4.48 mm** (−38 %). The feet genuinely leave the floor now.
- The cost is real: joint error **0.0382 → 0.0550, +44 %**.
- The sell-off is **slowed, not stopped** — `fx_footz` still falls from 1.09 at
  6–10M to 0.54 at 15–20M.
- **Heading is completely unmoved**: 85.1° with the term on versus 83.6° without,
  and `fx_all` at four times the weight is 85.2° while costing **+109 %** joint
  error. That is the third independent demonstration that this heading term does
  not control heading — after `M9_head05`/`M9_head20` (§2.5c) and the historical
  `dp_head_*` probes.

So: **one of the two unweighted behaviours is recoverable by re-weighting and one
is not.** The foot term works and needs a cost/benefit decision (44 % joint error
for feet that actually lift). The heading term is either mis-specified, applied to
the wrong quantity, or dominated — and that should be diagnosed by reading what
`root_heading_tracking_reward` actually computes, not by another weight sweep.

### 2.5f WHY the baseline is not good: heading is unrewardable by construction

The reward, measured rather than assumed. `M9_ref`, mean over its last 200
updates, total 1.611:

| term | mean | share |
|---|---:|---:|
| `rpos_tracking_reward` | 0.625 | **39 %** |
| `rquat_tracking_reward` | 0.402 | **25 %** |
| `joint_tracking_reward` | 0.392 | 24 % |
| `qvel_tracking_reward` | 0.227 | 14 % |
| `root_height_tracking_reward` | 0.017 | 1 % |
| every penalty combined | −0.08 | — |
| `root_heading_tracking_reward` | — | **absent** |
| `foot_height_reward` | 0.000 | **zero** |

So **64 % of the reward is root pose**, not joints. And here is what the root
pose is compared against, every step, in `reward_functions/tracking.py`:

```python
if self.env.has_free_base and self.deepmimic_heading_free:
    yaw = jnp.arctan2(...)              # the ROBOT'S OWN current yaw
    reference_qpos = reference_qpos.at[3:7].set(
        jnp.array([jnp.cos(0.5 * yaw), 0.0, 0.0, jnp.sin(0.5 * yaw)]))
```

**The reference's orientation is overwritten with the robot's own heading before
the pose terms are computed.** `rquat` therefore cannot penalise heading — the
target *is* wherever the robot happens to be facing — and `rpos` is measured in
that same rotated frame. Heading is not merely unweighted, it is **unrewardable
by construction**, and the explicit term intended to replace it ships at 0.0.

The design is deliberate and the code says so: yaw "carries no pose meaning while
dominating both world-frame comparisons", and heading "belongs in its own
explicit term". The counterpart term was simply never switched on.

> **The baseline is not broken. It is a heading-free, root-relative imitation,
> and it is doing that correctly.** It reproduces the clip in the robot's own
> frame. What it never promised — and what a viewer means by "dancing the clip" —
> is a world-frame reproduction. That is why the robot wanders 3.77 m where the
> clip travels 0.56 m and faces 84° away while scoring well.

This also explains why raising `root_heading_tracking_weight_ratio` fails
(§2.5c, §2.5e): it adds a *bounded reward* for facing correctly, but rotating
still costs nothing on the 64 % that is root pose, while disturbing the joint
terms. There is no pressure for it to work against.

**The untried experiment is `deepmimic_heading_free=False`**, which makes `rquat`
measure true world orientation and puts 25 % of the reward directly on heading.
Nobody has ever run it. `fx_hfree` and `fx_hfree_foot` (the latter with the foot
fix that worked) are training.

### 2.6 Randomized bodies are nearly free — and the token's gain does not survive them

`M5_2t_both_morph` and `M5_2t_ref_morph`: one policy, two topologies, five
motions, morphology randomization ramping to 0.3, with and without the token.
n=4, `super5`:

| arm | H1 | G1 |
|---|---|---|
| `M5_2t_ref_morph` | 0.1585 ± 0.0022 | **0.1245 ± 0.0024** |
| `M5_2t_both_morph` | **0.1577 ± 0.0011** | 0.1289 ± 0.0042 |
| token effect | −0.5 %, t ≈ 0.7 | +3.5 % *worse*, t ≈ 1.8 |
| `M5_2t_ref` (no randomization, n=3) | 0.1557 ± 0.0036 | 0.1311 ± 0.0022 |

Two separate readings.

**The good one, and it is the more important of the two for the pipeline.**
Randomizing the bodies is nearly free: against the fixed-body arm, H1 is 1.8 %
worse and G1 is 5.0 % *better*. One policy driving two topologies, five motions
and randomized bodies tracks about as well as one driving two topologies and five
motions on nominal bodies. Morphology randomization at 0.3 is not costing
tracking quality here.

**The null one.** The token's 2–3 % single-topology benefit does **not** survive:
it is indistinguishable from zero on H1 and marginally negative on G1. Two things
changed at once relative to `M9_both` / `M5_both` — a second topology and
randomized bodies — so this pair could not say which was responsible.

### 2.6b It is the second topology, not the randomization

`M9_both_morph` / `M9_ref_morph`: one topology, same clip and recipe as
`M9_both` / `M9_ref`, morphology randomization the only change. H1, `superM9`:

| bodies | reference only | both | token effect |
|---|---:|---:|---:|
| nominal | 0.1315 ± 0.0029 *(n=9)* | 0.1263 ± 0.0044 *(n=8)* | **−4.0 %** |
| randomized (0.3) | 0.1376 ± 0.0051 *(n=4)* | 0.1321 ± 0.0039 *(n=4)* | **−4.0 %** |
| cost of randomizing | +4.6 % | +4.6 % | — |

**The token's benefit is unchanged by randomization — −4.0 % in both rows — and
randomization costs both arms exactly the same +4.6 %.** The two effects are
independent and additive; nothing about a randomized body interferes with what
the token supplies.

So the null in §2.6 belongs to the **second topology**, not to the randomization.
That is the more interesting failure anyway.

### 2.6d The explanation I reached for is false, and checking cost two minutes

The obvious reading of §2.6b was that the per-joint code is fitted on H1's clip,
so a second body has to read a code never built for it. That predicts a clean
experiment — refit the token jointly on both bodies — so before submitting it I
checked the premise:

```
tokenizer_M9/config.json
  robots:          ['UnitreeH1', 'UnitreeG1']
  final_eval_loss: {UnitreeH1: 0.006399, UnitreeG1: 0.013017}

clips_super/UnitreeH1/super5dance_zq.npz   z_q (45125, 19, 32)
clips_super/UnitreeG1/super5dance_zq.npz   z_q (45125, 23, 32)   different content
```

**The tokenizer is already fitted jointly on both robots, and G1 already carries
its own 23-joint code.** The proposed experiment is the one that was already run.
Had it gone to the queue it would have spent three GPU-hours reproducing the
status quo and "confirmed" a hypothesis that was never at risk.

The same config supplies a better question. **G1's reconstruction is 2× worse
than H1's** (0.0130 vs 0.0064), which means the 2-topology null has two candidate
causes that no arm has yet separated:

- **(a) sharing** — one policy split across two bodies cannot exploit the extra
  channel. Consistent with the 2t arms being ~18 % worse on H1 than the 1t arms
  even before the token is considered;
- **(b) the G1 token** — it is simply a weaker code, and what looks like a
  two-topology null is really G1's null showing through the average.

A **single-topology G1 pair** decides it, and G1-alone has never been run.

### 2.6e ANSWERED: it is not sharing. The token simply does not work on G1.

`M5_g1_ref` / `M5_g1_both`, 576 envs, 99.5M steps, n=4 each on `super5dance`:

| arm | shape RMSE | heading |
|---|---:|---:|
| `M5_g1_ref` | 0.1233 ± 0.0013 | 79.5° |
| `M5_g1_both` | 0.1241 ± 0.0017 | 78.0° |

**Token effect on G1 alone: +0.6 %, t = 0.73.** Nothing, with G1 as the only
robot in the policy.

| condition | token effect |
|---|---:|
| H1 alone | **−4.0 %** |
| **G1 alone** | **+0.6 %** |
| two topologies (both robots) | ~0 % |

So hypothesis (a) is dead: **sharing a policy across two bodies is not what
destroys the gain.** The gain was never there for G1. What looked like "the
second topology erases it" was G1's null diluting H1's effect in the average, and
that name should be struck.

This lines up with the tokenizer's own reconstruction losses, which were in the
config the whole time: **H1 0.0064, G1 0.0130** — G1's code is twice as poor a
description of G1's motion as H1's is of H1's.

**This narrows the headline claim substantially.** The token's benefit is
demonstrated on *one robot*, not on humanoids in general. Before FSQ can be
called a component rather than a curiosity, the next question is whether the
G1 null is (i) a tokenizer-quality problem — retrain G1's code to H1's
reconstruction level and see if the gain appears — or (ii) something about G1
itself, in which case "when does FSQ work" gains a body-dependent term nobody has
characterised.

### 2.6c The 2×2, closed

`M5_2t_both` landed and completes it. Token effect (both vs reference-only):

| | 1 topology | 2 topologies |
|---|---:|---:|
| **nominal bodies** | **−4.0 %** | +1.0 % (H1), −2.9 % (G1) |
| **randomized bodies** | **−4.0 %** | −0.5 % (H1), +3.5 % (G1) |

The left column is identical top to bottom; the right column is noise around
zero in both rows and both robots. **Randomization is free to the token; the
second topology is fatal to it.** No interaction between the two.

### 2.2f The motion sweep, closed

`M1_ref` never existed and now does, so the auxiliary effect has a reference arm
at every M. All n=4, `both` vs `ref`:

| M | reference | both | token effect |
|---:|---:|---:|---:|
| 1 | 0.1343 ± 0.0025 | 0.1267 ± 0.0024 | **−5.7 %** (t ≈ 4.4) |
| 4 | 0.1363 ± 0.0044 | 0.1244 ± 0.0032 | **−8.7 %** (t ≈ 4.4) |
| 5 | 0.1424 ± 0.0023 | 0.1395 ± 0.0016 | −2.0 % (t ≈ 2.1) |
| 9 | 0.1315 ± 0.0029 | 0.1263 ± 0.0044 | −4.0 % (t ≈ 1.7) |

Negative at every M, significant at M=1 and M=4, and **not monotone** — the token
is not simply "more useful when there is more motion to compress". Whatever sets
the size of the effect, it is not the number of motions.

*(A caution recorded because it nearly went into this table: `M9_ref` also has a
`novel`-clip condition scoring 0.1208, and pooling by arm name without filtering
on the clip silently moved its mean from 0.1315 to 0.1282 — which would have
halved the apparent token effect. Always filter on `eval_condition["clip"]`.)*

---

## 3b. Interrupted at 05:45 — four arms were in flight

The ssh control sockets to `gate` and `viper11` disappeared at about 05:45
(`~/.ssh/` retained only `config` and `known_hosts`; most likely the WSL instance
restarted). Both hops need password + OTP, which this session cannot supply, so
collection stopped there. No `ssh -O exit` was run on either master; two of this
session's own hung *client* processes were cleared, which is a different thing.

**The jobs are unaffected** — they run on Viper and their `afterok` chains fire
without supervision. Everything above was already collected. Outstanding:

| arm | what it answers | why it matters |
|---|---|---|
| `M9_head05`, `M9_head20` | heading term on, ratio 0.5 / 2.0, temperature 2.0 | **the most consequential result pending** — is §2.5b's defect actually fixable, and what does it cost joint tracking |
| `M5_2t_both` | two topologies, nominal bodies | the fourth corner of §2.6b's 2×2; confirms the second topology is what erases the token |
| `M1_ref` | the M=1 reference arm, which never existed | closes the auxiliary sweep at M = 1, 4, 5, 9 |

To collect once access is restored:

```
ssh -MNf gate && ssh -MNf viper11 && ssh viper11 date
ssh viper11 'cd /ptmp/akalenik/urma && \
  module load python-waterboa/2025.06 && eval "$(conda shell.bash hook)" && \
  conda activate ./env && python tools/agg_wave2.py'
```

`tools/agg_wave2.py` pools by arm/condition/robot and flags anything at n<4.
Filter on `eval_condition["clip"]` before comparing arms (see §2.6b).

---

## 3. Open, running now

| question | arm | status |
|---|---|---|
| is the auxiliary gain information or smoothing? | `M9_both_scram`, `M5_both_scram` | training |
| is the decoder the wall for body-independent tokens? | `canon_dw2`, `canon_dw4` → `gate_canon` | training (first run that could ever answer it) |
| does a third topology cost the other two? | `scale_3t_d4` + `scale_2t_mb6144` control | resubmitted after the minibatch fix |
| does the auxiliary gain survive more M, and two robots? | `M1_both`, `M4_both`, `M5_2t_both` | chained |
| does the token still help when the bodies are randomized? | `M5_2t_both_morph` + `M5_2t_ref_morph` control | submitted |

`M5_2t_both_morph` is the most "several of everything" run the existing data
supports: one policy, two topologies, five motions, morphology randomization
ramping to 0.3, and the token on top of the reference. The full cross with three
topologies is not reachable tonight — the super-clip needs five dances per robot
and BoosterT1 has offline retargets for only two of them.

---

## 4. Ideas this night has already closed

- **"More than two topologies is a ROCm memory wall."** Falsified: the 3-robot
  env builds on the APU. The failure was `8192 % 3`.
- **"The 3-topology run needs a new retarget."** Falsified: the BoosterT1 clip
  was already on the cluster; only a dictionary entry was missing.
- **"Decoder width was tested."** Falsified: the knob could not run at any width
  but 1.0. Any earlier reasoning that treated decoder capacity as tested is void.
- **"We have seven families' worth of `dance2_subject4` retargets."** We have the
  files, but only three of them describe the same robot loco_mjx simulates.
  Atlas, Talos and ToddlerBot fail the FK screen.
- **"The sign tables don't collide, so the name-keyed lookup is fine."** True by
  accident only. Atlas collides with H1 on 13 names, 6 of them wrong.

## 6. THE DECODER GATE: failed, but the decoder was the right suspect

> **RETRACTED 2026-08-27 — the numbers in this section are train-contaminated.**
> `cmd_reconstruct` in `scripts/scaling/canonical_fsq_clip.py` looped
> `range(0, T)` over the whole clip while `--test-fraction` defaulted to 0.1, so
> every `qpos_rmse_rad` below — including the 0.1774 gate figure — was scored on
> ~90 % training data. The training histories show the ranking is not just noisy
> but **backwards**: widening the decoder cut train loss 2.8× and made held-out
> eval *worse*.
>
> ```
> canonical_tokenizer_w32: train 2.2784 -> 0.5530   eval[H1] 2.0496 -> 1.1637
> canonical_tokenizer_dw4: train 2.1970 -> 0.2010   eval[H1] 2.0041 -> 1.4086
> ```
>
> So "decoder width is the best lever, monotone and unsaturated" measures
> memorisation, not capacity. The script now emits `qpos_rmse_rad_heldout`;
> re-scoring every bundle is task **A2** in `docs/notes/FSQ_NEXT_RUNS_TODO.md`.
> The *gate failure* still stands — held-out is worse than the mixed figure, not
> better — but the *diagnosis of why* does not.

`gate_canon` ran at 00:17 and wrote:

```
dw2: H1 0.2178  G1(retarget) 0.2151
dw4: H1 0.1774  G1(retarget) 0.1787
w32: H1 0.2486  G1(retarget) 0.2434
BEST dw4 0.1774
GATE FAILED (0.177381 rad, need < 0.10).
No RL arm submitted -- a 3 h slot saved.
```

**A correction.** Mid-night this report inferred from the trainer's own `eval`
metric — which *rose* from 1.18 to 1.34 (dw2) and 1.23 to 1.41 (dw4) while train
loss fell — that the decoder was overfitting and that wider was worse. The
reconstruction report says the opposite on the metric that counts. The trainer
eval and `qpos_rmse_rad` are different quantities, which the text flagged, but
the directional conclusion drawn from the wrong one was still wrong.

What the decoder width actually buys, all at code width 32:

| decoder width | H1 reconstruction (rad) | vs previous |
|---:|---:|---:|
| 1 (`w32`) | 0.2486 | — |
| 2 (`dw2`) | 0.2178 | −12.4 % |
| 4 (`dw4`) | **0.1774** | −18.6 % |

So the decoder is **the best lever found for the body-independent token**, and by
some distance. Against the two dead hypotheses — a 32× bigger codebook bought
6 %, and going from 4 to 14 encoder sites bought 0 % — doubling the decoder twice
bought 29 %, monotonically, with no sign of saturating.

It still fails the gate by 1.77×. At ~18 % per doubling, reaching 0.10 rad needs
roughly three more doublings — a decoder of width 32 — and the gate exists
because below 0.10 rad an RL arm is worth three hours and above it, it is not.

The honest verdict is therefore narrower than "the decoder is the wall":

> Decoder capacity is a real and unsaturated limiter of the body-independent
> token, and it is the only one of the three tested causes that is. It is not
> enough on its own to make the token usable, and nothing tonight justifies
> spending an RL arm on it.

The gate did its job: it declined to spend the slot, on a number, without a human
in the loop.

### 6b. A2 RESULT (2026-08-27, job 11129367): re-scored held-out

Every tokenizer re-scored on held-out frames. `infl` = how much the old mixed
figure flattered the model.

| tokenizer | H1 held-out | H1 mixed | infl |
|---|---:|---:|---:|
| `w16` (decoder width 0.5) | 0.3377 | 0.2839 | 1.19x |
| `w32` (width 1) | 0.2921 | 0.2486 | 1.17x |
| `dw2` (width 2) | 0.2973 | 0.2178 | 1.36x |
| `dw4` (width 4) | **0.2897** | 0.1774 | 1.63x |
| `r1k` / `r32k` / `b12800` / `b32768` (codebook) | 0.38-0.39 | 0.32-0.37 | 1.04-1.13x |
| `tokenizer_M9` (PER-JOINT) | **0.0498** | 0.0508 | **0.98x** |
| `tokenizer_M9_la20` (per-joint, lookahead 20) | 0.1126 | 0.0933 | 1.21x |

**Decoder width buys nothing.** Held-out across widths 1 -> 2 -> 4 is
0.2921 -> 0.2973 -> 0.2897: flat within noise and non-monotone, against a mixed
figure that fell 29 % monotonically. The inflation factor climbs with width
(1.17x -> 1.36x -> 1.63x), which is the signature of a wider decoder buying
memorisation and nothing else. The retracted §6 claim is now not merely
unquotable but positively false: width is NOT a lever on this design.
The one real width effect is 0.5 -> 1 (0.3377 -> 0.2921); everything above
saturates immediately.

**The per-joint number was NOT contaminated** (0.0498 held-out vs 0.0508 mixed,
infl 0.98x). So only the canonical half of the old comparison was inflated, and
correcting it makes the per-joint advantage *larger*, not smaller:

> per-joint 0.0498 vs canonical 0.2897 held-out = **5.8x**, not the 3.5x
> the contaminated figures implied.

Per-joint on H1 also **clears the 0.10 rad gate** (0.0498); G1 does not (0.1160).
Lookahead 20 is worse than 10 on both bodies.

### 6c. A4 RESULT (2026-08-27, jobs 11129887-90): the per-joint 2x2

Kevin's rewritten encoder and decoder (autoencoder branch `8f04ee8`/`7a7fb1e`),
vendored beside the originals as an `FSQ_ENC`/`FSQ_DEC` switch so v1 stays the
default. Same clip (`superM9.npz`), epochs (400) and latent dim (32) as
`tokenizer_M9`, so this is comparable to the result it is trying to explain.
Held-out qpos RMSE, rad:

| | dec v1 (URMA softmax mask) | dec v2 (descriptor concat) |
|---|---|---|
| **enc v1** (`jnp.mean(x, axis=1)`) | H1 0.0523 · G1 0.1274 | H1 0.0333 · G1 0.0318 |
| **enc v2** (per-joint Conv1d over time) | H1 0.0430 · G1 0.1004 | **H1 0.0293 · G1 0.0281** |

**The decoder dominates; the encoder is secondary.** Holding the encoder at v1,
swapping the decoder is worth -36 % on H1 and **-75 % on G1**. Holding the
decoder at v1, swapping the encoder is worth only -18 % / -21 %. Together:
H1 1.8x, G1 4.5x.

This **inverts the prediction logged in this report on the same day**. The
argument was that our positive RL result (-3.5 % at M=9) rests on lookahead, that
enc v1 averages the lookahead window away before quantizing, and that the encoder
cell would therefore be the one that mattered to us. The encoder change is real
but it is the smaller half. The URMA joint-softmax mask was the bigger defect on
the per-joint path too -- not only on the canonical one.

**CORRECTION 2026-08-29 — "enc v2 keeps the time axis" is FALSE.** Read the
source: `algorithms/autoencoder/nn/encoder.py` runs two `Conv(kernel_size=3,
padding="SAME")` layers over time and then does `jnp.mean(x, axis=1)`. **Both
encoder versions pool the time axis away.** The v1/v2 difference is the
convolution BEFORE the pool, not the survival of the time axis. Every argument in
this report that turned on "v2 preserves lookahead where v1 averaged it away" was
built on a distinction that does not exist.

What the token actually carries: `build_windows` stacks frames t..t+9 (250 ms at
40 fps, genuinely FORWARD-looking), the convolution smooths ±2 frames, and the
mean collapses all of it to one vector per joint. So the token is **a smoothed
average of the next quarter-second**, not a trajectory and not a sample. That is
exactly why §2.2e found a raw future frame does not reproduce the token's gain,
and it predicts CH-N: a blurry 250 ms average is redundant beside an exact
per-frame reference (costs 10 %) and is the missing information beside a stale
one (pays 10-15 %).

**Bug in the same function.** The loop is `for k in range(lookahead)` with
`idx = arange(T) + k`, so row 0 and row 1 are BOTH frame t. The current frame is
duplicated and double-weighted in the mean, and the effective lookahead is 9
frames, not 10 — biasing the token toward the present, the opposite of its
purpose. NOT fixed in code: changing it would invalidate every existing `_zq`
sidecar, so it needs a deliberate regeneration, not a silent edit.

**It also fixes generalization, not just fit.** G1 under v1/v1 was inflated 1.42x
(0.1274 held-out vs 0.0896 mixed). Under v2/v2 the inflation is gone (0.0281 vs
0.0284, 0.99x). Contrast the canonical width sweep, where extra capacity bought
*only* memorisation (inflation climbing to 1.63x). The new decoder is a better
model, not a bigger one.

**Gate.** G1 previously FAILED the 0.10 rad gate at 0.1274 and now passes at
0.0281. Both bodies now clear it with ~3.5x margin.

**Control.** The v1/v1 cell reproduces `tokenizer_M9` (A2: H1 0.0498, G1 0.1160)
to within 5 % / 10 % across an independent run, which is the evidence that the
switch's v1 defaults did not silently change behaviour.

**On Kevin's ~100x claim.** It does not reproduce at that magnitude on our clips:
we measure 1.8x-4.5x held-out. The direction is real and the mechanism is now
localised, but the number is his data, his split and his SNR definition -- A6
still needs to ask whether his figure is held-out.

**Against canonical.** Best per-joint held-out is now 0.0293 vs best canonical
0.2897, a 9.9x gap. Whether the same decoder narrows it on the shared-code design
is exactly what `a7_canon_v1/v2` is running to answer.

---

# 7. NIGHT OF 27→28 AUGUST — the qvel defect, and what it rescues

Written by the OPERATOR during the run, in parallel with an INVESTIGATOR session
working the same board (`docs/notes/FSQ_FINDINGS_LOG.md`). Every number below is
either a completed job or a re-analysis of an artifact already on disk. Where a
claim was checked and failed, it is recorded as failed.

## 7.1 The headline: the FSQ loss was ~94% VELOCITY, and fixing it is the single largest lever found in this project

The tokenizer target is `(T, 11, J, 2) = [qpos, qvel]` and the loss was a flat,
unweighted `jnp.mean(jnp.square(recon - target))` over all of it — in BOTH
designs (`canonical_fsq_clip.py`, and upstream `step.py:50`/`:73` for the
per-joint design, which share `build_windows`). Targets are never normalised.

Measured directly on `canonical_bundle_M5rich`, the bundle `a7_canon` used:

| robot | qpos var | qvel var | ratio | qvel share of an unweighted MSE |
|---|---|---|---|---|
| UnitreeH1 | 0.3316 | 4.8992 | 14.8x | **93.7%** |
| UnitreeG1 | 0.2966 | 4.2976 | 14.5x | **93.5%** |

So ~94% of the optimisation went to velocity, while **every gate and the entire
downstream RL reference are position**. The investigator measured 96.8% on a
different bundle; 93.5-93.7% is the figure that governs the M5rich runs.

**The fix, as a dose-response on the per-joint design** (enc=v2, dec=v2, all else
matched; `QVEL_W` weights the velocity channel):

| QVEL_W | H1 qpos | G1 qpos | H1 qvel | G1 qvel |
|---|---|---|---|---|
| 1.0 (baseline) | 0.0293 | 0.0281 | 0.0742 | 0.0735 |
| 0.235 | 0.0256 (-12.6%) | 0.0237 (-15.7%) | 0.0785 (+5.8%) | 0.0783 |
| **0.05** | **0.0189 (-35.5%)** | **0.0170 (-39.5%)** | **0.0889 (+19.8%)** | 0.0874 |
| 0.0 | 0.0090 (-69.3%) | 0.0079 (-71.9%) | 1.9391 (**+2513%**) | 1.7999 |

Monotone in qpos across four points. **There is a cliff between 0.05 and 0.0**:
the last step buys 2.1x more position for 21.8x worse velocity.

**RECOMMENDATION: ship `QVEL_W = 0.05`, not 0.0.** The RL reference consumes
velocity (`deepmimic_qvel_temperature`, and a live `joint_velocity` reward term),
so the headline -69% arm is not deployable. This is the one operational change
from the night.

**Verification run against it.** (a) A pure loss-scale artefact was a live
alternative, because the patch normalises by `(1+QVEL_W)/2` and so doubles the
loss scale at `QVEL_W=0`. It is refuted by the qvel column: a scale change cannot
*selectively* destroy velocity while improving position. (b) All four gaps are far
outside the tokenizer-fit noise floor measured below (4.52% p2p). (c) The
operator predicted a NULL on this design ("already 3x inside the gate, little
headroom") and was wrong; the prediction is recorded as refuted.

## 7.2 The first error bar on `qpos_rmse_rad_heldout`

Five seeds at EP=40, everything else matched:

| | mean | sd | peak-to-peak |
|---|---|---|---|
| UnitreeH1 | 0.2942 | 0.0063 (2.13%) | **4.52%** |
| UnitreeG1 | 0.2734 | 0.0045 (1.64%) | **3.40%** |

**This is not the 4.95% figure quoted elsewhere in the project** — that one is
for crosseval *rollouts*. The tokenizer-*fit* floor had never been measured.

Consequences, applied retroactively:

| claim | gap | verdict |
|---|---|---|
| Kevin's decoder v1 vs v2, canonical | 10.3% / 8.2% | **SURVIVES** (~5 sd) |
| `QVEL_W` 1.0 to 0.0 | 69.3% | **SURVIVES** overwhelmingly |
| epochs 40 vs 150 | 1.2% / 0.4% | **INSIDE NOISE** |
| canonical decoder-width sweep | 2.6% | **INSIDE NOISE** |

Caveat held: measured at EP=40. ep40 and ep150 share a mean, so the variance is
probably comparable, but that is an assumption.

## 7.3 Canonical: closed on the gate, with one caveat still running

`a7_canon` — Kevin's rewritten decoder on the canonical (body-independent) token:

| decoder | H1 held-out | G1 held-out |
|---|---|---|
| v1 | 0.2967 | 0.2778 |
| v2 | 0.2661 | 0.2549 |

Both are above the 0.20 gate, so **the body-independent direction closes**.
Kevin's decoder buys only -10.3%/-8.2% here against -36%/-75% on the
per-joint token: the canonical bottleneck is the TOKEN, not the decoder.
Codebook use was 43094/45125, so this is not collapse. The canonical `QVEL_W`
arms were still running at write-up; if they land below 0.20 this verdict
reopens, and that is stated here so it cannot be quietly forgotten.

**The epoch axis is closed.** ep10 0.3504, ep20 0.3255, ep40 0.3003, ep80 0.2996,
ep150 0.2967 — position improves steeply to ~40 and is flat thereafter within
noise. The operator's hypothesis that the 150-epoch number was the *worst* point
of an overfitting curve is **refuted**: the rising `eval_loss` is ~94% a velocity
curve and says nothing about the position gate. Practical: **train canonical
tokenizers at 40 epochs, not 150** — same result, 3.75x cheaper.

## 7.4 BoosterT1 is a broken retarget, not a hard body

`BROBOTS nr=3` gave BoosterT1 held-out 1.0355 against H1 0.2605 and G1 0.2321.
The per-joint breakdown — already present in the report file, at zero compute:

| robot | median joint RMSE | max | top-5 share of variance |
|---|---|---|---|
| H1 | 0.2034 | 0.3400 | 51% |
| G1 | 0.2040 | 0.3181 | 42% |
| BoosterT1 | **0.2080** | **1.4005** | **81%** |

Its *typical* joint reconstructs exactly as well as H1's and G1's. Splitting by
the investigator's span-feasibility analysis:

| subset | n | RMSE |
|---|---|---|
| BoosterT1 span-FEASIBLE (hips/knees/waist/head) | 11 | **0.1687** |
| UnitreeH1 legs+waist | 9 | 0.1672 |
| UnitreeG1 legs+waist | 9 | 0.1661 |
| BoosterT1 span-INFEASIBLE (shoulders/elbows/ankles) | 12 | **0.5282** |

**Within 1.6% of the good robots on the joints its retarget can actually
represent.** The entire 4x penalty is the 12 joints the retargeter drove past
their physical limits. There is no residual capacity effect.

**This kills the capacity hypothesis at its source**: the "BoosterT1 has ~2x H1's
amplitude" figure that motivated it *is* the defect.

**Consequences.** The robot-count curve measures retarget quality, not tokenizer
capacity, and must not be reported as a scaling result. It also cannot currently
be extended: H1v2 and ToddlerBot fail descriptor build, Atlas and BoosterT1 are
broken, leaving H1+G1 — already measured.

**Operator error recorded:** the tick-1 clip audit mislabelled every joint,
because `joint_names` carries `n+1` entries with the free joint `root` first and
the code used `names[:n]`. Robot-level statistics are label-independent and
stand; the names did not. Corrected, BoosterT1's largest-amplitude joint is
`Left_Elbow_Pitch` — the same joint the investigator's independent span analysis
named, which is a stronger confirmation than either route alone.

## 7.5 Heading: a null, and a metric that must never be used again

`fx_head_t025` vs `fx_head_t20` both ran to completion (29,491,200 steps).

**The raw reading was an artefact and is retracted.** `root_heading_error` is a
per-episode MEAN, and heading drifts within an episode, so a policy that survives
longer posts a larger mean at identical control. Correlation of heading error
with episode length: **r = +0.90 to +0.94** in all four arm x robot cells, with
episode length growing ~16x.

Corrected to a drift rate (error / episode length), heading control **improves
77-83%** during training in both arms. And the arms end at comparable episode
length (3% and 0.9% apart) with drift rates 1.8% and 1.2% apart:
**the temperature null is genuine, now correctly measured.**

**Standing rule added: `root_heading_error` must never be compared across runs or
across training without dividing by episode length.** Every earlier heading claim
in this project rests on the raw mean.

`fx_head_off` (heading weight 0.5 to 0.0, else identical) is running as the
matched on/off control this project has never had.

## 7.6 The legs-vs-arms split, sharpened — and it converges on the ANKLE

Chain split of executed-vs-clip per-joint RMSE, and the same normalised by each
joint's reference amplitude (RMSE/std = 1.0 is exactly what emitting the
reference's *mean* would score):

| robot | chain | RMSE | ref std | RMSE/std |
|---|---|---|---|---|
| H1 | hip+knee | 0.2453 | 0.3128 | 0.784 |
| H1 | **ankle** | 0.2614 | 0.1856 | **1.409** |
| H1 | arm | 0.1715 | 0.5756 | **0.298** |
| H1 | trunk | 0.0993 | 0.1002 | 0.991 |
| G1 | hip+knee | 0.2120 | 0.2886 | 0.734 |
| G1 | **ankle** | 0.1728 | 0.1520 | **1.136** |
| G1 | arm | 0.1314 | 0.5079 | **0.259** |
| T1 | **ankle** | 0.2125 | 0.1589 | **1.338** |

Two things follow. **The split is ~2x larger than raw radians suggest** —
normalised leg/arm is 2.63x (H1) and 2.84x (G1) against raw 1.43x/1.61x, because
the arms move about twice as much and still track better in absolute terms.
And **the ankles are worse than a mean-predictor on all three robots**, with
H1's trunk sitting exactly on it at 0.991. Only the arms track with real margin.

**The ankle carries a large CONSTANT postural offset.** The centred and absolute
per-joint metrics differ by an implied constant bias `b = sqrt(abs^2 - centred^2)`:

| robot | chain | bias (rad) | bias (deg) |
|---|---|---|---|
| H1 | **ankle** | **0.5066** | **29.0** |
| H1 | hip+knee | 0.0666 | 3.8 |
| H1 | arm | 0.0831 | 4.8 |
| G1 | **ankle** | **0.1686** | **9.7** |
| T1 | **ankle** | **0.2439** | **14.0** |

Every robot's ankle carries the largest constant offset of any chain, 6-8x the
rest. The robot holds its ankle at a systematically different angle than the
reference asks, all the time — and the centred metric hides this completely
(H1's ankle looks unremarkable at 0.2566 centred, and is the worst number in the
table at 0.5679 absolute).

**Partial mechanism test.** Comparing the implied bias against the reference
ankle's own offset from neutral: G1's ratio is **0.91** — almost exactly the size
needed to put the executed ankle at zero, consistent with the policy standing
flat while the reference asks for -0.20 rad. But H1's ratio is **2.13** and T1's
**1.46**, so "the policy simply holds the ankle neutral" cannot be the whole
story. The sign of the bias is NOT determined by this test (`b` is a magnitude),
and settling it needs executed angles from a crosseval dump — not yet run.

The structural candidate now standing is the investigator's F13: the tracking
objective is chain-symmetric (one scalar `exp(-MSE/T)` over all joints, no
per-chain weight) while the *constraint* load is entirely on the legs — 9
leg-only reward terms, 0 arm-only. The ankle is the joint that holds the CoM over
the foot, so it is the most contended joint between tracking and those nine, and
a constant offset is what that tug-of-war should look like. `r8_control` vs
`r8_legfree` (the nine scaled 0.1x, everything else identical) is running to
test it.

Eliminated this night: capacity (one shared decoder, no per-chain parameters),
descriptor degeneracy (legs separate *better* than arms), and actuation — leg
actuators command 4-6% of available torque at the error where they fail, while
the joints that *do* saturate are ARM joints, which is the chain that passes.

## 7.7 What the night falsified

- **Canonical is decoder-limited.** No — the token is the bottleneck.
- **The 150-epoch canonical number is an overfitting artefact.** No — position
  saturates by epoch 40 and is flat to 150.
- **BoosterT1 is a hard body / one code cannot cover its excursion.** No — broken
  retarget; feasible joints match H1/G1 within 1.6%.
- **Dropping qvel will be a null on the per-joint design.** No — it is the
  largest single lever in the project (-69%), though 0.05 is the shippable point.
- **G1's floating reference explains why G1 beats H1.** No — G1's contact terms
  are active and it pays *more* ground-penetration than H1.
- **Heading degrades 6x during training.** No — an episode-length artefact;
  drift rate improves 77-83%.
- **Leg error is concentrated in ankles** (raw) / **is not** (normalised) — both
  true, different questions; stated explicitly rather than resolved by choosing.
- **`ctrlrange` explains the dead descriptor channels.** No — the vector never
  reads `ctrlrange`.
- **Per-frame crosseval RMSE is immune to the episode-length confound.** No —
  samples are bounded by episode length, so a longer-surviving arm samples deeper
  into episodes.

## 7.8 Traps found tonight

- **`alive_fraction` is not a survival metric.** It reads 0.9998 for a policy
  whose training episode length is 53.7 steps, because the environment
  auto-resets and only the terminal frame is masked. It is in every crosseval JSON.
- **The centred per-joint metric hides constant postural offsets** — up to 29
  degrees at H1's ankle. Report the absolute metric and the implied bias too.
- **One crosseval episode covers 0.39-1.68% of the clip.** "The policy learns
  the dance" is currently a claim about 1-4 second fragments.
- **`eval_loss` must never rank decoders** — it averages 11 lookahead rows x
  {qpos, qvel}; the gate is 1 row x qpos, ~1/22 of it, and the two rank Kevin's
  decoders in *opposite* directions.
- **The `b_scale` motion-count curve holds out a DIFFERENT motion at each n** and
  cannot be read as a data-scaling result.
- `reconstruct` crashes on any robot with unactuated joints in its clip
  (`(9023,30)` into `(9023,44)` on Talos) — a real bug in `cmd_reconstruct`.
- `$VAR` inside an inline `ssh` command expands on the WSL side. Cost one
  resubmission tonight; the working form is `ssh viper11 'bash -s' < script`, and
  checking that the job NAME carries the value is the cheap detector.
- Heredocs containing markdown backticks break in Git Bash — this section had to
  be written with the Write tool after `cat <<EOF` failed on exactly that.

---

# 8. THE TWO HEADLINE ANSWERS (27→28 August)

**Read 8.4-8.6 first if you read only part of this.** The campaign's stated
purpose was finding defects and it found real ones — the velocity loss, the
span-infeasible BoosterT1 retarget, the flat heading kernel, the saturated H1
ankle, and a crosseval contamination that invalidated eight runs. But **the four
measurement failure-families in 8.4-8.6 caused more corrected conclusions than
every code defect found**, and they are the part that transfers to work that is
not this project. A reader who takes only the defect list away gets the less
useful half.

## 8.1 Does the baseline work? YES — 2.3-2.6x better than doing nothing

Executed-vs-clip crosseval against the zero-action floor, the only comparison
this project accepts as a verdict:

| condition | UnitreeG1 | UnitreeH1 |
|---|---|---|
| zero-action floor (n=2) | 0.3885 ± 0.0063 | 0.4185 ± 0.0003 |
| **trained policy (n=4)** | **0.1494 ± 0.0034** | **0.1810 ± 0.0043** |
| **improvement** | **2.60x** | **2.31x** |

Distributions do not overlap by a wide margin. Measured on `r8_control`
(H1+G1, dance2_subject4, 29.5M steps). The 3-topology version — H1+G1+BoosterT1,
`local_3t_dance4` — **trained to completion for the first time on this project**
(19,660,800 steps, 10 snapshots, 1600 logged `nr_env_steps` blocks, verified by
artifact rather than exit code); its crosseval sweep was still running at
write-up.

**A defect caught before this number was read.** `r8_control` first pooled at
sd 0.1316 against its sibling's 0.0023 — a 57x discrepancy. Cause: the
zero-action floor runs carry `EXP=r8_control` with a `zero_*` tag, so grouping by
arm name pooled the FLOOR into the POLICY. The verdict above is post-fix.

## 8.2 Where does FSQ pay? The interface is FREE, and the loss was the lever

**The token interface costs nothing measurable.** Four seeds per arm, every arm
scored against the same true clip:

| arm | reference quality | G1 | H1 |
|---|---|---|---|
| `fsqrl_ref` | true clip | 0.1503 ± 0.0048 | 0.1764 ± 0.0052 |
| `fsqrl_token` | qpos 0.0293 | 0.1487 ± 0.0056 | 0.1808 ± 0.0079 |
| `fsqrl_tok0` | qpos 0.0090 | 0.1477 ± 0.0022 | 0.1771 ± 0.0068 |

All within noise (|t| < 1.0), **including on G1 where the manipulation is
1.6-5.6x clear of the reference-pipeline floor** — so this is not a measurement
artefact of the H1 pipeline.

**The positive finding inside that null: LINEAR error propagation is ruled out.**
If executed error inherited reference error linearly, `fsqrl_token` would show
t = 11.3-11.7. Observed: -0.43 and +0.93. **The policy ABSORBS reference error
rather than inheriting it.** Quadrature propagation remains fully consistent
(predicted t = 0.93 for H1, observed 0.93) and is NOT excluded.

**Precisely what is supportable:** *reference error does not propagate linearly,
and any quadrature-scale effect is below a 4-seed noise floor.* NOT
"reconstruction quality does not matter" — that needs ~15 seeds. The smallest
detectable reference error at n=4 is 0.0383 rad (G1) / 0.0431 (H1).

> **QUALIFIED BY §10 -- THIS NULL IS SPECIFIC TO M=1.** Every arm in this table
> runs on `dance2_subject4`, a SINGLE motion. On the nine-motion `superM9` clip
> the same interface costs **+9.7% G1 (t=4.46) and +20.6% H1 (t=10.18)**, and is
> worse than a *degraded* reference. **Never quote “the token interface is free”
> without stating the motion count it was measured at.**

**RESOLVED — THERE IS NO THRESHOLD. REFERENCE ERROR ADDS IN QUADRATURE.**
The degraded-reference controls (`fsqrl_degraded` 0.035, `fsqrl_deg2` 0.049)
were first crossevaled while still TRAINING and had to be re-run at convergence.
Re-run, the law is exact:

    executed_error = sqrt(policy_baseline^2 + reference_error^2)

| arm | recon | G1 predicted | G1 observed | H1 predicted | H1 observed |
|---|---|---|---|---|---|
| tok0 | 0.009 | 0.1533 | 0.1477 | 0.1745 | 0.1771 |
| tok05 | 0.019 | 0.1540 | 0.1532 | 0.1753 | 0.1768 |
| token | 0.029 | 0.1557 | 0.1487 | 0.1767 | 0.1808 |
| degraded | 0.035 | 0.1569 | 0.1531 | 0.1778 | 0.1772 |
| deg2 | 0.049 | 0.1599 | **0.1593** | 0.1810 | **0.1803** |

**Mean |residual| G1 0.0036 / H1 0.0019 — both inside the seed noise.**
Linear propagation is decisively excluded (it predicts `deg2` at 0.1991/0.2232).

**BUT THE LAW IS CONSISTENT WITH THE DATA, NOT ESTABLISHED BY IT — and this
qualification is essential.** The fitting regime has almost no discriminating
power. Quadrature's predicted rise against each arm's own sem:

| arm | ref | predicted rise | sigma (G1) | sigma (H1) |
|---|---|---|---|---|
| tok0 | 0.009 | 0.0002 | 0.06 | 0.07 |
| tok05 | 0.019 | 0.0010 | 0.26 | 0.32 |
| token | 0.029 | 0.0025 | 0.71 | 0.77 |
| degraded | 0.035 | 0.0037 | 1.06 | 1.11 |
| **deg2** | 0.049 | 0.0068 | **1.87** | **2.11** |

**Nine of ten arm x robot cells have no power.** "The law fits five arms on both
robots" is really "one cell has power and nine are consistent with anything", and
low residuals against a model predicting almost nothing are not evidence for it.

Compared against the NULL model (`executed = baseline`, reference error costs
nothing):

| robot | mean abs residual, quadrature | mean abs residual, NULL | better |
|---|---|---|---|
| G1 | 0.00356 | **0.00322** | **NULL** |
| H1 | **0.00189** | 0.00414 | **quadrature** |

**The robots appear to disagree — but that is an artefact of the test.** Two
distribution-free tests, answering different questions:

| robot | SIGN test (level shift) | RANK test (trend) |
|---|---|---|
| H1 | 5/5 above baseline, p = **0.031** | rho = +0.600, p = 0.175 |
| G1 | 2/5 above baseline, p = 0.81 | rho = **+0.700**, p = 0.117 |
| | robots contradict | **both positive; Fisher combined p = 0.100** |

**The sign test is not offset-free.** If a baseline estimate is slightly high,
every residual shifts down and the sign test collapses even with the trend
intact. G1's residuals are exactly that shape — scattered around zero but
INCREASING with reference error (-0.0054, +0.0001, -0.0044, +0.0000, +0.0062) —
and the apparent offset (~0.003) is about one sem (0.0036). **So G1, the robot
that fails the sign test, has the HIGHER rank correlation.** The robot
disagreement was a property of the test, not of the robots.

Reporting both is right, because they detect different things and their
disagreement was the informative part: **H1 shows both a level shift and a trend;
G1 shows only the trend.** Where there is no offset the sign test is the more
powerful of the two, which is why H1's sign p (0.031) beats its own rank p
(0.175).

**Honest status: a positive relationship is consistently indicated on both
robots at combined p ~ 0.10, the magnitude is untested everywhere, and no single
test reaches significance.** Weak but consistent — which it did not appear to be
before the offset-free test was run.

**Why the relative spec is still the right thing to hand over.** Its risk is
ASYMMETRIC: if the truth is the null, the spec is merely conservative. It would
only be dangerous under SUPER-quadrature, and linear — the natural
super-quadrature alternative — is decisively excluded. Quadrature is also far
more plausible a priori than the null, which is physically absurd at large
reference error: a reference a radian from the truth cannot cost nothing. **The
data are simply too weak to demonstrate it.**

**`fsqrl_deg1` (1-epoch tokenizer, ref ~0.10-0.15) is queued and is the first
test with real power** — quadrature predicts +19 % to +40 % there, 10-20 sigma
rather than 2, and it separates quadrature from the null by a margin the entire
existing dataset fails to reach.

**THE SPEC IS RELATIVE, NOT ABSOLUTE — and this inverts the usual reading.**
The law says the reference contribution is INDEPENDENT of the policy's own
error. So **the same tokenizer costs MORE on a better policy**, and this stack
expects the policy to keep improving (RMSE is still falling steeply at 19.66M
steps, alpha 0.23-0.29):

| policy baseline | ref 0.009 | ref 0.019 | ref 0.029 | ref 0.049 |
|---|---|---|---|---|
| **0.153 (today)** | 0.2 % | 0.8 % | 1.8 % | 5.0 % |
| 0.080 | 0.6 % | 2.8 % | 6.4 % | 17.3 % |
| 0.050 | 1.6 % | 7.0 % | **15.6 %** | 40.0 % |
| 0.030 | 4.4 % | 18.4 % | 39.1 % | 91.5 % |

**Today's "free" tokenizer is not free on a better policy.** The 0.029 reference
costing 1.8 % now would cost 15.6 % against a 0.05 policy. **So an absolute rad
threshold is the wrong FORM of spec** — it silently assumes the policy stays as
bad as it is today. That is the same class as substituting an aggregate for a
member: a number correct at one operating point standing in for a relationship.

**The spec in the right form**, to keep the cost below X:

    reference_error <= policy_baseline * sqrt((1+X)^2 - 1)

| target cost | ref must be | today (base 0.153) | at a 0.05 policy |
|---|---|---|---|
| < 1 % | 0.142 x base | 0.0217 | 0.0071 |
| < 2 % | 0.201 x base | 0.0308 | 0.0100 |
| < 5 % | 0.320 x base | 0.0490 | 0.0160 |

**Rule of thumb: keep the tokenizer's reconstruction error under about a fifth
of the policy's own tracking error and it costs under 2 %.** That covers ranges
never measured, moves with the policy, and needs no threshold.

**It also restates the QVEL_W result more strongly.** At 0.0090 the per-joint
tokenizer sits at **0.059x** today's baseline and still **0.18x** against a
policy three times better. So the loss fix is what keeps the tokenizer inside
spec for policies well beyond today's — a better claim than "3x inside the free
region", and one that does not depend on a threshold that no longer exists.

**So the spec is a formula, not a threshold**, and it is more useful than one.
At the G1 baseline of 0.153, reference error costs +0.17 % at 0.009 (our best
tokenizer), +1.8 % at 0.029, +5.0 % at 0.049, +11 % at 0.075 and +40 % at 0.153.
**The tokenizer does not need to hit any particular number — it needs to stay
well below the POLICY's own tracking error, and the cost is computable at any
quality.** Reference error becomes first-order only when it approaches ~0.15 rad,
three times worse than anything we produce by accident.

**Two earlier readings are withdrawn**, both traceable to the contaminated arms:
a "~0.03 rad knee" (from an n=4 arm that did not survive n=8) and a
"steeper than quadrature above the knee" (the excess was undertraining). With the
contamination removed there is no knee and nothing is steeper than quadrature.

**Superseded status note, kept because it is how the result was reached:**

**STATUS OF THE DEGRADED-REFERENCE CONTROLS (superseded above).** Two further arms were run to decide whether the null above means
"quality does not matter in this range" or "this comparison cannot detect
anything": `fsqrl_degraded` (recon 0.0353) and `fsqrl_deg2` (0.0489). **Their
first crossevals are INVALID.** `crosseval2.sbatch` resolves `latest.model` at
job start, and both arms were still TRAINING when those crossevals were
submitted — `degraded` by 13 minutes, `deg2` by 23 — so they scored
partially-trained policies. `deg2` is invalid in all eight seeds.

**Consequently: "the comparison is sensitive, therefore the flat region is a real
null" is UNPROVEN.** Do not quote a knee, a threshold, or a sensitivity claim
from this report until the `fin_s*` re-runs at convergence land. An earlier
"~0.03 rad" figure came from an n=4 arm that did not survive n=8, and a
subsequent "robot-dependent" reading rested on that same arm; both are withdrawn.

**What does NOT depend on any of that**, and is therefore what to use:
- The three arms in the table above (`ref`, `token`, `tok0`) plus `tok05` were
  ALL crossevaled after their training completed. They are clean.
- **Below ~0.028 rad the token reference is free on both robots.** A lower bound,
  resting only on clean arms, and it has survived every revision precisely
  because it is a bound rather than a point.
- The linear-propagation kill above is unaffected.
- Updated `ref` at n=8: G1 0.1531 ± 0.0051, H1 0.1743 ± 0.0045 — within noise of
  the n=4 values quoted in the table, which is itself a useful check that the
  4-seed estimate was honest.

**CAVEAT ON THE QVEL_W MAGNITUDE — the fits were cut off mid-descent.**
`history.json` for all three 400-epoch per-joint tokenizers shows eval loss
reaching its minimum at epoch **396-399 of 399** and still falling **9-14 %** over
the final quarter. None had converged. The cause is a schedule defect:
`create_train_state` derives `steps_per_epoch` from a SINGLE BATCH handed to it
for shape inference, so `decay_steps == epochs` (400) while the loop runs
**131,200** updates — **99.7 % of every fit ran at the cosine floor, 7.5e-5
instead of the requested 1.5e-3**, a 10.5x loss of learning-rate integral.
**So the -35.5 %/-39.5 % figure below is a comparison between two runs both cut
off mid-descent.** Direction is very unlikely to flip; the magnitude is not a
converged measurement and is being re-measured with the fix. It also means the
reconstruction floor of ~0.008 is an artefact of the schedule rather than of the
data or the model.

**Where FSQ actually pays: the LOSS, not the architecture.** The tokenizer loss
was ~94 % velocity while every gate and the RL reference are position. Fixing it
is worth **-35.5 %/-39.5 %** position at `QVEL_W=0.05`, the single largest lever
found in this project. **Ship 0.05, not 0.0** — the last step to zero buys 2.1x
more position for 21.8x worse velocity, and the RL reference consumes velocity.

**And the token IS consulted.** Functional ablation on the first trained
token-bearing urma2 policies: zeroing the token moves the action **9.1-17.8 %**
against a `keep_nominal` control at 1.5-2.7 % and a position channel at 3.8-6.1 %
— measured INSIDE the same network at the same training stage, so the within-arm
controls carry it. Per channel the reference still dominates by 12-23x.

**One reading is WITHDRAWN pending a matched re-run:** that removing the explicit
reference *doubles* the token's influence (17.8 % vs 9.1 %). That is a CROSS-ARM
comparison, and the two arms differ in training stage as well as design —
`latent` at 2,359,296 steps against `latonly` at 786,432, a 3x gap. If token
reliance falls as a policy trains (which is what "the network learns to route
around a channel" looks like), the less-trained arm shows higher influence with
no design difference at all. **`latonly` is both the less-trained arm and the
higher-influence one, so the confound points the same way as the claimed effect.**
The fix costs nothing: redo the ablation on matched converged checkpoints.

**The general tell, which cost three findings tonight:** in every case the
unintended second variable pushed the SAME direction as the claimed effect —
two crosseval batches with different weights, a tokenizer-degradation lever that
was the same variable as the contamination, and this. **The question to ask is
not "is there a confound" but "does the obvious confound happen to agree with my
result", because that is exactly the case where nothing in the data flags it.**

## 8.3 The legs-vs-arms split: REAL at matched amplitude, and ~2x not ~2.8x

**THE CLEANEST FORM THIS RESULT HAS EVER TAKEN.** Every earlier figure divided
per-joint RMSE by the joint's reference amplitude, which assumes error scales as
amplitude^1 and — because the dance clip's chains do not overlap in amplitude at
all (H1 legs 0.154-0.429, arms 0.447-0.724) — extrapolated across a gap.
**`walk1_subject1` has a genuine overlap band.** Verified directly: legs
0.121-0.362 against arms 0.307-0.620, giving a band at **0.307-0.362 containing 4
legs and 4 arms whose amplitudes interleave.**

Inside that band, H1, raw radians, no normalisation:

| joint | amplitude | RMSE | chain |
|---|---|---|---|
| l_arm_shx | 0.307 | **0.1169** | ARM |
| knee_angle_r | 0.312 | 0.2738 | LEG |
| knee_angle_l | 0.331 | 0.2919 | LEG |
| left_elbow | 0.340 | **0.1392** | ARM |
| r_arm_shz | 0.345 | **0.1305** | ARM |
| hip_flexion_r | 0.349 | 0.1832 | LEG |
| l_arm_shy | 0.356 | **0.0853** | ARM |
| hip_flexion_l | 0.362 | 0.2066 | LEG |

**Every leg above every arm. Zero RMSE overlap despite complete amplitude
overlap.** In-band means: legs 0.2389 vs arms 0.1180 = **2.02x**. G1's band is
narrower (0.294-0.306, 2v2) and also separates perfectly at 1.86x. Exact
permutation p = 1/C(8,4) = **0.0143** on H1; Fisher-combined with G1, **p = 0.017**.

**No normalisation, no assumed exponent, no extrapolation — on a clip whose
amplitude ordering nobody chose for this purpose. The split is real and is not a
normalisation artefact.**

**BUT IT IS SMALLER THAN PREVIOUSLY REPORTED:**

| method | ratio |
|---|---|
| matched-amplitude raw (this test) | **1.86-2.02x** |
| regression-controlled, two independent arm sets | 2.34x, 2.44x |
| amplitude-normalised (earlier headline) | 2.82x |

**The normalised figure is the outlier**, and the three routes that do NOT divide
by amplitude^1 agree at 1.86-2.44x. The reason: the arm chain's error is only
weakly amplitude-dependent (within-arm exponent **+0.25**, not the ~1 that
normalising assumes), so dividing by amplitude over-corrects and flatters the
arms. **Report ~2x.**

**Caveat that replaces the old one.** Only knee and hip_flexion fall inside the
band. Hip rotation, hip adduction and the ANKLES are all outside it on this clip
— and the ankles are exactly where a different mechanism was found (saturation at
103 % of limit). **The clean test does not cover the ankle.** A clip placing those
joints in an overlap band would extend it.

## 8.3b The ankle sub-effect (unchanged)

**The split is NOT a transient.** Amplitude-normalised leg/arm ratio at 29.5M
steps: **H1 2.75x, G1 2.80x** (mid-training 2.63x/2.84x).

**R8 — relaxing the nine leg-only reward terms 10x — does NOT close it:**

| robot | chain | control | legfree | t |
|---|---|---|---|---|
| H1 | **ankle** | 1.525 | **1.236** | **-18.12** |
| G1 | ankle | 1.154 | 1.094 | -2.02 |
| H1 | hip+knee | 0.634 | 0.646 | +0.83 |
| G1 | hip+knee | 0.622 | 0.606 | -1.33 |
| H1 | arm | 0.230 | 0.230 | 0.00 |
| G1 | **trunk** | 0.732 | **0.874** | **+6.54 (worse)** |

leg/arm ratio 2.75 -> 2.81 (H1), 2.80 -> 2.82 (G1): unchanged.

So the constraint-asymmetry hypothesis **fails at leg scope** — hips and knees do
not move — but its **mechanism is confirmed at ONE JOINT**: H1's ankle improves
18.9 % at t = -18.1 while every other chain is null. The ankle is the joint that
holds the CoM over the foot, hence the most contended between the tracking target
and the contact terms. It is not free: G1's trunk degrades 19.5 %.

**SCOPE CORRECTION — the ankle effect is an H1 RESULT, not a general one.**
The raw (un-normalised) per-chain decomposition, which reconciles with the
aggregate to 0.03-0.06 percentage points, shows the two robots differ in KIND:

| robot | ankle | hip+knee | arm | trunk | aggregate |
|---|---|---|---|---|---|
| H1 | **-18.9 %** | +1.8 % | **-0.4 %** | +1.8 % | -3.58 % |
| G1 | -5.1 % | -2.7 % | **-3.6 %** | +19.5 % | -3.22 % |

**H1 is decisive**: the entire aggregate is the ankle, everything else flat within
1.8 %. **G1 is not.** Its arms improve -3.6 %, as large as its whole aggregate —
and arms carry NONE of the nine leg-only constraints, so that cannot be a direct
effect. It is either a systemic benefit (a less-constrained body tracks
everything slightly better) or noise at the ~3 % detection floor. Either way
**the internal control that licensed the ankle claim FAILS on G1**: the claim
rested on arms and hips/knees being null in the same measurement, and on G1 the
arms are not null. G1's per-chain t-values are marginal throughout (ankle -2.02,
hip+knee -1.33).

So the supportable statement is: **one robot supports the localised ankle
mechanism decisively; the other is consistent with a weak systemic effect.** That
is materially weaker than the general claim and is stated this way so a reader
does not generalise from H1.

**Method note.** The amplitude-normalised view gave the same ORDERING on both
robots, so it looked like one phenomenon at two magnitudes. The RAW view shows
H1 localised and G1 diffuse — a difference in KIND. Report both: they answer
different questions, and here the raw one carried the finding.

That unifies three previously separate results: the ankles are the only chain
scoring worse than a mean-predictor (RMSE/std 1.14-1.41); they carry a large
common-mode postural offset (7-16 degrees, same physical direction on all three
robots once the mirrored ankle axes are accounted for); and they are the only
chain that responds to relaxing the contact constraints.

**THE ANKLE IS SATURATED, AND THAT IS THE MECHANISM.** Commanded torque from
the postural offset alone (`kp * bias`):

| robot | joint | limit | bias | torque | % of limit |
|---|---|---|---|---|---|
| **H1** | ankle_angle_l | 40 | 0.6897 | **41.4** | **103 % SATURATED** |
| H1 | ankle_angle_r | 40 | 0.6275 | 37.6 | 94 % |
| T1 | Left_Ankle_Pitch | 20 | 0.7040 | 17.6 | 88 % |
| G1 | left_ankle_pitch | 50 | 0.3138 | 14.1 | **28 %** |

H1's ankle commands more torque than its actuator can deliver, from the postural
offset alone, and sits half a joint-range from the reference. The offset is not a
choice — the policy is pinned at its force limit and the offset is where the
force balance settles. It also **trains itself into saturation**: 46 % of limit
at 1.97M steps, 103 % at 19.66M.

**This resolves the H1-vs-G1 scope question.** H1's R8 response is sharply
localised to the ankle and G1's is diffuse, because H1's ankle is pinned at
103 % and G1's is at 28 %. A saturated joint is exactly the one that responds
when competing demand is relaxed; an unsaturated one is not.

**And it changes the recommendation.** An ankle at its actuator limit **cannot be
fixed by reward tuning.** If this result is acted on, the lever is the force
limit or gear ratio, not the constraint weights — and the -18.9 % is better read
as *relaxing the competing demand let the ankle come back inside its limit* than
as a reward-balance win.

**The hips-and-knees half of the split remains unexplained** — now having
survived convergence, with the ankle sub-effect localised and quantified beneath
it. **Six mechanisms have been eliminated, each with the measurement that killed
it:**

| eliminated | by |
|---|---|
| network capacity | one shared decoder, no per-chain parameters |
| descriptor degeneracy | legs separate BETTER than arms in descriptor space |
| actuator saturation | legs at 4-6 % of limit; the chain that DOES saturate is the arms, which passes |
| reference reachability | H1's legs are 100 % within joint limits on every frame |
| the reward's leg-only constraints | relaxing nine terms 10x moves hips/knees by \|t\| < 1.4 |
| closed kinematic chain / ground contact | the arm-normalised ratio moves the WRONG way on both robots |

**Nobody currently has a live mechanism for it.** That is the honest state, and
six characterised eliminations are a more useful handoff than a seventh guess.
Two of those eliminations came from hypotheses proposed and killed the same
night — one of them within an hour, at zero compute, by the control its own
author had specified in advance. The pattern worth carrying is that a mechanism
which fits every known fact is cheap to invent and expensive to trust: both
`F13` (leg-only constraints) and the closed-chain story fitted everything known
at the time, and both were wrong — the first at the wrong SCOPE, the second in
the wrong DIRECTION.

## 8.4 THE METHODOLOGICAL RESULT — six survival-confounded measurements, one mechanism

The most transferable finding of the night is not any single number. **Six
measurements looked like properties of the system and were functions of how the
sample was taken**, on three different axes:

| # | quantity | axis | artefact |
|---|---|---|---|
| 1 | `root_heading_error` | visited states | per-episode mean; r=+0.90..0.94 with episode length. An apparent 6x heading degradation was survival; the drift RATE improves 77-83 %. |
| 2 | per-frame crosseval RMSE | visited states | samples bounded by survival; 5.2x artefact in a synthetic control at IDENTICAL tracking quality. |
| 3 | `alive_fraction` | visited states | 0.9998 for a policy surviving 53.7 steps — ~1.0 by construction under auto-reset. Retired. |
| 4 | shape reference floor | visited states | scored a PERFECT tracker at 0.0218 rad; the manufactured term is the gap between full-clip mean and visited-phase mean. |
| 5 | the "G1 survives 6.1x longer" gap | **training time** | a mid-training snapshot read as structure. At convergence G1/H1 = **1.03**. Five careful structural explanations were refuted against a transient. |
| 6 | "the sweep is running" | **rate vs state** | a job progressing and a job that will finish are different properties; only the second matters against a deadline. |

**Standing rule: any quantity that averages over VISITED states, or is sampled at
ONE point in training, is confounded until proven otherwise.** One crosseval
episode covers 0.39-1.68 % of the clip, so "visited" and "all" are very different
sets.

**The sharpest lesson is #5.** Five structural explanations were refuted with
care and none was right, because *the hypothesis space was structural and the
answer was dynamical*. Rigour inside a wrong hypothesis space does not converge,
and nothing about the care taken signalled the problem. Only running the baseline
to completion broke it.

## 8.5 THE SECOND FAMILY — substituting an AVERAGE for a MEMBER

Four times this night a per-item value was replaced by the aggregate of the set
it belonged to, and four times the conclusion was wrong. **Two instances by each
agent**, which is what makes it a family rather than one person's habit:

| # | substitution | consequence |
|---|---|---|
| 1 | aggregate centring term applied uniformly per joint | **INVERTED** which chain looked worst — the term is amplitude-proportional with a 12.5x spread, and the chain predicted "vulnerable" was the LEAST affected |
| 2 | canonical relative noise floor (2.13 % at mean 0.29) applied at mean 0.009 | understated noise 4.3x; caused a REAL result to be withdrawn, then un-withdrawn |
| 3 | chain-average RMSE from a DIFFERENT run used as a proxy | made two consistent metrics look 2x apart; the real numbers agreed to 0.03-0.06 pp |
| 4 | chain-average error (0.20 rad) applied to the ankle | concluded "no saturation" for a joint at **103 % of its actuator limit**; its bias is 3.4x the chain average |

**Why it recurs is a SELECTION EFFECT, not carelessness.** In every case the item
that mattered was an outlier within its own aggregate — ankle bias 3.4x the chain
mean, ankle MSE share 25.7 % against a 16 % proxy, the manufactured term spanning
12.5x, relative noise scaling with the mean. **You notice a joint, chain or arm
BECAUSE it is anomalous, so it is the least likely member of its set to be
represented by that set's average.** The aggregate is least reliable precisely
for the thing you are writing about.

**And the tell is identical every time: the per-item value already existed and
was not fetched.** `per_joint_rmse_rad`, `per_joint_bias_rad`, per-seed values,
per-joint force limits — each closed in one step. Cost of substituting: two
published findings wrong in DIRECTION, plus a withdrawal and an un-withdrawal.
Cost of fetching: minutes.

**The two families are duals.** The survival family (8.4) is about averaging over
the wrong SET — visited states, one point in training time, a process that is
alive rather than arriving. This one is about substituting an average for a
MEMBER. One collapses a distribution you should have CONDITIONED on; the other
collapses a distribution you should have INDEXED INTO.

**Between them they account for more retracted or corrected conclusions than
every genuine code defect found this night.** The campaign's stated purpose was
finding defects; its most transferable output turned out to be the failure modes
of the measurement.

**And a companion trap: a power calculation is a PRECONDITION of a decision rule,
not a companion to it.** A rule says "outcome X means conclusion Y" and silently
assumes the experiment can produce X. Twice this night a rule was registered
whose own precondition was unchecked — once where a null was the *predicted*
result for a perturbation below the detection threshold. It is cheap arithmetic
and it would have caught both cases before they ran.

## 8.6 THE THIRD AND FOURTH FAMILIES — uncertainty, and the design itself

**THIRD: reporting a POINT where an INTERVAL was warranted.** The tokenizer
threshold was published four times in four different states — "~0.03 rad" (n=4),
withdrawn, "robot-dependent, G1 1.4x stricter" (one marginal arm), and finally no
threshold at all. **Every pass used all the data available and was locally
correct.** The n=4 t genuinely WAS 2.81; the n=8 t genuinely IS 0.90. Both
measurements are right; neither was a verdict.

*The diagnostic form, usable in advance rather than in hindsight:* **which
statement would have survived all the passes unchanged?** Here it was "the knee
is somewhere in [0.029, 0.049]" — and in the event the only claim that DID
survive every revision was the lower bound, precisely because it was a bound.
**Prefer the bound that survives to the point that impresses. A claim you have to
withdraw was usually a point estimate wearing a verdict's clothes.**

**FOURTH: continuing to ANALYSE a dataset that cannot answer the question.**
Five arms were subjected to a residual comparison, a sign test and a rank
correlation across four exchanges, extracting a hint at combined p ~ 0.10 — while
the arm that settles it at 10-20 sigma sat unqueued.

**And the indictment is sharper than "we lacked a power calculation": we
computed it three times.** Two power analyses were run, and one of them stated
outright that nine of ten cells could not discriminate. **We had the number
saying the dataset could not answer the question, published it as a caveat, and
kept analysing.**

*The rule is therefore not "do a power calculation" — we did.* **A power
calculation is a DECISION, not a disclaimer. When it says sigma < 1.2, the next
action is a different experiment, not a better statistic.** Every test run after
that point was, in expectation, worth less than an arm nobody had queued.

*In fairness to the sequence:* it was not worthless. The rank test caught a real
error — an apparent robot disagreement that was a property of the statistic
rather than the robots — which would otherwise have reached this report. **The
failure is not that the analyses were wrong; it is that their expected value was
always below an experiment nobody had run, and the diagnostic saying so was in
the log the whole time.**

## 8.6b THE CONFOUND 2x2 — a procedure, not a warning

A confound's danger depends on its SIGN relative to the claim, and on whether the
claim is positive or null. The four cells behave differently:

| | confound OPPOSES the claim | confound AGREES with the claim |
|---|---|---|
| **positive finding** | **LOWER BOUND — safe to ship** | **UNBOUNDED — unsafe** |
| **null finding** | **UNINFORMATIVE — unsafe** | UPPER BOUND — safe-ish |

**The procedure:** for every claimed effect, list the variables that differ
between the compared conditions and write down the SIGN each would push. You do
not need to eliminate them all — you need to know whether any AGREES with you.
One that agrees must be eliminated before shipping; ones that oppose can be left
in and reported as a bound.

**All three of tonight's confound failures were top-right** (two crosseval
batches with different weights; a tokenizer-degradation lever that was the same
variable as the contamination; a cross-arm ablation whose arms differed in
training stage). **A fourth was bottom-left:** "leg actuators do not saturate"
was a NULL measured with a chain-average error 3.4x smaller than the ankle's true
bias — a confound that could only ever HIDE saturation, and the ankle turned out
to be at 103 % of limit. The cell says that measurement was never informative.

**Applied to the ankle result, per robot — and the answer differs by robot, so
the premise had to be checked rather than assumed:**

| robot | control `age_mean` | legfree `age_mean` | gap | ankle effect | status |
|---|---|---|---|---|---|
| **H1** | 488.6 | 487.4 | **-0.2 %** | **-18.9 %** | **confound ABSENT — clean point estimate** |
| G1 | 432.8 | 468.0 | +8.1 % | -5.1 % | confound opposes — **lower bound** |

`age_mean` is the direct measure of sampling depth; `mean_episode_length_steps`
is `total / n_terminations` and with `n_term = 3` on H1 it is far too noisy to
carry this argument. **On H1 the two arms sample from the same depth to within
0.2 %, so the drift confound is not operating and -18.9 % stands as a point
estimate.** On G1 it opposes, so the weaker -5.1 % is a lower bound.

**This is the one place tonight a confound made a finding SAFER rather than
shakier, and it shows the rule is not only a brake.**

## 8.7 THE LADDER

Four families, one error — **discarding a distribution** — applied to:

| applied to | failure |
|---|---|
| **members** | substituting an average for an outlier you noticed *because* it was anomalous |
| **sets** | averaging over visited states, or one point in training time, or a live process instead of an arriving one |
| **uncertainty** | reporting a point where an interval was warranted |
| **the design** | continuing to analyse when the power calculation already said stop |

And the ladder they imply, each rung strictly better than the one below:

> **Prefer the bound to the point.
> Prefer the law to the bound.
> Prefer the powered experiment to the clever test.**

---

# 9. THE 3-TOPOLOGY BASELINE, AND WHAT PER-JOINT ANALYSIS FOUND UNDERNEATH IT

## 9.1 The baseline works on H1 and G1. It does not work on BoosterT1.

`local_3t_dance4`, snapshot `snap_010_19660800` (19.66M steps, H1+G1+BoosterT1,
dance2_subject4) against the zero-action floor. Interleaved seeds, same
checkpoint path on both sides, n=4 matched pairs.

| robot | policy | zero-action floor | aggregate | t | bin-for-bin, matched age (0-4 / 5-9 / 10-19) |
|---|---|---|---|---|---|
| UnitreeH1 | 0.2424 +- 0.0047 | 0.4199 +- 0.0014 | **1.73x** | 72.5 | 1.54 / 1.81 / **2.01x** |
| UnitreeG1 | 0.2027 +- 0.0011 | 0.3901 +- 0.0035 | **1.92x** | 102.5 | 1.71 / 1.98 / 1.88x (holds to bin 80-159) |
| BoosterT1 | 0.6943 +- 0.0247 | 0.7803 +- 0.0010 | 1.12x | 7.0 | **0.94** / 1.12 / 1.16x |

n=4 matched pairs, complete. The n=3 interim reading (1.73 / 1.93 / 1.14) moved
by at most 0.02x, so the verdict is not seed-fragile.

**This is the first time three topologies have trained to completion on this
project**, and it is the campaign's first headline answer: for H1 and G1, yes,
the baseline tracks measurably better than doing nothing.

**BoosterT1 is the exception and it is not a close call.** At matched episode age
0-4 the policy is *worse than zero action* (0.94x). Its 1.14x aggregate is a
mixture of that early deficit with a modest later gain, not a uniform small win.
This is the first *behavioural* consequence of the span-infeasible retarget,
which until now was a purely kinematic finding: you cannot beat a floor at
tracking a target the body cannot reach.

Bins beyond 10-19 are not reported for H1 and T1 (floor n = 128 and 4).

### CAREFUL READING 9.1 AGAINST 8.1 -- and a caveat I got wrong first

**RETRACTED, with the correction below.** An earlier draft of this section claimed
that §8.1's `r8_control` and §9.1's 3T baseline "train different objectives",
listing seven reward coefficients differing by up to 200x. **That was wrong.** It
compared `viper_train.sbatch`'s DEFAULTS against `local_3t.sh`, without checking
what the submissions actually passed. They override the defaults to the local
values.

`r8_control`'s own log resolves **identically to `local_3t.sh` on all nine
terms** -- `joint_tracking_temperature=0.05`, `deepmimic_qvel_temperature=10`,
`foot_slip_coeff=20.0`, `ground_penetration_coeff=1000`,
`deepmimic_foot_height_weight_ratio=1.0`, `foot_z_velocity_coeff=1.0`,
`gait_coeff_mode=floor`, `gait_coeff_value=0.25`, `joint_tracking_coeff=30.0`.
**Same reward objective.**

**The real confound is TRAINING SCALE, not reward:**

| | r8_control (§8.1) | local_3t_dance4 (§9.1) |
|---|---|---|
| topologies | 2 (H1+G1) | 3 (+BoosterT1) |
| steps | **29,491,200** | **19,660,800** (0.67x) |
| envs | 768 | 576 (192/robot) |

So a topology-count reading of "G1 0.1494 -> 0.2027" is confounded by a **1.5x
difference in training steps** -- which is exactly the confound that refuted five
structural explanations elsewhere in this report (the G1/H1 survival gap was a
mid-training transient, 6.11x at 10M collapsing to 1.03x at convergence).
Episode length had converged by 19.66M in the 3T run, but tracking RMSE is a
different curve and is not shown to have.

**The eval harness IS commensurable, and that is checked rather than assumed:**
the zero-action floor does not depend on the policy, so it is a pure harness
probe, and the two campaigns agree to 0.3-0.4% (G1 0.3901 vs 0.3885; H1 0.4199
vs 0.4185).

**Method note, which is the transferable part.** Three of us produced three
different answers about this: "seven terms differ" (defaults vs local script),
"two terms differ" (counting assignments across all submission scripts), and
"zero terms differ" (reading the resolved config out of the arm's own log). Only
the third is about the experiment that actually ran. **A config claim about a
specific comparison must come from that arm's resolved log, not from a template's
defaults and not from a census of scripts.**

### Both known confounds oppose this result

That is the strongest position a positive finding can occupy, and it was checked
rather than assumed:

1. **Survival (visited-states family).** The policy is sampled at `age_mean`
   404-455, the floor at 5-25. Deeper sampling means deeper drift means worse
   RMSE, so the aggregate *penalises* the policy. The bin-for-bin column removes
   it directly, and the ratios survive.
2. **F66, the reference-phase wrap.** `TrackingClipCommands` does not override
   `update_command`, so phase wraps via the parent's `jnp.mod` and the reference
   teleports from the clip's last frame to its first mid-episode.
   `tracking_clip_cyclic` does not gate this. Exposure scales with episode
   length, so on a policy-vs-floor pair it is **~70x asymmetric** (<=8.86% for
   the policy against 0.12% for the floor) and inflates the policy's error only.

So 1.73x and 1.93x are **lower bounds under F66 and confirmed at matched age
under the survival confound.**

Severity is mild on this clip: the wrap discontinuity is 0.1295 rad against a
median 1-step motion of 0.07696 and a 99th percentile of 0.14352 -- 1.7x a normal
frame step, and *below* the clip's own 99th percentile of ordinary motion.
Frequency and severity are anti-correlated across clips (superM9: 1.71% of
episodes but a 13.7x jump), because dance clips tend to start and end in similar
poses. The frequent case is the mild one.

## 9.2 The legs-vs-arms split is TWO defects, and that is why seven explanations died

Per-joint policy/floor ratio from the same crossevals (>1 = policy better than
doing nothing), computed in **both** metric variants:

| chain | H1 centred | H1 absolute | T1 centred | T1 absolute |
|---|---|---|---|---|
| arm | 2.63 | 2.78 | 1.46 | 1.55 |
| knee | 1.20 | 1.31 | 1.16 | 1.08 |
| **ankle** | **1.25** | **0.54** | **0.92** | **0.41** |
| **hip** | **0.89** | **0.85** | **0.84** | **0.60** |
| trunk/head | 0.85 | 0.82 | 0.79 | 0.45 |

**The ankle fails on OFFSET.** Its shape is fine -- 1.25 centred on H1, better
than the floor -- but absolute is 0.54, roughly twice as bad as doing nothing.
The cause is a large mirrored common-mode postural bias: `ankle_angle_l`
-0.6923 rad (-39.7 deg) against `ankle_angle_r` +0.6330 (+36.3 deg), with the
same signature on T1 (-0.698 / +0.628). **The default per-joint metric removes
exactly the thing that is broken.**

**The hip fails on SHAPE -- on H1 and T1, but NOT on G1.** 0.89 centred and 0.85
absolute on H1, 0.84 and 0.60 on T1: below the zero-action floor in *either*
variant, with negligible bias (0.005-0.082 rad). Out-of-plane DoFs are worst
there: `hip_adduction` 0.75/0.81 on H1, `Hip_Roll` 0.74/0.69 and `Hip_Yaw`
0.83/0.81 on T1, while sagittal `hip_flexion` sits at 0.92-0.99.

**G1 refutes the generalisation.** Its hips are **1.27 centred / 1.16 absolute** --
comfortably ABOVE the floor in both variants. I checked H1 and T1, saw the
pattern, and wrote "on both topologies"; the third robot, which is also the one
that tracks best overall (1.92x), does not show it. Corrected here rather than
left standing.

| chain | H1 c / abs | T1 c / abs | G1 c / abs |
|---|---|---|---|
| ankle | 1.25 / **0.54** | 0.92 / **0.41** | 1.04 / **0.83** |
| hip | **0.89 / 0.85** | **0.84 / 0.60** | 1.27 / 1.16 |
| arm | 2.63 / 2.78 | 1.46 / 1.55 | 2.95 / 3.09 |

**What survives is the ANKLE defect, on all three**, always with absolute worse
than centred and always bias-driven (G1 mean |bias| 0.19 rad). **What is
robot-specific is the HIP defect.** The legs-vs-arms RATIO is present on all
three (G1 arm/hip = 2.3x); the hips-below-floor failure is H1 and T1 only.

That matters for the reward-conflict account: all three robots train under the
same reward configuration in this run, so a purely reward-side mechanism does not
by itself explain why G1's hips clear the floor and H1's do not.

**Why this dissolves the puzzle.** Seven explanations died against "legs vs arms"
because every one was a single-mechanism story fitted to a two-mechanism average.
Once split, the previously anomalous R8 result stops being anomalous: relaxing
the nine leg-only reward terms 10x moved the H1 ankle 18.9% at t=-18.1 and moved
nothing else, because it acted on the *offset* defect, which is ankle-only, and
could not touch the hip *shape* defect. Two results that appeared to compete are
measuring different things.

**Validity.** Cross-joint comparison within a robot is confound-free: every joint
comes from the same episodes, so survival cancels exactly. The absolute levels
remain subject to the survival and F66 confounds; the pattern does not.

**Not claimed: a sign flip.** It predicts this signature and was the first read,
but RMSE cannot distinguish "actively moving the wrong way" from "not tracking at
all". `per_joint_corr` (executed vs raw clip, Pearson) was added to
`crosseval_motion.py` and a final-checkpoint re-eval queued to carry it, at zero
marginal compute. corr ~ -1 = anti-tracked, a sign/axis defect; ~0 = the
reference carries no influence; >0 = tracked badly, and RMSE is amplitude or
phase. Open until that lands.

## 9.3 The R13 2x2: the two tokenizer fixes MULTIPLY

Held-out qpos RMSE (rad), per-joint design, four cells:

| | LR fix OFF | LR fix ON | fix is worth |
|---|---|---|---|
| **QVEL_W = 1.0** | H1 0.0293 / G1 0.0281 | 0.0282 / 0.0266 | -3.8% / -5.3% |
| **QVEL_W = 0.05** | 0.0189 / 0.0170 | **0.0170 / 0.0150** | **-10.1% / -11.8%** |

Combined: **-42.0% H1 / -46.6% G1** against the unfixed baseline.

**The LR fix is worth 2.2-2.6x more once the loss is reweighted to position.**
That is a genuine interaction, and it is mechanistically what you would expect:
when 94% of the loss is velocity, better optimisation mostly buys better velocity
fitting, which a held-out *position* metric barely registers. Reweight to
position and the same optimisation improvement converts into position gain.

**Methodological dividend.** These four cells were queued only to separate
QVEL_W from the LR fix. They also answer a question nobody designed them for --
whether F64's overfitting asymmetry is architectural or an artefact of comparing
a position-dominated eval against a ~97%-velocity one. A one-at-a-time design
would have measured the LR fix at whichever QVEL_W happened to be default and
mis-stated it by a factor of 2.5, *and* produced no interaction estimate to
notice the error with. The interaction cells are nearly free; occasionally they
answer a second question.

## 9.4 A sixth survival-confounded metric -- and it was the REPLACEMENT

`mean_episode_length_steps` is **not an episode length**. It is
`samples / n_terminations`, so when terminations are rare it exceeds the rollout
horizon:

| arm | samples | n_terminations | field reads | horizon |
|---|---|---|---|---|
| H1 policy | 31986 | 14 | **2285.7** | 1000 |
| G1 policy | 31978 | 22 | 1454.5 | 1000 |
| H1 floor | 29850 | 2150 | 14.9 | 1000 |

An episode cannot be 2285 steps long in a 1000-step rollout.

This field was introduced *this campaign* to replace `alive_fraction`, which was
retired for saturating at ~1.0 under auto-reset. **Both are reparametrisations of
the termination rate and both break when terminations are rare** -- one saturates
at 1.0, the other diverges above the horizon. The same defect in mirror image.
The replacement was chosen for having the right *units* rather than for being a
different measurement, and **units are not independence.**

It cost a real error: a wrap-exposure estimate of 28.03% where the true bound is
8.86%, published inside a correction to a colleague's own sizing. Their
arithmetic was right; the artifact field I checked it against was not.

`age_mean` is bounded by the horizon by construction and is what the bin-for-bin
analysis rests on, so nothing in 9.1 or 9.2 is affected.

**The general form, which is the transferable part:** *an exposure that is
uniform per FRAME is not uniform per ARM when arms differ in episode length.*
Every frame-rate hazard on this stack inherits the survival confound
automatically. That makes F66 a member of the visited-states family rather than
an independent artefact -- same mechanism, new surface -- and it is why the
alive@horizon trap this project recorded months ago kept feeling familiar.

## 9.5 The metric variant is a hidden analysis degree of freedom

`per_joint_rmse_rad` **is** `per_joint_rmse_rad_selfcentred` (verified equal to
four decimals; `_absolute` is the uncentred one). The default per-joint metric on
this stack silently removes postural offset.

Across the two robots, **six joints change which side of the zero-action floor
they land on** depending on the variant. Any per-joint claim must name its
variant, and a limb verdict read off the wrong one inverts.

The tell that exposed it: `|per_joint_bias_rad| / per_joint_rmse_rad` reaching
4.03. RMSE can never be smaller than the mean error on the same samples, so a
ratio above 1 is proof the two quantities live in different spaces. That
arithmetic impossibility is a cheap standing check on any RMSE/bias pair.

---

# 10. THE TWO HEADLINE SENTENCES

**1. Does the baseline work?**

> **Yes on H1 and G1, and no on BoosterT1.** One policy over three topologies,
> trained to completion for the first time on this project, tracks
> **1.73x** (H1, t=72.5) and **1.92x** (G1, t=102.5) better than the zero-action
> floor at n=4 matched pairs -- and both known confounds, survival and the F66
> reference wrap, push *against* that result, so those are lower bounds.
> BoosterT1 reaches only 1.12x and is *worse than doing nothing* at matched
> early episode age (0.94x), which is the first behavioural consequence of its
> span-infeasible retarget rather than a statement about topology count.

**2. Where does FSQ pay?**

> **In the loss, not the architecture -- but the interface stops being free as
> soon as the token has to carry more than one motion.** Reweighting the
> tokenizer loss from ~94% velocity to position (`QVEL_W=0.05`) and fixing the LR
> schedule are worth **-42.0% H1 / -46.6% G1** held-out together, and they
> *multiply*: the LR fix buys -3.8/-5.3% at QVEL_W=1.0 but -10.1/-11.8% once the
> loss is position-weighted. **The token interface is free at M=1** -- on
> dance2_subject4 the token, reference and near-perfect-token arms are
> indistinguishable at |t| < 1.0 -- **and costs 10-21% at M=9**: on the
> nine-motion superM9 clip, token-only is +9.7% G1 (t=4.46) / +20.6% H1
> (t=10.18) against a fresh reference, and is *worse than a degraded reference*
> (+4.4% / +10.0%). **So the tokenizer's reconstruction scales and its CHANNEL
> does not** -- the penalty grows with how much the single stream must carry,
> matching the earlier ordering of ~2% at one topology and ~6% at two.
>
> Meanwhile the RL side carries error of its own, in two separable reward
> defects: an ankle *offset* failure (general across all three robots) and a hip
> *shape* failure (H1 and BoosterT1 only, below the zero-action floor in every
> metric variant).

**The honest caveat on both — CORRECTED 2026-08-28 08:40.** The version of this
paragraph published earlier said the two campaigns' launchers "differ on seven
reward coefficients, two of them by 100-200x". *That was retracted at §7.9*: it
compared `viper_train.sbatch`'s DEFAULTS against `local_3t.sh`, and the
submission scripts override those defaults. `r8_control`'s own resolved log
matches `local_3t.sh` on **all nine** reward terms. The paragraph survived the
retraction here, in the section people read first, for half a day.

The caveat that actually holds is narrower and has three parts:

1. **Training scale.** `r8_control` ran 29,491,200 steps at 768 envs over two
   topologies; `local_3t_dance4` ran 19,660,800 at 576 over three. "The third
   topology cost ~35 % tracking" is confounded by a 1.5x difference in training,
   the same confound that made the G1/H1 survival gap look structural (6.11x at
   10M, 1.03x at convergence).
2. **Two knobs outside the nine that were checked.** `local_3t.sh` trains at
   `morphology_coeff_mode=fixed, value=0.0` and
   `tracking_post_contact_penalties=True`; the Viper dance4 arms do the opposite
   on both. The superM9 FSQ arms match *local*, not Viper. `e10_match` (CH-F)
   is the first Viper arm that removes this.
3. **The eval harness is commensurable and that was checked rather than
   assumed** — the zero-action floors agree to 0.3-0.4 %.

**The general lesson is (2), not (1).** A retraction scoped to the terms someone
happened to check is not a clean bill of health for the rest of the config; the
only sound comparison is a full resolved-config diff between the two arms' own
logs.

**A second correction to headline 1.** The 1.73x / 1.92x is real, and it is
carried by the arms and trunk. Per-joint correlation on that same checkpoint
(§11.1) puts every ankle at ~0.00 on all three robots and every leg joint below
the project's own fraction-of-variance-explained gate of 0.5; only the arms clear
it. Quote the baseline with that qualifier.

**A correction to headline 2.** The "token interface" arms carry no token: they
differ from the reference arm in one flag, `tracking_clip_dir`, pointing at a
*reconstructed* clip. The 10-21 % at M=9 is the cost of a token-RECONSTRUCTED
REFERENCE, not of the token as a channel — see §11.2.

---

# 11. NIGHT OF 28 AUGUST, MORNING SESSION — three unread result blocks, and a metric that inverts one of them

The 27→28 session stopped at 03:58. Three blocks arrived after that and were
still unanalysed at 08:00: the R16 dose (20 of 24 crossevals), the
`sub512`/`sub1024` RL arms, and `ce_corrfinal_s0.json`, written seven minutes
after the board was last rewritten.

## 11.1 Correlation, and what the RMSE metric was hiding

`crosseval_motion.py` gained `per_joint_corr` at 03:15 — Pearson correlation
between executed and raw-clip joint angle. RMSE cannot distinguish *"actively
moving the wrong way"* from *"not moving with the reference at all"*, and the
hips had scored below the zero-action floor in both metric variants, which a sign
flip and a learning failure both predict. Correlation separates them.

The field was in the **local** copy only. The Viper copy was 21 lines shorter, so
the entire R16 block was blind to it; it was pushed at 08:14 and R16 re-evaluated
under tag `c0..c3` (never `s0..s3` — overwriting a condition's artifact destroyed
a control on 24-08).

### (a) No joint is inverted, anywhere

The most negative correlation in any robot is BoosterT1's Left_Shoulder_Pitch at
−0.390, and that joint carries a −3.79 rad bias, i.e. it is the span-infeasible
retarget rather than a sign error. **The sign/axis hypothesis for the hips is
dead, at zero compute.**

### (b) Legs are far below the project's own variance gate — but "zero" is only true of the 3-topology arm

Fraction of variance explained, FVE = corr². This project set its gate at
FVE ≥ 0.5 after the dose-ceiling incident.

| chain | 3-topology local (n=1) | 2-topology Viper `r16_ctl` (n=8) |
|---|---|---|
| arms | 0.81 - 0.94 | **0.944** H1 / **0.943** G1 |
| hips | 0.06 - 0.47 | 0.340 H1 / 0.501 G1 |
| knees | 0.05 - 0.45 | 0.373 H1 / 0.502 G1 |
| **ankles** | **~0.001** | **0.038** H1 / **0.071** G1 |

**Both readings clear nothing but the arms, and they are an order of magnitude
apart at the ankle.** The 3-topology checkpoint's ankles are at corr ≈ 0.01-0.06
— genuinely no information — while the 2-topology arms reach 0.19-0.22. An
earlier draft of this section generalised the 3-topology number to "all three
robots"; that is wrong, and the two campaigns differ on four things at once
(topology count, 576 vs 768 envs, morphology fixed-0.0 vs schedule-0.7,
post-contact True vs False). `e10_match` (CH-F) is the arm that isolates it.

**Do not compare either against the zero-action floor's correlation.** The floor
reads hip FVE 0.265 on H1, which looks like a falling robot tracking the
reference. It is an artefact: `initial_state.type=reference` starts every episode
*on* the reference, the floor's episodes last 6.7 steps, so almost every frame it
contributes is an RSI-matched initial frame. **Correlation inherits the survival
confound that this project already documented for RMSE** — it averages over
visited states, and the shorter the episode the more the initialisation
dominates. This is the seventh member of that family.

### (c) The ankle reference is not the problem

It is the best-reconstructed joint in the robot:

| joint group | `ref_vs_raw` RMSE (H1) |
|---|---|
| ankles | **0.0023 / 0.0025 rad** |
| hips | 0.0043 - 0.0299 |
| arms | 0.0034 - 0.1045 |

The policy is ignoring a near-perfect target. `foot_metrics` says why:

| | reference wants airborne | policy achieves | gap |
|---|---|---|---|
| H1, 3-topology local | 0.1369 | 0.0311 | 4.4x too grounded |
| G1, 3-topology local | 0.2535 | 0.0396 | 6.4x |
| H1, `r16_ctl` | 0.1272 | 0.0176 | **7.2x** |
| G1, `r16_ctl` | 0.2795 | 0.0270 | **10.4x** |

**The feet are glued to the floor.** A foot in sustained contact closes a
kinematic chain; the ankle sits immediately proximal to that contact and cannot
follow a reference requiring the foot to leave the ground. The arms close no
chain and track at FVE 0.94. `ground_penetration_coeff` is 1000 and
`foot_slip_coeff` is 20, and neither has ever been dosed.

## 11.2 R16 read twice: RMSE says "global", correlation says "the ankle"

Dose verified in `r16_d3_s1`'s own resolved log (`rpos 0.4167 / rquat 0.25 /
footh 0.3333` against `1.25 / 0.75 / 1.0`). Pooled over both training seeds.

**In RMSE (n=8), the registered rule reads "legs improve, ratio holds → global effect":**

| | H1 ctl → d3 | t | G1 ctl → d3 | t |
|---|---|---|---|---|
| overall RMSE (absolute) | 0.2156 → 0.2000 **−7.3 %** | −4.71 | 0.1730 → 0.1676 **−3.2 %** | −2.08 |
| legs | 0.2588 → 0.2400 −7.3 % | −4.47 | 0.2068 → 0.2009 −2.9 % | −1.64 |
| arms | 0.1598 → 0.1462 −8.5 % | −3.16 | 0.1305 → 0.1242 −4.8 % | −1.88 |
| **leg/arm ratio (absolute)** | 1.626 → 1.644 **+1.1 %** | +0.38 | 1.589 → 1.621 **+2.1 %** | +0.68 |
| leg/arm ratio (self-centred) | 1.700 → 1.784 +5.0 % | **+2.56** | 1.682 → 1.746 +3.8 % | +1.92 |

Legs and arms improve by the same amount, the ratio does not close, and in the
self-centred metric it significantly *widens*. On this reading the per-chain
conflict account is unsupported.

**In variance explained, the same arms show a monotone gradient by kinematic
distance from the foot contact:**

| chain | H1 ctl → d3 → d10 | change | G1 ctl → d3 → d10 | change |
|---|---|---|---|---|
| **ankle** | 0.0378 → 0.0633 → **0.0763** | **+102 %** | 0.0714 → 0.1491 → **0.1440** | **+102 %** |
| knee | 0.3729 → 0.4429 → 0.4716 | +26 % | 0.5015 → 0.5559 → 0.5433 | +8 % |
| hip | 0.3396 → 0.3311 → 0.3148 | −7 % | 0.5007 → 0.5094 → 0.4656 | −7 % |
| arm | 0.9438 → 0.9522 → 0.9544 | +1.1 % | 0.9432 → 0.9497 → 0.9440 | +0.1 % |

Ankle **+102 % on both robots**, knee +8 to +26 %, hip −7 %, arm +0.1 to +1.1 %.
Identical ordering on both robots. And H1's feet start to come off the floor:
`foot_airborne` 0.0176 → 0.0552 → **0.0949** against a reference wanting 0.127,
i.e. from 7.2x too grounded to 1.34x. (G1's does not move: 0.0270 → 0.0251.)

**Neither reading is wrong. They answer different questions.** RMSE asks how big
the error is and reports a uniform improvement; correlation asks how much of the
reference the joint is following and reports that the gain is concentrated where
the pinned-contact account predicts. *A conclusion about a mechanism should be
read in the metric that measures information, not magnitude* — which is the same
lesson as F57's sign flip and the centred/absolute variant problem, arriving for
the third time.

This raises the prior on CH-A considerably: R16 obtained a 2x ankle gain **while
never touching `ground_penetration_coeff` or `foot_slip_coeff`**, which is what
CH-A doses directly.

**The survival confound was checked, not assumed.** H1's `d3` survives 2.8 % less
and terminates 2.2x more often, the direction that could manufacture the RMSE
improvement. Bin-for-bin it does not: the gain holds at every episode age on both
robots (H1 −3.1 % to −9.3 %, t = −0.96 to −2.96; G1 −4.0 % to −9.1 %, t to −4.22).

**RETRACTED, see Sec 12.3 — one of the two 10x training seeds collapsed to the
zero-action floor; the column below is the surviving seed only.**
Training-side the dose is monotone through 10x — joint tracking error −32/−36 %,
velocity −48/−60 %, for root position +17/+25 % and H1 foot height +26 %, with
root *orientation* and heading improving. **The root/foot block has been
over-weighted by between 3x and 10x for the entire campaign.** Note the
conversion gap: −17 to −26 % in the training metric buys −3 to −7 %
executed-vs-clip; the two are not the same quantity.

## 11.3 The "token interface" arms carry no token

`fsqrl_token` differs from `fsqrl_ref` in exactly one flag:

```
fsqrl_ref    --environment.command.tracking_clip_dir=.../clips/clips_M
fsqrl_token  --environment.command.tracking_clip_dir=.../rescore/tok_pj_ev2_dv2rec
```

`fsqrl_token` trains against a clip *reconstructed* from FSQ codes. No latent
enters that policy's observation. So §10's second headline —

> the token interface is free at M=1 and costs 10-21 % at M=9

— measures **reconstruction fidelity propagating into control**, which the
quadrature law already covers, and says nothing about the token as a channel.
Restated: *a token-reconstructed reference is free at M=1 and costs 10-21 % at
M=9.*

The arms that do carry a token, `fsqrl_latent` and `fsqrl_latonly`, set
`tracking_clip_latent_obs=True` **and** point `tracking_clip_dir` at the
reconstructed clip, so their reward target is itself a reconstruction. Token
effect and reconstruction error are inseparable in them. Every design-B number
this project has quoted carries that defect.

Third occurrence of this shape, after `latent_obs` existing with no launcher
passing it and the reference channel believed absent until F22. **Standing rule:
before quoting an architecture result, diff the two arms' resolved configs and
confirm the flag you believe is the independent variable is the one that
differs.**

### The fix, built and running

`clips/clips_M_rawzq/` pairs the **raw** `superM9.npz` (md5-identical to
`clips_M`) with the **ev2_dv2** `_zq.npz` sidecar (md5-identical to the
reconstruction directory's). Frames verified equal at 46770, latent dim 32, both
robots. The `_zq.npz` already in `clips_M` is from a different, older tokenizer —
md5 differs — and must not be used.

| arm | reward target | observation | isolates |
|---|---|---|---|
| `fsqrl_ref` (n=4, have) | raw | reference | control |
| `fsqrl_token` (n=4, have) | **reconstructed** | reference | reconstruction fidelity |
| **`bl_both`** (CH-C) | raw | reference **+ token** | the token as ADDED information |
| **`bl_only`** (CH-C) | raw | **token only** | the token as a REPLACEMENT channel |

## 11.4 `sub512`/`sub1024`: quadrature is a small-error approximation

superM9, n=4, contamination guard clean (`model_mtime` 13-17 s before job end).

| arm | G1 | vs fresh ref | H1 | vs fresh ref |
|---|---|---|---|---|
| fresh reference | 0.1554 | — | 0.1718 | — |
| `deg1` (hold=1) | 0.1633 | +5.1 % | 0.1885 | +9.7 % |
| token-reconstructed reference | 0.1705 | +9.7 % | 0.2073 | +20.6 % |
| **`sub512`** | **0.1786** | **+14.9 %** | **0.2064** | **+20.1 %** |
| **`sub1024`** | **0.2861** | **+84.1 %** | **0.3231** | **+88.1 %** |

Against `executed² = base² + ref²` on H1:

- `sub512`, ref error 0.1312 → predicts 0.2162, measures 0.2064: **−4.5 %**
- `sub1024`, ref error 0.3507 → predicts 0.3905, measures 0.3231: **−17.3 %**

The law was fitted over reference error 0.009-0.049 and is sub-additive by 17 %
at 0.35 — the policy partially discounts a bad reference. **Quote it with its
range of validity.** Whether that is smooth saturation or a regime change needs
2-3 points between 0.05 and 0.13, not yet queued.

## 11.5 What wave 4 decides

25 training arms and 126 crossevals over six Viper chains, plus a two-arm local
3-topology queue; per-chain rules in board §0c. The three that matter:

1. **CH-A and the local queue — does dosing the contact coefficients restore the
   legs?** Registered rule: `foot_airborne` rises toward `ref_foot_airborne`
   **and** ankle corr clears 0.3. §11.2 raises the prior substantially. A null
   sends the search upstream of contact, to actuator saturation and reference
   feasibility. The 30x arm exists to find the **ceiling** — past some point
   "degrade the contact penalty" silently becomes "remove it", and
   `foot_penetration_m` is the only thing that will say so.
2. **CH-C — can the token carry the motion at all?** `bl_only` near `fsqrl_ref`
   reopens design B at M=9; at or below `fsqrl_token`, one shared stream cannot
   carry nine motions and the next move is architectural, not more scaling.
3. **CH-F — are the local and Viper campaigns comparable?** `e10_match` is the
   first Viper dance4 arm at `local_3t.sh`'s morphology and post-contact
   settings, and §11.1(b) shows the two campaigns' ankles differ by an order of
   magnitude.

One caution to carry in. The morphology hypothesis is **weakened before it runs**
— `local_3t.sh` already trains at morphology 0.0 and its ankles are the ones at
corr ≈ 0 — so E8's value is commensurability, not mechanism, and a null there
says nothing about the ankle.

---

# 12. WAVE 4 RESULTS — 29 arms, 171 jobs, all completed

Submitted 08:20-08:55, drained by 22:00. One FAILED job on the shared account
belonged to another user; every wave-4 job completed. Read with
`analyse_wave4.py` (pooling rules) and `wave4_diagnose.py` (per-member and
age-binned checks).

## 12.1 The contact dose is a NULL on two topologies and a LARGE WIN on three

**CH-A, H1+G1, `foot_slip_coeff` x `ground_penetration_coeff` at 1/3/10/30x,
n=8-12 per cell.** The registered rule was: `foot_airborne` rises toward
`ref_foot_airborne` AND ankle corr clears 0.3.

| H1 | ctl | 3x | 10x | 30x |
|---|---|---|---|---|
| RMSE (absolute) | 0.2074 | 0.2073 | 0.2090 | 0.2147 |
| `foot_airborne` (ref wants 0.127) | 0.0155 | 0.0208 | 0.0114 | 0.0269 |
| ankle FVE | 0.032 | 0.041 | 0.044 | 0.027 |

Bin-for-bin at 10x, every age bin is within ±1.2 % at |t| < 0.8 and `age_mean`
is unchanged (495 → 494). **The rule fails on both clauses.** The feet did not
come off the floor, and tracking did not move. On two topologies, dosing the
contact coefficients does nothing.

**The same dose on THREE topologies, run locally, is one of the largest effects
this project has measured** (n=4, control `local_3t_dance4`):

| robot | ctl | 10x | change | t | age |
|---|---|---|---|---|---|
| UnitreeH1 | 0.3532 | 0.3149 | **−10.9 %** | −17.5 | +1.8 % |
| UnitreeG1 | 0.2511 | 0.2058 | **−18.0 %** | −18.3 | **+13.0 %** |
| BoosterT1 | 1.3575 | 0.8056 | **−40.7 %** | −30.7 | −0.6 % |

G1's ankle FVE goes 0.0055 → **0.0638** (11.6x), BoosterT1's 0.0027 → 0.0220
(8.2x), and BoosterT1's *arm* FVE 0.42 → 0.74. G1 also survives 13 % **longer**,
which is the conservative direction for an RMSE improvement.

## 12.2 The reconciliation: `tracking_post_contact_penalties`, and a double count

The two campaigns differ on the knob flagged in §11 as a commensurability
problem, and it turns out to be the mechanism rather than a nuisance.

**CH-F measured it directly** (`e10_pc`, identical to `e1_ctl` but
`POSTCONTACT=True`), n=8, `age_mean` unchanged:

| | `e1_ctl` | `e10_pc` | change |
|---|---|---|---|
| H1 RMSE | 0.2074 | 0.2431 | **+17.2 %**, bin-for-bin +5.0 to +9.3 %, t to +4.30 |
| H1 ankle RMSE (absolute) | 0.2797 | 0.4550 | **+62.7 %** |
| H1 `foot_penetration_m` | 0.0131 | 0.0052 | **−60 %** |
| H1 `joint_tracking_error` (training) | 0.0682 | 0.0898 | +31.7 % |

So post-contact penalties **buy contact quality and pay for it in tracking, and
the ankle pays most.** G1 is much less affected (+2.3 %).

**And in the code the three terms are counted twice.** `DefaultReward` already
includes `foot_slip_reward + ground_penetration_reward + feet_orientation_reward`
inside `reward_penalty`; `TrackingReward` then adds
`internal_state["tracking_post_contact_penalty"]`, which is *those same three
terms*, after the imitation bonuses:

```python
# default.py:321   the penalty that will be re-applied
internal_state["tracking_post_contact_penalty"] = (
    foot_slip_reward + ground_penetration_reward + feet_orientation_reward)
# default.py:330   ...and the same three are already inside reward_penalty
reward = tracking_reward + reward_penalty + alive_clipped_reward
reward = jnp.maximum(reward, 0.0) + alive_unclipped_reward
# tracking.py:555  applied a SECOND time, past the clip
reward = jnp.maximum(reward - alive_floor + internal_state[
    "tracking_post_contact_penalty"], 0.0) + alive_floor
```

The stated intent is to move these terms *past* the clip so they can "compete
with bonuses that were historically added only after clipping". The
implementation **adds** them rather than **moving** them: they are never removed
from `reward_penalty`. The effective contact penalty under `POSTCONTACT=True` is
therefore strictly larger than under `False` — up to 2x, less wherever the base
clip had already absorbed them (and it often had: `reward/total` under `False`
sits at 0.0011).

**That reconciles §12.1 completely.** `local_3t.sh` runs `POSTCONTACT=True`, so
its contact penalties are up to doubled and it sits far above the optimum —
dosing them down 10x recovers 11-41 %. The Viper dance4 arms run `False`, are
already near the optimum, and the same dose does nothing. One mechanism, two
opposite results, no contradiction.

**Status: strongly supported, one arm short of confirmed.** The confirming
experiment is the 2x2 `POSTCONTACT` x contact-dose on two topologies: with
`POSTCONTACT=True` the dose should recover most of the +17.2 %. Not yet run.

### A logging trap found on the way

`reward/total` is **not comparable across the `POSTCONTACT` setting**: 0.0011
(`False`) vs 1.5022 (`True`), a 1380x difference that is pure bookkeeping.
`DefaultReward` logs `reward/total` *before* `TrackingReward` adds every
imitation bonus, and only the post-contact branch overwrites it with the final
value. **So under `POSTCONTACT=False`, `reward/total` excludes every imitation
term.** Any reward-budget arithmetic taken from `reward/total` on such an arm is
a fraction of a partial total.

## 12.3 The dose ceiling caught two arms, and RMSE alone would have missed both

**`e1_c30` / `l3t_c30` (30x) is removal, not degradation.** Locally:

| | H1 ctl | H1 c30 |
|---|---|---|
| `age_mean` | 449 | **43** |
| `foot_penetration_m` | 0.0033 | 0.0119 |
| RMSE (absolute) | 0.3532 | **0.2826 (−20 %)** |

**RMSE alone would have named c30 the best H1 arm of the wave.** It is a policy
that falls over in 43 steps and is therefore only ever scored on the easy early
frames. `age_mean` and `foot_penetration_m` co-reported are what caught it —
exactly the guard the 30x point was included to trigger.

**`r16_d10_s2` collapsed outright.** Per training seed:

| arm | robot | age | RMSE | arm FVE |
|---|---|---|---|---|
| `r16_d10_s1` | H1 | 450 | 0.1955 | (healthy) |
| **`r16_d10_s2`** | **H1** | **7** | **0.5208** | **0.110** |
| zero-action floor | H1 | 6.7 | 0.5182 | — |

`r16_d10_s2` is indistinguishable from doing nothing. **One of two training
seeds at the 10x root/foot dose died.** §11.2 reported the d10 column from `s1`
alone, before `s2` existed, and described the dose as monotone through 10x. The
corrected statement is: *the 10x dose is past the stability limit — it trains
sometimes.* The pooled `r16_d10` mean (0.3582) is meaningless and must never be
quoted; this is the aggregate-hiding-a-member trap, caught by a per-member table.

## 12.4 E2b: the R16 gain was FOOT HEIGHT, not root tracking

R16 dosed `rpos` + `rquat` + `foot_height` together. Split, n=8-12:

| arm | H1 | vs ctl | G1 | vs ctl |
|---|---|---|---|---|
| `e1_ctl` | 0.2074 | — | 0.1773 | — |
| **`e2_fh3`** (foot-height only) | **0.1944** | **−6.3 %** | **0.1679** | **−5.3 %** |
| `e2_rp3` (root pos+quat only) | 0.2125 | +2.5 % | 0.1738 | −2.0 % |

`e2_fh3` holds bin-for-bin at every age on both robots (H1 −2.4 to −5.2 %, t to
−2.73; G1 −3.4 to −7.0 %, t to −3.23) with `age_mean` unchanged, and it has the
best ankle FVE of any dance4 arm (H1 0.058, G1 0.110) and the best hip FVE
(G1 0.601). **`deepmimic_foot_height_weight_ratio` at 0.333 is the single
recipe change this wave supports shipping.** Root-tracking weight is not the
lever; dosing it alone is null-to-harmful.

## 12.5 Design B is decisively negative, and design A is free

**The first clean design-B arms this project has run** — raw clip as reward
target and reference, token added only to the observation. n=8, both training
seeds healthy, bin-for-bin at every age.

| arm | H1 | vs `fsqrl_ref` | G1 | vs `fsqrl_ref` |
|---|---|---|---|---|
| `fsqrl_ref` (control, n=12) | 0.1972 | — | 0.1613 | — |
| `fsqrl_token` (design A: reconstructed reference) | 0.2047 | +3.8 % (t=1.51, **ns**) | 0.1628 | +0.9 % (t=0.48, **ns**) |
| **`bl_both`** (reference + token) | 0.2307 | **+17.0 %** (t to +7.8/bin) | 0.1741 | **+8.0 %** (t to +6.1/bin) |
| **`bl_only`** (token replaces reference) | 0.2493 | **+26.4 %** (t to +10.3/bin) | 0.1818 | **+12.7 %** (t to +7.7/bin) |

**Adding the token to a perfect reference makes the policy worse, on both robots,
at every episode age.** The earlier H1-only anomaly ("adding information hurt
H1") is now a general result, measured on a design where it cannot be blamed on
reconstruction error.

**And the rate curve refutes my own prediction.** F70 measured the code to be a
near-lossless summary of a ~10-frame window, so I registered that `LATENT_HOLD=10`
should be nearly free:

| arm | H1 | G1 | arm-chain FVE |
|---|---|---|---|
| `bl_only` (hold 1) | 0.2493 | 0.1818 | 0.93 |
| `blr_h10` | 0.2668 (+7.0 %) | 0.2013 (+10.7 %) | 0.87 |
| `blr_h40` | 0.3753 (+50.5 %) | 0.3135 (+72.4 %) | **0.47 — collapsed** |

Holding the token for its own window length costs 7-11 %, not nothing. Holding
it for 40 frames collapses the policy on both seeds.

### The leading explanation is observation WIDTH, and it is testable

`environment.py:1330`:

```python
self.joint_observation_size = (
    4 + int(self.tracking_reference_channel_active)
      + int(self.tracking_reference_velocity_active) + self.tracking_latent_dim)
```

The token takes the per-joint dynamic observation block from **5 channels to 36**
— a 7x widening — to deliver a signal that a *linear* probe shows is R² = 0.94
redundant with the reference channel sitting beside it (F70). That is a
capacity cost paid for duplicated information, and it predicts every sign in the
table: `bl_both` worse than `fsqrl_ref`, `bl_only` worse still (the reference it
replaced was cheaper and sufficient), and design A free (no width change at all).

**This is not yet separated from "the policy cannot use the token".** The
separating experiment is a `latent_dim` ablation — PCA-compress the 32-d code to
4/8/16 and re-run `bl_both`. If the penalty scales with width, it is capacity; if
it is flat, it is the content. **Until that runs, do not conclude that FSQ
conditioning cannot work — conclude that at 32 dims per joint it costs more than
it pays.**

## 12.6 E8 morphology: mostly a survival artefact, as registered

Pooled, `e8_m0` beats `e1_ctl` by −3.8 % H1 / −3.2 % G1. But `age_mean` moved
−3.4 % / −5.0 % (t = −4.90 / −3.35). Conditioned on age, **G1's effect vanishes
and reverses** — only bin 0-4 is significant (−6.9 %, t = −3.20) and bins 20+ run
+0.1 to +2.7 %. H1's is weakly real (−2.5 to −6.3 %, all |t| < 1.9). The null was
registered in advance for the right reason (`local_3t.sh` already trains at
morphology 0.0 and its ankles are the ones at corr ≈ 0), and the arm's value was
always commensurability.

## 12.7 What this wave changed, in order

1. **Ship `FOOTH=0.3333`.** It is the only clean, survival-free, bin-for-bin
   recipe win in the wave (−5 to −6 %) and it is the half of R16 that paid.
2. **Treat `POSTCONTACT=True` as a measured Pareto choice, not a default** — and
   fix the double count so the trade can be made at a deliberate weight. It costs
   H1 17 % tracking and 63 % ankle error to buy 60 % less penetration.
3. **The 3-topology baseline is running at a badly wrong contact weight.**
   `l3t_c10` is −11 to −41 % and is the new 3-topology baseline.
4. **Design B, as built, does not pay.** Design A is free. The width ablation
   decides whether that is a statement about FSQ or about 32 channels per joint.
5. **Retract** "the R16 dose is monotone through 10x" (one seed of two collapsed)
   and **retract** the pooled `r16_d10` numbers.

---

# 13. WAVE 5 RESULTS — 33 arms landed, and the FSQ direction closes

**Submitted 2026-08-28 22:20 CEST, read 2026-08-29 09:00.** Four chains, 37 arms
/ 185 jobs. 33 arms landed with full 4-rollout-seed crossevals; `fhd_10_s3`,
`blw_d16_s3` and `m9f_ref_s3` were still training at read time, and
`m9f_recon_s2`'s evals were in flight, so `m9f_recon` is n=4 (one training seed)
and everything else is n=8 or n=12. **No crashes, no collapsed members** — every
arm's arm-chain FVE is 0.93-0.97 and every `age_mean` is 440-500, which is the
first wave where the per-member guard found nothing to report.

Artifacts pulled to `wave5_artifacts/` (+ `wave4_controls/` for the contrasts);
reproduce with `analyse_wave5.py`.

## 13.1 CH-J — the design-B penalty is RECIPE-INDEPENDENT

The whole FSQ axis re-run at `FOOTH=0.3333`, M=9, against a raw-reference
control. Absolute RMSE, pooled over 2 training seeds, **and it holds in all nine
age bins** with ages within 3 %:

| arm | what it isolates | H1 | G1 | same axis at `FOOTH=1.0` |
|---|---|---|---|---|
| `m9f_recon` | design A — token-RECONSTRUCTED reference | +1.6 % (t=0.58, ns) | +4.4 % (t=3.54) | +3.8 / +0.9, both ns |
| `m9f_both` | design B — raw reference **+** 32-d token | **+10.5 %** (t=4.24) | **+10.5 %** (t=6.96) | +17.0 / +8.0 |
| `m9f_only` | design B — token REPLACES the reference | **+16.6 %** (t=6.41) | **+16.3 %** (t=9.41) | +26.4 / +12.7 |

The mis-weighted baseline was not the explanation. The penalty survived the
recipe fix and merely **symmetrised** across robots (H1 fell 17 -> 10.5, G1 rose
8 -> 10.5). Section 12.5's verdict can now be stated without the `FOOTH` caveat.

Design A is free on H1 and costs a small but real +4.4 % on G1 — call it 0-4 %,
near-free rather than exactly zero.

## 13.2 CH-H — THE DECIDING EXPERIMENT SAYS *CONTENT*, NOT CAPACITY

The rule registered before the arms landed: penalty FALLS with K -> capacity,
compress the token; FLAT or RISES -> content, design B closes. Against the
no-token reference at matched `FOOTH=1.0`:

| K (retained variance) | 4 (65.5 %) | 8 (84.0 %) | 16 (94.7 %) | 32 (100 %) |
|---|---|---|---|---|
| per-joint obs block | 9 ch | 13 ch | 21 ch | 36 ch |
| H1 penalty | +11.9 % (t=8.5) | +16.9 % (t=13.2) | +12.5 % (t=5.4) | +17.0 % (t=9.3) |
| G1 penalty | +2.7 % (t=2.2) | +6.0 % (t=3.9) | +4.1 % (t=2.4) | +8.0 % (t=5.7) |

**Non-monotone, and nowhere near zero.** An 8x narrower code recovers at most
~5 pp of a 17 pp penalty on H1, and K=8 is indistinguishable from K=32. Even at
four channels carrying 65 % of the information the token still costs +11.9 %.
There is a small width component — it is not the mechanism.

So "compress the token" is not the fix, and 12.5's alternative — *the policy
cannot use this token* — is what remains. Two caveats to carry: a PCA projection
of an FSQ code is not quantized, and the G1 `blw` ages moved -5 %
(survival-flattered), so **H1 carries this conclusion.**

## 13.3 CH-I(a) — the POSTCONTACT reconciliation is CONFIRMED

The prediction registered in 12.2 and on the board: under `POSTCONTACT=True` a
raised contact dose recovers most of `e10_pc`'s +17.2 % H1 cost. Against the
(False, 1x) control `e1_ctl`:

| cell | H1 | G1 | H1 penetration |
|---|---|---|---|
| `e10_pc` (True, 1x) | +17.2 % (t=5.16) | +2.3 % (t=2.69) | 5.2 mm |
| `pcx_c3` (True, 3.3x) | **+1.2 % (t=0.75, ns)** | +6.5 % (t=7.81) | **9.6 mm** |
| `pcx_c10` (True, 10x) | **+2.3 % (t=1.48, ns)** | +2.0 % (t=1.09, ns) | 14.5 mm |
| `e1_ctl` (False, 1x) | — | — | 13.1 mm |

The whole tracking cost of the double count is cancelled by re-dosing, on the
robot that paid it. Bin-for-bin on the shape metric: `e10_pc` +5 to +9 %,
`pcx_c10` +1 to +5 %, `pcx_c3` about 0 in every bin. **Section 12.2's
local/Viper reconciliation stands and the 3-topology -11 to -41 % win keeps its
explanation.**

And an unlooked-for result: `pcx_c3` is **the only arm in two waves that lowers
foot penetration at no tracking cost** — 13.1 -> 9.6 mm on H1 (t=-4.99, +1.2 % ns
on tracking). It is the only point measured under the project's 10 mm gate.

## 13.4 CH-I(b) — "FOOTH=0.3333 is the fix" is WRONG; the curve is flat and the term can go to ZERO

Dose curve against `FOOTH=1.0` (`e1_ctl`), 2 seeds per point:

| FOOTH | 0.3333 | 0.10 | 0.0333 | **0.0 (OFF)** |
|---|---|---|---|---|
| H1 | -2.5 % (ns) | -5.3 % (t=-3.39) | -2.5 % (ns) | -2.7 % (ns) |
| G1 | -5.6 % (t=-3.68) | -7.3 % (t=-4.44) | -6.0 % (t=-5.33) | **-9.7 % (t=-6.65)** |
| H1 `foot_airborne` | 0.0133 | 0.0433 | 0.0205 | **0.0532** |
| G1 ankle FVE | 0.147 | 0.109 | 0.096 | **0.156** |

Flat from 0.33 down, and **switching the term off is the best G1 point measured**
— against a `gait_coeff` lesson that said *dose, not switch*. It does not
generalise: here the switch-off is the winner.

The mechanism is the interesting half. At `FOOTH=0` H1 `foot_airborne` goes
0.0155 -> 0.0532 (+242 %, t=12.74) against a reference wanting 0.125, and G1's
ankle FVE more than doubles (t=8.48). **The foot-height reward term is part of
what glues the feet to the floor** — the 0b-4 pinned-contact mechanism has its
first named reward-side cause.

Caveats: at `FOOTH=0` H1's age fell 3.0 % (t=-4.40) and the H1 gain is -1 to -4 %
and ns bin-for-bin, so **the H1 half is small and uncertain; G1's holds
bin-for-bin** (-4 to -7 %, t up to -3.0).

**And a partial replication failure to log.** Section 12.4 quoted `FOOTH=0.3333`
at -6.3 % H1 from two seeds. The third seed (`fhd_33_s3`) reads -2.5 %. Pooled
over three seeds each: **-5.0 % H1 (t=-3.22) / -5.4 % G1 (t=-5.99)**. The claim
survives; the H1 magnitude was 25 % optimistic at n=2 training seeds.

## 13.5 CH-K — COMPUTE IS EXHAUSTED ON THIS RECIPE

| step budget | H1 | G1 |
|---|---|---|
| 19.7M -> 39.3M | -9.0 % (t=-6.06) | -13.5 % (t=-9.45) |
| **39.3M -> 78.6M** | **-2.8 % (t=-2.35)** | **-2.0 % (t=-1.08, ns)** |

Doubling the budget buys 2-3 %, and it buys it **entirely in the chains that
already work**: arms -10.4 % H1 / -10.6 % G1, legs -0.9 % / +0.4 % (both ns).
Foot penetration gets *worse* with compute (H1 20.3 -> 22.3 mm). Ankle FVE rises
(0.072 -> 0.090 H1) but is still 5x from its gate.

**The residual error is structural, not budgetary.** Buying steps past ~40M on
this recipe is not a lever, and any claim that the project "just needs more
compute" is now falsified on the fixed recipe at n=8-12.

## 13.6 THE PIPELINE, HONESTLY, AT THE END OF WAVE 5

`best79` — one policy, two topologies, 78.6M steps, the fixed recipe:

| | H1 | G1 |
|---|---|---|
| absolute RMSE vs raw clip | 0.1833 | 0.1503 |
| zero-action floor, same setup | 0.518 | 0.501 |
| **ratio over the floor** | **2.83x** | **3.33x** |
| arm FVE (gate 0.5) | 0.968 PASS | 0.974 PASS |
| hip / knee FVE | 0.41 / 0.45 FAIL | 0.63 / 0.63 PASS |
| **ankle FVE** | **0.090 FAIL** | **0.144 FAIL** |
| foot penetration (10 mm gate) | **22.3 mm FAIL** | 2.3 mm PASS |
| heading error, mean | **82.8 deg** | 63.3 deg |

Three holes, and wave 5 showed compute closes none of them:

1. **The ankles carry almost no reference information** on either robot. Rising
   (+176 % vs 19.7M) and still 5x from the gate.
2. **Global heading is untracked in every arm this project has ever run** —
   78-87 deg mean, unmoved by 4x compute. It is off by construction
   (`HEADING_RATIO=0.0`), and at `HEADING_TEMP=0.25` the exponential kernel is
   dead past ~60 deg, which is where all our errors live. Never actually tested.
3. **Tracking and penetration are on a Pareto the recipe has never deliberately
   traded.** Every arm that tracks better sinks further into the floor;
   `pcx_c3` is the single measured point that moves both the right way.

## 13.7 WHAT WAVE 5 CHANGED, IN ORDER

1. **Design B closes as formulated.** The penalty is recipe-independent
   (13.1) and does not track observation width (13.2). Do not spend another
   night on tokenizer scaling or code compression for the conditioning path.
2. **Design A is the FSQ result the project actually owns**: a
   token-reconstructed reference costs 0-4 %. That is a *compression* claim —
   motions can be stored and streamed as codes with no control cost — not a
   performance claim. State it that way.
3. **Ship `pcx_c3` + `FOOTH` near 0** as the recipe candidate: the only
   combination whose parts each improve tracking, un-glue the feet, and stay
   under the penetration gate. Never yet run together.
4. **Stop buying steps past ~40M** on this recipe (13.5).
5. **Heading has never been tested**, only left off. It is the cheapest
   untouched error in the pipeline.
6. **Retract** "`FOOTH=0.3333` is worth -6.3 % on H1" -> -5.0 % at three seeds;
   and retract "0.3333 is the operating point" -> the curve is flat to zero.

## 13.8 A GUARD THAT IS MEASURING THE WRONG THING

`age_binned_rmse_rad` is computed from `diff = ex - rw` — the **self-centred
shape** metric with the F20 mismatched centring — while every headline in this
report is `raw_rmse_rad_absolute`. So the bin-for-bin guard, which exists to
decide whether a pooled effect is real, runs on a *different and attenuated*
metric: `e1_ctl -> e10_pc` is +17.2 % pooled-absolute and +5 to +9 % binned.
Directions have always agreed, so no past conclusion changes — but the two are
not magnitude-comparable and were being read as if they were. Fix
`crosseval_motion.py` to bin the absolute diff as well.

---

# 14. THE FOUR STRUCTURAL ITEMS — 2026-08-29, local, while wave 6 runs

Wave 5 showed compute is exhausted and the remaining error is structural
(Sec 13.5). These are the four structural items, done in order. Three of the four
killed a hypothesis this project was carrying; the fourth is a code fix with a
regression test. None of them cost cluster time.

## 14.1 BoosterT1 — the retarget IS infeasible, and it is the ARMS

**Two cheap explanations died first.**

*Not a wrap.* `Left_Elbow_Pitch` wanting 8.051 rad is 461 degrees, and
8.051 = 2*pi + 1.768 — the exact signature of an angle emitted on the other
branch, which inflates a measured SPAN by 2*pi while the physical motion is
unchanged. A span test is wrap-sensitive and had been read as a kinematics
result. `t1_retarget_audit.py` unwrapped every joint of both clips along time and
searched the best 2*pi branch offset per joint: **nothing changes.** 12 span-
infeasible joints before, 12 after; 13.75 % of frames out of range before,
13.02 % after. F4 stands, and now for the right reason.

Beside it, the comparison the earlier audit did not print:

| clip | limited joints | span-infeasible | frames out of `jnt_range` |
|---|---|---|---|
| BoosterT1 dance2_subject4 | 23 | **12** | **13.75 %** |
| BoosterT1 dance2_subject1 | 23 | **8** | **2.50 %** |
| UnitreeH1 dance2_subject4 | 19 | 0 | 0.41 % |
| UnitreeG1 dance2_subject4 | 23 | 1 (by 2 %) | 0.00 % |

*Not reach either.* `t1_reach_check.py` measured the retarget's demanded
shoulder-to-hand distance against T1's own maximum arm extension, swept over the
corners of the arm's joint box: **0.0 % of frames demand more than the arm has**
(peak demand 254.6 mm against 254.6 mm available). So the limb-length story — the
human's arm mapped on without the length ratio — is wrong too. The hand is always
at a reachable DISTANCE and at an unreachable DIRECTION: T1's shoulder and elbow
ranges confine it to a much smaller solid angle than the human's.

**The regeneration.** `regenerate_t1_clip.py` re-solves each frame for the joint
angles closest to the retargeter's own body poses *subject to the joint limits*,
by projected Gauss-Newton on MuJoCo body Jacobians, warm-started frame to frame.
The stored `xpos`/`xquat` are exactly that body-space solution — FK from the
signed qpos reproduces them to **2.9e-08 m**, checked before anything is written —
so the target is preserved, not invented. The root free joint is not re-solved
(no limits, and leaving it alone keeps every root statistic identical), and every
derived body field is rebuilt by FK.

| clip | out-of-range | span-infeasible | body error vs the retarget target |
|---|---|---|---|
| dance2_subject1 | 2.50 % -> **0.00 %** | 8 -> **0** | mean **2.5 mm**, worst body 11.9 mm (left foot) |
| dance2_subject4 | 13.75 % -> **0.00 %** | 12 -> **0** | mean 19.8 mm, **left hand 231 mm, right hand 139 mm**, feet 3.4 mm |

**And the key negative: IK buys nothing over naive clamping** — 230.8 vs 230.9 mm
on the left hand, 19.8 vs 20.0 mm mean. A solver that reproduces feasible frames
to 0.0 mm does not fail by 23 cm because it is a bad solver. There is no
redundancy left in a 4-DOF arm at its limits, so the whole limb cannot compensate.

**What this means, concretely.**

1. **T1's legs are FIXED and usable now.** Feet land within 3.4 mm of the
   retargeter's target on subject4 and 11.9 mm on subject1. The ankle
   infeasibility cost foot ORIENTATION, not foot position.
2. **T1's arms on dance2_subject4 are NOT recoverable by post-processing.** They
   need the source retarget redone with T1's joint ranges as constraints, which
   is loco-mujoco work, not clip work.
3. **dance2_subject1 is essentially free to fix** (2.5 mm) and is the T1 clip to
   use for a topology-count claim in the meantime.
4. Every previous T1 number remains what it was: scored against an impossible
   target. The fixed clips live in `external_data/amass_converted/LAFAN1_t1fix/`
   (the clamp baseline is kept beside them in `LAFAN1_t1clamp/` so the "IK buys
   nothing" claim can be re-checked).

### 14.1b A float32 ulp, and why the generator now verifies the FILE

The generator first reported 0.00 % out-of-range and 0 span-infeasible. An
independent re-audit of the WRITTEN clips read **8.95 % and 3** on subject4.

Both were right about different objects. The IK clips exactly ONTO the limit, the
clip is stored float32, and a value sitting exactly on a bound rounds outward by
up to one ulp on the way to disk -- 1.2e-07 rad at 1 rad. Worst violation
anywhere was **5.7e-08 rad**, i.e. 3e-06 degrees, on precisely the joints that had
been pinned. Physically nothing; arithmetically a limit violation on most pinned
frames, and a span that exceeds its range by a rounding error.

Two changes, because "it is only 1e-08" is how a real defect gets waved through:

- the solver's box is inset by `MARGIN = 1e-6` rad (6e-05 degrees), so the
  written value is strictly inside;
- the generator **re-loads the file it just wrote and re-checks it from disk**,
  and refuses to leave a bad artifact behind. Both clips now read
  `out-of-range 0.00 %, worst violation 0.00e+00 rad, span-infeasible 0` on
  re-read.

This is the project's own "verify by ARTIFACT, not by exit code" rule one level
down: not by exit code, and not by the array still in memory either.

## 14.2 The contact model — the feet are fine standing and 12 mm soft on IMPACT

Eight reward-side stories have failed to explain the ankle. This measures the
physics instead (`contact_audit.py`, on the loco_mjx urma2 models the project
actually trains on — the earlier `scripts/scaling/foot_contact_audit.py` audits
the loco-mujoco stack, which is a different model).

**Finding 1 — settled penetration is NOT the problem.** At the shipped
`solref = 0.02 1`, the nominal pose settles at 0.55 mm (H1), 4.13 mm (G1),
4.05 mm (T1). Training reports **22.3 mm mean on H1**. So the 22 mm is not a soft
spring compressing under weight; it is dynamic.

**Finding 2 — and on IMPACT the contact is very soft.** Peak penetration after a
5 cm drop, and what the env's own unused knob does to it:

| `contact_solref_timeconst` | H1 settled / peak | G1 settled / peak | T1 settled / peak |
|---|---|---|---|
| **0.0 (as trained — knob OFF)** | 0.55 / **11.56 mm** | 4.13 / **12.72 mm** | 4.05 / **8.10 mm** |
| 0.004 | 0.23 / **3.73 mm** | 1.09 / **1.73 mm** | 0.09 / **2.42 mm** |
| 0.008 | 0.21 / 4.51 mm | 4.76 / 5.30 mm | 0.54 / 2.65 mm |
| 0.02 (solimp only) | 1.21 / 11.64 mm | 4.07 / 12.88 mm | 3.96 / 10.35 mm |

A 5 cm fall drives the foot **over a centimetre** through the floor. The knob is
already plumbed (`environment.py:166`, `CONTACT_TIMECONST` in the sbatch), has
been passed as **0.0 in every run this project has made**, and at 0.004 cuts
impact penetration **3.1x on H1 and 7.3x on G1**. The 0.02 row is the control
that says the win is the TIMECONST, not the solimp raise that comes with it.

**Finding 3 — there is no self-collision anywhere.** The env deletes every geom
that is not a foot, the floor, or a `reward_collision_sphere`
(`environment.py:106-111`). Each robot therefore has **exactly 3 collidable
geoms**, plus 26-29 reward-only spheres with `contype=0` that never physically
collide. Legs can pass through each other, arms through the torso, and nothing
but the two foot boxes can touch the ground. For a dance with crossing limbs that
is a real fidelity gap, and it means the `ground_penetration` reward term is
scoring geometric intersection of bodies that cannot collide.

**Finding 4 — H1's foot is a 2.8 cm-wide blade, and this is new.**

| robot | footprint | aspect | area | **per kg** |
|---|---|---|---|---|
| **UnitreeH1** (51.6 kg) | 0.295 x **0.028 m** | **10.5:1** | 83 cm2 | **1.60 cm2/kg** |
| UnitreeG1 (33.3 kg) | 0.190 x 0.054 m | 3.5:1 | 103 cm2 | 3.08 cm2/kg |
| BoosterT1 (31.6 kg) | 0.223 x 0.100 m | 2.2:1 | 223 cm2 | 7.05 cm2/kg |

The real Unitree H1 foot is about 0.09-0.10 m wide. loco_mjx ships it at 0.028 m
— roughly **3x too narrow** — giving H1 half of G1's support per kilogram and a
quarter of T1's, on the heaviest and tallest of the three. With one box per foot
that half-width IS the entire lateral support polygon.

**The same defect was found and FIXED once already, in the other stack.**
`scripts/scaling/foot_contact_fix.py` replaced H1's two collapsed capsules with a
0.24 x 0.09 m box for loco-mujoco, with the note that the support polygon had
"zero fore-aft extent" and that H1 "topples in 1.0-1.7 s ... 45 of 45 cases". That
fix was never carried across to loco_mjx, which is the stack urma2 trains on.

This is a candidate mechanism for the H1-specific asymmetries §0b flagged as
unexplained and that no reward story has covered: H1 has the worst hip and knee
FVE (0.41/0.45 against G1's 0.63/0.63), 10x G1's penetration, and the 36-40 degree
mirrored ankle postural bias. An ankle spending its authority on lateral balance
over a 2.8 cm blade is an ankle that is not tracking.

**Registered before the arms exist**, so it cannot be fitted afterwards: widening
H1's foot to 0.09 m (underside height unchanged, so nominal standing height and
every reference-height statistic are untouched) should raise H1's ankle FVE toward
G1's and shrink the H1/G1 penetration ratio. If it does not, the blade is not the
mechanism and the H1/G1 split needs a ninth explanation.

## 14.3 Generalization — the claim the project is for, measured for the first time

Every number in this report is scored on the clip and the bodies the policy
trained on. `best79` trained on **dance2_subject4 alone, H1+G1 only**, so:

- **GEN-A, held-out MOTION:** `best79_s1/s2` on `dance2_subject1`,
  `dance2_subject3`, `walk1_subject1`, trained bodies.
- **GEN-B, held-out BODY:** `best79_s1` on `unitree_h1v2`, `atlas`, `talos`, on
  the clip it does know — which isolates body transfer from motion transfer. One
  robot per job: four topologies in one graph peaked at 22 GB on a 24 GB box.

**Every cell has its own matched zero-action floor on the same clip and the same
body.** Without it a held-out score is uninterpretable — on `robotis_op3` the
policy once turned out to be worse than doing nothing, and that was only visible
against a floor. 60 crosseval jobs, no training, submitted alongside wave 6.

They write to `crosseval_gen/`, not `crosseval_ladder/`, because the artifact name
is `${EXP}__${CE_TAG}.json` and every analyser keys the arm off `EXP` alone: a
held-out-BODY eval of `best79_s1` on the clip it trained on would otherwise pool
straight into its trained result. That is the shape of the collision that
destroyed the 2026-08-24 FSQ control. `crosseval2.sbatch` gained `CE_OUTDIR` for
it (and `CE_HEADING_OBS`, for wave 6).

**Reading guard for when these land: co-report `age_mean`.** A held-out body that
falls over immediately is scored only on easy early frames and posts a flattering
RMSE.

## 14.4 The post-contact double count is fixed in code

`foot_slip + ground_penetration + feet_orientation` were stashed for
TrackingReward to re-apply past its clip **and** left inside `reward_penalty`, so
`tracking_post_contact_penalties=True` charged them twice. They are now removed
from `reward_penalty` exactly when TrackingReward is going to add them — a MOVE,
which is what the flag was always meant to be — guarded by
`getattr(self, "post_contact_penalties", False)` so `DefaultReward` stays
standalone-safe.

`tests/test_post_contact_not_double_counted.py` pins the arithmetic: the three
terms are absent from the unconditional sum, restored under exactly one
`if not move_contact_quality_past_clip:` branch, applied in exactly one
`if self.post_contact_penalties:` branch in `tracking.py`, and the stash holds
exactly those three. It is deliberately a structural test — the numeric path needs
a compiled MJX environment with a clip, which is a training run, not a unit test.

**NOT DEPLOYED TO VIPER.** Viper has its own `loco_mjx` copy and wave 6 is mid-
flight; pushing a reward change now would mean arms in one chain training under
different code, which is the confound this project has paid for more than once.
Deploy after wave 6 drains, then re-measure the tracking-vs-penetration Pareto at
a deliberate weight — which is the measurement the double count has been
preventing all along.

## 14.5 What these four changed

1. **T1's legs are usable now; T1's arms need an upstream retarget.** A
   topology-count claim can include T1 on `dance2_subject1` immediately.
2. **Two more T1 hypotheses are dead** (wrap, reach), and constrained IK is shown
   to buy nothing over clamping — so nobody should try either again.
3. **`CONTACT_TIMECONST=0.004` is a one-flag experiment that cuts impact
   penetration 3-7x** and has never been run at a non-zero value.
4. **H1's foot is 3x too narrow in loco_mjx**, the fix already exists in the other
   stack, and it is the first named model-side candidate for the H1/G1 split.
5. **The double count is gone from the code**, with a test, pending deployment.

## 14.6 OPTION 1 WORKS — the limits were enabled, they were just not stiff enough

§14.1 established that T1's retarget is genuinely infeasible and that no
post-processing of the clip recovers the arms. This tests the upstream fix.

**The diagnosis.** loco-mujoco fits a motion by PHYSICS, not by IK: it welds
mocap targets to the robot's mimic sites and runs `mj_step` 25 times per frame
(`fit_smpl_motion`, `motion_iterations: 25`). `disable_joint_limits` is `False`
throughout — the limits were never switched off. But **MuJoCo joint limits are
soft constraints** (`jnt_solref = [0.02, 1]`, `jnt_solimp[0] = 0.9`), and a stiff
weld out-pulls them. That is how a retargeter that was told about the limits
produced a clip that violates them.

**The knob.** `fit_smpl_motion` now reads two optional params from the robot
conf, both defaulting to `None` so every existing retarget reproduces unchanged:
`limit_solref_timeconst` and `limit_solimp_dmin`, applied to `env._model.jnt_*`
immediately before the fit loop (they have to go there — the init step reloads
the model and would discard them). Patch lives in the `loco-mujoco` submodule,
`smpl/retargeting.py`.

**Why this is a real test and not §14.1's IK again.** The IK projected the
retargeter's *finished* pose onto the feasible set, which cannot recover freedom
already thrown away. This re-runs the SOLVE from the SMPL motion, so the whole
body is free to find a different configuration. The expensive half of a
robot2robot retarget is fitting SMPL to the source robot; a cached fit from the
morphology campaign supplied that, so only the target-side physics fit was re-run.

**Result.** 600 frames of dance2_subject4, four cells, everything else fixed:

| cell | out-of-range frames | span-infeasible joints |
|---|---|---|
| **STOCK (as shipped)** | **8.83 %** | **5** |
| `solref 0.004` | 2.49 % | 2 |
| `solref 0.004` + `dmin 0.99` | **2.40 %** | 1 |
| stiff + 4x solver iterations | 2.65 % | **0** |

**A 3.7x reduction, and span-infeasibility essentially eliminated.** More solver
iterations buy nothing (2.65 vs 2.40, slightly worse) — it is the constraint
stiffness, not the convergence budget.

**And the cost, which is the half that decides whether to ship it.** A stiff
limit can "fix" feasibility by simply refusing to move the joint, trading an
impossible reference for a wrong one. Joint excursion, stiffest vs stock:

| chain | mean span | vs available range | out-of-range |
|---|---|---|---|
| **arm** (n=8) | 3.646 -> 2.713 rad (**-25.6 %**) | 3.705 rad | **21.76 % -> 3.87 %** |
| leg (n=12) | 1.263 -> 1.233 rad (-2.4 %) | 1.930 rad | 2.40 % -> 2.47 % |
| trunk (n=3) | 0.894 -> 0.904 rad (+1.0 %) | 2.617 rad | 0.06 % -> 0.17 % |

The arm gives up a quarter of its excursion and 82 % of its violation. Read that
against the range rather than against stock: the stock arm span was 3.646 rad
against 3.705 rad of *total* travel — the fit was running the arms at essentially
100 % of their range and past it on individual joints (`Right_Shoulder_Pitch`
4.927 rad from a 4.530 rad joint). The stiff fit uses 73 % of range. The arm
still moves; it stops demanding the impossible.

**What it does NOT fix.** The legs are untouched — `Left_Hip_Roll` 11.20 % ->
11.87 %, `Right_Hip_Roll` 15.38 % -> 15.72 %. Their spans (0.94-1.13 rad) sit
comfortably inside a 1.77 rad range, so this is an OFFSET, not a span problem:
the human's hip adduction lands outside T1's *asymmetric* range
([-0.200, 1.570] in the fit convention) and stiffness has nowhere to put it.
Same for elbow yaw (~10-12 % residual). Those need option 3 — scaling or
re-centring the source motion — not a stiffer constraint.

**Caveats, in order of size.**
1. The cached SMPL fit is a 600-frame window of dance2_subject4 fitted from the
   **H1** source; the shipped T1 clip is 9023 frames from **UnitreeH1v2**. Stock
   reads 8.83 % here against 13.75 % shipped — same order, not the same number.
   The stock-vs-stiff contrast is internally valid; the absolutes are not
   transferable.
2. The shipped clips have NOT been regenerated with this setting. Doing so needs
   the UnitreeH1v2 source fit (torch, per clip) and is the next step.
3. A near-rigid limit changes the contact/constraint conditioning of the fit;
   nothing here checks that the resulting motion is still smooth. Check
   before shipping.

**A measurement trap that cost the first reading.** The fit emits qpos in
loco-mujoco's convention, and loco_mjx's T1 differs from it by a sign flip on 9
of 23 joints (identical spans, mirrored ranges — `t1_model_range_diff.py`).
Audited against the loco_mjx ranges, every flipped joint reads ~100 %
out-of-range with a span comfortably INSIDE its range, and `Right_Knee_Pitch`
duly reported 100.00 % in both cells. **A joint that is 100 % out of range with a
span that fits is a convention error, never a limit violation.** The probe now
audits against the model the fit actually used.

**Recommendation.** Ship `limit_solref_timeconst = 0.004`,
`limit_solimp_dmin = 0.99` in `BoosterT1.yaml` and regenerate both clips from the
H1v2 source. That gets T1's arms from unusable to ~4 % out-of-range for a quarter
of their excursion, on top of legs that §14.1 already recovered. The hip-roll
residual stays open and is a different fix.

## 14.7 GEN-A LANDED — held-out motion is essentially FREE

36 artifacts, `best79` (trained on **dance2_subject4 alone**), n=8 per cell, each
against its own matched zero-action floor on the same clip:

| clip | robot | policy | floor | ratio | t | age | arm FVE |
|---|---|---|---|---|---|---|---|
| dance2_subject4 *(trained)* | h1 / g1 | 0.1833 / 0.1503 | 0.518 / 0.501 | **2.83x / 3.33x** | — | 499 / 492 | 0.97 / 0.97 |
| dance2_subject3 | h1 / g1 | 0.1697 / 0.1448 | 0.533 / 0.501 | **3.14x / 3.46x** | -162 / -188 | 500 / 499 | 0.98 / 0.98 |
| walk1_subject1 | h1 / g1 | 0.1692 / 0.1523 | 0.441 / 0.431 | **2.60x / 2.83x** | -77 / -139 | 499 / 497 | 0.98 / 0.98 |
| dance2_subject1 | h1 / g1 | 0.2340 / 0.1948 | 0.508 / 0.533 | **2.17x / 2.73x** | -99 / -109 | 494 / 470 | 0.96 / 0.96 |

**A policy trained on one motion tracks motions it has never seen at the same
2-3.5x over the floor** — and on `dance2_subject3` it is BETTER than on the clip
it trained on. `walk1_subject1` is a different motion CLASS (walking, not dancing)
and still reads 2.6-2.8x. Ages are 470-500 everywhere, so none of this is the
survival confound, and arm FVE is 0.96-0.98 throughout.

**The policy is following the reference, not memorising the clip.** That is the
generalization claim this project is for, and it had never been measured.

**Consequence for BoosterT1, which is the largest one.** T1 has only two clips
against H1/G1's ten, and "T1 is data-poor" has been an unexamined background
assumption. It is now falsified as a limiting factor: H1 and G1 generalize from
**one** motion. T1 does not need more motions, more compute (its step curve is
-8.8 % against H1's -41 %), or more clip post-processing. It needs a reference
that is physically reachable.

Caveat: two training seeds, held-out MOTION on TRAINED bodies. The held-out
BODY half (GEN-B: atlas, talos, unitree_h1v2) is still queued and is the harder
claim.

## 14.8 THE REFERENCE-FEASIBILITY MATRIX — the defect is SYSTEMIC, and BoosterT1 is not the worst

BoosterT1's impossible reference was found by accident. `reference_feasibility_matrix.py`
checks every robot x every clip we own, so the same defect cannot be sitting
undetected in another family. It reports two things:

- **span-infeasible** (primary): a joint whose peak-to-peak travel in the clip
  exceeds its total range. **Sign- and offset-invariant** — `|max-min|` does not
  change if the convention is negated or the zero point shifts — so it needs no
  sign table and cannot be faked by a convention mismatch. No re-homing,
  re-signing or re-centring rescues such a joint.
- **out-of-range** (secondary, a LOWER BOUND): reported as the best of the two
  sign choices per joint. Getting this wrong cost two false readings in one day.

A joint over its range by <5 % is called MARGINAL: at that size a retarget edge
effect and a model-range rounding are indistinguishable, and lumping it in would
put H1 (1.12x on one clip of eleven) with Atlas (2.74x on all five).

| robot | clips | clips with a REAL bad joint | REAL bad joints | worst ratio | oor (lower bd) | chains | verdict |
|---|---|---|---|---|---|---|---|
| **UnitreeG1** | 10 | **0** | **0** | 1.04x | **0.00 %** | (ankle, marginal) | **GOOD — every clip reachable** |
| **UnitreeH1** | 11 | 1 | 1 | 1.12x | 0.05 % | hip | **MOSTLY GOOD — one bad clip** |
| ToddlerBot | 5 | 5 | 17 | 1.18x | 1.97 % | arm | DEGRADED — systematic but shallow |
| Talos | 2 | 2 | 17 | 1.61x | 2.88 % | arm, trunk | BROKEN |
| BoosterT1 | 2 | 2 | 19 | 2.35x | 5.08 % | ankle, arm | BROKEN |
| **Atlas** | 5 | 5 | **67** | **2.74x** | **12.12 %** | arm, hip, trunk | **BROKEN — the worst of all** |
| UnitreeH1v2 | 5 clips | — | — | — | — | — | **NO loco_mjx MODEL — cannot be a topology here at all** |

### What this says

1. **Only the two Unitree humanoids have a reachable reference.** G1 is clean on
   all ten clips; H1 is clean on ten of eleven (`dance2_subject2` puts two hips
   1.12x over). Every non-Unitree family is broken to some degree.
2. **BoosterT1 is not an outlier — it is the middle of a systemic failure.**
   **Atlas is worse on every axis**: 67 impossible joints against T1's 19, 2.74x
   against 2.35x, 12.12 % of frames out of range against 5.08 %, and it fails on
   all five of its clips. This retires the fallback proposed in §14.5 — swapping
   T1 for Atlas would make the problem worse, not better.
3. **It is one defect, not six.** §14.6 showed the mechanism is loco-mujoco
   fitting motions with SOFT joint limits that the mocap welds out-pull. That
   mechanism is robot-independent, so it hits every family; severity just tracks
   how restrictive a body's ranges are against the motion being asked of it.
   The `limit_solref_timeconst` / `limit_solimp_dmin` fix therefore applies to
   ALL of them, not just T1.
4. **`UnitreeH1v2` has clips but no loco_mjx model.** It cannot be trained or
   evaluated in this stack — and the GEN-B cells asking for it will fail rather
   than return a wrong number. Do not count it as an available topology.

### WALKING IS MUCH EASIER THAN DANCING, and by a wide margin

Worst span ratio per motion class, on the two clean robots:

| motion | H1 | G1 |
|---|---|---|
| `walk_cycle_*` (4 clips) | **0.36 - 0.85x** | **0.36 - 0.89x** |
| `walk1_subject1` | 0.99x | 0.98x |
| `dance2_subject*` | 0.98 - **1.12x** | 0.98 - **1.04x** |

The synthetic walk cycles use barely a third of the available joint travel; the
dances run at 98-112 % of it. **Dance is the motion class that saturates the
joint ranges**, which is why every retarget defect in this project has surfaced
on a dance clip and none on a walk. It also means a walking claim is far more
robust to retarget error than a dancing one — and that the non-Unitree families
have NO walk clips at all, only dances, i.e. only the hardest case.

### Which robot for which motion, as it stands today

| | dance | walk |
|---|---|---|
| **UnitreeG1** | **USE** — all 5 dances reachable | **USE** — 5 clips, large margin |
| **UnitreeH1** | **USE, minus `dance2_subject2`** | **USE** — 5 clips, large margin |
| BoosterT1 | only after re-retarget (§14.6); legs already fixed (§14.1) | no walk clips exist |
| ToddlerBot | shallow arm-only defect; cheapest non-Unitree to fix | no walk clips exist |
| Talos | needs re-retarget; also has a known loader bug | no walk clips exist |
| Atlas | needs re-retarget most of all | no walk clips exist |
| UnitreeH1v2 | unavailable (no model) | unavailable |

## 14.9 FIX 2 APPLIED — one change repairs three of four broken families

§14.8 showed the impossible-reference defect is systemic and §14.6 showed the
mechanism is robot-independent (soft MuJoCo limits out-pulled by the mocap
welds). So the fix should repair every family at once. `refit_all_families.py`
runs that, on 600 frames of dance2_subject4 from the cached SMPL fit.

**The check that was missing last time is now in.** A stiff limit can buy
feasibility by simply refusing to move a joint -- trading an impossible reference
for a wrong one -- and no feasibility metric would notice. `fit_smpl_motion` now
records per-frame **mimic-site tracking error** (how far the welded sites ended
up from what SMPL asked for) into `LAST_FIT_DIAGNOSTICS`, so fidelity is measured
rather than assumed.

And out-of-range is now reported at a **magnitude threshold of 1e-3 rad
(0.06 deg)**. Below that it is solver residue at a pinned joint, not a pose the
body cannot hold; counting it would repeat the float32-ulp false alarm of §14.1b.

| robot | out-of-range >0.06 deg | span-infeasible | worst span ratio | mimic-site error |
|---|---|---|---|---|
| **BoosterT1** | 8.79 % -> **0.09 %** | 5 -> 1 | 2.28x -> **1.00x** | 28.6 -> **27.3 mm** |
| **Atlas** | 6.41 % -> **0.31 %** | 9 -> 6 | 1.85x -> **1.00x** | 43.0 -> 44.5 mm |
| **ToddlerBot** | 5.62 % -> **0.16 %** | 5 -> 5 | 1.14x -> **1.00x** | 20.8 -> 21.0 mm |
| Talos | 7.61 % -> 7.07 % | 6 -> 1 | 1.98x -> **1.00x** | 55.0 -> 55.9 mm |

**Every family loses its span-infeasibility.** Worst ratio goes to exactly 1.00x
on all four: no joint is asked to travel further than it has, anywhere. The
remaining `span-inf` counts are joints sitting at exactly 1.00-1.05x, i.e. pinned
against their bounds, which is what a working constraint looks like.

**And the motion SURVIVED.** Mimic-site error moves by -1.3 to +1.6 mm against
21-55 mm baselines -- under 4 % relative. Nothing was flattened to buy
feasibility. **BoosterT1's fit actually got BETTER** (28.6 -> 27.3 mm): the stock
solver was wasting effort dragging joints through their limits, and stopping it
improved the fit it was there to produce.

### Talos is the exception, and it is not a stiffness problem

Talos keeps a 7.07 % residual, and **more stiffness makes it worse**:

| Talos setting | out-of-range >0.06 deg | mean violation | max violation |
|---|---|---|---|
| `0.004 / 0.99` | 7.07 % | 0.49 deg | 5.97 deg |
| `0.002 / 0.999` | 8.56 % | 1.38 deg | 7.48 deg |
| `0.002 / 0.9999` | 9.09 % | 1.38 deg | 8.71 deg |

So the residual is the same class as BoosterT1's hip roll (§14.6): a demanded
pose sitting outside an asymmetric range, which no constraint stiffness moves. It
is worth keeping in proportion though -- **a mean violation of 0.5 deg** against
a best-case policy tracking error of 8-10 deg is small. Talos went from four
joints wanting up to 1.98x their entire range to a handful of frames asking for
half a degree past a bound. That is a qualitative change even though it is not
zero. Note also that Talos declares 30 limited joints against the others' 23 and
its clip loader is already known to drop 14 unactuated ones, so a coupled-joint
interaction is the first thing to check if anyone wants the last 7 %.

### Shipped as the default for the four families

`limit_solref_timeconst: 0.004`, `limit_solimp_dmin: 0.99` are now in
`BoosterT1.yaml`, `Atlas.yaml`, `Talos.yaml`, `ToddlerBot.yaml`, and documented
as `null` in `defaults.yaml`. **H1 and G1 are deliberately NOT changed** -- their
clips are already feasible and there is no reason to perturb known-good
references.

### Are these topologies usable now?

| robot | before | after fix 2 | verdict |
|---|---|---|---|
| BoosterT1 | 12 impossible joints, 2.35x | 0, 0.09 % out | **USABLE** |
| Atlas | 15 impossible joints, 2.74x | 0, 0.31 % out | **USABLE** |
| ToddlerBot | 4 impossible joints, 1.18x | 0, 0.16 % out | **USABLE** |
| Talos | 11 impossible joints, 1.98x | 0, 7.07 % out at 0.5 deg mean | **USABLE with a caveat** |
| UnitreeH1 / UnitreeG1 | already clean | unchanged | **USABLE** |

**SCOPE, and it is a real limit.** This is 600 frames of dance2_subject4 from the
locally cached SMPL fit. Re-issuing the *shipped* clips needs each clip's
UnitreeH1v2 source trajectory, and only `dance2_subject4` is on this machine --
the rest download from HuggingFace on demand. So what is established is that the
fix works ACROSS FAMILIES on the flagship motion; the full re-issue is a batch
job, not another investigation.

## 14.10 THE FULL RE-ISSUE — 44 clips, and every family becomes usable

### A correction first

§14.9 said only `dance2_subject4` could be regenerated locally, because the
retargeter's default source env is `UnitreeH1v2` and only that clip was on this
machine. **That was wrong.** The source is a PARAMETER, not a constant, and this
repo holds **11 full-length UnitreeH1 clips and 10 UnitreeG1 clips** locally --
both families verified feasible in §14.8, and H1 is already what produced the
cached SMPL fits in `deepmimic_morphology/`. No downloads were ever needed.

`motion_transfer_robot_to_robot` splits into an expensive SOURCE fit (torch,
cached via `path_to_fitted_motion_source`) and a cheap per-target physics fit, so
11 motions x 4 robots costs **11 source fits, not 44**. The whole batch ran in
minutes. Output is `external_data/amass_converted/LAFAN1_fixed/` -- deliberately a
NEW directory, because `LAFAN1/` is what every trained checkpoint and crosseval
baseline was scored against and overwriting it would silently invalidate
comparisons to work already done.

### Result: 44 clips, before and after

| robot | shipped | re-issued |
|---|---|---|
| **BoosterT1** | 2 clips, 19 impossible joints, worst **2.35x** | **11 clips**, 3 joints on 1 clip, worst **1.09x** |
| **Atlas** | 5 clips, 67 impossible joints, worst **2.74x** | **11 clips, 0 impossible joints**, worst **1.02x** |
| **Talos** | 2 clips, 17 impossible joints, worst **1.61x** | **11 clips**, 2 joints on 1 clip, worst **1.11x** |
| **ToddlerBot** | 5 clips, 17 impossible joints, worst **1.18x** | **11 clips**, 11 joints on 6 clips, worst **1.06x** |

Two things happened at once. The references became reachable -- worst span ratio
falls from 1.18-2.74x to **1.02-1.11x** everywhere -- and the clip count went from
2/5/2/5 to **11 each**, because sourcing from H1 brings the walk motions these
families never had. **Atlas goes from the worst family in the project to
"GOOD -- every clip reachable".**

Spot-checked full-length rather than trusting the 600-frame probe:
`BoosterT1/dance2_subject1`, 9023 frames, max violation **123.8 deg -> 2.8 deg**.

### WALKING IS NOW COMPLETELY CLEAN ON EVERY FAMILY

The §14.8 prediction -- that walking's 3x joint headroom would survive even a
buggy retarget -- holds, and after the fix it is unambiguous:

| motion class | span-infeasible joints, all 4 families | worst ratio |
|---|---|---|
| `walk_cycle_*` (16 cells) | **0 in 15 of 16** | **0.23 - 0.95x** |
| `walk1_subject1` (4 cells) | 1-5, all marginal | 1.01 - 1.03x |
| `dance2_*` (24 cells) | 1-13, mostly marginal | 1.01 - 1.11x |

**All four previously-broken families now have four perfectly clean walk cycles
each**, using a quarter to a third of their joint travel. The residual is
entirely in the dances, and it is marginal: the only cells over 5 % are
`BoosterT1/dance2_subject1` (1.09x), `Talos/dance2_subject3` (1.11x) and
ToddlerBot's dances (1.06x, i.e. 6 % over, barely across the threshold).

### Caveat on the residual out-of-range

Span is now clean everywhere, so the remaining out-of-range (Atlas 4.81 %,
Talos 3.06 %) is an OFFSET effect -- the trajectory fits in width but sits
shifted. That is either genuine (an asymmetric joint range, like BoosterT1's hip
roll in §14.6) or a zero-point difference between the loco-mujoco model the fit
used and the loco_mjx model this audit reads. The audit's lower bound handles
SIGN flips but not offsets, so those two are not separated here. Two clips are
suspicious in the same way -- `walk_cycle_s8883` reads 12.0 % on Atlas and 12.7 %
on Talos while being span-clean on both -- which points at that clip rather than
at those robots.

### Which robot for which motion, after the fix

| | walk cycles | walk1 | dance |
|---|---|---|---|
| UnitreeH1 / UnitreeG1 | USE | USE | USE (H1 minus `dance2_subject2`) |
| **BoosterT1** | **USE, clean** | USE | USE (`subject1` marginal) |
| **Atlas** | **USE, clean** | USE | **USE, all reachable** |
| **Talos** | **USE, clean** | USE | USE (`subject3` marginal) |
| **ToddlerBot** | **USE, clean** | USE | USE, 1.06x on the arms |

The project went from two usable topologies to six, and from dance-only data on
the non-Unitree families to a clean walking set on all of them. What is NOT yet
established is that a policy can TRACK any of them -- that needs a training run,
and the reference being reachable is a precondition, not a result.

## 14.11 A FEASIBLE REFERENCE IS NOT A USABLE ONE — Atlas, Talos and ToddlerBot do not map onto their training models

§14.10 established that the re-issued clips are physically reachable. That is one
of three preconditions, and the other two fail for the three new families.

**Sign screening (`derive_clip_signs.py`).** loco-mujoco and loco_mjx keep the
same joint names and reverse many axes; a family added without its sign table
defaults every joint to +1.0 and the policy chases a part-mirrored reference.
That has already cost three campaigns. The tool recovers the signs by asking
which sign vector makes loco_mjx's FK reproduce the clip's own recorded world
poses. Validated first on the three known families -- it reproduces their
published tables exactly, residual ~1.5e-4. Then:

| family | negated | residual | verdict |
|---|---|---|---|
| UnitreeH1 / UnitreeG1 / BoosterT1 | 13 / 15 / 14 | ~1.5e-4 | USABLE, matches published table |
| **Atlas** | 16 | **0.0430** (21x) | **NOT REPRODUCIBLE BY SIGNS ALONE** |
| **Talos** | 17 | **0.8302** (415x) | **NOT REPRODUCIBLE BY SIGNS ALONE** |
| **ToddlerBot** | 5 | **2.3953** (1198x) | **NOT REPRODUCIBLE BY SIGNS ALONE** |

**Structural agreement (`clip_model_agreement.py`)** says why:

| family | clip joints | mjx joints | matched by name | shared bodies | max link offset |
|---|---|---|---|---|---|
| BoosterT1 | 23 | 23 | **23** | 23/25 | **0.0 mm** |
| Atlas | 27 | 27 | 23 | 29/33 | 0.0 mm |
| Talos | **44** | **30** | 30 | 44/46 | 0.0 mm |
| **ToddlerBot** | **56** | **44** | **30** | **6/58** | **63.0 mm** |

- **ToddlerBot ships two different robots under one name.** Only 6 of 58 body
  names are shared and the link offsets differ by up to 63 mm. No sign table can
  fix that; it is not the same kinematic tree.
- **Talos's clip carries 44 joints against a 30-joint training model** -- the 14
  unactuated ones its loader was already known to drop. A 0.83 residual says the
  remaining mapping is wrong too, not merely incomplete.
- **Atlas is the closest**: identical geometry, but 4 of 27 joint names do not
  match, and a 0.043 residual (4.3 cm) means the sign story is incomplete.

**Consequence, and it corrects §14.10's headline.** "The project went from two
usable topologies to six" is wrong. The reference is now REACHABLE for six
families; it is CORRECTLY MAPPED for three. Only **UnitreeH1, UnitreeG1 and
BoosterT1** can be trained on today. And it means the feasibility numbers for
Atlas/Talos/ToddlerBot in §14.10 compare a clip column against a joint range that
may belong to a different joint -- they are not trustworthy either.

**A precondition ladder, so this is not re-learned a fourth time.** Before a
family counts as a training topology:
1. the clip's joints map onto the training model's by name (`clip_model_agreement.py`)
2. the two models are the same kinematic tree (shared bodies, matching link offsets)
3. FK from the signed clip angles reproduces the clip's recorded world poses
   (`derive_clip_signs.py --check`, residual ~1e-4)
4. the reference is physically reachable (`reference_feasibility_matrix.py`)
5. only then: does a policy track it

This project has repeatedly measured rung 4 or 5 on a family that had not passed
rungs 1-3, and every time the numbers were uninterpretable.

## 14.12 CH-L LANDED — HEADING IS SOLVED, and OBSERVABILITY was the whole story

Wave 6 CH-L, n=4 rollout seeds, against the matched control (`e2_fh3_s{1,2}` +
`fhd_33_s3`, n=12, same recipe, reward off and observation off).

| arm | H1 heading | G1 heading | H1 dRMSE | G1 dRMSE | what it isolates |
|---|---|---|---|---|---|
| CONTROL | 80.7 deg | 72.7 deg | — | — | reward OFF, obs OFF |
| `hd_r20n` r=0.20 T=2.0, **obs OFF** | **82.8 deg** | **73.1 deg** | +1.9 % | -3.7 % | reward WITHOUT observation |
| `hd_r20` r=0.20 T=2.0, obs ON | **8.7 deg** | **9.7 deg** | +8.1 % | +3.8 % | the working cell |
| `hd_r20t` r=0.20 **T=0.25**, obs ON | **6.1 deg** | **5.2 deg** | +19.0 % | +14.4 % | the "dead kernel" control |
| `hd_r75` r=0.75 T=2.0, obs ON | **4.5 deg** | **5.4 deg** | +20.5 % | +18.1 % | dose |

**1. OBSERVABILITY WAS THE BLOCKER, and it is unambiguous.** `hd_r20n` rewards
heading exactly as `hd_r20` does but without the 2 observation channels, and it
is **indistinguishable from the control** (82.8 vs 80.7 deg, 73.1 vs 72.7 deg).
Rewarding a quantity the policy cannot see does literally nothing. Every previous
"heading does not work" result on this project is explained by this one line.

**2. Heading is now solved.** 80.7 -> 8.7 deg on H1 and 72.7 -> 9.7 deg on G1,
a 9.3x and 7.5x reduction, far inside the registered 40 deg gate, with age
unchanged (491 -> 494).

**3. RETRACTION -- the "dead kernel" analysis was wrong.** I argued that at
`TEMP=0.25` the exponential kernel scores `exp(-1.4^2/0.25) = 4e-4` at our real
80 deg errors and therefore could not teach anything. `hd_r20t` runs exactly that
and gives the **best heading of the three obs-ON cells** (6.1 / 5.2 deg). The
reasoning assumed the error stays at 1.4 rad; once the policy can SEE the error,
even a tiny gradient starts it moving, and a sharp kernel is then sharper near
zero. Temperature trades heading accuracy against joint accuracy, it does not
gate learning. The memory note claiming otherwise is corrected.

**4. RETRACTION -- "heading may be futile until the feet move" is also wrong.**
I hypothesised that turning requires stepping, so heading could not improve while
the feet are glued. It improved 9x with the feet still mostly planted
(`foot_airborne` 0.0139 -> 0.0181). Heading tracking does lift the feet slightly
(+30 % H1, and +80 % at r=0.75) but nowhere near the reference's 0.127, so the
two defects are far more separable than I predicted.

**5. There is a real trade, and `r=0.20 / T=2.0` is the operating point.**

| setting | heading | joint RMSE cost |
|---|---|---|
| r=0.20 T=2.0 | 8.7 / 9.7 deg | **+8.1 % / +3.8 %** |
| r=0.20 T=0.25 | 6.1 / 5.2 deg | +19.0 % / +14.4 % |
| r=0.75 T=2.0 | 4.5 / 5.4 deg | +20.5 % / +18.1 % |

Against the registered gate (heading < 40 deg, RMSE no worse than +5 %, age
within 3 %): **G1 passes on every clause. H1 passes heading and age and misses
the RMSE clause** (+8.1 % against a +5 % budget). Arm FVE also slips 0.950 ->
0.936. So heading is not free -- it is bought.

### Is it worth buying? Yes, and the clips say why

Total accumulated yaw per clip, and the fraction of frames spent more than
90 deg from the starting heading:

| clip | total turn | frames >90 deg from start |
|---|---|---|
| dance2_subject1 | **21 535 deg** | **86.8 %** |
| dance2_subject4 | **12 058 deg** | **74.8 %** |
| dance2_subject5 | 13 057 deg | 58.0 % |
| walk1_subject1 | 7 375 deg | 94.5 % |
| `walk_cycle_*` (all 4) | **16 - 23 deg** | **0.0 %** |

**The dances turn constantly** -- three quarters of dance2_subject4 is spent
facing more than 90 deg away from where it started. A policy at 80 deg mean
heading error is not performing that choreography, and **our headline metric
cannot see it**, because joint angles are body-local. That is why an 8 % joint
RMSE cost for a 9x heading improvement is a good trade for dance: it buys the
half of the motion the metric was blind to.

**And it is nearly irrelevant for the synthetic walk cycles**, which turn 16-23
deg in total. If the next campaign is the walking claim on `LAFAN1_fixed/`,
heading can stay off and the 8 % kept.

# 15. THE GUARD, AND THE FINAL WAVE — 2026-08-29 afternoon

## 15.1 `pipeline_guard.py` — the config->effect check the project never had

Eight times a knob was configured and did not do what it said, and every one was
found by a human reading code: the post-contact double count, a heading reward
that was exactly 0.000000 for a campaign, a cosine kernel that ignores its own
temperature, `FOOTSLIP`/`GROUNDPEN` defaulting to 0.1/10 against controls at
20/1000, `latent_obs` with no launcher passing it, a stripped `joints_present`
flag, `CE_REFBIAS` defaulting to 1.0 against arms trained at 0.0, and a crosseval
that never passed `--root_heading_obs`. Roughly one knob in eight has lied.

Two checks, built on artifacts we already produce:

- **CHECK 1, dead terms.** A reward coefficient set non-zero whose logged reward
  is identically zero over the run is doing nothing; a term that is CONSTANT has
  a weight but no gradient. Catches bugs 2 and 3.
- **CHECK 2, train/eval agreement.** Every field an evaluation must reproduce
  from training, compared between the arm's resolved flags and the crosseval
  artifact's own `eval_condition`. Catches bugs 7 and 8. Absent flags resolve
  against `default_config.py`, read at runtime -- without that, a field the
  training never passed is skipped rather than compared, which is exactly how
  bug 8 hides.

**Validated with positive AND negative controls**, each reproducing a historical
bug verbatim on a real log and a real artifact, pinned in
`tests/test_pipeline_guard.py` (5 tests, all passing):

| control | result |
|---|---|
| real arm, unmodified | **0 problems** |
| heading ratio claimed 0.20, reward logged zero (bug 2) | **caught** -- "IDENTICALLY ZERO over 800 samples" |
| eval `refbias=1.0` vs train 0.0 (bug 7) | **caught** |
| eval `root_heading_obs=True` vs the default False (bug 8) | **caught** |

### The sweep: 94 arms, and the tool's own first result was a false alarm

Run over every arm with a training log:

> **CLEAN 94, FLAGGED 0.** Three wave-7 arms are NOT CHECKABLE yet (no per-term
> reward table logged until training gets going).

The first run flagged **19 arms** -- every `bl_*` and `blw_*` design-B arm, on
`latent`, `latent_dim` and `latent_replaces`. That was the TOOL, not the arms:
the flag parser only read lines STARTING with `--`, and the sbatch passes those
flags on an `EXTRA_ARGS_ARR=--a=1 --b=2` line. Had I reported it, I would have
retracted the design-B verdict on the strength of a regex. The fix is pinned by
its own test.

So: no arm this project has run disagrees with its own configuration on any field
the guard can see. That is a real clean bill of health for the numbers already
published -- and the first one backed by a check rather than by reading.

**What it does NOT cover**, stated so the clean result is not over-read: bug 1
(arithmetic inside a reward term -- covered separately by
`test_post_contact_not_double_counted.py`), bug 4 (a launcher default differing
from a control's -- needs an arm-to-arm diff, not an arm-to-itself check), bug 5
(a flag no launcher passes -- invisible, because the arm's log is self-consistent),
and bug 6 (an observation channel dropped inside the network).

## 15.2 What was shipped and what is running

**Deployed to Viper** (LF-converted and verified with `file`; the CRLF trap has
cost this project a submission before): the post-contact double-count fix, and
the new `foot_half_width_override` knob. Safe to deploy because no TRAINING arm
was pending -- only crossevals, which do not touch the reward.

**`foot_half_width_override`** widens the foot box's y half-extent only.
Verified surgical (`verify_foot_width.py`): H1 0.028 -> 0.090 m, support
1.60 -> 5.14 cm2/kg (past G1's 3.08), `geom_pos` delta 0.0, and **nominal
standing height identical to 12 decimal places**, so no reference-height
statistic moves. Default 0.0 = off, so nothing reproduces differently.

**WAVE 7 on Viper** -- 8 arms / 62 jobs, every comparison against a wave-7
control because `POSTCONTACT=True` now means something different than it did in
waves 4-6:

| chain | arms | question |
|---|---|---|
| CH-P | `w7_ctl`, `w7_foot` x2 seeds | does widening H1's foot pay? |
| CH-Q | `w7_head` x2 | heading on the fixed code |
| CH-R | `w7_both` x2 | do the two compose? |
| GEN-B' | crossevals | **BoosterT1 held out**, on the regenerated reference |

GEN-B' replaces the first GEN-B, which produced nothing: `unitree_h1v2` has no
loco_mjx model and Atlas/Talos clips do not map onto theirs (Sec 14.11).
BoosterT1 is the one correctly-mapped body `best79` never trained on, so it is
the only honest zero-shot test available.

**LOCAL, 3 topologies** -- `l3t_fix` then `l3t_fix_head` on `LAFAN1_3t` (H1 and
G1 from the original clips, BoosterT1 from the regenerated ones). Viper cannot
run three topologies at all, so this is the only path. Arm 1 changes exactly one
thing against `local_3t_dance4`: T1's reference is now reachable. H1 and G1 read
the same clips they always did and are a built-in null control -- if they move,
something other than the T1 reference changed.

## 15.3 HEADING CONFIRMED ON VIDEO — and what that decides

`w7_ctl_s1` and `w7_head_s1` rendered beside their reference and reviewed by the
operator 2026-08-29: **the heading arm visibly looks better.** That is the first
qualitative check any result on this project has had.

It matters because it resolves a conflict the headline metric cannot. Heading
costs **+14.4 % joint RMSE on H1**, and on the strength of that number alone the
heading arm is the worse policy. But joint angles are body-local, so joint RMSE
is *structurally blind* to facing — dance2_subject4 accumulates 12 058 deg of yaw
and spends 74.8 % of its frames more than 90 deg from its starting heading, and
none of that enters the metric. The +14.4 % is real and the metric is
incomplete; the video is what adjudicates between them.

**Standing rule this gives us:** when the headline metric and the rendered motion
disagree, and the metric is structurally blind to the axis in dispute, the video
wins. The metric is not wrong — it is silent, and silence was being read as
approval.

**Decision: heading ships** in the final recipe (`HEADING_RATIO=0.20`,
`HEADING_TEMP=2.0`, `tracking_clip_observe_root_heading=True`), and the +14.4 %
is reported as its price rather than hidden.

**What this does NOT close.** Only the heading axis was confirmed. The feet, the
ankles, and root drift are untouched by it — and H1's drift is if anything WORSE
with heading on (root travel 1.18 m -> 1.69 m against a reference wanting
0.56 m). BoosterT1 has not been rendered on the fixed reference at all. "We
watched a video" is now true of one axis on two robots, not of the system.
