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

- **The body-independent token closes.** Its gate measured 0.1774 rad against a
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
| **On top of the reference, not instead of it** | the only configuration that ever wins. Token-*replacing* is 1–3 % worse than reference | every M, §2.2f |
| **The reference channel is degraded** | value **rises** — 4.0 % → 14.0 % → 11.6 % as the reference drops 40 → 8 → 4 Hz | §2.2d |
| **The body is randomized** | **no effect at all** — −4.0 % with and without | §2.6b |
| **A second topology is present** | value **collapses to zero**, on both robots | §2.6c |

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
monotonically throughout. This is not a failure to acquire either behaviour — it
is a trade, made twice, against the only term carrying real weight
(`joint_tracking_coeff=30` at temperature 0.05).

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

A **single-topology G1 pair** decides it, and G1-alone has never been run. If it
shows the same ~−4 % the H1-alone arms show, the token works fine on G1 and the
loss is caused by sharing. If it shows ~0, the token never worked on G1 and
"the second topology erases it" was the wrong name for this result.
`M5_g1_ref` / `M5_g1_both` are training.

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
