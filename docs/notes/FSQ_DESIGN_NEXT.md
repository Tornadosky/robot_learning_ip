# How to make FSQ actually pay — written from the 26-08 measurements

Read `experiments/fsq_khaendler/REPORT_FSQ_SCALE.md` first. This note is only
the design conclusion: given what was measured, where can a discrete motion token
earn its place, and how should it be built and tested.

## The framing error that made every campaign a loss

Five campaigns asked the same question: *does the token beat the explicit
reference at tracking a clip we already have?*

That question is rigged, and not by a little. The token is **computed from** the
reference. A compressed derivative of X cannot beat X at delivering X — the data
processing inequality says so before any GPU is booked. The ~2 % we measure is
simply the compression tax, and the only surprising thing is that it is as small
as 2 %.

**So: never again benchmark the token against the reference on a clip that
exists.** Every place FSQ can pay is a place where the reference channel is
*absent, unusable, or too expensive* — not a place where it is sitting right
there and merely being ignored.

## Four places FSQ can actually pay, in order of how convincing they'd be

### 1. As a PRIOR you can sample from — the one the reference cannot enter at all

This is the real reason to want a discrete code, and we have never tested it.

You cannot sample "a plausible 19-dimensional joint trajectory" from nothing.
You *can* fit a small autoregressive model over a 1000-symbol vocabulary and
sample from it — that is the entire reason tokenization won in language and
audio. The product is **motion the robot has never been given**: text-to-motion,
in-painting a gap, continuing a motion past the end of a clip, searching over
codes for a motion that satisfies a constraint.

The control here genuinely cannot be built. There is no "explicit reference"
version of *invent a new dance*, which makes it the strongest available claim —
far stronger than the cross-topology one, where a control (each robot's own
reference) does exist and beat us.

**Cheapest decisive experiment.** The z-only policies already exist and Kevin's
tokenizer reconstructs at 0.05 rad, so the channel is faithful. Fit an n-gram or
a two-layer transformer over the existing `_zq` code stream — minutes of CPU.
Sample a fresh code sequence. Feed it to the trained z-only policy. Then ask two
questions, both measurable: does the robot stay up (alive fraction vs the
zero-action floor), and is the motion *new* (nearest-neighbour distance from
every training clip, in code space and in joint space). If the robot dances
something plausible that is not in the training set, that is a result no
reference-channel experiment can match, and it costs one afternoon.

### 2. As a rate-distortion result, which is a claim we have already half-proved

We measured: **8 tokens/s costs 5–13 %, 4 tokens/s costs ~24 %.**

Reframe that as bandwidth. The explicit reference is 19 floats at 40 Hz ≈ 24
kbit/s. A 1000-code book at 8 Hz is ~80 bit/s. That is **~300x less bandwidth
for ~6 % more tracking error** — and *that* is a sentence a systems person cares
about, where "loses 2 %" is a sentence nobody wants.

**How to report it properly.** Stop reporting fidelity at one rate. Plot error
against bits/second, sweep the codebook size and the hold together, and show
where the knee is. FSQ's job is to sit at a good point on that curve, not to beat
an uncompressed channel. Our data already has three points on it; a proper sweep
is cheap because hold is a config flag.

**Design implication:** the rate floor sits between 4 and 8 codes/s. If a
downstream planner can only emit 4/s, the fix is not a bigger book — it is a
token that spans a longer window (the decoder should output a short *trajectory*
per code, not one frame), so held codes stop being stale.

### 3. Embodiment-agnostic tokens — but fix the FEATURES, not the codebook

This is measured and unambiguous now. Canonical reconstruction against codebook
size, same data, same epochs:

| codes | used | H1 | G1 |
|---|---|---|---|
| 1 000 | 1000 (saturated) | 0.3661 | 0.3402 |
| 12 800 | 9 654 | 0.3501 | 0.3272 |
| 32 768 | 16 711 | 0.3442 | 0.3223 |

32x the codebook buys 6 % less error and the book stops saturating. **Capacity is
not the constraint.** The 22 task-space dimensions — root height, attitude,
velocities, and 4 end-effector positions — do not determine elbow and knee
angles, and no codebook can index information the encoder never saw.

Three repairs, cheapest first, all decided by *reconstruction error alone* with
no RL arm:

- **Add sites.** Go from 4 end-effectors to 8–10 (elbows, knees, chest). One
  tokenizer fit, ~1 h. Gate: reconstruction below 0.10 rad. If task space with
  elbows and knees still cannot reconstruct, the whole task-space idea is dead
  and we stop paying for it.
- **Descriptor-keyed joint tokens.** Encode the joint state itself, but identify
  each joint by URMA's existing 47-dim descriptor rather than by index. The token
  is then a set of (descriptor, value) pairs quantized jointly — per-joint
  information, but body-agnostic because the *descriptor*, not the slot, says
  which joint it is. This is the honest middle between Kevin's design (faithful,
  body-specific) and ours (body-agnostic, uninformative).
- **Coarse token + per-body residual head.** The shared code carries gross
  motion; a small per-body decoder fills in the fine joints. This is close to
  what Kevin's tokenizer already does, and it reconstructs 7x better than ours.

**And do not rerun `M5_2t_canon` with a bigger book.** That is now a known dead
slot.

### 4. As an AUXILIARY channel, not a replacement — the test we never ran

Every experiment so far set `tracking_clip_latent_replaces_reference=True`. That
forces the token to *substitute* for the reference, which is simultaneously the
hardest test and the least useful configuration.

Give the policy **both** and see what the token adds. It is a config change, no
new code. Three outcomes, all informative: the token is redundant (delta 0, and
we stop); it helps robustness when the reference is degraded (drop the reference
to 4 Hz, keep the token at 8 — does the pair beat either alone?); or it helps
generalization to a body the reference is wrong for. That last one is where a
lossy-but-body-agnostic code has a real job: **as a regularizer or a fallback,
not as the primary command**.

## CORRECTION: the window argument WAS tested, and it lost on its merits

An earlier draft of this note claimed the environment short-circuits the window
argument by also handing the policy `reference_phase`, so the reference channel
never has to be ambiguous. **That is wrong, and reading
`environment.py:1055-1081` settles it.** The policy's joint observation is
assembled from exactly: joint position deltas, joint velocities, the previous
action, the keep-nominal flag, then — under `tracking_active` — the reference
delta (only when `tracking_reference_channel_active`), the reference velocity
(only when `tracking_reference_velocity_active`, and `REFVEL_OBS=False` all
night), and the latent block. `reference_phase` is internal state used to look up
the reference. **It is never observed.**

So the comparison was exactly the intended one: an instantaneous per-joint target
with no timeline context, against a token encoding an 11-frame window. The
window's premise held, and the window still bought nothing across a 9x range in
M. That makes the falsification *stronger* than the version in the earlier draft,
and it removes the phase-ablation experiment — there is no phase to ablate.

What remains genuinely untested about the window is its **length**: 11 frames at
100 Hz is 110 ms, which may simply be too short to contain a continuation worth
knowing. That is now running (`tok_la20` → `M9_z_la20_h{1,10}`), and it doubles
as the rate-floor fix below.

## Rules for measuring any of this

- **n ≥ 4 rollout seeds, always.** A single seed moves a score by up to 4.95 % —
  bigger than every effect we have ever argued about. Crossevals are ~15 min;
  budget four per arm before budgeting a second training arm.
- **Report reconstruction error before the RL arm runs.** It cost nothing and it
  correctly predicted the C2 failure in advance.
- **Score against the zero-action floor**, every time. "It tracks" is meaningless
  without a denominator, and the canonical arm looked like a policy right up
  until the floor showed it was worse than doing nothing.
- **Report bits/second next to error.** It is the axis on which the token is
  supposed to win, and we were not measuring it.

## The one-line recommendation

Stop asking the token to replace a reference we already have — it will always
lose by ~2 %, and now we know that number is stable and small. Spend the next
slots on the **generative prior** (sample codes, get motion nobody recorded) and
on the **rate-distortion curve** (300x less bandwidth for 6 % error). Fix the
canonical tokenizer's *features* before ever running another embodiment-agnostic
arm. And put the window argument out of its misery by testing it without the
phase input.
