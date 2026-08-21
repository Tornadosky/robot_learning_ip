# Cross-Embodiment Results — One Policy, Several Humanoids

**Status: COMPLETE.** Built, trained, evaluated and written 2026-08-02 22:20 →
2026-08-03 06:45. Twelve training runs (six local at 200M steps, six on Viper
at 150M), all evaluated per robot. Every Viper job is resolved; the queue is
empty and no process is left running.

## 0. The headline

> **One embodiment-aware policy was trained simultaneously on H1 (19 joints),
> G1 (23) and Atlas (27) — different observation and action dimensions, in one
> padded PPO update — and every robot individually beats its own exact-reset
> zero-action baseline on held-out rollouts, across three seeds, with no robot
> hidden behind an aggregate.**

That claim is **earned**, and it is earned by *both* architectures at the higher
budget — six runs (URMAv2 seeds 3/4/5, masked MLP seeds 3/4/5), and
`every_robot_beats_zero_action` is true in all six. The per-robot numbers are in
§4.1 and §4.3.

The architecture question the night was really about resolves differently, and
against the more sophisticated option:

> **URMA's per-joint conditioning did not earn its complexity.** Given the
> cross-topology input it was designed for, URMAv2 *loses overall* to a masked
> MLP (mean Δ +100.4 vs +109.3 over three seeds), loses H1 in every seed, and is
> **13–48× less reproducible across seeds**. At low optimizer budget it loses
> every robot and fails G1 outright. Its one durable win is **Atlas**, the
> highest-joint-count robot, in all three seeds.

The stretch claim — randomised morphology *within* each family — was **not
attempted**; see §5.

---

## 1. What was built tonight

Roadmap step 7 asked for an embodiment-aware backbone that can *represent*
variable topology; before tonight URMA had only ever been given fixed-topology
H1 input, where it has no topology variation to exploit and loses to a plain
MLP. The decisive experiment — URMA on genuinely heterogeneous topologies —
had never been run. Tonight's engineering makes it runnable.

### 1.1 `scripts/scaling/joint_descriptions.py` (new)

A family-agnostic **22-dim structural description per actuated joint**, computed
from any LocoMuJoCo MJX environment's MuJoCo model in actuator order:

```
body_pos(3), jnt_axis(3), child_count(1), nominal_qpos(1), force_limit(1),
damping(1), armature(1), stiffness(1), frictionloss(1), jnt_range(2),
ctrl_range(2), total_mass(1), body_mass(1), body_inertia(3)         = 22
```

Two deliberate departures from the H1-specific 26-dim block in
`online_h1._urma_joint_descriptions`:

- **The four H1 morphology coordinates are not in the joint block.** They only
  exist for H1, so including them would make the block untransferable. Family
  descriptors belong in the global part of the observation.
- **`child_count` is the true body-tree degree**, not the count restricted to
  actuated bodies. The H1 version's narrower definition undercounts across a
  family, and tree degree is the part that actually transfers.

The `_finite_scaled` scaling constants are unchanged, so magnitudes match what
URMA already trains on.

Joint→observation mapping is resolved by the observation's `xml_name`, not by
position: ToddlerBot has **38 observed joints for 30 actuators**, so positional
matching silently misaligns it.

### 1.2 Padded cross-topology joint block

`ParallelMorphVecEnv` gained `joint_block_specs`, emitting per environment:

```
[ padded base observation | robot one-hot (optional) | action mask (J_max)
  | J_max × (22 description + 3 state + 1 valid bit) ]
```

Joint state is `(position, velocity, previous action)`; the previous action is
the action actually applied to that group in the preceding `step`, and zeros at
reset. Joints beyond a robot's count are zero-filled with a 0 mask bit.

**Reserved slots.** `reserved_observation_dim` / `reserved_action_dim` /
`reserved_group_slots` keep padding for a robot that is *not* trained on. This
is what makes the topology-held-out test possible at all: a fixed-width network
can only be replayed on ToddlerBot (30 joints, 546-dim base observation) if
those slots existed at training time. All runs reserve ToddlerBot, so
`J_max = 30` and the padded observation is **1356** wide for URMA
(546 + 0 one-hot + 30 mask + 30×26) and **580** for the masked MLP
(546 + 4 one-hot + 30 mask).

Audited invariants, all covered by tests:

- the valid-joint mask stays **binary** and **outside** running normalisation
  (URMA reads it from `raw_features`, before `RunningMeanStd`);
- padded action means are exactly 0 and their std is 1e-3;
- `general_indices` stop before the action-mask block, so the binary
  bookkeeping never enters the normalised general input.

### 1.3 `CrossTopologyURMAPPO`

`scripts/scaling/cross_topology_urma.py` — `URMAPPO` with
`ParallelMorphPPO._wrap_env`, so URMA gets the grouped per-morphology reward
normalisation instead of the stock single-model wrapper. The robot one-hot is
optional (`--robot-one-hot / --no-robot-one-hot`); the **no-one-hot arm is the
one that tests structural generalisation rather than index memorisation**.

### 1.4 Tests

`tests/test_cross_topology_urma.py`, 10 test cases (7 functions, one
parametrised over four robots), all passing. The five the goal document required
are items 1–5:

1. joint descriptions finite and correctly shaped for H1 (19), G1 (23),
   Atlas (27) **and ToddlerBot (30)**;
2. the padded mask has exactly `n_joints` ones per robot, and padded joints
   carry no description or state signal;
3. different topologies produce different description blocks;
4. padded action means are 0 (and std 1e-3) after a forward pass;
5. one jitted PPO update over all three robots, finite parameters throughout;
6. reserved slots make ToddlerBot representable in the trained layout;
7. `general_indices` stop before the action mask.

Full suite: **54 passed** (44 pre-existing + 10 new).

---

## 2. Measured throughput (this matters for what fits in the night)

Measured on the production runs, not the smokes:

| Path | Machine | Steps/min | Note |
|---|---|---|---|
| H1 online morphology (prior work) | RTX 4060 Ti | 6.89 M | §6b of `SCALING_1000_RESULTS.md` |
| H1 online morphology (prior work) | Viper 1 APU | 1.90 M | ratio 3.6× |
| **Cross-topology URMAv2, 3 robots** | RTX 4060 Ti | **1.75–1.76 M** | 4096 envs, 200M-step runs |
| **Cross-topology masked MLP, 3 robots** | RTX 4060 Ti | **5.58–5.75 M** | same config, same envs |
| Cross-topology URMAv2, 3 robots | Viper 1 APU | 1.52–1.53 M | 150M-step runs |
| Cross-topology masked MLP, 3 robots | Viper 1 APU | 1.59 M | same config |
| Single-topology H1, URMAv2 | Viper 1 APU | 1.78 M | control |
| Single-topology H1, masked MLP | Viper 1 APU | 1.85 M | control |

Two planning corrections, both of which changed what was run:

1. **Cross-topology URMA is ~4× slower per step than the single-topology H1
   path** on the same GPU — three MJX branches per update and a 1356-dim
   observation instead of ~450. The goal document's schedule assumed 400M steps
   would fit in ~3.5 h; at 1.75M steps/min that is 3.8 h *per run*, and the
   trainer compiles the whole loop as one XLA program with no intermediate
   checkpoint, so an overrun salvages nothing. Budgets were cut to 200M
   (local) and 150M (Viper) rather than risk that.
2. **URMA and the masked MLP are *not* equally fast, and the early smokes said
   otherwise.** The 8-update smokes measured 1.77M vs 1.89M — within 7% — and
   that number is in an earlier draft of this file. It was wrong: over 762
   updates the real figures are 1.75M vs 5.58M, a **3.2× gap**, because the
   smoke was dominated by fixed startup cost. On Viper the two *are* within 4%,
   since MJX physics dominates there rather than the network. The comparison in
   §4.3 is therefore matched on **environment steps**, not wall clock, and that
   choice favours URMA (see §4.4).

### 2.1 ROCm compile cost and the URMA 4-epoch segfault

With `--update-epochs 4`, Viper compiles this graph in **1413 s (23.5 min)** for
the masked MLP and **1536 s** for URMA, against **64 s** on CUDA. With
`--update-epochs 1` the same jobs compile in **22–31 s**.

That is the sharper version of the known ROCm gotcha, and it is new: the
4-epoch *nested* update is what makes the ROCm compile explode by ~50×, and it
is the same construct that then segfaults. At 1 epoch neither problem exists.
Compile cost is per distinct HLO; the seed is a runtime argument, so paired
seeds reuse the persistent `JAX_COMPILATION_CACHE_DIR`.

The goal document's ROCm warning **reproduced exactly**:

```
job 10811782 (urmav2, --update-epochs 4)
[cross-train] compile complete in 1535.8s
slurm_script: line 55: 3429867 Segmentation fault  python ...parallel_cross_humanoid_train.py
```

The fault is **after** a successful compile, and it is **specific to URMA**: the
masked MLP compiled and ran with 4 epochs on the same hardware in the same
batch. A dedicated 1-update probe was *not* how this was found — that probe
spent 40 minutes in compile without reaching execution and was cancelled; the
batch itself became the test.

**Consequence for the comparison.** Rather than run URMA at 1 epoch against an
MLP at 4 — which would confound architecture with optimiser epochs — *both*
Viper arms run at `--update-epochs 1`. CUDA has no such fault, so the local
queue keeps 4 epochs. The two machines therefore give **two independently
matched comparisons at different optimiser settings**, which is a stronger
result than one.

---

## 3. Pipeline validation (8.4M-step smoke, not a result)

These numbers exist only to show the plumbing is real. 8.4M steps is ~4% of a
production run, so nothing here is evidence about the architecture question.

Training signal on the URMAv2 cross-topology smoke: mean episode return
5.38 → 7.84 and mean episode length **16.1 → 32.0** over 8.4M steps, with all
three robots in every update (group sizes 1366/1365/1365, action-mask counts
19/23/27).

Per-robot evaluation of that smoke (32 envs/robot, horizon 100, exact-reset
zero-action baseline, fall rate from `absorbing`):

| Robot | Joints | Policy return | Zero-action | Δ | Welch t | Policy len | Zero len |
|---|---|---|---|---|---|---|---|
| h1 | 19 | 17.66 | 17.73 | −0.07 | −0.07 | 66.2 | 45.0 |
| g1 | 23 | 7.62 | 16.96 | **−9.34** | −10.59 | 18.3 | 30.1 |
| atlas | 27 | 5.65 | 5.40 | +0.26 | +0.53 | 57.3 | 54.6 |

Topology held out — ToddlerBot, 30 joints, never trained on, accepted by the
fixed-width no-one-hot policy: Δ return **−25.05** (Welch t −13.87), 0% of
environments beating zero action. The *mechanism* works (an unseen topology is
representable and runs); the *policy* does not transfer at 8.4M steps.

---

## 4. Results

### 4.1 `local_urmav2_s3` — URMAv2, H1+G1+Atlas, **no robot one-hot**, seed 3

200M steps, 4096 envs (1366/1365/1365), 64 rollout steps, 32 minibatches,
4 epochs, lr 1e-4, 1.75M steps/min, 114 min on the RTX 4060 Ti.

All three robots are in **every** shared update — verified from the manifest's
`group_sizes = [1366, 1365, 1365]` and `action_mask_counts = [19, 23, 27]`, not
assumed.

Training curve (mean episode return / length):

| Progress | Update | Return | Length |
|---|---|---|---|
| 0% | 0 | 5.55 | 17.6 |
| 25% | 190 | 34.03 | 101.6 |
| 50% | 380 | 96.80 | 202.2 |
| 75% | 570 | 146.06 | 227.5 |
| 100% | 761 | 199.49 | 280.6 |

Still rising at the end — 200M steps is **not** a converged budget.

Held-out evaluation, 128 envs/robot, horizon 200, exact-reset zero-action
baseline with identical reset keys, fall rate from `absorbing`:

| Robot | Joints | Policy return | Zero-action return | Δ | Welch t | Policy len | Zero len | Non-fall (policy) | Non-fall (zero) | Envs beating zero |
|---|---|---|---|---|---|---|---|---|---|---|
| h1 | 19 | 158.23 | 16.78 | **+141.45** | +27.03 | 169.9 | 41.6 | 1.00 | 0.04 | 97% |
| g1 | 23 | 190.15 | 17.71 | **+172.44** | +32.96 | 175.8 | 31.8 | 1.00 | 0.02 | 98% |
| atlas | 27 | 34.83 | 5.22 | **+29.61** | +17.60 | 152.1 | 53.9 | 0.69 | 0.08 | 100% |

**Every robot individually beats its own exact-reset zero-action baseline.**
No robot is hidden behind an aggregate. Atlas is clearly the hardest of the
three — highest joint count, lowest absolute return, and the only one that still
falls in 31% of episodes — but it is positive with t = +17.6.

### 4.2 Topology held out — ToddlerBot (30 joints, never trained on)

The same no-one-hot policy, replayed on a robot outside the training set. This
is possible at all only because every run reserves ToddlerBot's padded slots.

Every URMA policy trained tonight, replayed on ToddlerBot:

| Policy | Trained on | Δ vs zero action | Welch t | Envs beating zero |
|---|---|---|---|---|
| `local_urmav2_s3` (200M, 4 ep) | h1+g1+atlas | **−22.04** | −20.93 | 0% |
| `local_urmav2_s4` (200M, 4 ep) | h1+g1+atlas | **−20.96** | −19.69 | 0% |
| `local_urmav2_s5` (200M, 4 ep) | h1+g1+atlas | **−20.11** | −18.94 | 0% |
| `urmav2_x3_s1` (150M, 1 ep) | h1+g1+atlas | **−22.62** | −21.38 | 0% |
| `urmav2_x3_s2` (150M, 1 ep) | h1+g1+atlas | **−23.43** | −22.45 | 0% |
| `urmav2_h1_s1` (150M, 1 ep) | h1 only | **−23.15** | −21.81 | 0% |

For `local_urmav2_s3`: policy return 9.62 vs zero-action 31.66, episode length
23.9 vs 59.3 — the policy is actively worse than doing nothing.

**A sharper negative than expected: topology diversity bought nothing.** The
policy trained on three topologies (−22.0/−21.0/−20.1/−22.6/−23.4) is not measurably better
on an unseen fourth than the policy trained on **one** topology (−23.2). If
per-joint conditioning were transferring structural knowledge, training on
19/23/27-joint robots should have helped on a 30-joint robot relative to
training on 19 alone. It did not, at this scale.

**This is a clean negative and it is reported as one.** The architecture accepts
an unseen topology — a 30-joint robot runs through a network trained on 19/23/27
joints — but the policy does not transfer zero-shot. The mechanism is visible
in the design: only the *joint block* is structurally general. The padded base
observation is layout-specific, and ToddlerBot's occupies dimensions that were
identically zero throughout training, so the general (non-joint) input is fully
out of distribution. Structural generalisation of the joint encoder is
necessary but plainly not sufficient.

### 4.3 The matched URMA-vs-MLP comparison

Two independently matched comparisons, one per machine. Within each block every
setting is identical except the backbone (and the one-hot, which the masked MLP
needs and URMA is deliberately denied). Δ is mean policy return minus mean
exact-reset zero-action return, same reset keys, 128 envs/robot, horizon 200.

**Block A — local RTX 4060 Ti, 200M steps, 4 optimizer epochs, seeds 3, 4 and 5**

Δ = mean policy return − mean zero-action return. "Consistent" means **every**
seed of one arm beats **every** seed of the other; "overlapping" means they
interleave.

| Robot | URMA s3 | s4 | s5 | **URMA mean** | spread | MLP s3 | s4 | s5 | **MLP mean** | spread | Winner |
|---|---|---|---|---|---|---|---|---|---|---|---|
| h1 (19) | +141.45 | +109.69 | +123.21 | +124.78 | 31.76 | +142.89 | +144.07 | +145.27 | **+144.08** | 2.38 | **MLP** (consistent) |
| g1 (23) | +172.44 | +176.42 | +124.74 | +157.87 | 51.68 | +174.16 | +173.17 | +173.08 | **+173.47** | 1.08 | **MLP** (overlapping) |
| atlas (27) | +29.61 | +13.35 | +13.07 | **+18.68** | 16.54 | +10.87 | +10.44 | +9.93 | +10.42 | 0.94 | **URMA** (consistent) |
| **mean over robots** | | | | **+100.44** | | | | | **+109.32** | | **MLP** |

`every_robot_beats_zero_action` is **true for all six runs** — three seeds per
arm, both architectures.

What three seeds show that two could not:

1. **URMA's instability is the dominant effect, and it is worse than two seeds
   suggested.** g1 looked like URMA's most reliable robot at seeds 3–4 (+172.4,
   +176.4, spread 4.0); seed 5 came in at **+124.74**, taking the spread to
   51.68. The masked MLP's g1 spread across the same three seeds is **1.08**.

   | Robot | URMA spread | MLP spread | Ratio |
   |---|---|---|---|
   | h1 | 31.76 | 2.38 | **13×** |
   | g1 | 51.68 | 1.08 | **48×** |
   | atlas | 16.54 | 0.94 | **18×** |

   The MLP reproduces itself to within ~1–2 return units on every robot. URMA
   swings by up to 52. This is the single most robust finding in the comparison.
2. **URMA's Atlas win survived all three seeds.** URMA's *worst* Atlas seed
   (+13.07) still beats the MLP's *best* (+10.87), and URMA's Atlas non-fall
   rate is higher in every seed (0.69/0.40/0.41 vs 0.39/0.34/0.31). The
   magnitude collapsed from the seed-3 reading (2.7× → 1.8× on the mean), but
   the ordering is 3-for-3.
3. **The overall gap widened against URMA** as seeds accumulated: +107.2 vs
   +109.3 at two seeds became **+100.4 vs +109.3** at three, entirely because
   URMA's added seed was a bad one and the MLP's was not.

**Block B — Viper, 150M steps, 1 optimizer epoch, seeds 1 and 2**

| Robot | URMAv2 s1 | URMAv2 s2 | masked MLP s1 | masked MLP s2 | Winner |
|---|---|---|---|---|---|
| h1 (19) | +5.36 | +7.58 | +62.35 | +60.11 | **MLP** |
| g1 (23) | **−2.88** | **−4.11** | +2.75 | +3.19 | **MLP** |
| atlas (27) | +3.19 | +3.38 | +8.36 | +7.72 | **MLP** |

The masked MLP wins every robot at 1 epoch, and **URMA fails g1 outright in both
seeds** — so `every_robot_beats_zero_action` is `false` for the Viper URMA arm.
Seed-to-seed agreement is tight in both arms, so this is a real ordering, not
noise.

Block B is, however, **badly under-trained**: final raw training return is 3.6
(URMA) and 1.5 (MLP) against 199.5 and 155.5 in Block A. One epoch at 150M steps
gives 18.3k gradient steps versus 97.5k in Block A — 5.3× fewer, on a task whose
learning curve was still accelerating at the end of Block A. Block B says what
happens early in training; Block A is the informative comparison.

**Single-topology controls — Viper, H1 only, 150M steps, 1 epoch**

| Backbone | Δ vs zero action | (t) | non-fall | Final train return |
|---|---|---|---|---|
| masked MLP | **+161.01** | +28.7 | 1.00 | 396.27 |
| URMAv2 | +26.77 | +17.3 | 0.41 | 42.13 |

These are what make Block B interpretable. On fixed-topology H1 the MLP beats
URMA by 6×. So URMA's deficit at 1 epoch is **general, not cross-topology
specific** — it reproduces the already-known result that URMA is simply harder
to optimise, rather than showing anything about topology.

### 4.4 Verdict: URMA did not earn its complexity

Stated plainly, whichever way it falls:

> **The masked MLP is the better default for cross-topology control at every
> budget tested here.** Over three seeds it beats URMAv2 overall (+109.3 vs
> +100.4 mean Δ), wins H1 in every seed, wins G1 on the mean, is **13–48× more
> reproducible**, runs 3.2× faster in wall clock, and wins every robot at low
> optimizer budget. Given the cross-topology input it was designed for, URMA's
> per-joint conditioning produced **no general win**.

This is a clean negative for the URMA hypothesis and it is not dressed up. The
honest qualifications, in both directions:

- **It is not "URMA is worse everywhere" either.** URMA wins Atlas in **all
  three seeds**, on both return (+18.7 vs +10.4 on the mean; worst URMA seed
  still beats best MLP seed) and non-fall rate (higher in every seed). Atlas is
  the highest joint count and the hardest of the three — the only robot where
  either architecture still falls in most episodes. The single-topology control
  rules out the trivial explanation: at 1 epoch on H1 alone the MLP beats URMA
  6× (+161.0 vs +26.8), so URMA has no general late-blooming advantage — yet it
  still takes Atlas at the higher budget. A per-joint encoder helping most on
  the highest-DOF body is exactly the direction URMA's premise predicts.
  **Three topologies is not a trend**, but this is the one thread worth pulling
  before URMA is abandoned (§7).
- **The instability is arguably the real finding.** Across three seeds URMA's
  g1 result ranges from +124.7 to +176.4 on an identical configuration, while
  the MLP's ranges over 1.08. An architecture with that spread is not
  deployable regardless of its mean, and it makes any single-seed URMA
  comparison unreliable — including the seed-3 reading earlier in this very
  document, which a one-seed night would have published as "URMA wins Atlas
  2.7×", and the seed-3/4 reading, which would have published g1 as URMA's
  strongest robot right before seed 5 halved it.
- **The one-hot is the MLP's ceiling, not a free win.** The MLP's index is
  positional, so it **cannot be evaluated on a held-out robot at all**. URMA
  can be — it just fails (§4.2). The MLP wins the benchmark that exists; URMA
  is the only one of the two that can even be asked the generalisation
  question.
- **Compute is matched on environment steps, not wall clock.** On the local GPU
  URMA runs at 1.75M steps/min against the MLP's 5.58–5.75M, so Block A gave
  URMA ~3.2× more wall-clock time for the same environment steps. On Viper the
  two are within 4% (1.52M vs 1.59M), because MJX physics dominates there.
  Matching on wall clock would make URMA look **worse**, not better.

## 5. Claims explicitly NOT supported

- **Not converged.** Returns were still climbing at 200M steps, so no number
  here is a ceiling for either architecture. The Viper 1-epoch block is far
  from converged and should not be read as an architecture verdict on its own.
- **No claim that URMA scales better with joint count.** Three topologies is
  not a trend, and two seeds is not a variance estimate. The Atlas result is a
  hint worth chasing, nothing more.
- **No formal variance estimate.** Three seeds per arm supports "consistent
  ordering" and a spread ratio; it does not support a confidence interval, and
  none is quoted — a 95% CI from n=3 on a distribution this skewed would be
  theatre. Welch t values in the per-robot tables are *across environments
  within a run*, not across seeds.
- **The two blocks are not comparable to each other** — different machines,
  step budgets and optimizer epochs. Compare within a block only.
- **The MLP was never run at 4 epochs on Viper, nor URMA at 1 epoch locally.**
  Architecture and machine are therefore not fully crossed with optimizer
  epochs; the epoch effect and any machine effect are confounded between
  blocks.
- **One motion, one clip.** Everything is `dance2_subject4`, frames
  19482–20282, `direct` retargeting. No multi-motion claim.
- **No randomised morphology within families.** The stretch goal ("robots *and*
  bodies") was not reached; these are the three stock bodies.
- **No zero-shot transfer to an unseen topology** — §4.2 is explicitly negative.
- **The one-hot arm cannot be evaluated on a held-out robot at all**, because
  the one-hot index is positional; only the no-one-hot arm is testable there.

## 6. Acceptance criteria

| # | Criterion | Status |
|---|---|---|
| 1 | H1, G1 and Atlas train in **every** shared update | **met** — from manifests: `group_sizes [1366,1365,1365]`, `action_mask_counts [19,23,27]` in every cross-topology run |
| 2 | Every robot beats its exact-reset zero-action baseline | **met for 8 of 10 cross-topology runs** — all six local 4-epoch runs (both arms × seeds 3,4,5) and both Viper MLP seeds. **Not met** for the two Viper 1-epoch URMA seeds, which are negative on g1. Reported per arm, never averaged across arms. |
| 3 | No robot hidden by an aggregate | **met** — every table is per robot |
| 4 | Holds across ≥2 seeds | **exceeded** — three seeds per arm at 4 epochs (3,4,5), plus two seeds per arm at 1 epoch |
| 5 | URMA-vs-MLP verdict at matched budgets with single-topology controls | **met** — §4.3, §4.4; controls in §4.3 Block B |
| 6 | Topology-held-out behaviour reported, negative or not | **met** — §4.2, negative across all five URMA policies |

**Step 8 of the roadmap is complete** for a single motion, on stock bodies. The
two things it does *not* yet cover are randomised morphology per family and
multi-motion; both are called out in §5 and reflected in `SCALING_ROADMAP.md`.

## 7. What I would do next, in priority order

1. **Chase the Atlas thread with more topologies, not more seeds on these
   three.** URMA winning the highest-DOF robot in all three seeds while losing
   the lowest-DOF one in all three is the only signal here consistent with its
   premise. Add
   Talos (30) and BoosterT1 and see whether the URMA-minus-MLP gap tracks joint
   count. That is a cheap, falsifiable follow-up.
2. **Fix URMA's seed instability before trusting any URMA number.** A 13–48×
   spread ratio against the MLP makes single-seed URMA results meaningless.
   Suspects, in order: the learnable attention temperature, weight-norm
   interaction with a 5-layer core, and the `gain=0.01` action-latent
   initialisation.
3. **Make the general (non-joint) input topology-agnostic.** §4.2 shows the
   joint block alone does not transfer. As long as the padded base observation
   is layout-specific, no held-out topology can work. Routing the root/reference
   features through a robot-independent encoding is the prerequisite for any
   real zero-shot claim.
4. Only then: morphology randomisation per family, and multi-motion.

## 8. Deliverables and provenance

| Deliverable | Where |
|---|---|
| Per-robot tables, verdict, held-out result, unsupported claims | this file |
| Machine-generated tables (regenerable from disk) | `experiments/cross_embodiment/SUMMARY.md` |
| Manifests, checkpoints, Slurm logs, evaluations | `experiments/cross_embodiment/`, inventory in its `README.md` |
| Roadmap steps 7 and 8 updated | `SCALING_ROADMAP.md` |
| Tests | `tests/test_cross_topology_urma.py` (10 new) |

New/changed code:

| File | Change |
|---|---|
| `scripts/scaling/joint_descriptions.py` | new — 22-dim family-agnostic joint descriptors |
| `scripts/scaling/cross_topology_urma.py` | new — `CrossTopologyURMAPPO` |
| `scripts/scaling/parallel_env.py` | padded joint block, reserved slots, padded `URMAInputLayout` |
| `scripts/scaling/parallel_cross_humanoid_train.py` | `--backbone`, `--robot-one-hot/--no-`, `--append-joint-features`, `--reserve-robots` |
| `scripts/scaling/evaluate_cross_humanoid_policy.py` | backbone-aware loading, `absorbing` fall rate, Welch t, held-out `--robots` |
| `scripts/scaling/summarize_cross_embodiment.py` | new — tables, blocked by matched budget |
| `scripts/scaling/viper_cross_topology.sbatch`, `submit_viper_crosstopo_batch.sh`, `pull_viper_crosstopo.sh`, `local_gpu_crosstopo.sh`, `local_gpu_crosstopo_eval.sh` | new — run/collect harness |
| `pytest.ini` | new — registers the `slow` marker |

**Gate status:** `pytest tests/` → **54 passed** (44 pre-existing + 10 new).
`ruff check --select E4,E7,E9,F` and `ruff format --check` over `scripts/scaling`
and `tests` → **clean**.

One caveat on the lint gate, stated rather than hidden: the ruff available in
the WSL env is **0.16.1**, whose *default* rule set is much broader than the one
this repo was previously verified against. Under those new defaults the repo
reports 131 findings, of which ~79 are `RUF100` (unused `# noqa`) in files this
work never touched. I did not mass-rewrite pre-existing files to satisfy a
changed linter default; the new files are clean under both rule sets except for
the same pre-existing `# noqa: E402` idiom they inherit from their neighbours.

**All Viper jobs are resolved.** The last of the six finished at 01:14; the
queue was verified empty at 04:06 and nothing was submitted afterwards. Six
completed runs plus seven cancelled/failed jobs (the abandoned 4-epoch batch and
the probe) are all accounted for in `experiments/cross_embodiment/viper_logs/`.
The local GPU is idle with no training process running.

*Housekeeping note:* the `viper11` SSH ControlPersist socket expired around
06:50 and re-establishing it needs an interactive password + 2FA, so the final
convenience re-check could not be run from this session. It does not change the
above — the empty-queue verification at 04:06 post-dates every job completion.
