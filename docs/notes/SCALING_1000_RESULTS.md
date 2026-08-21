# 1,000-Robot Scaling Results

**Status: in progress.** This file is written as the experiments land. Sections
marked *pending* have jobs running or queued; nothing in them is claimed yet.

Continuation of [`SCALING_CONTINUATION_GOAL.md`](SCALING_CONTINUATION_GOAL.md),
which continues [`SCALING_ROADMAP.md`](SCALING_ROADMAP.md). Prior measurements
are in [`SCALING_PROGRESS_2026-08-01.md`](SCALING_PROGRESS_2026-08-01.md).

---

## 1. What a "robot" is here

1,000 robots means 1,000 **fixed, reproducible embodiment records** in one H1
topology family, not 1,000 topologies and not 1,000 unreconstructable random
draws. Each record has a stable integer body ID, a unique four-coordinate
morphology descriptor, and a deterministic mapping to distinct MJX model arrays
inside a single compiled graph.

`scripts/scaling/embodiment_catalog.py` (schema 1, generator revision
`h1-online-morph-4coord-v1`) serializes: schema and generator revision, family,
topology signature, RNG seed, sampling method, raw descriptors, split, morphology
bounds, body IDs, a SHA-256 content hash, and a validity summary. Loading
recomputes the hash and refuses a tampered file.

### Frozen catalogs

Generated once by `scripts/scaling/make_catalogs.py`, before any training, into
`experiments/scaling_1000/catalogs/`. Topology signature for all of them is
`nq=26,nv=25,nu=19,njnt=20,nbody=21,ngeom=39,nmesh=17,nsite=10`.

| Catalog | Bodies | Split | Sampling | Seed | Content hash (12) |
|---|---:|---|---|---:|---|
| `train_1000` | 1000 | train | latin_hypercube | 1000001 | `169ef428d2bb` |
| `validation_128` | 128 | validation | latin_hypercube | 5000001 | `01b73332ee73` |
| `iid_256` | 256 | test | latin_hypercube | 2000001 | `46b7ed738d66` |
| `boundary_256` | 256 | test | boundary | 3000001 | `2f6213566c6d` |
| `ood_128` | 128 | ood | extrapolation | 4000001 | `f3ebe9d47a38` |

`train_2` and `train_16` are nested prefixes of `train_1000` used as the AMP
scale ladder's lower rungs.

Morphology bounds are the recorded four coordinates: leg, arm and shoulder
length scales in [0.85, 1.20] and torso mass scale in [0.70, 1.50].

All 1,000 training descriptors are unique at 1e-9. Boundary bodies pin at least
one coordinate into a thin band at a face, edge or corner (every body has
max |normalized coordinate| > 0.9). OOD bodies place **exactly one** coordinate
outside its training bound, all scales positive.

### Physical audit — the bodies really differ

`scripts/scaling/audit_catalog_bodies.py` checks the MJX arrays, not just the
descriptors, on representative plus most-extreme entries of each catalog.

| Catalog | Audited | Distinct mass+offset arrays | Kinematic leg length (m) | Total mass (kg) | Invariants |
|---|---:|---|---|---|---|
| `train_1000` | 47 | 47/47 | 0.680 – 0.960 | 45.34 – 61.83 | all hold |
| `iid_256` | 47 | 47/47 | 0.681 – 0.959 | 45.86 – 61.82 | all hold |
| `boundary_256` | 47 | 47/47 | 0.680 – 0.960 | 43.96 – 63.13 | all hold |
| `ood_128` | 47 | 47/47 | **0.541 – 1.093** | 43.27 – 68.31 | all hold |

Invariants checked per body: positive body masses and inertias, ordered joint
ranges, finite `body_pos`/`body_ipos`/`body_mass`/`body_inertia`/`site_pos`, and
finite reset observations, step observations and step rewards.

**Stated limitation.** Collision and visual meshes are the shared nominal H1
meshes. This is kinematic and inertial diversity, not mesh-level morphology
diversity, and it must not be described as the latter.

---

## 2. Systems proof — 1,000 bodies × 200 replicas (capacity only)

Slurm job `10803153` on `vipa1170`, exit 0.

| Quantity | Value |
|---|---:|
| Catalog bodies | 1,000 |
| Replicas per body | 200 |
| Total environments | 200,000 |
| Rollout steps | 16 |
| PPO updates | 10 |
| Gradient minibatch updates | 1,280 |
| Total steps | 32,000,000 |
| **Exposure per body (min / max)** | **200 / 200** |
| Bodies with zero exposure | 0 |
| XLA compile | 988.4 s |
| Train | 1,014.9 s |
| Throughput | 1.892 M steps/min |
| Peak live device memory | 12.64 GiB (82.50 GiB budget) |
| Host MaxRSS | 14.20 GB |
| Wall clock | 43:30 |

Balance is exact by construction (`slot % num_bodies` with 200,000 = 200 × 1,000)
and was additionally verified on device: assignment matched the schedule,
descriptors matched the catalog rows, and each reset observation's descriptor
channel matched its assigned body.

**`capacity_only = true`.** Ten updates is a throughput and memory result. Its
final training return (0.334) and length (42.6) are *not* a learned
1,000-robot controller and are not compared with quality runs.

### Failures archived — the standalone slot-reset graph is unstable on ROCm

Two independent failures, both inside a **standalone**
`jit(vmap(mjx_reset_with_slot))` executed before training as a pre-flight check.
This graph is not part of training; the identical reset runs correctly *inside*
the PPO training graph, which is how job 10803153 completed.

| Job | Slots checked | Symptom |
|---|---:|---|
| 10803152 | 8,000 | aborted: `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`, preceded by `gpusolverDnCreate` hipSolver error |
| 10806775-77 | 2,048 | compiled successfully (cache written after ~2 min), then **hung in execution** at 99.8% CPU for 1h50m with no further output; cancelled |
| 10803153 | 200,000 | succeeded, 579.7 s |

Size does not predict the outcome, so this is a flaky runtime interaction rather
than a capacity limit. The device check is now **off by default**
(`--catalog-check-envs 0`). What replaces it:

- exposure is exact by construction (`slot % num_bodies`) and reported
  analytically in every manifest;
- `tests/test_online_h1.py` verifies on CPU that slot assignment is exactly
  balanced, that descriptors match the catalog rows, and that each reset
  observation's descriptor channel describes its assigned body;
- job 10803153 already verified all of the above on device at 200,000 slots.

Slurm outputs archived under `experiments/scaling_1000/system_1000x200/`.
Jobs 10806775-77 were re-submitted as 10807503-05 with the check disabled.

---

## 3. Quality-scaled 1,000-body walk training — *pending*

Pre-registered before launch, six concurrent one-APU jobs:

- clip `walk1_subject1`; 8,000 environments; 64 rollout steps; 32 minibatches;
  4 PPO epochs; descriptor-conditioned MLP `[256, 128]`; lr 1e-4; init std 0.2;
  300,000,000 steps (586 updates, 75,008 gradient minibatch updates);
- arm A `fixed_balanced` on `train_1000` — 8 replicas per body, exposure exactly
  8/8 — seeds 1, 2, 3 (jobs 10806775-77);
- arm B matched `continuous` control with per-episode resampling, identical
  budgets and bounds, seeds 1, 2, 3 (jobs 10806778-80).

The step budget is 3× the roadmap floor because the recorded 100M-step
continuous run was still far from convergence (final length 99.4); 300M fits one
8-hour job at the measured 1.9 M steps/min.

Acceptance criteria are the roadmap's §2D list and are evaluated in section 5.

### Arm B — continuous control, complete

All three seeds finished with exit 0, no NaNs, 299,520,000 steps each,
585 updates, 74,880 gradient minibatch updates, 1.90 M steps/min,
2.01 GiB peak live device memory, 2:55 wall clock.

| Seed | Job | Return (last 10 updates) | Length (last 10 updates) | Final length | Body exposures |
|---:|---:|---:|---:|---:|---:|
| 1 | 10806778 | 66.89 | 731.9 | 747.2 | 1,415,026 |
| 2 | 10806779 | 69.00 | 747.3 | 736.7 | 1,429,459 |
| 3 | 10806780 | 66.36 | 726.2 | 728.8 | 1,421,938 |

Seed agreement is tight (length spread 726–747, ~3%).

**The 3× step budget was the decisive factor.** Against the recorded 100M-step
continuous run on the same environment family:

| Budget | Training return | Training episode length |
|---|---:|---:|
| 100M (recorded baseline) | 40.58 | 99.4 |
| 300M (this run, 3-seed mean) | 67.4 | 735 |

Quarter-point length values for seed 1 are 34 → 173 → 429 → 614 → 747, so the
curve was **still rising at 300M** and this is not a convergence claim. Training
episode length is also not held-out control quality; see section 5.

---

## 4. Online morphology × multi-motion AMP — bridge implemented

`scripts/scaling/online_amp.py` and `online_amp_train.py`. Body, motion clip and
motion phase are all chosen at reset inside **one** H1 graph: the body by the
catalog schedule, the clip and phase by LocoMuJoCo's trajectory handler. There is
no static body × motion branch product — `static_branches = 1` in the manifest.

### Discriminator leakage guard

Expert transitions come from one canonical body and are 434 columns wide; policy
observations are 438 because the online environment appends the morphology
descriptor. Padding the expert to 438 makes the descriptor channel a **constant**
that separates expert from policy perfectly — the exact failure the roadmap warns
about.

`DescriptorBlindNet` drops those columns before the running mean/std layer and
before any weights, so no gradient path and no normalisation statistic touches
them. The manifest records a runtime check: perturbing only the descriptor
columns changes the discriminator logits by exactly 0.0.
`--no-blind-discriminator` keeps the leaky path available as a negative control.

`tests/test_online_amp.py` covers the blind case, the constant-padding case, the
unblinded negative control, and that normalisation statistics span only the kept
columns.

### Scale ladder

| Rung | Bodies × motions | Environments | Status |
|---|---|---:|---|
| 1 | 2 × 2 | 32 | **passed** (local, leakage check 0.0) |
| 2 | 16 × 4 | 1,024 | **ran, but the AMP objective was inert — see below** |
| 2r | 16 × 4 | 1,024 | rerunning with two corrections (jobs 10808055/56) |
| 3 | 1,000 × 4 | ~8,000 | gated on 2r |

### Rung 2r — with a non-saturating discriminator, AMP does contribute

Matched pair, joint-space reward in both, 99,942,400 steps each, ~1.50 M steps/min,
leakage check 0.0 in both.

| Arm | Discriminator | Final logits (policy / expert) | AMP style reward | Episode length (last 10) |
|---|---|---|---:|---:|
| gentle (10808055) | 2 epochs, lr 1e-5 | **-0.08 / +0.48** | **≈0.71** | **451.3** |
| strong (10808056) | 10 epochs, lr 5e-5 | -1.00 / +1.00 | **0.00** | 413.3 |

The style reward is `max(0, 1 - 0.25 (logits - 1)^2)`, so the strong arm's
saturated discriminator contributes exactly nothing and that arm is a pure
environment-reward run. This pair is therefore a controlled AMP-active vs
AMP-inert contrast with every other factor matched: **the adversarial term is
worth about +9% episode length** (451.3 vs 413.3), and it is only available when
the discriminator is prevented from winning.

### Rung 2r evaluation — AMP is *not* what breaks the plateau

`scripts/scaling/evaluate_online_amp_policy.py`, 16 bodies × 16 replicas = 256
environments, horizon 1,000, deterministic actions, held-out clip
`dance2_subject5`.

| Arm | Train motions: length | non-fall | Held-out: length | non-fall | Beats zero |
|---|---:|---:|---:|---:|---:|
| gentle (AMP active) | **420.2** | **0.609** | 239.6 | 0.230 | 100% |
| strong (AMP inert) | 413.0 | 0.566 | **328.7** | **0.266** | 100% |
| zero action | — | 0.023 | — | 0.031 | — |

Both arms clear the registered thresholds on training clips (length > 250,
non-fall > 0.10) and both beat zero action on every body. But the arm whose style
reward is identically zero clears them too — so **the registered thresholds are
met without AMP contributing anything**. On the held-out motion the AMP-active
arm is actually *worse* (239.6 vs 328.7).

The honest conclusion is the opposite of the earlier training-curve reading: AMP
is worth about +2% train episode length here, not +9%, and it costs held-out
motion transfer. Whatever lifted these runs from the recorded ~140-150 four-clip
plateau to ~415, it is not the adversarial objective. The two remaining
candidates, both changed relative to the recorded plateau runs, are the
joint-space-only reward and morphology randomisation acting as augmentation.

### CORRECTION — the matched control refutes the reward explanation

An earlier version of this section attributed the plateau break mainly to the
joint-space reward. **A matched control refutes that.** Running the *original
full-mimic reward* on 1 body × 4 clips at the identical 100M budget reaches
episode length **400.2** — higher than the joint-space version's 366.9, not
lower.

| Configuration | Bodies | Reward | AMP live | Length |
|---|---:|---|---|---:|
| Recorded 4-clip plateau (unmatched) | 1 | full mimic | n/a | ~140–150 |
| Ablation (matched control) | 1 | **full mimic** | no | **400.2** |
| Job 10808498 | 1 | joint-space | no | 366.9 |
| Job 10808056 | 16 | joint-space | no | 413.3 |
| Jobs 10808055/452 | 16 | joint-space | yes | 451.3 / 444.1 |
| AMP rung 3 (local) | **1000** | joint-space | yes | **487.1 / 484.2** |

So none of the three candidates explains the gap on its own:

- **not the reward** — full-mimic reaches 400.2 in this setup;
- **not morphology randomisation** — a single body already reaches 400;
- **not AMP** — the discriminator-saturated arm reaches 413.

What remains is that the **recorded ~140–150 plateau is not a matched baseline**.
It came from a different trainer, environment configuration and step budget. Any
of the modern environment's properties — random-phase reference-state
initialisation from the trajectory handler, the current termination criterion,
observation construction — could account for it, and this continuation has not
isolated which. The honest statement is:

> In the current environment, one policy reaches ~400–490 mean episode length on
> four dance clips, well above the previously recorded ~140–150. The earlier
> figure was produced by a different setup and is not a controlled comparison, so
> the improvement cannot be attributed to any single change made here.

The lesson is that the plateau was treated as a property of DeepMimic-style
tracking when it may have been a property of one particular configuration.

### Superseded decomposition (kept for the record)

Job 10808498 settles it: **1 body × 4 clips**, joint-space reward, discriminator
saturated (style reward 0), reaches episode length **366.9**. With one body and
no working adversarial term it is already 2.4× the recorded plateau.

| Configuration | Bodies | Reward | AMP contributing | Episode length |
|---|---:|---|---|---:|
| Recorded 4-clip DeepMimic plateau | 1 | full mimic | n/a | ~140–150 |
| Job 10808498 | 1 | joint-space | no | **366.9** |
| Job 10808056 (strong) | 16 | joint-space | no | 413.3 |
| Jobs 10808055/452 (gentle) | 16 | joint-space | yes | 451.3 / 444.1 |

Attribution, largest first:

1. **Reward change** (dropping site/root terms): ~140–150 → 367, by far dominant;
2. **Morphology randomisation** (1 → 16 bodies): 367 → 413, ~+13%, i.e. body
   variety acts as useful augmentation;
3. **AMP** (working discriminator): 413 → ~448, ~+9% on training clips, and
   *negative* on held-out motion.

Caveat: the ~140–150 plateau figure comes from earlier runs with a different step
budget and configuration, so row 1 is a strong indication rather than a matched
ablation. Rows 2-4 are matched.

The multi-motion plateau is therefore best explained as a **reward-specification
artefact**, not a limit of DeepMimic-style tracking. Site and root position terms
are computed against a reference body's proportions; removing them removes a
penalty the policy could not satisfy.

### AMP gentle, both seeds

| Seed | Train length | Train non-fall | Held-out length | Held-out non-fall | Beats zero |
|---:|---:|---:|---:|---:|---:|
| 1 | 420.2 | 0.609 | 239.6 | 0.230 | 100% |
| 2 | 428.5 | 0.680 | 280.4 | 0.391 | 100% |

Reproducible, including the non-saturated discriminator (-0.07/+0.48 both seeds).

### Per-motion, where the aggregate hides the most

| Arm | subject1 | subject2 | subject3 | subject4 |
|---|---:|---:|---:|---:|
| gentle non-fall | 0.634 | **0.197** | 0.631 | 0.983 |
| strong non-fall | **0.155** | 0.295 | 0.877 | 1.000 |

Per-clip competence is wildly uneven and the two arms fail on *different* clips.
A single mean over four clips would have concealed both facts.

### Rung 3 — 1,000 bodies × 4 motions, one branch

Trained locally, 8,000 environments, 99,840,000 steps, 6.9 M steps/min, two seeds.
**4,000 body-motion cells through a single static XLA branch**, discriminator
unsaturated throughout (policy 0.03 / expert 0.07, so the style reward stayed
live), leakage check 0.0 in both seeds.

Held-out evaluation, 8 replicas per body, horizon 1,000, deterministic actions:

| Seed | Train motions: length | non-fall | Held-out `dance2_subject5`: length | non-fall | Beats zero |
|---:|---:|---:|---:|---:|---:|
| 1 | 279.6 | 0.460 | 244.4 | 0.153 | 100% |
| 2 | **432.0** | **0.683** | 258.5 | 0.345 | 100% |
| zero action | — | 0.038 | — | 0.042 | — |

Per motion (seed 2 / seed 1 non-fall): `subject1` 0.814/0.690, `subject2`
0.481/0.364, `subject3` 0.887/0.538, `subject4` 0.558/0.257. Clip difficulty
ordering is consistent across seeds even though the levels are not.

Two honest caveats:

- **Seed variance is large.** Training curves were nearly identical (487.1 vs
  484.2) yet evaluation differs by 54% (279.6 vs 432.0). Training episode length
  is measured with stochastic actions over resampled bodies; evaluation is a
  deterministic first episode on fixed bodies. The two seeds agree on the
  threshold verdict but not on the magnitude, so no precise number should be
  quoted from a single seed.
- **The per-body breakdown is over *training* bodies**, not held-out ones: the
  AMP evaluator reads the training catalog from the manifest. It satisfies the
  registered per-body requirement, but the body-generalisation question for AMP
  is untested. Only the *motion* axis is held out here.

### Registered §4C status

Evaluated at the top rung (1,000 bodies × 4 motions, two seeds):

| Criterion | Threshold | Status |
|---|---|---|
| Mean episode length across training clips | > 250 | **met** (279.6 / 432.0) |
| Non-fall rate | > 0.10 | **met** (0.460 / 0.683) |
| Held-out motion evaluated | required | **done** (244.4 / 258.5, non-fall 0.153 / 0.345) |
| Per-body and per-motion breakdown | required | **done** (over training bodies) |
| At least two seeds | required | **done** |
| Plateau attributable to AMP | implied | **not met — the controls refute it** |

Every registered numeric threshold is cleared. The claim that this *breaks the
multi-motion plateau* is still **not** supported, because the ~140–150 baseline
was produced by a different setup and the matched controls show the gain does not
come from AMP, the reward change, or morphology randomisation individually.

The length curve also peaked near 484 mid-run and settled at 451, so training was
not monotone.

### Rung 2 (job 10807518) — do not read the episode length as an AMP success

The run completed cleanly: 99,942,400 steps, 1,525 updates, 1.50 M steps/min,
64 body × motion cells through **one** static branch, leakage check 0.0. Final
mean episode length was **432.7**, well above the recorded four-clip DeepMimic
plateau of ~140–150.

That comparison does not survive inspection. The discriminator curves are:

| | first | mid | last 10 |
|---|---:|---:|---:|
| discriminator on policy | -0.00 | -0.99 | **-1.00** |
| discriminator on expert | 0.01 | 0.99 | **1.00** |

The discriminator saturated to a perfect separation. AMP's style reward is
`max(0, 1 - 0.25 (logits - 1)^2)`, which at `logits = -1` is exactly **0**. So
for most of training the adversarial term contributed nothing and the policy was
driven by the environment tracking reward alone. The episode length is a
DeepMimic result wearing an AMP label; it is **not** evidence that AMP breaks the
multi-motion plateau. Per the roadmap's own rule — do not declare success from
discriminator behaviour alone — the converse also applies here.

A second defect: this run used LocoMuJoCo's default mimic reward, which includes
site and root-position terms. Those assume the reference body's limb lengths, so
across online morphologies they penalise a body for being a different size. The
1,000-body walk runs deliberately use the joint-space subset
(`qpos 0.6 / qvel 0.4`, all site and root weights 0); the AMP path now does too.

Rung 2 is therefore rerun with both corrections and a matched pair that changes
exactly one factor:

| Job | Reward | Discriminator |
|---|---|---|
| 10808055 | joint-space only | gentle: 2 epochs, lr 1e-5 |
| 10808056 | joint-space only | as before: 10 epochs, lr 5e-5 |

If saturation persists in both, the next levers are discriminator gradient
penalty and reference-state initialisation, not more steps.

Registered AMP acceptance thresholds (recorded DeepMimic four-clip plateau is
~140–150 mean episode length at very low non-fall rate): mean episode length
> 250 across training clips, non-fall rate > 0.10, held-out motion evaluation,
per-body and per-motion breakdown, and at least two seeds before calling the
plateau broken.

---

## 5. Held-out evaluation

`scripts/scaling/evaluate_catalog_policy.py` evaluates every arm on the frozen
catalogs with the same bodies, reset keys, initial phases and horizon, because
bodies are assigned by environment slot rather than drawn randomly. It refuses to
report if the policy and zero-action arms saw different bodies or if the
assignment deviated from the balanced schedule. Output includes per-body rows,
not only aggregates.

### How to read episode length here — it is doubly censored

An episode ends for three different reasons, and only one is a failure:

- the **1,000-step environment horizon** is reached (`done`, not `absorbing`);
- the **reference clip runs out** — `loco-mujoco/loco_mujoco/environments/base.py:154`
  sets `done` when `subtraj_step_no >= len_trajectory - 1`, again without
  `absorbing`. The clip is 3,000 control steps at 100 Hz and the initial phase is
  uniform over it, so a third of episodes have fewer than 1,000 steps of
  reference left before they start;
- the robot **falls** (`absorbing`).

Predicted fraction reaching the horizon from the phase distribution alone is
2/3 = 0.667; measured is 0.657 on IID. So mean and median episode length
*understate* stability, and **non-fall rate is the honest stability metric**.

An earlier version of this evaluator reported `non_fall_rate` from `done` rather
than `absorbing`. Because LocoMuJoCo defines `done = absorbing or step >= horizon`
and the eval horizon equalled the env horizon, that metric was 0.000 by
construction. It is fixed; the affected numbers were never published outside this
file's draft.

### Arm B — continuous control, 3 seeds, deterministic actions, horizon 1,000

Aggregated across seeds 1-3 (mean; per-seed spread was under 2% everywhere).

| Catalog | Return | Zero-action return | Median length | **Non-fall** | Zero non-fall | Reached horizon | Bodies beating zero | Max abs morphology corr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| IID-256 | 188.5 | 0.97 | 788.9 | **0.941** | 0.015 | 0.657 | **100%** | 0.089 |
| Boundary-256 | 184.6 | 0.98 | 785.8 | **0.937** | 0.019 | 0.653 | **100%** | 0.139 |
| Validation-128 | 187.0 | 0.96 | 802.6 | **0.942** | 0.020 | 0.646 | **100%** | 0.123 |
| OOD-128 | 178.5 | 1.01 | 773.2 | **0.908** | 0.016 | 0.620 | **100%** | 0.360 |

Against the roadmap's §2D thresholds, for this arm:

| Criterion | Threshold | Result |
|---|---|---|
| All seeds finish without NaNs or dropped bodies | required | met |
| IID bodies beating zero-action return | ≥ 90% | **100%** |
| Boundary bodies beating zero action | ≥ 75% | **100%** |
| No single coordinate explains the result | \|corr\| < 0.25 | met in-distribution (≤ 0.139) |
| OOD reported honestly | required | reported; 100% beat zero, non-fall 0.908 |
| Aggregated across seeds with uncertainty | required | met |

The single coordinate-correlation exceedance is OOD (0.360), which is expected
by construction: extrapolation bodies push exactly one coordinate outside its
bound at a time, so return necessarily correlates with the pushed axis.

### Arm A vs Arm B — the controlled catalog loses, clearly and reproducibly

Same bounds, budgets, network, optimiser, seeds and evaluation. Arm A trains on
the 1,000 fixed catalog bodies (exposure exactly 8/8, zero unexposed); Arm B
resamples freely from the same bounds and saw ~1.42 M distinct bodies.

| Catalog | Arm | Return | Median length | **Non-fall** | Beats zero |
|---|---|---:|---:|---:|---:|
| IID-256 | fixed | 153.2 ± 10.2 | 673.9 | 0.678 | 100% |
| IID-256 | continuous | **188.5 ± 2.2** | **788.9** | **0.941** | 100% |
| Boundary-256 | fixed | 149.0 ± 10.8 | 671.7 | 0.666 | 100% |
| Boundary-256 | continuous | **184.6 ± 2.2** | **785.8** | **0.937** | 100% |
| Validation-128 | fixed | 150.8 ± 8.0 | 662.0 | 0.660 | 100% |
| Validation-128 | continuous | **187.0 ± 1.9** | **802.6** | **0.942** | 100% |
| OOD-128 | fixed | 145.3 ± 9.1 | 661.6 | 0.644 | 100% |
| OOD-128 | continuous | **178.5 ± 2.3** | **773.2** | **0.908** | 100% |

The gap is ~19% of return and ~15% of median length on every catalog, and it is
far outside seed noise:

| Catalog | Difference (fixed - continuous) | SE | t | 95% CI |
|---|---:|---:|---:|---|
| IID-256 | -35.3 | 6.02 | -5.9 | [-47.1, -23.5] |
| Boundary-256 | -35.6 | 6.33 | -5.6 | [-48.0, -23.2] |
| OOD-128 | -33.2 | 5.42 | -6.1 | [-43.9, -22.6] |

The fixed arm is also ~4-5× more seed-variable (± 8-11 vs ± 2).

**§2D verdict: the "within 10% of the matched continuous arm" criterion FAILS,
and the failure is explained rather than noise.** The controlled 1,000-body
curriculum is genuinely worse than uncontrolled resampling over the same bounds.
The plausible mechanism is coverage: 1,000 bodies versus ~1.42 M. This agrees
with the repository's earlier finding that a held-out morphology gap closes
through broader training coverage rather than through conditioning.

### Per-body, so no robot hides behind an average

Seed 1, worst case over individual bodies:

| Arm | Catalog | Min per-body length | Bodies with non-fall < 0.5 | Bodies not beating zero |
|---|---|---:|---:|---:|
| fixed | IID-256 | 211.2 | 42 / 256 | **0** |
| fixed | Boundary-256 | 194.0 | 48 / 256 | **0** |
| fixed | OOD-128 | 57.5 | 31 / 128 | **0** |
| continuous | IID-256 | 209.8 | **0 / 256** | **0** |
| continuous | Boundary-256 | 212.5 | **2 / 256** | **0** |
| continuous | OOD-128 | 52.0 | **3 / 128** | **0** |

The aggregate was hiding something real for the fixed arm: 16-24% of its bodies
fall more often than not, while the continuous arm has almost none. Both arms
still beat zero action on **every single body of every catalog**.

### Named static XML variants — where generalisation actually stops

10 of the 11 `generated_variants` H1 morphologies, 64 replicas each.
`big_feet` is excluded by name: it differs from nominal only in foot geometry,
which the four online coordinates cannot express. Evaluating these descriptors
through the online path evaluates those compiled robots, because
`test_dynamic_arrays_match_existing_static_generator` asserts the dynamic arrays
equal each static XML's `body_pos`, `body_ipos`, `body_mass`, `body_inertia` and
`site_pos`.

| Variant | In bounds | Continuous length | non-fall | Fixed length | non-fall |
|---|---|---:|---:|---:|---:|
| nominal | yes | 766.0 | 0.953 | 606.7 | 0.625 |
| tall_legs | yes | 786.3 | 0.953 | 633.5 | 0.625 |
| short_legs | yes | 749.6 | 0.875 | 613.2 | 0.531 |
| long_arms | yes | 802.8 | 0.969 | 629.2 | 0.641 |
| broad_shoulders | yes | 718.1 | 0.922 | 551.0 | 0.609 |
| heavy_torso | yes | 765.9 | 0.953 | 586.5 | 0.594 |
| combined | yes | 857.6 | 0.953 | 654.5 | 0.484 |
| extreme_short_heavy | **no** | 150.5 | 0.125 | 183.9 | 0.141 |
| extreme_combined | **no** | 95.6 | 0.047 | 103.3 | 0.047 |
| extreme_tall_light | **no** | **1.0** | **0.000** | **1.0** | **0.000** |

All seven in-bounds variants are controlled well by the continuous policy
(non-fall 0.88–0.97). All three out-of-bounds extremes fail badly, and
`extreme_tall_light` (leg scale 1.5 vs training bound 1.20) fails completely.

### Why the extremes fail — the terminal handler, not the policy

`extreme_tall_light` records length 1.0 for the **zero-action baseline too**, so
it dies on step 1 no matter what the policy does. The cause is not the reset
offset, which is correct: nominal leg length is 0.8 m, so raising the root by
`0.8 * (leg_scale - 1)` is exactly right, and the audit confirms leg length
0.68–1.20 m over scale 0.85–1.50.

The cause is that `RootPoseTrajTerminalStateHandler`
(`loco-mujoco/loco_mujoco/core/terminal_state_handler/traj.py:100`) derives its
root-height window from the **reference trajectory alone**:

```text
root_height_range = (traj_min - 0.3, traj_max + 0.3)
```

That window is morphology-independent. A taller body legitimately stands higher,
so once `0.8 * |leg_scale - 1|` approaches the 0.3 m margin the body is declared
absorbing at or near reset regardless of its actions. To first order this
confines leg scale to roughly **[0.625, 1.375]**; the precise cliff is
phase-dependent, because the window spans the reference gait's own height
variation, which is why `extreme_combined` (1.42) survives partially at 95.6
while `extreme_tall_light` (1.50) does not survive at all.

Where every catalog sits relative to that window:

| Catalog | Leg scale range | Bodies past the cliff |
|---|---|---:|
| train_1000 | 0.850 – 1.200 | 0 / 1000 |
| iid_256 | 0.851 – 1.199 | 0 / 256 |
| boundary_256 | 0.850 – 1.200 | 0 / 256 |
| ood_128 | 0.676 – **1.366** | 0 / 128 |
| named_variants | 0.680 – **1.500** | 2 / 10 |

So every training and held-out result above is unaffected — though the OOD
catalog's 1.366 maximum sits just inside the ~1.375 cliff, which is closer than
is comfortable and was not by design.

**This is the single most important limitation found in this continuation.** The
morphology range is currently capped by a termination criterion, not by the
policy's ability to control the body. Any attempt to widen the leg-length
distribution toward the "thousands of arbitrary robots" goal must first make the
terminal handler morphology-aware — comparing each body against its own grounded
reference height rather than the nominal reference. Until then, the two failing
named variants are environment-initialisation failures and must not be reported
as evidence about policy generalisation.

### The fix, and what it revealed

`scripts/scaling/morphology_terminal.py` adds
`MorphologyAwareRootPoseTrajTerminalStateHandler`, which subtracts each body's own
grounding offset before the height comparison, so every robot is judged against
*its* expected standing height. Reset and termination now share one
`OnlineMorphMjxUnitreeH1.root_height_offset()` definition and cannot drift apart.
Only `_is_absorbing_compat` is overridden, so the CPU and MJX paths stay in sync,
and environments without a morphology carry fall through to the base behaviour.

Re-evaluating the **unchanged** 300M continuous walk checkpoint under each
handler — same policy, same bodies, same reset keys, only the termination
criterion differs:

| Variant | Bounds | Stock length | Aware length | Stock non-fall | Aware non-fall |
|---|---|---:|---:|---:|---:|
| nominal | in | 778.3 | 778.2 | 0.969 | 0.969 |
| tall_legs | in | 786.3 | 786.1 | 0.953 | 0.953 |
| short_legs | in | 781.9 | 781.8 | 0.922 | 0.922 |
| long_arms | in | 802.8 | 802.8 | 0.969 | 0.969 |
| broad_shoulders | in | 718.3 | 718.3 | 0.922 | 0.922 |
| heavy_torso | in | 765.9 | 765.9 | 0.953 | 0.953 |
| combined | in | 868.3 | 868.2 | 0.969 | 0.969 |
| **extreme_tall_light** | **out** | **1.0** | **613.6** | **0.000** | **0.594** |
| extreme_short_heavy | out | 148.0 | 330.5 | 0.109 | 0.266 |
| extreme_combined | out | 94.9 | 201.4 | 0.047 | 0.094 |
| **aggregate beats-zero** | | **90%** | **100%** | | |

Two things follow, and the first is the safety property that makes the second
believable:

1. **The fix is a no-op inside the training box.** All seven in-bounds variants
   are unchanged to within 0.2 steps. It only affects bodies the old window was
   wrongly rejecting.
2. **The policy was far better than the measurement.** `extreme_tall_light` has
   leg scale 1.5 — 25% beyond the training bound of 1.20 — and the *same
   unchanged checkpoint* controls it for 613.6 steps at 59.4% non-fall. Nothing
   about the policy changed; it had always been able to do this, and a
   morphology-independent termination rule was hiding it.

This substantially widens the honest generalisation claim: the policy extrapolates
well past its training bounds, and the previously reported failures at the
extremes were an artefact of the evaluation environment. It also means the
measured morphology ceiling in this repository was never the policy's.

### What the primary claim can and cannot say

Earned:

> One descriptor-conditioned policy trained with simultaneous, balanced
> experience from 1,000 physically distinct online H1 embodiments (exposure
> exactly 8/8, no body unexposed) controls seed-disjoint IID, boundary and mild
> extrapolation embodiments far better than zero action - 100% of bodies on every
> frozen catalog, across three seeds.

Not earned, and contradicted by the matched control:

> ...and a controlled 1,000-body catalog is the better way to train it.

It is not. On this evidence the catalog's value is reproducibility, auditability
and honest held-out accounting - not sample efficiency.

---

## 6. Dance at 1,000 bodies — training complete, evaluation not run

`dance2_subject4`, the clip where the recorded four-body round-robin result
reached expert-level tracking. All five jobs exit 0, 299,520,000 steps each,
exposure exactly 8/8 for the fixed arm.

| Arm | Seed | Job | Return (last 10) | Length (last 10) |
|---|---:|---:|---:|---:|
| fixed | 1 | 10808474 | 129.03 | 498.7 |
| fixed | 2 | 10808475 | 132.02 | 504.0 |
| fixed | 3 | 10808476 | 125.83 | 498.6 |
| continuous | 1 | 10808477 | 135.77 | **526.2** |
| continuous | 2 | 10808478 | 135.36 | **529.1** |

Findings:

- **Continuous beats fixed again**, by ~5% length and ~4% return. The direction
  replicates the walk result, though the margin is much smaller here (5% vs 24%).
- Seed agreement is tight within each arm (≤ 1%).
- The preferred success bar was **median episode length ≥ 500 on IID bodies**.
  Training length is 499–529, straddling that line — but training length on the
  training distribution is *not* the IID held-out measurement the bar refers to,
  and it is doubly censored (§5). **The bar is not yet evaluated.**
- Quarter-point lengths for continuous seed 1 are 41/147/481/509/503, so the
  curve had largely flattened by 300M, unlike the walk runs.

### Dance held-out evaluation — the success bar is met

All five checkpoints on the frozen catalogs, 4 replicas per body, horizon 1,000,
deterministic actions (run on the local GPU, not the cluster).

| Catalog | Arm | Return | Median length | Non-fall | Beats zero | Max abs corr |
|---|---|---:|---:|---:|---:|---:|
| IID-256 | fixed | 163.7 | 529.6 | 0.539 | **100%** | 0.088 |
| IID-256 | continuous | 169.5 | **546.9** | 0.549 | **100%** | 0.097 |
| Boundary-256 | fixed | 160.2 | 526.1 | 0.542 | **100%** | 0.096 |
| Boundary-256 | continuous | 163.9 | 545.2 | 0.548 | **100%** | 0.071 |
| OOD-128 | fixed | 158.8 | 527.2 | 0.544 | **100%** | 0.281 |
| OOD-128 | continuous | 163.2 | 546.2 | 0.555 | **100%** | 0.322 |

The registered bar was *median episode length ≥ 500 on IID bodies and a majority
of boundary bodies beating zero action*:

| Criterion | Threshold | Result |
|---|---|---|
| Median IID episode length | ≥ 500 | **529.6 / 546.9** — met by both arms |
| Boundary bodies beating zero | majority | **100%** — met by both arms |

**Priority 3 succeeds.** One policy tracks an expressive LAFAN1 dance across
1,000 physically distinct H1 bodies and generalises to seed-disjoint, boundary
and mildly extrapolated bodies.

Continuous again beats fixed, but only by ~3% median length here versus ~15% on
walk — the ranking replicates, the magnitude does not.

---

## 6c. Widening the morphology bounds — naive widening fails

With the terminal-handler ceiling removed, the morphology distribution can be
widened for the first time. New bounds were chosen to make the extreme named
variants in-distribution:

| Coordinate | Narrow | Wide |
|---|---|---|
| leg / arm / shoulder scale | 0.85–1.20 | 0.68–1.50 / 0.70–1.55 |
| torso mass scale | 0.70–1.50 | 0.55–2.40 |
| kinematic leg length | 0.68–0.96 m | **0.54–1.20 m** |
| total mass | 45–62 kg | **42–80 kg** |

All 1,000 wide bodies pass the physical audit. Training from scratch on them,
same recipe and budget as the narrow runs (300M steps, 8,000 envs, 2 seeds,
morphology-aware terminal handler), **collapses**:

| Metric | Narrow fixed | Wide fixed |
|---|---:|---:|
| Training episode length | 554 | **61.0 / 62.4** |
| Held-out IID return | 153.2 | **2.05** (zero action: 1.02) |
| Held-out IID median length | 673.9 | **67.4** |
| Held-out non-fall | 0.678 | **0.029** |
| IID bodies beating zero | 100% | 64.1% |

The failure is **uniform across all four coordinates**, which is the informative
part:

| Axis | Bottom-quartile length | Top-quartile length |
|---|---:|---:|
| leg | 69.1 | 69.4 |
| arm | 65.8 | 67.6 |
| shoulder | 68.2 | 67.3 |
| torso mass | 71.8 | 65.3 |

No body type is uniquely responsible; the policy simply failed to learn. This is
**not** a representational limit: the narrow-bounds policy already controls
`extreme_tall_light` (leg 1.5, outside its own training range) for 613 steps. It
is an optimisation/curriculum failure, and it is the exact failure the progress
log warned about — *"do not widen all axes blindly"* — with the roadmap
prescribing progressive widening. All axes were widened at once here.

The fix under test is a warm-started bounds curriculum
(`--init-checkpoint`, wired to loco-mujoco's existing `build_resume_train_fn`):
narrow checkpoint → mid bounds → wide bounds. Early signal is strong: a
warm-started mid-bounds run reaches episode length 273 after **0.64M** steps,
versus 61 after **300M** from scratch.

## 6b. Compute: the local GPU beats the cluster for this workload

Measured on the same code, same catalog, same shapes:

| Configuration | RTX 4060 Ti (WSL2) | Viper MI300A | Ratio |
|---|---:|---:|---:|
| 8,000 envs × 64 steps (production) | **6.89 M steps/min** | 1.90 M | **3.6×** |
| 2,000 envs × 64 steps | 8.06 M steps/min | — | — |
| Cold XLA compile | ~30 s | 681–1,007 s | ~30× |
| Peak live memory | 1.91 / 12.0 GiB | 2.01 / 82.5 GiB | — |

A 300M-step run is ~44 minutes locally versus ~2.6 hours on one APU. MJX on a
19-DoF humanoid is latency- and occupancy-bound rather than FLOP-bound, and the
ROCm path additionally runs with `--xla_gpu_enable_command_buffer=` disabled for
stability. **The cluster is only worth using here for many concurrent seeds, not
for single-run speed.**

Setup note: JAX ships no Windows CUDA wheels, so the GPU is reachable only from
WSL2. The environment is `~/jaxgpu` inside the `Ubuntu` distribution, built on
the pre-existing `/home/smirn/.local/bin/python3.12` (WSL's default 3.10 is below
JAX's 3.11 floor), with `jax[cuda12]==0.10.1` matching the Windows and Viper
versions. Run with `PYTHONPATH=$PWD/scripts ~/jaxgpu/bin/python ...`.

## 7. Roadmap status

| Step | Before this continuation | Now |
|---:|---|---|
| 3 | partial | catalog + systems proof done; quality training running |
| 4 | partial | online AMP bridge implemented and leakage-tested; long run pending |
| 5 | partial | body × motion now sample independently in one graph; unproven at scale |
| 6–10 | unchanged | unchanged |

## 8. Verification

```powershell
$env:PYTHONPATH="$PWD\loco-mujoco;$PWD\scripts"
.\.venv\Scripts\python.exe -m pytest tests\test_parallel_morph_env.py tests\test_online_h1.py `
  tests\test_amp_targets.py tests\test_urma_networks.py tests\test_masked_mlp.py `
  tests\test_embodiment_catalog.py tests\test_online_amp.py -q
ruff check scripts\scaling tests
ruff format --check scripts\scaling tests
```

Current result: **39 passed**, ruff check and format clean.
