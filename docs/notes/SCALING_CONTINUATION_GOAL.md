# Continuation Goal — From Online Capacity to 1,000-Robot Scaled Training

## Mission

Continue [`SCALING_ROADMAP.md`](SCALING_ROADMAP.md) from the first unfinished
milestone. The immediate goal is not another environment-count stress test. It is
to finish the online embodiment system, construct a reproducible set of **1,000
distinct robots**, train one policy on all of them with balanced simultaneous
coverage, and demonstrate held-out control quality. After that, combine online
embodiments with multi-motion AMP and proceed toward cross-family control.

The primary claim this continuation must earn is:

> One descriptor-conditioned policy was trained with simultaneous, balanced
> experience from 1,000 physically distinct online H1 embodiments, and it
> controls seed-disjoint and boundary-held-out embodiments substantially better
> than zero action across multiple seeds.

The stretch claim is:

> One policy controls online-randomized H1 embodiments across several motions,
> breaking the recorded multi-motion DeepMimic robustness plateau.

Read these before changing code or launching work:

- [`SCALING_ROADMAP.md`](SCALING_ROADMAP.md)
- [`SCALING_PROGRESS_2026-08-01.md`](SCALING_PROGRESS_2026-08-01.md)
- [`experiments/goal_20260710_223728/report.md`](experiments/goal_20260710_223728/report.md)
- [`experiments/frontier_20260714_224204/report.md`](experiments/frontier_20260714_224204/report.md)

## What “1,000 robots” means

For this goal, 1,000 robots means **1,000 fixed, reproducible embodiment records**,
not 1,000 copies of one robot and not merely 1,000 random draws that cannot be
reconstructed.

Each embodiment must have:

- a stable integer body ID;
- a unique normalized morphology descriptor;
- concrete, distinct physical model arrays used by MJX;
- a recorded family, generator revision, bounds, seed, and split;
- physical-validity checks;
- deterministic regeneration or catalog loading.

The first 1,000-robot milestone is one topology family: H1 with the same joint,
observation, action, geom, and mesh counts. It does **not** mean 1,000 unrelated
robot topologies. Cross-topology scale is a later roadmap gate.

## Do not redo completed work

The following are established and should be treated as baselines:

- Grouped-static simultaneous PPO works, including the supervisor-style
  10 bodies × 200 environments layout.
- Online H1 morphology parameters change per environment and can resample at
  asynchronous episode reset.
- The current online parameters are leg length, arm length, shoulder width, and
  torso mass scales.
- Online descriptor generation is negligible compared with compilation and
  training.
- A descriptor-conditioned MLP is the current same-topology winner.
- The 8,192-env × 64-step, 100M-step Viper MLP exposed an estimated 1.305M
  embodiments and beat zero action on 96.9% of 1,024 seed-disjoint bodies.
- One MI300A completed MLP capacity runs at 65,536×64, 131,072×32, and
  262,144×16. These are capacity results, not evidence that the largest batch is
  the best learning configuration.
- URMAv2 is safe at 32,768×64 and fails at 40,960×64 on the tested Viper setup.
- The present URMAv2 adaptation is worse than the MLP on fixed-topology H1 and
  must not replace it without a matched held-out win.
- Static four-body × four-motion AMP executes through one shared policy and
  discriminator.
- H1/G1/Atlas execute through one padded/masked PPO policy; ToddlerBot passes a
  four-topology preflight.
- Cross-humanoid reference caches exist for nine humanoid targets.

Key existing entry points:

- `scripts/scaling/online_h1.py`
- `scripts/scaling/online_h1_train.py`
- `scripts/scaling/evaluate_online_policy.py`
- `scripts/scaling/parallel_env.py`
- `scripts/scaling/parallel_multimorph_amp_train.py`
- `scripts/scaling/parallel_cross_humanoid_train.py`
- `scripts/scaling/cross_humanoid_retarget.py`
- `scripts/scaling/urma_networks.py`
- `scripts/scaling/masked_mlp.py`

## Remaining roadmap status

Steps 1 and 2 are implemented. Everything below remains unfinished to some
degree:

| Roadmap step | Remaining work | Done when |
|---:|---|---|
| 3 | Controlled 1,000-body catalog, balanced scaled training, dance and OOD evaluation | A quality-trained policy has auditable exposure to all 1,000 train bodies and passes held-out tests across seeds |
| 4 | Long, post-fix AMP experiment proving whether AMP breaks the multi-motion plateau | AMP beats the registered DeepMimic robustness threshold or produces a conclusive matched negative result |
| 5 | Combine online morphology and multi-motion AMP | Body and motion are both sampled/balanced inside one dynamic H1 training graph |
| 6 | Compare the existing SMPL-based references with actual GMR-quality retargeting | Quantitative kinematic/contact comparison and selected cached pipeline for at least H1/G1/Atlas |
| 7 | Make the embodiment-aware backbone useful across variable topology | URMA/URMAv2 consumes the padded cross-topology representation and beats relevant baselines |
| 8 | Strong cross-family, few-motion policy | H1/G1/Atlas each beat zero action and their short masked-MLP baseline on held-out rollouts |
| 9 | Many-motion representation and command interface | A gated experiment shows the AMP/ASE/SONIC-style path scales beyond a few clips |
| 10 | Foundation controller | Only after steps 3–9 have evidence; do not begin as an integration-only exercise |

## Priority 1 — Finish the online embodiment system

### 1A. Add deterministic catalogs

Extend the online H1 path rather than creating 1,000 XML branches. Add a small,
versioned catalog abstraction. Suggested files are
`scripts/scaling/embodiment_catalog.py` plus focused extensions to
`online_h1.py` and `online_h1_train.py`; use different names if a cleaner design
fits the repository.

Required catalog modes:

1. `continuous`: current per-reset random sampling.
2. `fixed_balanced`: body ID is assigned deterministically to an environment and
   remains fixed while reset state/motion phase changes.
3. `catalog_resample`: reset selects only from a specified catalog, with a
   deterministic balanced schedule rather than uncontrolled frequency drift.

Required serialized fields:

- schema and generator revision;
- robot family and topology signature;
- RNG seed and sampling method;
- raw and normalized descriptors;
- train/validation/test/OOD split;
- morphology bounds;
- body IDs and a descriptor/content hash;
- physical-validity summary.

Do not use a 1,000-way one-hot policy input. The policy should receive the compact
morphology descriptor; body IDs are for accounting and reproducibility.

### 1B. Make “different” physically auditable

The catalog test must verify more than descriptor uniqueness. For representative
and boundary bodies, assert that the intended MJX arrays differ from nominal and
match the descriptor:

- segment offsets/kinematic lengths;
- body mass and inertia;
- any added actuator, damping, friction, joint-range, or primitive-geometry
  parameter;
- positive masses/inertias and ordered joint ranges;
- finite reset and step outputs.

The nominal mesh is currently shared. Keep that limitation explicit. Do not claim
mesh-level morphology diversity unless collision/visual geometry actually changes.

After the four-coordinate baseline is reproduced, consider additional
same-topology coordinates such as segment-specific mass/inertia, actuator gain,
joint damping, friction, joint range, and primitive foot dimensions. Add one
parameter group at a time with validation; do not widen all axes blindly.

### 1C. Required tests

Add tests for:

- identical seed/revision produces an identical catalog and hash;
- the 1,000-body training catalog contains exactly 1,000 unique descriptors and
  IDs;
- fixed-balanced assignment gives every body the requested replica count;
- catalog resampling remains balanced over a registered window;
- reset observations describe the newly active body;
- invalid mass/inertia/range inputs are rejected;
- representative model arrays differ in the expected locations;
- saved catalogs reload without numerical drift.

## Priority 2 — Explicit 1,000-body scale experiments

Two separate experiments are mandatory because they answer different questions.

### 2A. Supervisor-layout systems proof

Run exactly:

- 1,000 catalog bodies;
- 200 simultaneous environment replicas per body;
- 200,000 total environments;
- 16 rollout steps;
- one descriptor-conditioned MLP;
- at least 10 PPO updates.

This is expected to fit because 262,144 unique environments already completed at
16 steps, but the new run must prove **balanced 200× replication of a fixed
1,000-body catalog**. It is a throughput/memory/system result only.

The manifest must record:

- `num_catalog_bodies = 1000`;
- `replicas_per_body = 200`;
- exact min/max exposure count across body IDs;
- catalog path and hash;
- compile, train, throughput, live memory, and Slurm metadata;
- update/minibatch counts;
- clear `capacity_only = true` labeling.

Do not describe its return as a learned 1,000-robot controller after only ten
updates.

### 2B. Quality-scaled 1,000-body training

Use the already validated quality/throughput neighborhood:

- 1,000 fixed training bodies;
- 8 replicas/body = 8,000 environments;
- 64 rollout steps;
- descriptor-conditioned MLP;
- 32 minibatches and four PPO epochs unless a matched ablation justifies a change;
- at least 100M steps;
- three seeds;
- one selected single motion, starting with `walk1_subject1`.

Run a matched continuous-resampling arm with the same environment, step, network,
optimizer, and seed budgets. This tests whether a controlled 1,000-body curriculum
matches or improves the existing continuously sampled policy.

Six independent one-APU jobs—three fixed-catalog seeds and three continuous
seeds—fit under the observed eight-job account limit and should run concurrently.
Do not implement distributed PPO merely to launch this comparison.

### 2C. Frozen evaluation sets

Create evaluation catalogs before training and never regenerate them after seeing
results:

- 256 IID interpolation bodies inside the training bounds;
- 256 boundary bodies concentrated near faces/edges/corners of the bounds;
- 128 physically valid mild extrapolation bodies outside one training bound at a
  time;
- 10–50 existing named XML variants for a static-model check.

For every checkpoint, evaluate the exact same bodies, reset keys, initial motion
phases, and horizon with:

- deterministic policy actions;
- zero actions;
- the current 100M continuous MLP checkpoint;
- a nominal-only policy if a compatible checkpoint exists.

Report mean, median, quantiles, completion/non-fall rate, episode length, return,
fraction beating zero, and correlations with every morphology coordinate. Include
per-body results, not only aggregates.

### 2D. Step-3 acceptance criteria

The 1,000-body walk milestone is complete only if:

- all three training seeds finish without NaNs or dropped catalog bodies;
- every train body has balanced exposure within the declared tolerance;
- at least 90% of IID held-out bodies beat zero-action return;
- IID mean return and episode length are no worse than 10% below the matched
  continuous-resampling arm, or the difference is explained with confidence
  intervals;
- at least 75% of boundary bodies beat zero action;
- OOD results are reported honestly even if they fail;
- no single morphology coordinate alone explains the result (`|correlation| <
  0.25` is the initial audit threshold);
- results are aggregated across seeds with uncertainty, not selected from the
  best seed.

If 100M steps are still improving, extend only the best pre-registered recipe to
300M. Extend toward 600M–1B only when learning curves and held-out checkpoints
justify the cost. Do not compensate for a flat curve by blindly adding steps.

## Priority 3 — Finish single-motion scale with an expressive dance

After the 1,000-body walk result, repeat the controlled catalog experiment on one
expressive LAFAN1 dance clip. Use the existing round-robin four-body dance result
as the competence reference, not as a directly comparable environment setup.

Required comparisons:

- fixed 1,000-body catalog versus continuous resampling;
- nominal and moderate bodies versus boundary/extreme bodies;
- raw canonical grounded reference versus one justified retargeted-reference arm
  if the reference geometry is suspected to dominate;
- checkpoints at registered step budgets.

The preferred success bar is median episode length ≥500 on IID bodies and a
majority of boundary bodies beating zero action. If this is not reached, deliver a
conclusive scaling curve showing whether failure is optimization, stability, or
reference mismatch. A well-controlled negative result still completes the
experiment; it does not justify claiming morphology-independent dance control.

## Priority 4 — Online morphology × multi-motion AMP

Build the missing bridge between `online_h1.py` and the verified static
multi-motion AMP path.

### 4A. Correct design

At reset, independently or through a registered curriculum, select:

- a body descriptor/body ID;
- a motion/clip ID;
- a motion phase.

Do not create a static body × motion XLA branch product. The body remains dynamic
inside one H1 model graph and the motion comes from a multi-trajectory handler.

Prevent discriminator leakage. Either:

- exclude body IDs/morphology metadata from discriminator observations; or
- attach correctly matched morphology metadata to both expert and policy
  samples.

Add an explicit test that the discriminator cannot distinguish expert/policy by a
missing or constant descriptor channel. Preserve the audited AMP -1/+1 labels,
discriminator gradient clipping, and current-reward normalization.

### 4B. Scale ladder

Use this order:

1. 2 bodies × 2 motions, local smoke.
2. 16 catalog bodies × 4 motions, 256–1,024 environments.
3. 1,000 catalog bodies × 4 motions, approximately 8,000 environments with a
   balanced body-motion schedule.
4. Extend motion count only after robustness improves.

The 1,000×4 run does not require 200 replicas for every body-motion pair. Keep the
quality-validated total environment count and balance exposure over time.

### 4C. AMP acceptance criteria

The recorded four-clip DeepMimic plateau is roughly 140–150 mean episode length
with very low non-fall rate. Register before the long run:

- mean episode length >250 across training clips;
- non-fall rate >0.10;
- held-out motion evaluation;
- per-body and per-motion breakdown;
- at least two seeds before calling the plateau broken.

If AMP does not beat the registered threshold, inspect discriminator balance,
reference initialization, termination curriculum, and motion conditioning before
moving to ASE/SONIC. Do not declare success from discriminator loss alone.

## Priority 5 — Cross-family work after the 1,000-H1 gates

### 5A. Retargeting quality, not only speed

The existing cross-humanoid pipeline is fast enough. The unfinished question is
quality. Compare the current direct/SMPL route against actual GMR or an equivalent
high-quality retargeter for at least H1, G1, and Atlas on the same clips.

Measure:

- joint and end-effector tracking error;
- root trajectory/orientation error;
- foot penetration, sliding, and contact timing;
- retarget wall time and cache reuse;
- downstream short-policy learning, if kinematic metrics disagree.

Select and cache one reference method per topology/motion. Same-DOF online
variants should continue sharing a canonical reference unless evidence shows that
per-body retargeting materially improves control.

### 5B. Merge URMA with the cross-topology wrapper

The next URMA milestone is not another fixed-H1 capacity run. Combine:

- padded per-joint descriptions and states;
- the raw binary valid-joint mask;
- masked action distributions;
- per-family/topology identity only when needed;
- H1/G1/Atlas grouped simulator branches.

Keep padded action means at zero and practical standard deviation near zero for
invalid joints. Keep the mask binary and outside running normalization.

Compare against the existing masked MLP with matched steps/seeds. Do not promote
URMA/URMAv2 unless it improves per-robot held-out return or enables a topology the
MLP cannot represent reliably.

### 5C. Cross-family acceptance gate

Before step 8 is called complete:

- H1, G1, and Atlas all train in every shared update;
- every robot beats its exact-reset zero-action baseline on mean return;
- no robot is hidden by an aggregate average;
- results hold across at least two seeds;
- one or more motions are evaluated separately;
- topology-held-out behavior is reported, even if negative.

Only after this gate should online per-family H1/G1 morphology randomization be
combined in one shared cross-family policy.

## Deferred roadmap — Steps 9 and 10

Do not begin a SONIC/BeyondMimic/foundation-controller integration merely because
the repositories or papers are available. Begin step 9 only when online
multi-motion AMP is stable and motion count—not body control—is the measured
bottleneck.

The step-9 experiment must have a concrete data scale, command representation,
baseline, and held-out skill metric. Step 10 begins only after the 1,000-body,
multi-motion, and cross-family gates above have evidence.

## Viper execution rules

Working access path:

```powershell
wsl -d Ubuntu -e ssh -o BatchMode=yes viper11 "hostname; squeue -u akalenik"
```

Remote environment:

```bash
source /ptmp/akalenik/frontier/venv/bin/activate
```

Operational requirements:

- Use `Ubuntu`, not the default `Ubuntu-20.04` WSL distribution.
- Pass Slurm overrides through `sbatch --export=ALL,...`; unexported inline
  variables previously fell back to launcher defaults.
- Preserve `XLA_FLAGS=--xla_gpu_enable_command_buffer=` and the persistent XLA
  cache for ROCm.
- Verify the remote script hash/config before every production submission.
- Start with a reset/step preflight and one-update run for every new graph shape.
- Run independent seeds/controls concurrently; the observed account limit is
  eight running jobs.
- Do not call independent jobs “distributed training.” One shared multi-APU
  policy still requires explicit gradient synchronization.
- URMAv2 four-epoch nested updates segfaulted on ROCm in the measured setup. Use
  one epoch plus more rollout updates until that runtime issue is fixed and
  re-verified. The MLP four-epoch production path is stable.
- Cancel redundant capacity probes once a tighter success/failure bracket exists.
- Do not leave jobs untracked at handoff. Archive the Slurm output, request JSON,
  manifest, checkpoint, and evaluation locally.

## Experiment and artifact contract

Every non-smoke run must write a machine-readable manifest containing:

- git/submodule revision or explicit dirty-worktree fingerprint;
- exact command and Slurm job ID/node;
- catalog revision/path/hash and split;
- body/motion counts and exposure distribution;
- environment count, rollout length, updates, epochs, minibatches, and total
  gradient minibatch updates;
- compile/training time and throughput;
- host and device memory;
- checkpoint path;
- train curves and per-evaluation-set summaries;
- whether the run is capacity-only or quality-oriented.

Use a new results root such as:

```text
experiments/scaling_1000/
  catalogs/
  system_1000x200/
  walk_fixed_1000/
  walk_continuous_control/
  dance_fixed_1000/
  online_amp/
  evaluations/
```

Create a final `SCALING_1000_RESULTS.md` that includes:

- the exact claims supported and unsupported;
- the 1,000-body exposure audit;
- seed-aggregated learning/evaluation tables;
- IID, boundary, extrapolation, and named-XML results;
- system versus quality scaling curves;
- wall-clock and memory accounting;
- failures, cancelled jobs, and negative results;
- the updated status of roadmap steps 3–8.

## Verification before handoff

At minimum, run:

```powershell
$env:PYTHONPATH="$PWD\loco-mujoco;$PWD\scripts"
.\.venv\Scripts\python.exe -m pytest tests\test_parallel_morph_env.py tests\test_online_h1.py tests\test_amp_targets.py tests\test_urma_networks.py tests\test_masked_mlp.py -q
ruff check scripts\scaling tests
ruff format --check scripts\scaling tests
```

Add new catalog/assignment/online-AMP tests to the command. Validate every JSON
artifact and reload every final checkpoint.

Preserve unrelated user changes. Do not reset the dirty worktree or overwrite
existing experiment artifacts. An earlier generator check touched existing
nominal/extreme H1 XML and gallery metadata; do not treat those files as clean
baselines without reviewing the progress report caveat.

## Stop conditions and decision discipline

- Capacity success is not policy-quality success.
- One successful seed is not a generalization result.
- IID interpolation is not extrapolation.
- Same-topology online morphology is not arbitrary-topology generation.
- A discriminator that trains is not proof that multi-motion robustness improved.
- A longer run is justified only by a learning curve or a pre-registered budget,
  not by hope.
- Retargeting and XML generation are not the current dominant costs; prioritize
  training objective, evaluation, and XLA-shape reuse.
- If a job fails, archive the exact error before changing multiple variables.
- Change one scientific factor at a time and keep a matched control.

## Completion definition for the next agent

The next continuation is complete when all of the following are true:

1. A deterministic, validated 1,000-body H1 training catalog and frozen held-out
   catalogs exist.
2. The exact 1,000 bodies × 200 replicas systems run completes and is labeled as
   capacity-only.
3. Three-seed, 100M+ quality training on 1,000 bodies completes alongside matched
   continuous-resampling controls.
4. IID, boundary, extrapolation, and named-XML evaluations are archived with
   per-body results and zero-action comparisons.
5. One expressive single-dance scale experiment reaches a registered conclusion.
6. The online-morphology × multi-motion AMP bridge is implemented, tested, and at
   least reaches the 16-body × four-motion rung.
7. All Viper jobs are resolved, outputs are local, tests pass, and
   `SCALING_1000_RESULTS.md` gives an honest verdict.

Steps 6–10 of the original roadmap remain follow-on research unless the primary
1,000-body and online-AMP gates finish early with strong evidence.
