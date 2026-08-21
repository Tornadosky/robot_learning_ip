# What can be run immediately

The current code is sufficient for an **8M integration smoke** that checks:

* reference generation;
* H1/G1 environment construction;
* online morphology;
* URMAv2 input layout;
* action masks;
* one shared update;
* checkpoint serialization;
* evaluation and rendering plumbing.

Do not use this smoke as the final learning result because of the goal/reward mismatch above.

I prepared a complete helper bundle:

[Download the H1/G1 URMA pipeline bundle](sandbox:/mnt/data/h1g1_urma_pipeline_bundle.zip)

Extract it and enter the directory:

```bash
unzip h1g1_urma_pipeline_bundle.zip
cd h1g1_urma_pipeline
```

Configure the local repository:

```bash
export REPO=/mnt/c/Users/smirn/Desktop/robot_learning_ip

# JAX/MJX training environment
export JAX_PY=$HOME/jaxgpu/bin/python

# Use a different environment here if retargeting has Torch/SMPL dependencies
export RETARGET_PY=$HOME/jaxgpu/bin/python

export BACKBONE=urmav2
export SEED=3

# One PPO optimizer pass over each mixed H1+G1 rollout batch.
export UPDATE_EPOCHS=1

# Needed for local rendering.
export MUJOCO_GL=egl
```

To use URMA v1 instead:

```bash
export BACKBONE=urma
```

## Run each stage separately

This is preferable to launching the whole chain because the first failing stage immediately identifies the layer at fault.

```bash
bash run_h1g1_urma_pipeline.sh init
bash run_h1g1_urma_pipeline.sh retarget
bash run_h1g1_urma_pipeline.sh verify
bash run_h1g1_urma_pipeline.sh tests
bash run_h1g1_urma_pipeline.sh preflight
bash run_h1g1_urma_pipeline.sh smoke
```

Then evaluate the smoke checkpoint:

```bash
RUN_NAME=catalog2_smoke_s3 \
bash run_h1g1_urma_pipeline.sh evaluate
```

Render it:

```bash
RUN_NAME=catalog2_smoke_s3 \
bash run_h1g1_urma_pipeline.sh render
```

Run the artifact auditor:

```bash
python audit_h1g1_pipeline.py \
  --experiment \
  "$REPO/experiments/urma_h1g1_singlemotion_dance2_subject4_s3"
```

The equivalent convenience command is:

```bash
bash run_h1g1_urma_pipeline.sh all-smoke
```

The staged version remains better for diagnosing integration problems.

---

# What those stages actually execute

## 1. Reference retargeting

The retarget stage executes:

```bash
"$RETARGET_PY" \
  "$REPO/scripts/scaling/cross_humanoid_retarget.py" \
  --source h1 \
  --targets h1 g1 \
  --clip dance2_subject4 \
  --duration 8.0 \
  --start-frame 19482 \
  --mode direct \
  --output-root "$REPO/external_data/cross_humanoid"
```

Expected reference files:

```text
external_data/cross_humanoid/
└── h1_source/
    └── dance2_subject4/
        ├── h1/start19482_800f_direct.npz
        └── g1/start19482_800f_direct.npz
```

This is cross-topology retargeting: source/H1 motion to H1 and source/H1 motion to G1.

For randomized H1 and G1 bodies, do not retarget the human/source clip again. Apply each topology’s joint reference to the sampled body and derive spatial targets by FK. The repository’s later retargeting analysis reached the same conclusion for same-topology morphology: joint-angle semantics with body-specific FK are the well-posed default; whole-body similarity retargeting can be geometrically infeasible. 

## 2. FK verification

The verifier executes approximately:

```bash
"$JAX_PY" \
  "$REPO/scripts/scaling/verify_fk_targets.py" \
  --robots h1 g1 \
  --source h1 \
  --clip-windows dance2_subject4:19482:800 \
  --reference-root "$REPO/external_data/cross_humanoid" \
  --phases 3 111 400 731 \
  --output \
  "$REPO/experiments/urma_h1g1_singlemotion_dance2_subject4_s3/metrics/fk_verification.json"
```

It checks:

* independent CPU MuJoCo against JAX/MJX;
* nominal, low-corner, and high-corner bodies;
* several phases;
* JIT against eager execution;
* corner targets moving relative to nominal;
* phase (p) differing from (p+1);
* finite outputs.

The current verifier’s established tolerance is approximately (10^{-5}) metres, while the latest measured worst CPU/JAX positional error was around (5.5\times10^{-7}) metres. 

## 3. Preflight

The runner performs two preflights:

```text
catalog2
    H1 low corner
    H1 high corner
    G1 low corner
    G1 high corner

continuous
    random body per environment/reset
```

The `catalog2` morphology values are:

```text
low  = [0.90, 0.70, 0.50, 0.70]
high = [1.12, 1.50, 2.00, 1.30]
```

The preflight must report:

```text
robots: ["h1", "g1"]
number of topology groups: 2
action-mask counts: [19, 23]
reward type: MorphMimicReward
backbone: urma or urmav2
robot one-hot: false
finite observations and rewards
```

The policy still uses padded vectors. Invalid action entries must remain exactly zero. That is required even though no MLP is being trained.

## 4. Smoke training

The current smoke expands to the following contract:

```bash
"$JAX_PY" \
  "$REPO/scripts/scaling/parallel_cross_humanoid_train.py" \
  --backbone urmav2 \
  --robots h1 g1 \
  --source h1 \
  --clip dance2_subject4 \
  --start-frame 19482 \
  --frames 800 \
  --reference-mode direct \
  --reference-root "$REPO/external_data/cross_humanoid" \
  --morphology catalog2 \
  --reward-type MorphMimicReward \
  --no-robot-one-hot \
  --append-joint-features \
  --learnable-std \
  --total-envs 512 \
  --total-timesteps 8000000 \
  --num-steps 64 \
  --num-minibatches 16 \
  --update-epochs 1 \
  --hidden 256 128 \
  --lr 1e-4 \
  --seed 3 \
  --run-tag catalog2_smoke_s3 \
  --output-dir \
  "$REPO/experiments/urma_h1g1_singlemotion_dance2_subject4_s3/checkpoints/catalog2_smoke_s3"
```

No FSQ-related argument is present:

```text
no --blank-goal
no --goal-for-critic
no --actor-latent-dim
no --latent-codes
```

“`--update-epochs 1`” means one optimizer pass over every collected mixed H1/G1 rollout batch. It does not mean one total training iteration and it does not mean H1 and G1 are trained sequentially. Every PPO update contains both topology groups.

---

# Required long-run procedure after the P0 patch

Once the shared body-correct goal provider, terminal forwarding, and manifest fields have been implemented and the new tests pass, configure:

```bash
export GOAL_TYPE=MorphGoalTrajMimicRootErr
export TERMINAL_HANDLER=MorphologyAwareRootPoseTrajTerminalStateHandler
export MAX_ROOT_DEVIATION=0.5
```

Then run a 60M probe:

```bash
bash run_h1g1_urma_pipeline.sh train60m
```

Evaluate:

```bash
RUN_NAME=continuous_60m_s3 \
bash run_h1g1_urma_pipeline.sh evaluate
```

Render:

```bash
RUN_NAME=continuous_60m_s3 \
bash run_h1g1_urma_pipeline.sh render
```

Audit with the learning gate enabled:

```bash
python audit_h1g1_pipeline.py \
  --strict-learning \
  --experiment \
  "$REPO/experiments/urma_h1g1_singlemotion_dance2_subject4_s3"
```

Only after the 60M run passes should you launch 300M:

```bash
bash run_h1g1_urma_pipeline.sh train300m

RUN_NAME=continuous_300m_s3 \
bash run_h1g1_urma_pipeline.sh evaluate

RUN_NAME=continuous_300m_s3 \
bash run_h1g1_urma_pipeline.sh render

python audit_h1g1_pipeline.py \
  --strict-learning \
  --experiment \
  "$REPO/experiments/urma_h1g1_singlemotion_dance2_subject4_s3"
```

The runner blocks long runs unless the new goal and terminal settings are provided. There is an escape hatch:

```bash
export ALLOW_STOCK_GOAL_BASELINE=1
```

That should only be used to reproduce a clearly labelled known-mismatch baseline. It must not be used for the production conclusion.

## Independent training seeds

Evaluation seeds from one checkpoint do not measure training stability. Run at least three independent training seeds:

```bash
for s in 3 7 11; do
  SEED="$s" \
  EXP="$REPO/experiments/urma_h1g1_singlemotion_dance2_subject4_s${s}" \
  GOAL_TYPE=MorphGoalTrajMimicRootErr \
  TERMINAL_HANDLER=MorphologyAwareRootPoseTrajTerminalStateHandler \
  MAX_ROOT_DEVIATION=0.5 \
  UPDATE_EPOCHS=1 \
  bash run_h1g1_urma_pipeline.sh train300m
done
```

Run evaluation and audit separately for each experiment directory.

---

# Viper commands

The provided Viper launcher is corrected for the current recorded constraints:

* exactly H1 and G1;
* no stale ToddlerBot reserve;
* maximum 768 environments;
* exactly one PPO optimizer epoch;
* no FSQ flags;
* no robot one-hot;
* `MorphMimicReward`.

The repository records a 768-environment ceiling, a one-optimizer-epoch requirement, and at most two topology branches on the tested Viper ROCm path. Four update epochs caused very long compilation followed by a segfault.   

## Current-code 8M smoke

After syncing the code, references, and launcher to Viper:

```bash
sbatch \
  --export=ALL,\
TAG=h1g1_catalog2_s3,\
BACKBONE=urmav2,\
MORPHOLOGY=catalog2,\
TOTAL_ENVS=768,\
TOTAL_TIMESTEPS=8000000,\
UPDATE_EPOCHS=1,\
SEED=3 \
  viper_h1g1_urma.sbatch
```

## 60M production probe after the P0 patch

```bash
sbatch \
  --export=ALL,\
TAG=h1g1_continuous_60m_s3,\
BACKBONE=urmav2,\
MORPHOLOGY=continuous,\
TOTAL_ENVS=768,\
TOTAL_TIMESTEPS=60000000,\
UPDATE_EPOCHS=1,\
SEED=3,\
GOAL_TYPE=MorphGoalTrajMimicRootErr,\
TERMINAL_HANDLER=MorphologyAwareRootPoseTrajTerminalStateHandler,\
MAX_ROOT_DEVIATION=0.5 \
  viper_h1g1_urma.sbatch
```

## 300M production run

```bash
sbatch \
  --export=ALL,\
TAG=h1g1_continuous_300m_s3,\
BACKBONE=urmav2,\
MORPHOLOGY=continuous,\
TOTAL_ENVS=768,\
TOTAL_TIMESTEPS=300000000,\
UPDATE_EPOCHS=1,\
SEED=3,\
GOAL_TYPE=MorphGoalTrajMimicRootErr,\
TERMINAL_HANDLER=MorphologyAwareRootPoseTrajTerminalStateHandler,\
MAX_ROOT_DEVIATION=0.5 \
  viper_h1g1_urma.sbatch
```

Viper outputs:

```text
Slurm log:
  /ptmp/akalenik/frontier/h1g1_urma_<job-id>.out

Run directory:
  /ptmp/akalenik/frontier/crosstopo_<TAG>_<job-id>/
```

Do not attempt three or more topology branches on Viper until the recorded ROCm limitation is resolved. Use the local CUDA path for a three-plus-topology composition smoke.

---

# Experiment output locations

With the default local configuration:

```text
experiments/urma_h1g1_singlemotion_dance2_subject4_s3/
├── DESIGN.md
├── STATE.md
├── RESULTS.md
├── logs/
│   ├── 01_retarget.log
│   ├── 02_verify_fk.log
│   ├── 03_targeted_tests.log
│   ├── 04a_preflight_catalog2.log
│   ├── 04b_preflight_continuous.log
│   ├── catalog2_smoke_s3.log
│   └── ...
├── manifests/
│   ├── provenance.txt
│   └── retarget_manifest.json
├── checkpoints/
│   ├── preflight_catalog2/preflight.json
│   ├── preflight_continuous_512/preflight.json
│   ├── catalog2_smoke_s3/
│   │   ├── manifest.json
│   │   └── checkpoint_final/
│   ├── continuous_60m_s3/
│   └── continuous_300m_s3/
├── metrics/
│   ├── fk_verification.json
│   ├── policy_vs_zero_*.json
│   ├── fk_tracking_random_*.json
│   ├── fk_tracking_nominal_*.json
│   ├── fk_tracking_catalog_*.json
│   ├── contact_*.json
│   └── pipeline_audit.json
├── media/
│   ├── *_randomized_dynamics_nominal_mesh.mp4
│   ├── *_nominal.mp4
│   ├── *_zero_action.mp4
│   ├── target-vs-achieved contact plots
│   └── RENDER_CAVEAT.txt
└── handoff/
```

The evaluation catalog includes:

```text
nominal:
  [1.00, 1.00, 1.00, 1.00]

low training corner:
  [0.90, 0.70, 0.50, 0.70]

high training corner:
  [1.12, 1.50, 2.00, 1.30]

mild OOD leg length:
  [1.15, 1.00, 1.00, 1.00]
```

---

# Target-sphere videos

Yes, target visualization should be part of the primary evidence.

However, the spheres must come from the exact same provider, morphology, topology reference, and trajectory phase that the reward uses. A visually plausible reconstruction is not sufficient.

The repository already established the correct principle in `c15_render_targets.py`: target markers must come from the same reference provider and frame index as the reward. It used mocap spheres because the standard visualizer was unreliable on the tested MuJoCo version.  

For the cross-topology renderer, I recommend two visual layers:

1. Five large semantic markers matching the actual reward sites:

   * upper body;
   * left hand;
   * right hand;
   * left foot;
   * right foot.

2. An optional translucent reference skeleton or small joint markers showing the full target pose.

The five reward markers are mandatory. Full per-joint spheres alone can look correct while the task-space site or orientation target is wrong.

Each rendered frame should also include:

```text
family
four morphology values
trajectory ID
trajectory phase
morphology generation
mean target error
absorbing/reset state
```

It should draw lines from achieved sites to target sites and write a sidecar JSON containing the exact phase and morphology for every frame.

The current general cross-topology renderer replays rollout `qpos` on the nominal CPU model. Therefore:

* randomized dynamics affected the rollout;
* randomized geometry is not faithfully represented in the MP4;
* the existing MP4 is useful for behavior inspection;
* it is not sufficient evidence that sampled-body reference targets are correct.

Dynamic MJX morphology deliberately leaves meshes nominal in this path. 

I wrote the complete renderer patch specification, including acceptance tests:

[Target-sphere renderer specification](sandbox:/mnt/data/h1g1_urma_pipeline/TARGET_SPHERE_RENDERER_PATCH.md)

Until that patch is implemented, `eval_contact_quality.py` is the more trustworthy visual source. It reconstructs the sampled CPU body and produces target-versus-achieved foot and hand overlays. 

---

# What to inspect after the run

## Systems checks

The pipeline is not ready for a learning conclusion unless all of these pass:

```text
exactly two topology groups: H1 and G1
one shared parameter tree
one optimizer state
action masks exactly [19, 23]
padded action means exactly zero
all observations, rewards, losses and actions finite
sampled-model descriptions correspond to the sampled model
H1 and G1 reference files are different topology-specific files
CPU/JAX reference target error <= 1e-5 m
corner-body targets differ from nominal targets
phase p differs from p+1
checkpoint reload reproduces actions
```

After the P0 patch, also require:

```text
goal target sites == reward target sites <= 1e-5 m
joint reward and FK target use the same clamped qpos
reset observation and reward use the same trajectory phase
valid tall bodies are not absorbing at reset because of nominal height
goal type and terminal type appear in manifest.json
root-deviation threshold appears in manifest.json
```

## Learning checks

Never inspect only the aggregate return. Examine H1 and G1 separately.

For each topology and each evaluation morphology, require:

```text
policy return > exact-reset zero-action return
policy site-tracking error < zero-action site-tracking error
policy episode length/survival > zero-action
no immediate-reset corner bodies
finite contact metrics
no severe stance-foot sliding
no systematic floor penetration
no high-frequency action or joint-position jerk
```

The auditor checks the explicit `every_robot_beats_zero_action` verdict where the evaluator supplies it.

The strongest first production conclusion requires the same direction across three independent training seeds. A single good seed should be described as an existence result, not a stable controller.

## Video checks

Inspect specifically for:

* spheres freezing while the robot/reference phase advances;
* phase resetting one frame too early or late;
* H1 markers accidentally being used for G1;
* markers staying at nominal-body positions after morphology changes;
* morphology changing while the observation still describes the previous body;
* a robot imitating joint motion while drifting away from the global reference;
* one topology learning while the other repeatedly falls;
* feet tracking their joint pose but missing the intended ground contact.

---

# Body skipping and rejection statistics

The current conservative cross-family sampler does not implement an admission/rejection loop. It samples a body and uses it. Consequently, the current code cannot truthfully report “zero bodies skipped”—there is no measured skip counter.

The provided auditor marks this as:

```text
UNSUPPORTED: body_rejection_accounting
```

rather than silently interpreting the absence of a field as zero.

Before adding wider or feasibility-changing randomizations—joint ranges, joint axes, foot geometry, severe dimensions—add bounded reset-time screening with:

```text
draws_total
accepted_total
rejected_total
resamples_total
resample_exhausted_total
rejections_by_reason
```

Recommended rejection reasons:

```text
nonfinite_model
invalid_mass_or_inertia
invalid_joint_range
reference_limit_violation
initial_penetration
initial_absorbing
fk_nonfinite
reference_screen_failed
```

Use a fixed maximum number of redraws. Exhaustion should fail closed rather than silently replace the body with nominal.

For the current four-dimensional safe box, the likely result is that every body is accepted, but that still needs to be logged as evidence.

---

# Selecting the motion

Use `dance2_subject4`, frames `19482–20282`, for the first integration smoke because it is the frozen cross-family window already exercised by the repository.

Do not treat it as the strongest final locomotion test. It has only about 10.9% stance frames, whereas the existing `walk1_subject1` 800-frame window has about 91.6% stance. 

After the dance integration passes, repeat the pipeline with the existing walk window:

```bash
export CLIP=walk1_subject1
export START_FRAME=10521
export FRAMES=800
export DURATION=8.0
export EXP="$REPO/experiments/urma_h1g1_singlemotion_walk1_subject1_s3"
```

Then rerun:

```bash
bash run_h1g1_urma_pipeline.sh init
bash run_h1g1_urma_pipeline.sh retarget
bash run_h1g1_urma_pipeline.sh verify
bash run_h1g1_urma_pipeline.sh preflight
bash run_h1g1_urma_pipeline.sh smoke
```

Before a long walk run, add a reference screen for:

```text
joint-limit violations
joint velocity/acceleration
foot-floor penetration
stance fraction
self-collision
root/path continuity
loop seam if looping
first-frame absorbing state
```

The broader repository contains several walk-reference constructions, including newer conditioned references. Therefore, the exact walk artifact used must be content-hashed and recorded rather than referred to only by clip name and frame range.

---

# Scaling to additional topologies

Add topologies one at a time.

For a new robot family, the admission sequence should be:

```text
1. Add the topology/environment specification.
2. Add action and observation layout information.
3. Define the family body names used by morphology mutation.
4. Define or map the common mimic sites.
5. Generate one topology-specific reference for the selected motion.
6. Screen that reference before RL.
7. Verify body-correct FK at nominal/low/high morphology.
8. Run a new-topology-only preflight.
9. Run H1 + new-topology catalog2 smoke.
10. Run H1 + G1 + new-topology composition smoke.
11. Only then include it in a long continuous-morphology run.
```

A topology must not enter the production set merely because its environment builds. Atlas is the existing counterexample: the environment and shared policy machinery worked, but the selected references violated joint limits, so its poor control was not interpretable as a policy failure.

The repository already passed a five-family systems smoke with H1, G1, Atlas, Talos, and Booster T1 using one tree and masks `[19, 23, 27, 30, 23]`. That demonstrates composition, not control quality. 

For a topology with more joints than the current maximum, increase the padded maximum and retrain. A checkpoint cannot represent slots that did not exist when its parameter shapes were created. For a topology-held-out test, those slots must either have been reserved during training or the architecture/runtime must support genuinely dynamic parameterized decoding.

---

# What to send back for analysis

Create a compact handoff archive:

```bash
bash collect_h1g1_handoff.sh \
  "$REPO/experiments/urma_h1g1_singlemotion_dance2_subject4_s3"
```

The archive excludes large checkpoint weights but includes:

```text
DESIGN.md
STATE.md
RESULTS.md
exact stage logs
git revision and dirty status
submodule state
retarget manifest
reference paths
FK verification JSON
preflight JSON
training manifests and gate evidence
policy-vs-zero evaluation
nominal/random/corner/OOD FK tracking
contact metrics
target overlay images
videos
pipeline audit
```

For a Viper run, include the Slurm `.out` file separately.

The most important files to return are:

```text
metrics/fk_verification.json
checkpoints/*/preflight.json
checkpoints/<run>/manifest.json
metrics/policy_vs_zero_*.json
metrics/fk_tracking_*.json
metrics/contact_*.json
metrics/pipeline_audit.json
logs/<training-run>.log
media/*
```

Checkpoint weights are only needed if the metrics or media reveal a problem that requires replay.

---

# Prepared artifacts

* [Complete helper bundle](sandbox:/mnt/data/h1g1_urma_pipeline_bundle.zip)
* [Bundle SHA-256](sandbox:/mnt/data/h1g1_urma_pipeline_bundle.zip.sha256)
* [Local staged runner](sandbox:/mnt/data/h1g1_urma_pipeline/run_h1g1_urma_pipeline.sh)
* [Corrected Viper launcher](sandbox:/mnt/data/h1g1_urma_pipeline/viper_h1g1_urma.sbatch)
* [Post-run artifact auditor](sandbox:/mnt/data/h1g1_urma_pipeline/audit_h1g1_pipeline.py)
* [Compact handoff collector](sandbox:/mnt/data/h1g1_urma_pipeline/collect_h1g1_handoff.sh)
* [No-FSQ production fixes](sandbox:/mnt/data/h1g1_urma_pipeline/PRODUCTION_FIXES_BEFORE_LONG_RUN.md)
* [Target-sphere renderer specification](sandbox:/mnt/data/h1g1_urma_pipeline/TARGET_SPHERE_RENDERER_PATCH.md)
* [Validation record](sandbox:/mnt/data/h1g1_urma_pipeline/VALIDATION.json)

The shell files passed `bash -n`, the auditor passed Python compilation, both long-run guards failed closed as intended, and the auditor passed synthetic smoke and production-contract fixtures. I did not execute MuJoCo/JAX training against the actual repository because the uploaded materials are Repomix snapshots and omit the executable checkout, binary reference files, and installed runtime environments.
