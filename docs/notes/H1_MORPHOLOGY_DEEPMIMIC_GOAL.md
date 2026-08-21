# Robust H1 DeepMimic Across Randomized Bodies

**Status:** implementation and experiment goal, not yet started  
**Written:** 2026-08-08  
**Primary family:** Unitree H1  
**Primary motion:** `dance2_subject4`  
**Primary policy:** ordinary LocoMuJoCo PPO MLP; URMA is an optional late comparison

## Mission

Build the smallest defensible pipeline in which **one policy** learns one
specific motion across multiple randomized bodies from the **same H1 family**.
Use LocoMuJoCo's standard imitation/DeepMimic machinery wherever possible,
retarget the motion appropriately for every body, and produce quantitative
evidence plus a video for every evaluated body in which the controlled robot and
its body-specific reference targets are visible together.

The first claim to earn is deliberately narrow:

> One ordinary MLP policy, trained with LocoMuJoCo's standard trajectory,
> `GoalTrajMimic`, `MimicReward`, and PPO components, tracks one fixed
> `dance2_subject4` window on at least four reproducibly randomized H1 bodies,
> using independently verified offline-retargeted references.

Only after this baseline works should the project test reference-cache scaling,
rolling batches of newly retargeted bodies, morphology grids/interpolation,
body-normalized task-space references, and an amortized retargeter. Exact online
IK and URMA are late experiments, not prerequisites for the main claim.

## Fixed scope and terminology

- Use `dance2_subject4`, because it is the established clip throughout this
  repository. The historical matched window is frames `19482:20282`; confirm
  that this is still the intended 800-frame window and record its frequency and
  content hash before training. Do not silently use an automatically selected
  window in one arm and the historical window in another.
- Use only H1 bodies with the same joint/action topology in this goal.
- Start with the nominal H1 plus four deterministic random bodies sampled from
  conservative, physically valid bounds. Expand to eight random bodies only
  after the four-body run passes its gates.
- "One policy" means one actor parameter set and one optimizer/training state.
  Separate policies rendered in one grid do not satisfy the claim.
- "Offline retargeting" means solving or constructing a reference once for a
  `(motion, body)` pair and storing the result. Selecting that reference during
  a rollout is an indexed lookup, not another retargeting solve.
- "Online retargeting" means producing a reference for a newly sampled body
  during the training process. Distinguish exact optimization from an
  amortized-network prediction.
- A "morphology refresh round" is the outer operation that produces a new body
  pool. Do not call it a PPO epoch, because PPO optimizer epochs have a separate
  meaning.

## Preserved technical judgment

### Is online retargeting simple?

For hundreds of fixed generated bodies, morphology-aware references are
practical, but full nonlinear retargeting should not run independently inside
every environment at every reset or control step. The scalable design is cached
or amortized retargeting.

Separate the randomizations first:

- **Mass, inertia, damping, friction, and actuator strength:** no kinematic
  retargeting is required. The reference can stay unchanged, although the policy
  still has to become robust to the new dynamics.
- **Moderate limb-length or torso-scale changes with unchanged topology:** full
  IK may be avoidable. Apply the nominal joint trajectory to the current body,
  use that body's forward kinematics, and construct body-specific task-space
  targets from the result.
- **Large proportions, asymmetry, changed joint limits, or altered feet/contact
  geometry:** these need a genuinely morphology-aware reference or an explicit
  feasibility projection.

For hundreds or thousands of enumerated bodies, the preferred procedure is:

1. Generate each body deterministically.
2. Retarget the selected motion once.
3. Store `qpos`, `qvel`, root state, body/site positions and orientations,
   linear/angular velocities, contact schedule, phase metadata, and provenance.
4. During training, sample a body and retrieve the matching reference tensors by
   body ID.

Previous measurements in this repository were approximately 1--9 seconds to
retarget an eight-second clip/target. At that rate, 1,000 bodies times four
motions costs roughly 1.1--10 hours once, after which resets are constant-time
lookups. Re-benchmark this cleanly; do not assume the old number applies to a new
retargeter or machine.

For continuously sampled, almost-never-repeated bodies, test these progressively:

1. **Current-body forward kinematics** with body-normalized semantic targets.
2. **A morphology grid:** four dimensions with five values each gives
   `5^4 = 625` cached bodies; interpolate between neighboring references.
3. **An amortized retargeter:** train a small model such as
   `reference = f(morphology, phase, motion_id)` from offline teacher solutions.
4. **Exact online optimization:** only if the cheaper representations cannot
   meet the measured accuracy requirement.

Do not retarget at every control step. If arbitrary continuous morphology truly
requires a new solve, produce the whole clip at a morphology refresh boundary,
or use an amortized predictor inside training.

### Can the project reuse a default DeepMimic implementation?

Yes, most of it. LocoMuJoCo is the closest fit and is already vendored and used
by this repository. It supplies trajectory handling, imitation environments,
`GoalTrajMimic`, `MimicReward`, PPOJax, robot-to-robot retargeting utilities, and
MJX execution. The relevant existing entry points include:

- [`loco-mujoco/README.md`](loco-mujoco/README.md)
- [`scripts/train_deepmimic_dance.py`](scripts/train_deepmimic_dance.py)
- [`scripts/scaling/online_h1_train.py`](scripts/scaling/online_h1_train.py)
- [`scripts/scaling/online_h1.py`](scripts/scaling/online_h1.py)
- [`loco-mujoco/loco_mujoco/core/reward/trajectory_based.py`](loco-mujoco/loco_mujoco/core/reward/trajectory_based.py)
- [`loco-mujoco/loco_mujoco/smpl/retargeting.py`](loco-mujoco/loco_mujoco/smpl/retargeting.py)

There is no zero-code combination of LocoMuJoCo, arbitrary randomized MJX
models, URMA2, RL-X, and morphology-aware references. The clean boundary is:

```text
LocoMuJoCo imitation environment
  + standard trajectory handler / phase / RSI / termination
  + standard GoalTrajMimic and MimicReward implementation
  + standard PPOJax MLP baseline
  + H1 randomized-body provider                 [custom adapter]
  + morphology-aware reference/cache provider   [custom adapter]
  + diagnostic target renderer                  [custom, visualization only]
```

Do not copy only `MimicReward` into the current custom `loco_mjx` task and call
the result standard DeepMimic. The trajectory handler, reference-state
initialization, phase semantics, site definitions, root state, termination, and
reference velocities are part of the method too.

The best minimal-change path is therefore to make the LocoMuJoCo imitation
environment the source of truth and attach the smallest possible morphology and
reference adapters. If integration forces a choice, preserve the standard
imitation stack and adapt the policy/body plumbing around it.

### Other repositories worth treating as references

- [LocoMuJoCo](https://github.com/robfiras/loco-mujoco) is the preferred working
  foundation for this JAX/MJX/ROCm project.
- [MimicKit](https://github.com/xbpeng/MimicKit), from the original DeepMimic
  author, is a useful modern gold-standard reference, but adopting its main
  parallel backends would be a stack migration.
- [ProtoMotions](https://github.com/NVlabs/ProtoMotions) is a useful reference
  for scalable motion imitation and PyRoki retargeting, but its main training
  stack is NVIDIA-oriented.
- [mjlab](https://github.com/mujocolab/mjlab) and
  [BeyondMimic](https://github.com/HybridRobotics/whole_body_tracking) are useful
  independent baselines for modern whole-body tracking, reference-state
  initialization, adaptive sampling, and full-body rewards. They are not the
  first implementation choice for the current AMD/ROCm JAX stack.
- [GMR](https://github.com/YanjieZe/GMR) is a candidate offline retargeting
  teacher with H1/H1-2 support.
- [PyRoki's humanoid-retargeting example](https://chungmin99.github.io/pyroki/examples/10_humanoid_retargeting/)
  is a candidate basis for batched JAX retargeting or teacher-data generation.

## Standard-versus-custom provenance rule

The final report must include a provenance table like this, populated with exact
module paths, upstream commit hashes, and local diff status:

| Component | Required starting point | Custom work allowed |
|---|---|---|
| Motion loading/cropping | LocoMuJoCo `Trajectory` and dataset configuration | Fixed clip/window selection and manifest only |
| Goal observation | LocoMuJoCo `GoalTrajMimic` or its supported successor | Adapter for body-indexed reference data |
| Imitation reward | LocoMuJoCo `MimicReward` | Parameter configuration; no rewritten reward kernel in the baseline |
| RSI/phase/termination | LocoMuJoCo behavior | Configuration only unless a documented defect is proven |
| Optimizer/policy | LocoMuJoCo PPOJax ordinary MLP | Multi-body environment scheduling adapter |
| Body generation | Existing H1 generator/online morphology code | Validation, deterministic catalog, dynamic model adapter |
| Retargeting | LocoMuJoCo/GMR/PyRoki where applicable | Morphology parameter binding, caching, validation |
| Target visualization | LocoMuJoCo visualizer if compatible | Visualization-only spheres/ghost if required |
| URMA | Existing URMA2 implementation | Optional late matched comparison only |

Configuration is not the same as rewriting an algorithm. Explicitly setting
reward weights while using the upstream reward implementation is acceptable,
but the source and reason for each non-default value must be recorded. The first
nominal control should use the closest runnable upstream example/configuration
before any tuning.

Do not edit vendored framework code unless a minimal, tested upstream defect
requires it. Prefer extensions or adapters outside `loco-mujoco/`. If a vendor
patch is unavoidable, isolate and document it so the experiment can also run
against an unmodified checkout.

## Required experiment ladder

Every phase below must produce a small manifest and a decision. Negative results
are valid. Skipping a phase requires a concrete written reason and evidence that
it is inapplicable, not merely inconvenient.

### Phase 0 -- Audit, freeze, and pre-register

1. Record the root repository revision, LocoMuJoCo submodule revision and diff,
   JAX/MuJoCo versions, hardware/backend, exact launch command, and RNG seeds.
2. Confirm the motion is `dance2_subject4` and freeze one exact frame window.
   Save a hash of the source and cropped trajectories.
3. Trace every behavior-defining component used by the baseline and label it
   `upstream`, `configuration`, `existing local adapter`, or `new custom code`.
4. Freeze the initial conservative morphology bounds before observing training
   results. The existing online H1 dimensions are leg length, arm length,
   shoulder width, and torso mass. Confirm which model arrays and collision or
   visual geometry each dimension actually changes.
5. Define metrics and pass/fail thresholds before the shared-policy long run.
6. Inspect existing artifacts and reuse valid components, but do not declare an
   old run matched unless its clip, reward, reference, body, control mode, and
   evaluation protocol are identical.

Output: `audit.md`, `provenance.json`, frozen experiment configuration, and a
short inventory of reusable versus superseded code.

### Phase 1 -- Unmodified nominal LocoMuJoCo sanity baseline

Run the nearest stock LocoMuJoCo H1 imitation example on the frozen motion
window using its standard trajectory/goal/reward/PPO stack and an ordinary MLP.
Use the smallest smoke budget that proves reset, one compiled update, evaluation,
and rendering; then run a sufficient baseline only if the smoke is clean.

Required controls:

- Kinematic playback of the reference.
- Zero-action rollout from matched reset states.
- Deterministic trained-policy rollout.
- Per-term imitation metrics, not only summed return.

**Gate:** if nominal H1 cannot learn the selected motion with the standard
stack, stop progression and diagnose the clip, control mode, reference,
termination, and reward configuration. Do not add morphology, URMA, a learned
retargeter, or a new reward to hide a broken nominal baseline.

### Phase 2 -- Deterministic initial body catalog

Create the nominal control plus four randomized H1 bodies from fixed seeds.
Start within conservative bounds and keep topology, joint ordering, observation
layout, and action layout identical. If all four pass, optionally extend to
eight randomized bodies.

Every catalog record must include:

- stable body ID and generation seed;
- raw and normalized morphology descriptor;
- generator revision, bounds, and source configuration;
- hashes of the resulting model-defining arrays or XML;
- changed kinematic, mass/inertia, actuator, joint-limit, site, collision, and
  visual arrays;
- topology signature and action/joint ordering;
- physical-validity results: finite values, positive mass/inertia, valid joint
  ranges, no impossible initial penetrations, and a finite reset/step.

Do not call bodies different based only on four descriptor values. Prove that
the intended physical model arrays changed. State explicitly if visual meshes
or collision geometry remain nominal.

### Phase 3 -- Offline retargeting oracle and visual QA

Retarget the exact frozen motion separately for every catalog body. Prefer the
standard LocoMuJoCo retargeting route where it supports the required H1 variant;
otherwise use GMR or PyRoki as a teacher and document the adapter. A nominal
joint-angle/FK construction may be included as a distinct baseline, not silently
called optimized retargeting.

Store a versioned, immutable reference artifact for every body containing at
least:

- root translation and orientation;
- joint positions and velocities;
- selected body/site positions and orientations;
- selected linear and angular velocities;
- phase/frequency and frame count;
- foot-contact schedule or the signals used to derive it;
- body ID, body hash, motion hash, retargeter/config hash, and quality metrics.

Before RL, produce **two diagnostic videos per body**:

1. Kinematic reference playback on that randomized body.
2. The randomized body plus the exact target markers that the reward will use.

The target marker coordinates must come from the same reference provider and
frame index as the reward. Do not reconstruct a separate approximate target
only for rendering. Recommended visualization:

- translucent reference body if a supported standard renderer provides it;
- otherwise colored spheres at pelvis/upper-body, hands, and feet, optionally
  connected by thin lines;
- consistent colors and sphere sizes across bodies;
- a clear body ID, morphology values, motion phase/frame, and mode label;
- visually distinct current sites and target sites if both are drawn.

The existing scripts disable `GoalTrajMimic(visualize_goal=True)` because the
old arrow path can crash with the installed MuJoCo 3.9 bindings. First test the
current supported LocoMuJoCo goal visualization (including `GoalTrajMimicv2` if
appropriate). If it is incompatible, implement a visualization-only overlay or
mocap-sphere path; do not replace the upstream reward for this reason.

Quantify retarget quality before training:

- root-relative and world-space site MPJPE;
- root position/orientation error;
- site orientation and velocity error;
- joint-limit margin and projection count;
- foot sliding, penetration, and contact-timing mismatch;
- discontinuity at clip loop boundaries, if looping is used.

**Gate:** a reference that is visibly or quantitatively invalid must be fixed or
excluded with a documented reason. Do not ask PPO to compensate for a bad
teacher.

### Phase 4 -- Controllability controls

Before the shared policy, train the minimum diagnostic set of single-body
specialists using the same standard stack:

- nominal H1;
- the hardest-looking initial random body after retarget QA;
- any body on which kinematic/reference checks are marginal.

Specialists are diagnostic controls, not the deliverable. They separate
"reference or actuator infeasible" from "shared training failed."

**Gate:** if a specialist cannot learn its own valid reference, diagnose
actuation, joint limits, ground/root alignment, reward scales, and termination.
Do not proceed directly to interpolation or an amortized retargeter, because
those are less exact than the offline oracle.

### Phase 5 -- Main baseline: one MLP policy, four then eight bodies

Train one ordinary PPO MLP policy across the nominal body and four randomized
bodies, with every body represented in the same training run. Reuse existing
multi-body scheduling/vectorization code if it is correct, while keeping the
LocoMuJoCo trajectory, goal, reward, and PPO logic intact.

Requirements:

- Balanced exposure per body, counted and reported. Round-robin is acceptable
  for the tiny static baseline; simultaneous grouped environments are preferable
  when practical.
- The body and its reference must always be selected by the same immutable body
  ID. Assert against mismatches.
- Start with the standard unconditioned MLP. Add a compact morphology descriptor
  as a matched ablation only after the baseline. Do not start with URMA.
- Use identical clip, control mode, reward, rollout horizon, and evaluation
  reset keys for all bodies.
- Save intermediate checkpoints and choose the reported checkpoint using a
  pre-registered aggregate that cannot hide a failing body, such as worst-body
  normalized score followed by mean score.
- Run enough independent seeds for the final claim; three is the default. A
  single seed is only a smoke or feasibility result.

Evaluate the shared policy separately on every training body and at least two
seed-disjoint held-out randomized bodies. Compare it with matched zero-action
rollouts and, where available, a single-body specialist.

For every body, save:

- one deterministic policy video with target spheres/ghost visible;
- one reference-only diagnostic video;
- return and every reward term;
- episode length and absorbing/fall rate, distinguishing horizon completion
  from falling;
- root/site/joint tracking errors and contact metrics;
- morphology/body ID and checkpoint hash.

Also save a synchronized grid video, but never use the grid as a substitute for
the per-body videos.

Suggested success bar, to be finalized in Phase 0 before long training:

- every initial random body clearly beats its matched zero-action control;
- no body is hidden behind an average;
- the shared policy reaches at least 80% of its matched specialist score or
  another pre-registered normalized tracking threshold on each body;
- performance holds across at least three seeds;
- target videos and numerical metrics agree about whether retargeting is valid;
- at least two held-out moderate bodies are reported even if they fail.

If four randomized bodies pass, repeat at eight. If specialists work but the
shared policy fails, investigate sampling balance, observation conditioning,
normalization, capacity, and curriculum before changing the imitation method.

### Phase 6 -- Offline cache and retargeting scalability benchmark

Turn the validated offline oracle into a content-addressed cache keyed by at
least:

```text
(robot family, topology signature, body/model hash,
 motion/window hash, retargeter version/config hash, output schema version)
```

Benchmark cold and warm paths for body counts such as `1, 4, 8, 32, 128, 512,
1000`, stopping early only for a documented resource limit. Measure:

- wall time and CPU time per body-motion pair;
- throughput in bodies/hour;
- peak memory and output bytes per body-second of motion;
- cache-hit lookup latency;
- serial versus parallel speedup and efficiency;
- failure/retry rate and invalid-body rate;
- training throughput with cache lookup included;
- time spent in body generation, model loading/compilation, IK/optimization,
  forward kinematics, validation, and serialization.

Run controlled worker-count sweeps rather than launching one process per body.
Record the machine and CPU/GPU allocation. Verify cache determinism by
regenerating a sample and comparing hashes and numerical tolerances.

### Phase 7 -- Rolling parallel retarget-and-train pool

Test the user's proposed scaling mode: periodically create `X` new morphologies,
retarget them in parallel, then continue training from their cached references.
Use morphology refresh rounds, not per-control-step solves.

Start with `X = 8`, then try `32` and `128` if the smaller arm is stable. The
producer and trainer should be decoupled:

```text
round k trainer consumes immutable validated pool k
                         |
                         +-- producer prepares pool k+1 in parallel
next boundary: validate + atomically publish pool k+1, then switch by body ID
```

Required safeguards:

- Training never reads a partially written reference.
- Each body/reference pair is hash-checked before publication.
- Failed bodies are recorded, not silently replaced in a way that changes the
  sampling distribution.
- Retarget workers are bounded and cannot starve the trainer or shared cluster.
- Replacing the body pool must not force one XLA compilation per XML/body. If it
  does, retain the static baseline but move this scaling arm to dynamic
  same-topology model arrays and indexed reference tensors.
- Preserve replay of older bodies or use a mixed old/new pool so catastrophic
  forgetting can be measured.

Compare at least these schedules at matched total environment steps:

- fixed cached catalog;
- rolling replacement;
- rolling mixture of old and new bodies;
- continuous dynamic bodies with the cheapest valid reference method.

Measure policy quality on a fixed audit set after every refresh, forgetting,
body coverage, reference-production delay, total wall time, and training
throughput overhead. Success means reference production is either hidden behind
training or adds a small pre-registered overhead while maintaining body-specific
tracking quality.

### Phase 8 -- Body-normalized task-space reference

Build the cheapest morphology-aware approximation before attempting a learned
retargeter:

- express semantic site positions root-relative;
- normalize positions by appropriate nominal segment lengths or height;
- reconstruct them through the current H1 kinematic tree;
- keep orientations and phase/contact schedules explicit;
- adjust root height from current foot/leg geometry;
- derive current-body velocities consistently rather than reusing incompatible
  nominal velocities;
- project or mask infeasible joint targets in a documented way.

Compare this method with the offline-retargeted oracle on the same bodies before
RL, then in matched policy training. Report accuracy and speed. If it does not
meet the oracle-relative threshold, keep it as a negative result rather than
tuning the reward until the difference disappears.

### Phase 9 -- Morphology grid and interpolation

Use the existing four-dimensional H1 morphology parameterization. Begin with a
small grid or space-filling design inside the validated bounds; only scale to
the illustrative `5^4 = 625` table after a smaller smoke passes.

Test interpolation on bodies excluded from the grid. Interpolate a
representation appropriate to the quantity: treat quaternions/orientations
correctly, preserve phase, recompute velocities where needed, and run current-
body FK after interpolation. Do not linearly interpolate arbitrary rotations or
contact labels without validation.

Compare:

- nearest cached body;
- multilinear/barycentric interpolation;
- normalized task-space reconstruction;
- exact offline retarget oracle.

Report lookup/interpolation cost, reference error, policy quality on grid and
held-out bodies, and storage scaling.

### Phase 10 -- Amortized retargeter

Only after the offline oracle and cache/grid baselines exist, train a small
morphology-conditioned reference model from teacher retargets. A reasonable
interface is:

```text
inputs:  morphology descriptor, phase, optional motion ID
outputs: root state, joint state, semantic body/site targets, optional contact
```

For a single motion, motion ID can be omitted. Use train/validation/test splits
by body seed, with boundary and moderate OOD sets. Preserve periodic phase using
an appropriate encoding. Enforce or measure quaternion normalization, temporal
smoothness, joint limits, contact consistency, and loop continuity.

First evaluate it as a supervised retargeter against the exact offline teacher.
Only then use its frozen outputs in RL. Do not jointly train the retargeter and
policy in the first experiment, because policy reward can hide target drift.

Compare reference generation throughput, memory, oracle error, and downstream
policy tracking with the cache/grid/task-space methods.

### Phase 11 -- Exact online optimization, only if still justified

If the previous methods fail on important continuous morphologies, prototype a
batched exact solver at morphology refresh boundaries or episode reset. Never
start with per-control-step optimization. Benchmark whether JAX/PyRoki batching
or CPU worker pools are faster and whether solver latency can overlap training.

An exact-online arm is successful only if its additional reference accuracy
produces a meaningful policy-quality gain over the offline/amortized methods at
an acceptable wall-time cost.

### Phase 12 -- Optional architecture ablations

URMA is not required for the single-family, fixed-topology goal. Once the
standard MLP baseline is sound, run matched comparisons in this order:

1. ordinary MLP, no explicit morphology descriptor;
2. ordinary MLP plus compact morphology descriptor;
3. existing URMA2 with the same observation information, reward, body schedule,
   steps, seeds, and evaluation set.

Do not spend the primary experiment budget on URMA unless it beats the simpler
MLP consistently on held-out bodies. Report parameter count and throughput as
well as tracking quality. Existing project evidence already warns that URMA2 is
not automatically better for fixed-topology H1.

## Failure-routing rules

Use this sequence instead of blindly moving to a more complicated method:

| Failure | Next action |
|---|---|
| Nominal stock baseline fails | Fix environment/reference/control/reward integration; do not add morphology |
| Retarget video or metrics are wrong | Fix or reject the reference; do not ask RL to compensate |
| A random-body specialist fails | Check feasibility, actuators, joint limits, root/grounding, and termination |
| Specialists pass but shared policy fails | Check body balance, normalization, observation/descriptor, curriculum, and capacity |
| Offline cache works but is slow to build | Parallelize/cache/profile; then test grid or amortization |
| Grid/interpolation is inaccurate | Use denser/adaptive samples, task-space reconstruction, or amortized teacher model |
| Amortized model is inaccurate | Improve teacher coverage/constraints; exact online solve is the last fallback |
| MLP already succeeds robustly | Keep it; URMA becomes an optional scientific ablation |

Approximations are not fallbacks for a broken oracle. The exact offline path is
the validation target against which the scalable alternatives must be judged.

## Metrics and reporting requirements

Videos are diagnostic evidence, not the only evidence. Report per body and per
seed; never hide a failed body in an aggregate.

### Reference quality

- world and root-relative position error per semantic site;
- orientation geodesic error;
- joint position and velocity error;
- root pose and velocity error;
- foot-contact precision/recall or timing mismatch;
- foot sliding and ground penetration;
- joint-limit violations/projection frequency;
- temporal acceleration/jerk and loop discontinuity.

### Policy quality

- total return and every imitation reward component;
- episode length;
- absorbing/fall rate, separate from horizon/clip completion;
- phase coverage before termination;
- tracking metrics above during policy rollout;
- matched zero-action and specialist ratios;
- worst-body, median-body, and mean-body score;
- held-out-body results;
- three-seed distribution for any positive robustness claim.

### Systems/scaling quality

- compile time and training steps/second;
- retarget seconds/body-motion and bodies/hour;
- cache size and lookup latency;
- parallel speedup and worker efficiency;
- memory peak;
- producer/trainer overlap and percentage wall-time overhead;
- invalid/failing morphology rate;
- exact number and frequency of exposures per body.

## Required deliverables

Create one timestamped experiment directory, for example:

```text
experiments/h1_morphology_deepmimic_<timestamp>/
  README.md
  report.md
  audit.md
  provenance.json
  config/
  body_catalog/
  references/
  checkpoints/
  evaluations/
  scaling/
  videos/
  images/
  logs/
```

The final `report.md` must contain:

- the exact claim earned and claims not earned;
- standard-versus-custom provenance;
- body catalog and motion/window identity;
- phase-by-phase decisions, including negative results and skipped-arm reasons;
- per-body/per-seed tables and learning curves;
- retarget accuracy and scaling plots;
- links to every body's reference and controlled videos;
- cache/rolling-pool/grid/task-space/amortized comparisons;
- MLP/descriptor/URMA comparison if Phase 12 was reached;
- reproducible commands and environment/hardware information;
- known limitations, especially unchanged mesh/collision geometry, bounded
  morphology, single motion, and single topology.

Tests must cover at least:

- deterministic body regeneration and hashes;
- body/reference ID alignment;
- reference schema, shapes, finite values, normalized quaternions, and phase;
- cached versus freshly generated reference equivalence;
- reward and renderer consuming the same target frame/provider;
- balanced body scheduling;
- one jitted multi-body reset/step/update without NaNs;
- checkpoint reload/evaluation determinism within tolerance;
- atomic rolling-pool publication and rejection of partial/mismatched caches.

Do not overwrite prior experiment artifacts. Smoke-test locally before launching
long jobs. Respect shared-cluster limits and record every submitted job and
result, including failures.

## Definition of done

The goal is complete only when all of the following are true:

1. A nominal standard LocoMuJoCo DeepMimic baseline is reproduced.
2. At least four deterministic randomized H1 bodies have validated offline
   body-specific references.
3. One ordinary MLP policy trains across all initial bodies and is evaluated per
   body across at least three seeds, with matched controls.
4. Every evaluated body has a reference-validation video and a trained-policy
   video showing the exact reward targets as spheres, a ghost, or an equivalent
   clear overlay.
5. The offline retargeting cache has a measured scalability curve.
6. A rolling parallel retarget-and-train pool has been tested at more than one
   pool size and compared with the fixed cache.
7. Body-normalized task-space references, a morphology grid/interpolation, and
   an amortized retargeter have each been implemented and evaluated against the
   offline oracle, or a concrete technical reason proves an arm inapplicable.
8. Exact online optimization has either been evaluated or explicitly rejected
   from measured evidence showing it is unnecessary or unacceptably expensive.
9. URMA has either received a matched late comparison or is explicitly omitted
   because the simpler MLP already meets the single-family goal and compute is
   better spent on retargeting evidence.
10. A final report clearly identifies upstream framework code, configuration,
    and custom adapters, and states negative results and limitations honestly.

## Goal command to use later

Copy and send the following as a new message when ready to start the work. This
wording explicitly creates a persistent goal; it does **not** start anything by
being stored in this file.

```text
Create a goal with this objective: Read H1_MORPHOLOGY_DEEPMIMIC_GOAL.md completely and execute it end-to-end. Build the robust single-policy, single-motion, randomized-H1 DeepMimic pipeline described there, starting with the simplest unmodified LocoMuJoCo nominal baseline and deterministic offline-retargeted bodies. Use dance2_subject4 and an ordinary MLP first; URMA is optional and late. Preserve LocoMuJoCo's standard trajectory, goal, MimicReward, RSI/termination, and PPO implementations wherever possible, and explicitly document every custom adapter. Train one shared policy on at least four randomized H1 bodies, save per-body reference-validation and trained-policy videos with the exact reward targets visible as spheres or a reference ghost, and report per-body/per-seed metrics without hiding failures. Then execute and compare the fixed cache, offline retargeting scalability benchmark, rolling parallel retarget-and-train pools, body-normalized task-space references, morphology grid/interpolation, and amortized retargeter, using the validated offline retarget as the oracle. Try exact online optimization only if measured evidence justifies it. Follow the file's failure gates, tests, provenance rules, acceptance criteria, and definition of done; start with smokes, preserve existing work, do not overwrite old artifacts, and continue until every required arm is completed or is shown with concrete evidence to be inapplicable. Produce the timestamped experiment artifacts and final report specified in the file.
```

If the client supports slash-command syntax, the equivalent compact form is:

```text
/goal Read H1_MORPHOLOGY_DEEPMIMIC_GOAL.md completely and execute it end-to-end, following its staged experiments, gates, deliverables, and definition of done. Begin with the standard LocoMuJoCo MLP plus offline-retargeted dance2_subject4 H1 bodies; do not begin with URMA or exact online IK.
```
