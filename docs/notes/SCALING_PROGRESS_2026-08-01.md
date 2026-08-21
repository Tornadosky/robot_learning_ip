# Morphology and Motion Scaling Progress — 2026-08-01

**Forward roadmap (10 steps to morphology-independent multi-skill control):**
[`SCALING_ROADMAP.md`](SCALING_ROADMAP.md)

## Executive result

The repository now has two complementary ways to replace round-robin training:

1. **Grouped-static parallel training** puts several validated MJX/XML bodies in
   every rollout and every PPO update. It exactly supports the existing H1
   variants, but every body adds another static XLA branch.
2. **Online same-topology training** uses one H1 MJX graph and makes morphology
   parameters dynamic per environment. It supports hundreds of thousands of
   unique bodies at once and can resample a new body after every episode. This is
   the route to millions of embodiment exposures.

The main scaling conclusion is that XML generation and reference retargeting are
not the bottlenecks. For static bodies, XLA compilation grows with the number of
distinct model branches. For online bodies, PPO/MJX training and accelerator
memory dominate; online body sampling itself is effectively free.

For the current fixed-topology H1 distribution, the descriptor-conditioned MLP
is the strongest measured policy as well as the fastest. URMA/URMAv2 are now
working research backbones, but their main justification is the future
variable-joint/topology problem, not this four-coordinate H1 case.

On one Viper MI300A, that MLP completed with 65,536 environments at a full
64-step rollout and with 262,144 environments at a 16-step rollout. The
quality-validated production point remains 8,192×64: its 100M-step run exposed
an estimated 1.305M bodies and beat zero action on 96.9% of 1,024 held-out bodies.

Most capacity rows are deliberately short systems runs. The 100M-step online MLP
now adds strong held-out interpolation evidence across the sampled range, but it
is not yet a convergence claim or an extrapolation test beyond those bounds.

## Repository starting point and research progress

The new parallel paths build on substantial prior training rather than replacing
an empty baseline:

At this snapshot, `generated_variants/` contains 12 named H1, 14 named G1, and
seven named Go2 morphology directories (plus two randomized/auxiliary H1
directories). Training evidence is concentrated on H1 and G1; the Go2 set is a
static morphology/gallery line, not a demonstrated locomotion policy family.

| Existing line of work | Measured result | What it means for scaling |
|---|---|---|
| Independent morphology matrix | The driver defines 20 H1/G1 × body × motion cells and launches one retarget/training subprocess per cell | Useful experts and controls, but one policy per cell and sequential execution do not share experience |
| Round-robin shared H1 walk | One policy rotated through four bodies for 240M steps; trained-body episode lengths were 670–704 and a moderate held-out body reached 381 | A shared morphology policy works; coverage is the remaining generalization lever |
| Round-robin shared H1 dance | Four bodies, 629M steps; episode lengths 639–684 and 0.84–0.87× the available dedicated experts | This is the repository's strongest multi-body result, but each 10M-step segment still sees only one body |
| Six-body coverage / FiLM control | A held-out moderate short-leg body improved from length 381 to 688 without FiLM and 703 with FiLM; the extreme short-heavy body remained near 105–117 | Bracketing the test body with more training morphologies mattered; FiLM itself did not explain the gain |
| Multi-motion DeepMimic | Four-clip H1 training plateaued near length 140–150 at both 600M and 2B steps; clip-ID conditioning did not break it | More steps in the same goal-conditioned tracking formulation are not the next lever; AMP/adversarial motion learning is justified |
| Offline shared-policy BC | Five G1 variants and 2M transitions; best FiLM model reached 0.53× expert return and failed zero-shot on the held-out extreme body | Distillation alone does not solve closed-loop multi-body robustness |

The old round-robin trainer is scientifically valid and produced the flagship
result, but its optimizer sees one body's rollout distribution for a long segment
before moving to the next. The supervisor's proposed layout instead puts every
body in the same rollout and PPO minibatch. The grouped-static implementation below
does exactly that, while the online implementation removes the need to enumerate
bodies at all.

The detailed source evidence is in
[`experiments/goal_20260710_223728/report.md`](experiments/goal_20260710_223728/report.md),
[`experiments/frontier_20260714_224204/report.md`](experiments/frontier_20260714_224204/report.md),
and
[`experiments/overnight_shared_policy_20260629_094141/report.md`](experiments/overnight_shared_policy_20260629_094141/report.md).

## What is implemented

| Capability | Entry point | Current scope |
|---|---|---|
| Parallel PPO over static bodies | `scripts/scaling/parallel_multimorph_train.py` | Same observation/action topology; one homogeneous vmap per XML body, combined in one JIT and one PPO update |
| Online morphology generation | `scripts/scaling/online_h1_train.py` | H1 with dynamic leg length, arm length, shoulder width, and torso mass per environment; one selected LAFAN1 clip per run |
| Per-episode online resampling | `--resample-per-episode` | A reset samples a new body and returns an observation from that new body |
| URMA-style backbone | `--backbone urma` | Joint-description encoder, joint-conditioned decoder, padded valid-joint mask |
| URMAv2-style backbone | `--backbone urmav2` | Joint-set-normalized attention, dot-product action decoder, WeightNorm core, joint-conditioned standard deviation |
| Held-out embodiment evaluation | `scripts/scaling/evaluate_online_policy.py` | Reloads a checkpoint, samples a disjoint fixed-body batch, and performs body-for-body policy versus zero-action comparison |
| Multi-body, multi-motion AMP | `scripts/scaling/parallel_multimorph_amp_train.py` | Shared policy and discriminator over four H1 bodies and four LAFAN1 dance clips in the verified run |
| Cross-humanoid references | `scripts/scaling/cross_humanoid_retarget.py` | H1, G1, H1v2, Atlas, Talos, Apollo, Booster T1, ToddlerBot, and Fourier GR1T2 targets |
| Cross-topology shared PPO | `scripts/scaling/parallel_cross_humanoid_train.py` | Trains H1/G1/Atlas together; H1/G1/Atlas/ToddlerBot pass a four-topology preflight |
| Cross-topology evaluation | `scripts/scaling/evaluate_cross_humanoid_policy.py` | Reports same-seed policy/zero-action return and lifetime separately for every robot |
| Static generation benchmark | `scripts/scaling/benchmark_morphology_generation.py` | Generates into a temporary directory, writes XML/assets, and validates with MuJoCo |
| Viper launchers | `scripts/scaling/*.sbatch` | ROCm APU partition through the `viper11` SSH alias |

The online H1 body modifies compact kinematic and inertial model arrays. Version
1 deliberately shares the nominal collision/visual mesh because copying the
143k-vertex mesh per environment would remove most of the memory advantage. The
reference is canonical same-DOF joint-space motion; this is embodiment
randomization, not a fresh mesh-level retarget for every sampled body.

The four online coordinates are a conservative first distribution, not a hard
limit. Further same-graph candidates include segment-specific masses/inertias,
actuator gains/gears, joint damping/ranges, contact friction, and primitive foot or
limb dimensions. Each needs physical-validity bounds and a held-out sensitivity
test. Changing joint/geom counts or duplicating full mesh vertex arrays remains a
static-topology change and should not be treated as a cheap extra coordinate.

The URMA implementations are repository-native Flax adaptations of the published
ideas, not claims of byte-for-byte reproduction of the authors' training stack.
The network already accepts padded joint slots and a validity mask. The online
URMA simulator path currently exercises that design on the fixed 19-DoF H1
topology. A separate padded MLP bridge now executes across H1/G1/Atlas; merging
its topology masks with the URMAv2 actor is the next architecture step.

As an external reference point rather than a promise for this repository, the
URMAv2 paper reports one policy across 50 robot designs, 25,600 parallel
environments, up to 10 million training embodiments, and a 5-billion-step run in
about 40 hours on one A100. Their system includes the cross-topology morphology
generator and training infrastructure that this repository still needs.

## Measured scaling

### Why static branching is not the thousand-body solution

All local rows below use 2,000 total environments and the same short PPO budget.

| Static bodies × envs/body | Build | XLA compile | Train | Throughput |
|---:|---:|---:|---:|---:|
| 1 × 2,000 | 1.70 s | 19.34 s | 1.08 s | 7.13 M steps/min |
| 4 × 500 | 5.22 s | 54.87 s | 2.08 s | 3.69 M steps/min |
| 10 × 200 | 13.25 s | 141.13 s | 4.27 s | 1.80 M steps/min |

The exact supervisor-requested layout, 10 morphologies with 200 environments
each, therefore works in parallel. It is substantially better semantically than
round robin because every update sees every body, but branch count reduces
throughput and increases compile time.

On Viper, the same 10 × 200 layout completed 2,048,000 steps:

| XLA compile | Train | Throughput | Peak live device memory | Job |
|---:|---:|---:|---:|---:|
| 1,768.21 s | 157.36 s | 0.781 M steps/min | 0.578 GiB | 10800890 |

The ROCm compiler, rather than memory, dominated this short static run.

### Online bodies remove the per-body branch

Two Viper online MLP runs used 8,192 environments and 8,192 distinct initial
bodies:

| Run | Steps | Build | XLA compile | Train | Throughput | Peak live / allocator budget | Job |
|---|---:|---:|---:|---:|---:|---:|---:|
| Initial systems control | 10,485,760 | 9.30 s | 1,000.25 s | 333.28 s | 1.888 M steps/min | 2.01 / 82.50 GiB | 10800887 |
| Production, resample at reset | 99,614,720 | 9.07 s | 870.95 s | 3,132.10 s | 1.908 M steps/min | 2.00 / 82.50 GiB | 10801263 |

Online generation scales with environment count without adding model branches.
All 8,192 initial bodies remained unique after rounding each of the four
descriptors to 1e-6.

On the local RTX 4060 Ti, URMAv2 was stress-tested with one 64-step PPO update:

| Concurrent bodies/envs | XLA compile | Train | Throughput | Peak live memory |
|---:|---:|---:|---:|---:|
| 2,000 | 43.44 s | 2.14 s | 3.59 M steps/min | 1.43 GiB |
| 4,096 | 45.30 s | 3.86 s | 4.08 M steps/min | 2.73 GiB |
| 8,192 | 46.36 s | 7.79 s | 4.04 M steps/min | 5.60 GiB |
| 10,240 | 46.26 s | 9.48 s | 4.15 M steps/min | 6.82 GiB |
| **11,264** | **46.65 s** | **10.50 s** | **4.12 M steps/min** | **7.50 GiB** |

11,264 succeeds with the 12.0 GiB JAX allocator budget. 11,776 and 12,288 fail
during the training allocation, so the measured single-GPU boundary for this
exact rollout/minibatch/network configuration is between 11,264 and 11,776.
Changing rollout length, minibatches, optimizer epochs, network width, or memory
preallocation changes this ceiling.

At 256 environments, the short backbone comparison was:

| Backbone | Parameters | XLA compile | Throughput | Peak live memory |
|---|---:|---:|---:|---:|
| MLP | 130,215 | 32.59 s | 1.86 M steps/min | 0.161 GiB |
| URMA | 259,557 | 40.91 s | 1.59 M steps/min | 0.257 GiB |
| URMAv2 | 213,510 | 41.89 s | 1.51 M steps/min | 0.256 GiB |

The embodiment-aware models have a modest cost at small batch sizes; at thousands
of environments, simulation/batch work dominates and URMAv2 reaches about 4.1 M
steps/min locally in the stress configuration.

The equivalent 256-environment ROCm probes completed on Viper as follows:

| Backbone | XLA compile | Train | Throughput | Peak live memory | Job |
|---|---:|---:|---:|---:|---:|
| URMA | 2,520.20 s | 3.52 s | 0.558 M steps/min | 0.254 GiB | 10801020 |
| URMAv2 | 2,228.74 s | 4.15 s | 0.474 M steps/min | 0.253 GiB | 10801021 |

These probes make the amortization point especially clear: a roughly 37–42
minute compile surrounded a few seconds of work. They, and the earlier local
URMA rows, were launched before a network audit separated the raw binary
valid-joint mask from continuous observation normalization. Their resource and
throughput measurements remain representative, but their return curves must not
be used as learning evidence.

Post-audit local runs supply corrected optimization signals for both backbones:

| Backbone | Environments | Steps | Estimated body exposures | XLA compile | Train | Throughput | Return first → last | Episode length first → last |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MLP | 1,024 | 9,961,472 | 202,088 | 38.56 s | 147.21 s | 4.060 M steps/min | 1.269 → 7.445 | 29.9 → 76.3 |
| URMA | 512 | 1,998,848 | 59,353 | 42.06 s | 59.11 s | 2.029 M steps/min | 1.267 → 2.219 | 29.8 → 35.7 |
| URMAv2 | 1,024 | 9,961,472 | 300,416 | 47.15 s | 202.36 s | 2.954 M steps/min | 0.811 → 2.244 | 26.0 → 34.2 |

URMAv2's last 12 return measurements remain near 2.17–2.28 rather than reflecting
one outlier. The increasing online curves verify that optimization executes while
bodies change at resets; the held-out result below shows why this must not yet be
described as a converged or superior controller.

### Held-out control result

The matched 10M-step MLP and URMAv2 checkpoints were evaluated deterministically
on 1,024 seed-disjoint bodies held fixed for their first episode. The exact same
bodies and reset states were evaluated with zero actions:

| Metric | MLP policy | URMAv2 policy | Zero action |
|---|---:|---:|---:|
| Mean first-episode return | **12.629** | 2.716 | 2.835 |
| Mean first-episode length | **82.03** | 33.83 | 33.29 |
| Median return | **11.340** | 2.778 | 2.689 |
| Bodies beating zero-action return | **93.2%** | 39.1% | — |

All 1,024 bodies were unique at 1e-6 descriptor precision. Return correlations
with leg, arm, shoulder, and torso scales were 0.099, 0.018, -0.048, and 0.056,
so no single sampled coordinate explains most performance variation. The honest
conclusion is that the descriptor-conditioned MLP is currently both faster and
far stronger for this fixed-topology, four-parameter H1 problem. The URMAv2
adaptation has **not** beaten the strong zero-action initialization baseline at
10M steps/152 updates. URMAv2 remains strategically useful for the future
variable-topology setting, but it needs architecture/optimization work before it
should replace the MLP here.

The production Viper MLP strengthens that result. With 8,192 environments,
99,614,720 steps and per-episode body resampling, it exposed an estimated
1,304,843 bodies. Compilation took 870.95 s and training took 3,132.10 s
(1.908M steps/min); the final training return/length were 40.58/99.41. On the
same 1,024 held-out bodies used above:

| Metric | 100M MLP policy | Zero action |
|---|---:|---:|
| Mean first-episode return | **46.966** | 2.835 |
| Mean first-episode length | **99.30** | 33.29 |
| Median return | **53.062** | 2.689 |
| Bodies beating zero-action return | **96.9%** | — |

Its return correlations with leg, arm, shoulder and torso scales are all within
±0.030. This is the first online-million-exposure run in this scaling effort with
clear held-out control evidence, not just a throughput signal.

### Online resampling makes the number of seen bodies much larger than env count

A verified 256-environment URMAv2 run resampled at episode boundaries and exposed
an estimated 2,326 bodies during 65,536 environment steps. A compiled batch reset
generated 256 bodies in 0.000854 s, or approximately 300,000 bodies/s. The
corrected 1,024-environment run measured approximately 953,000 bodies/s in the
same steady reset microbenchmark. These numbers include reset work and
demonstrate that random descriptor generation is not a meaningful runtime
bottleneck.

The completed Viper URMAv2 capacity search used one optimizer epoch and an
approximately 82.5 GiB JAX allocator budget under the 100 GB Slurm request:

| Environments | Rollout steps | Outcome | Compile | Execute / failure | Job |
|---:|---:|---|---:|---:|---:|
| **32,768** | 64 | **completed 20,971,520 steps** | 2,492.85 s | 844.19 s, 1.491M steps/min, 51.18 GiB peak live | 10801507 |
| 40,960 | 64 | OOM on first call | 2,524.4 s | 63.96 GiB failed allocation | 10801509 |
| 49,152 | 64 | OOM on first call | 2,452.5 s | 70.84 GiB failed allocation | 10801267 |
| 65,536 | 32 | OOM on first call | 1,459.0 s | 51.50 GiB failed allocation | 10801513 |
| 65,536 | 64 | OOM on first call | 2,546.1 s | 76.06 GiB failed allocation | 10801130 |

Thus 32,768 is measured safe and 40,960 is measured unsafe for the 64-step
URMAv2 configuration on one APU. The 65,536×32 result also shows that its large
joint-set temporary buffers do not scale only with environment×rollout product.
Job 10801130 predates the binary-mask audit, but the audit does not materially
change these allocation shapes. Its identical post-audit duplicate 10801170 was
deliberately cancelled before repeating a known OOM.

The preferred MLP is far lighter. Holding each PPO update at 4,194,304
transitions while trading environment count against rollout length gave:

| Environments | Rollout steps | Compile | Train 41,943,040 steps | Throughput | Peak live memory | Estimated body exposures | Job |
|---:|---:|---:|---:|---:|---:|---:|---:|
| **65,536** | **64** | 707.20 s | 1,303.09 s | 1.931M steps/min | 14.50 GiB | 1,314,041 | 10801552 |
| **131,072** | **32** | 1,001.01 s | 1,316.26 s | 1.912M steps/min | 14.88 GiB | 1,418,876 | 10801556 |
| **262,144** | **16** | 681.42 s | 1,317.38 s | 1.910M steps/min | 15.67 GiB | 1,905,437 | 10801695 |

Every initial descriptor batch was unique at 1e-6 precision and every job exited
successfully. Therefore 262,144 is a measured lower bound on raw resident-body
capacity for this MLP, not a memory ceiling. The search deliberately stops at a
16-step rollout: reducing it further would produce a less representative PPO
quality test. These ten-update runs establish systems capacity only; their low
final returns must not be compared with the 190-update production MLP.

An earlier resubmission attempt, job 10801166, did not inherit inline environment
overrides and therefore ran the safe 256-environment defaults; no oversized
accidental workload was launched.

A learning-oriented 8,192-environment URMAv2 job requested approximately 100M
steps, 190 PPO updates, 32 minibatches and four optimizer epochs (job 10801220).
Its executable compiled in 2,025.7 s but segfaulted on its first call (exit 11)
without a JAX OOM. Reducing it to ~20M steps/38 updates retained the segfault when
four optimizer epochs were used (job 10801422, cached compile 47.0 s).

The single-epoch diagnostic completed successfully (job 10801470). It kept the
same 8,192 environments and ~20M steps, and its 32 minibatches across 38 updates
still supplied 1,216 gradient minibatch updates, matching the successful local
10M run's count. Cached compilation took 47.75 s; training took 654.8 s at 1.826M
steps/min, exposed an estimated 631,629 bodies, and peaked at 4.70 GiB live device
memory. This isolates the ROCm instability to the nested four-epoch update graph,
not the 8,192-body simulator batch. The safe Viper recipe is currently one epoch
and more rollout updates.

Its held-out control result remains poor: mean first-episode return was 2.308
versus 2.835 for zero action, mean length was 33.05 versus 33.28, and only 20.0%
of 1,024 disjoint bodies beat zero-action return. More exposure did not make this
URMAv2 adaptation competitive with the MLP.

The matched 100M-step MLP production control completed on `vipa1143` as job
10801263 (exit 0, 1:07:17 wall clock, 2.00 GiB peak live device memory). Its
1.305M estimated body exposures and held-out result are reported above; it is the
preferred same-topology training path.

## Motion and retargeting measurements

The multi-motion AMP path has been locally verified with four H1 morphologies,
four dance clips, 256 total environments, and 2,032 expert transitions. It
compiled in 69.3 s and trained its 8,192-step smoke budget in 1.1 s.

The larger Viper run also completed successfully:

| Bodies × motions | Environments | Expert transitions | XLA compile | Train | Throughput | Job |
|---:|---:|---:|---:|---:|---:|---:|
| 4 × 4 | 1,024 (256/body) | 7,984 | 1,971.80 s | 21.47 s for 262,144 steps | 0.733 M steps/min | 10801022 |

Its four body groups and all four motion clips contribute to the same AMP
policy/discriminator training call. The Slurm job completed in 33:48 with exit
code 0 and 12.6 GiB maximum host RSS. It was launched just before the audit fix
that removes a one-step lag in grouped reward-variance normalization, so use it
as valid compile/throughput evidence; use a post-fix long run for learning claims.

### Cross-topology shared-policy smoke test

The retargeted references are now connected to a padded grouped trainer, not only
stored offline. One local PPO call included three different simulator topologies:

| Robot | Native observation dim | Native action dim | Environments |
|---|---:|---:|---:|
| H1 | 434 | 19 | 32 |
| G1 | 450 | 23 | 32 |
| Atlas | 466 | 27 | 32 |
| ToddlerBot | 546 | 30 | preflight only |

Observations are padded to 466 and receive a 3-way robot one-hot plus a 27-slot
binary action mask (496 total). Actions are padded to 27 and sliced back to
19/23/27 before each simulator step. The masked Gaussian forces unused action
means to zero and fixes their standard deviations at 1e-3, so smaller robots do
not learn noise in nonexistent joints.

The 96-environment shared PPO smoke compiled in 52.18 s and completed its
1,536-step training call in 0.92 s; its masked checkpoint also reloads. This
proves that one update can now contain different robot families/topologies and
their own retargeted references.

A larger local run used 384 environments (128/robot), 81 shared updates and
995,328 steps. It compiled in 57.64 s, trained in 119.65 s (0.499 M steps/min),
and increased aggregate training return from 8.74 to 10.03 and episode length
from 21.6 to 40.5.

The same-seed, per-robot held-out evaluation gives the necessary qualification:

| Robot | Return difference vs zero | Length difference vs zero | Starts beating zero return |
|---|---:|---:|---:|
| H1 | -0.435 | +8.93 steps | 40.6% |
| G1 | -1.957 | +11.63 steps | 20.3% |
| Atlas | +0.492 | +5.04 steps | 72.7% |
| **Overall** | **-0.633** | **+8.53 steps** | **44.5%** |

The shared policy is numerically stable and changes control on all three robots,
but one million steps do not yet beat the aggregate zero-action return. Generic
per-joint descriptions, URMAv2 integration, longer optimization and balanced
per-robot evaluation are still required.

A separate four-topology preflight adds ToddlerBot. Eight environments (two per
robot) produce finite rewards/observations in one JIT step; the combined policy
input is 580 wide and the padded action is 30 wide. Thus four distinct humanoid
topologies are simulator-compatible today, while repeated training evidence is
currently limited to H1/G1/Atlas.

The cross-humanoid pipeline supports two useful modes:

- `direct`: reuse an already converted SMPL-based cache for the target robot.
- `robot2robot`: H1 reference → fitted SMPL → target robot, analogous in purpose
  to GMR-style cross-embodiment retargeting while using LocoMuJoCo's available
  retargeters.

Cold timing for one 7.99 s, 800-frame reference was:

| Target/mode | Output | Cold time |
|---|---:|---:|
| H1 direct | 800 samples at 100 Hz | 1.17 s |
| G1 direct | 800 samples at 100 Hz | 2.73 s |
| Atlas direct | 800 samples at 100 Hz | 3.59 s |
| ToddlerBot direct | 800 samples at 100 Hz | 4.33 s |
| H1 → G1 robot-to-robot | 159 samples at 20 Hz | 3.99 s |
| H1 → Atlas first uncached target | 159 samples at 20 Hz | 8.81 s |

The first uncached Atlas run includes 1,000 CPU shape-optimization iterations.
Retargeted files are cached, so this is normally an offline, amortized cost.

Static XML/asset generation plus MuJoCo validation took 1.69 s for ten bodies,
0.166 s/body or 5.91 bodies/s. Consequently, creating thousands of XML variants
is possible in minutes, but compiling thousands of separate MJX branches is not
efficient. Generate static XMLs for validation/evaluation; sample compact body
parameters online for large-scale training.

For order-of-magnitude planning, 1,000 static H1 XMLs would take about 2.8
minutes serially at the measured rate. Retargeting 1,000 independent 8-second
robot targets would take roughly 0.3–2.5 hours serially across the measured
1–9-second range; 1,000 bodies × four motions would take roughly 1.1–10 hours
serial before CPU parallelism. Both outputs are cacheable. Online same-DOF H1
training needs neither operation per sampled body, which is why millions of
exposures are feasible.

## Where the wall clock goes

A useful first-order cost model is:

`total = generation + retarget/cache creation + XLA compile + simulation/PPO training`

| Stage | Measured scale | Interpretation |
|---|---|---|
| Static body generation | 0.166 s/body | Cheap and parallelizable on CPU |
| Cross-robot retarget | roughly 1–9 s per 8 s clip/target when cold | Cheap relative to training and cacheable; shape fitting is the expensive part |
| Local CUDA compile | roughly 19–141 s static; 33–47 s online | Dominates smoke runs, amortizes over long training |
| Viper ROCm compile | 681–1,001 s for large MLP shapes; 1,459–2,546 s for large URMAv2 shapes; 1,768 s for 10 static branches | Current largest startup bottleneck; persistent XLA cache and fewer graph shapes matter |
| PPO/MJX training | 1.5–4.1 M steps/min in measured online runs | Dominates once runs contain hundreds of millions/billions of steps |

The persistent Viper XLA cache is effective when the graph is reusable. A
post-audit 256-environment URMAv2 rerun (job 10801166) compiled in 43.1 s, versus
2,228.7 s for the first equivalent backbone probe. The morphology count, rollout
shape, network configuration, and software versions therefore need to remain
stable between production runs if compilation is to be amortized.

The production Viper MLP measurement replaces the earlier projection: 99.61M
steps took 52.20 minutes of training and 14.52 minutes of compilation (67.3
minutes of Slurm wall time including setup). At 1.908M steps/min, one billion
would take about 8.73 hours after compile. The current eight-hour launcher can fit
about 888M steps after a cold compile at this exact rate. Optimizer epochs, rollout
length, validation, policy size and cache reuse all change that estimate.

At the measured 4-body × 4-motion AMP rate, 100 million steps would take about
2.27 hours after its 32.9-minute compile, while one billion would take about 22.8
hours. Under the current eight-hour job script, that exact configuration could
fit roughly 328 million steps after compilation. More discriminator/PPO epochs
would reduce that number.

## How many morphologies can we train?

There are three different answers:

1. **Unique bodies over a run:** effectively unbounded for the continuous online
   H1 parameterization. Millions are practical because bodies are regenerated at
   reset and never stored as XMLs. Even quantizing each scale to 0.01 within the
   current bounds yields 36 × 36 × 36 × 81 = 3,779,136 descriptor combinations.
2. **Bodies resident simultaneously:** local URMAv2 is safe at 11,264 and fails at
   11,776 for its 64-step configuration. On one Viper APU, URMAv2 is safe at
   32,768 and fails at 40,960. The preferred MLP completes 65,536×64,
   131,072×32 and 262,144×16; 262,144 is a demonstrated lower bound rather
   than the memory ceiling.
3. **Truly different robot topologies:** not yet thousands in one executable.
   H1/G1/Atlas now work in one padded grouped PPO executable at smoke scale, and
   references exist for additional humanoids. Each topology remains a separate
   static simulator branch, so branch compile cost—not padding—is the limiting
   factor. URMAv2 is not yet wired to this cross-topology wrapper.

The practical near-term target is therefore millions of sampled embodiments from
one robot family, tens of static validation bodies, and a small number of distinct
robot topologies in padded groups. “Millions of arbitrary robots” still requires
online topology generation and masked embodiment-aware control, not more XML
generation.

Maximum resident environments are not automatically the best learning setting.
At 8,192 environments × 64 rollout steps, the successful 100M MLP run received
190 PPO updates. At 65,536 × 64, the same 100M budget would contain only 23 full
rollout updates, each with 4.19M transitions. Very large batches improve hardware
occupancy but reduce optimizer-update cadence and can hurt sample efficiency.
Use the capacity ceiling for headroom; choose production environment count,
rollout length and minibatches from held-out learning curves. Today, 8,192 × 64
is the strongest measured quality/throughput point.

## Recommended scale-up sequence

1. Use online H1 + descriptor-conditioned MLP + per-episode resampling for the
   main same-topology training runs. Keep URMAv2 as the cross-topology research
   branch until it beats the matched held-out baseline. Start at 2,048
   environments and use the measured 8,192 × 64 setting as the current production
   point; treat larger capacity rungs as batch-shape experiments, not automatic
   upgrades.
2. Run at least 100M steps per candidate before interpreting returns. The present
   production MLP meets that floor; the capacity manifests do not. Always run
   `evaluate_online_policy.py` against the same-seed zero-action baseline.
3. Add curricula over morphology bounds. Begin near nominal, then widen leg, arm,
   shoulder, and mass ranges only when return remains stable. Factor the H1 index
   map into a per-family specification and add an equivalent online G1 family;
   do not mix H1/G1 topology changes into the same dynamic model graph.
4. Keep 10–50 fixed XML bodies outside the sampling stream as held-out evaluation
   morphologies. This separates memorization from interpolation/extrapolation.
5. Cache one retarget per robot topology and motion. Same-DOF online H1 variations
   can reuse canonical joint-space references; do not retarget every sampled body.
6. Combine the verified pieces into online H1 AMP: sample both a body descriptor
   and motion ID at reset, reuse the canonical multi-clip expert cache, and either
   exclude morphology metadata from the discriminator or provide matched metadata
   to expert and policy samples. This avoids a static body × motion branch product.
7. Extend the working H1/G1/Atlas padded/masked grouped interface with generic
   per-joint descriptions, then feed it to URMAv2. Initially keep one static
   branch per topology and the topology count small.
8. Add multi-device synchronous PPO only after the single-APU ceiling is measured.
   The current Viper launchers request one GPU; several independent jobs scale
   experiments immediately, but they do not yet synchronize one policy across
   GPUs/nodes.

## Viper access

Viper access is working from the WSL distribution named `Ubuntu`:

```powershell
wsl -d Ubuntu -e ssh -o BatchMode=yes viper11 "hostname; squeue -u akalenik"
```

The default `Ubuntu-20.04` distribution does not contain the working SSH setup,
which caused the earlier apparent access failure. `viper11` is the SSH alias/login
host; inside jobs the Python environment is activated with:

```bash
source /ptmp/akalenik/frontier/venv/bin/activate
```

Pass scale overrides explicitly through Slurm; inline shell variables were not
propagated by this cluster setup in job 10801166:

```powershell
wsl -d Ubuntu -e ssh viper11 "cd /ptmp/akalenik/frontier/repo && sbatch --export=ALL,BACKBONE=urmav2,NUM_ENVS=65536,TOTAL_TIMESTEPS=41943040,NUM_STEPS=64,HIDDEN_1=128,HIDDEN_2=64,RESAMPLE=1 scripts/scaling/viper_urma_online.sbatch"
```

Verified allocations use account `mage_apu`, partition `apu1` (requested as
`apu`), `gres/gpu:1`, and ROCm-backed JAX. Job 10801130 ran on `vipa1110`;
the node advertises two GPUs and 220 GB allocatable memory. The official
[Viper-GPU guide](https://docs.mpcdf.mpg.de/doc/computing/viper-gpu-user-guide)
identifies them as AMD Instinct MI300A APUs with 128 GB HBM3 and 24 CPU cores per
APU. The current JAX process exposes an approximately 82.5 GiB allocator budget
after the 100 GB Slurm memory request and runtime reservations.

The live partition query reports 298 nodes/596 GPUs, close to the official final
configuration of [300 nodes/600 MI300A APUs](https://www.mpcdf.mpg.de/services/supercomputing/viper).
This account's association reports at most eight concurrently running jobs and
300 submitted jobs; its available QoS tiers permit jobs ranging from one to 128
nodes. Therefore, up to roughly eight independent one-GPU seeds/ranges can be
run immediately under the observed account limit. Using multiple APUs for one
shared policy requires synchronous distributed PPO; the current launchers do not
perform gradient all-reduce.

## Evidence and caveats

Machine-readable manifests are under:

- `experiments/scaling_parallel/`
- `experiments/scaling_urma/`
- `experiments/scaling_retarget/`
- `experiments/scaling_cross_topology/`
- `experiments/scaling_viper/`

The scaling test suite is:

```powershell
$env:PYTHONPATH="$PWD\loco-mujoco;$PWD\scripts"
.\.venv\Scripts\python.exe -m pytest tests\test_parallel_morph_env.py tests\test_online_h1.py tests\test_amp_targets.py tests\test_urma_networks.py tests\test_masked_mlp.py -q
```

Current result: 20 passed. The grouped-static normalization wrapper has a
regression test for independent per-morphology return statistics. AMP also has a
test that verifies GAIL uses 0/1 labels and AMP uses -1/+1 labels. URMA tests keep
the valid-joint mask binary across running-statistic updates and suppress both the
mean and practical sampling variance of padded actions.

Important limitations before claiming policy generalization:

- the largest runs are deliberately short capacity tests;
- the thousand-body online path and the four-motion AMP path are separately
  verified, but are not yet one combined online-morphology AMP trainer;
- the corrected 10M-step URMAv2 checkpoint does not yet beat zero-action mean
  return on 1,024 held-out bodies;
- online H1 shares nominal meshes and uses a canonical same-DOF reference;
- the four-parameter distribution is useful but far smaller than the morphology
  space in the published URMA work;
- the shared H1/G1/Atlas masked policy is trained for only about one million steps;
  its held-out lifetime improves over zero action, but aggregate return does not;
- current launchers are single-device, not multi-node data-parallel PPO.

Worktree note: an early generation check touched the already-present nominal and
extreme-tall H1 derivative XMLs (and the related gallery metadata). They were not
used to infer timing and were not reverted because the repository already
contained user work. The recorded generation benchmark uses a private temporary
directory and cannot overwrite research variants.

## Research references

- URMA paper: <https://proceedings.mlr.press/v270/bohlinger25a.html>
- URMA project: <https://nico-bohlinger.github.io/one_policy_to_run_them_all_website/>
- Official URMA code: <https://github.com/nico-bohlinger/one_policy_to_run_them_all>
- URMAv2 paper: <https://arxiv.org/abs/2509.02815>
