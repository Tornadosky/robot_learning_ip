# URMA + FSQ overnight — RESULTS

**Goal:** one shared URMA policy controlling H1 + G1 + Atlas with online
within-family morphology randomization, commanded ONLY through a swappable
motion latent z (fake → oracle → learned FSQ). No AMP anywhere.
Built 2026-08-13 22:45 → 2026-08-14 ~03:45 Berlin. Spec: `DESIGN.md`;
running log: `STATE.md`.

## Definition-of-done scoreboard — ALL 8 CORE ITEMS MET

| # | Item | Status | Evidence |
|---|---|---|---|
| 1 | One shared URMA policy tree runs H1/G1/Atlas with a SEPARATE swappable motion-latent input | **MET** | one `CrossTopologyURMAPPO` parameter tree; z is a second positional actor argument, never an observation entry; `gate_evidence.json` in both training checkpoints |
| 2 | 3 fam × ≥2 morph × 2 motions fake-z: ≥1 jitted PPO update, training smoke, checkpoint round-trip, video | **MET** | `checkpoints/smoke_3fam_morph_fakez` (one update) + `train_fakez4_100m` (100M steps, 58 min, return 5.5→30.8) + reload Δ=0.0 + 5 videos in `media/fakez_*` |
| 3 | Online randomization verified per family; no M/K branch product | **MET** | 8 distinct bodies/family in one batch (continuous), exactly the 2 corner bodies under catalog2; memaudit: catalog2/catalog64/continuous → identical compile (26.9/27.1/26.8 s reset) and identical peak memory (0.186 GB) at 384 envs — M changes data, never branches. K=2 motions live in one trajectory (split_points), same graph |
| 4 | Real FSQ trained, evaluated, loaded into the same pipeline | **MET** | Viper job 10909013: val 0.776 vs constant 1.340; codes load via the same `TrajectoryLatentBuffer` interface (`--latent-codes`) |
| 5 | Learned-z 3-family smoke + policy TRAINED on learned codes | **MET** | `train_learnedz4_100m`: 100M steps FROM SCRATCH on learned codes — return **35.0 vs 30.8** for fake-z at identical budget; gates re-run on that checkpoint all pass |
| 6 | Joint/alternating FSQ+PPO ≥1 correct update | **MET** | `metrics/design_c_alternating_proof.json`: PPO update finite on in-graph-encoded z + supervised FSQ step + real code refresh; honestly labeled ALTERNATING (no policy gradient into encoder) |
| 7 | Tests: shapes, masks, cursor alignment, resets, family-invariant codes, morphology descriptions, sensitivity, serialization | **MET** | 59 passed (15 new tonight: `test_urma_motion_latent.py` ×9, `test_fsq_motion.py` ×6) + runtime gate evidence; 2 PRE-EXISTING failures in the user's dirty worktree documented below, untouched |
| 8 | Media, metrics, manifests, Viper records, honest RESULTS | **MET** | 10 labeled videos, `metrics/`, `manifests/`, job ledger below, this file |

## Architecture as built

```
canonical window (22-dim root-relative features × 10 frames, stride 4, 100 Hz)
  → provider (fake table | oracle projection | learned FSQ [8,5,5,5], d=4)
  → ONE shared TrajectoryLatentBuffer keyed (motion_id, timestamp)   ← identical
    z for every family by construction (split_points equality enforced)
  → URMAActorCritic(observation, actor_latent)                       ← separate arg
       actor core: z --Dense(64)--elu--> concat with general input
       critic: never sees z (bit-identical values across z)
  → masked per-joint actions (19/23/27 valid of 27 padded)
```

- 3 compiled topology branches (`ParallelMorphVecEnv`), morphology and
  motion sampled dynamically inside each branch.
- Morphology (4 shared coords/family, family adapters in
  `family_morphology.py`): leg_length_scale [0.90,1.12] — genuinely
  geometric (knee/ankle attachment offsets + volume-consistent mass and
  inertia) — plus torso_mass, damping, strength (gain AND forcerange).
- Per-joint URMA descriptions (26-dim = 22 structural + 4 morphology) are
  recomputed IN-GRAPH from the SAMPLED model; emitted block equals an
  out-of-graph recomputation to ≤1.2e-7 and differs between bodies.
- The mimic goal block (`GoalTrajMimic` — the only reference/phase carrier
  in the observation) is zeroed via `blank_goal_observation`: z is the
  actor's ONLY future-motion channel. Documented consequence: the critic is
  NOT privileged either; reward and termination read the reference env-side.

## Gate evidence (all six gates)

1. **Latent contract** — on trained checkpoints: shuffled z changes valid
   actions (Δ 0.343 fake-z / 0.455 learned-z; zero-z 0.359), padded means
   exactly 0.0, padded std pinned at 1e-3, critic Δ exactly 0.0. At
   `actor_latent_dim=0` the parameter tree is byte-compatible with
   pre-latent checkpoints (unit-tested; Flax auto-numbering unshifted).
2. **Rollout integrity** — pre-step cursor → z row exact at every probed
   step incl. reset boundaries (off-by-one impossible without detection);
   upstream tests cover minibatch shuffle pairing and done-boundary;
   serialize→reload action Δ = 0.0; no NaNs.
3. **Topology + morphology** — every batch holds all three families with
   exact mask counts 19/23/27; ≥2 (up to 8/8) distinct bodies per family;
   descriptions track the sampled model. Canonical grid: FSQ features are
   bit-identical (Δ=0.0) to the RL crop rows; `ParallelMorphVecEnv.th`
   hard-errors on any family misalignment.
4. **Scaling/memory** — M∈{2,64,∞}: identical compile times and peak
   memory (0.186 GB @384 envs); full training: 2.77 GB peak in-use @2048
   envs, 1.73M steps/min (dynamic descriptions ≈ free vs the 1.75M/min
   historical no-morph rate). Serialization measured on disk: full agent
   checkpoint (params + config + token table) 6.0 MB; FSQ encoder/decoder
   708 KB; RL token cache 28 KB; all 10 videos 3.6 MB — bounded.
5. **Learned FSQ** — held-out-clip reconstruction 0.776 vs 1.340 constant
   (42% better); every level of every dim used; 889/1000 codes, 8.7 bits;
   RL-window cache: 123 unique codes, 18% temporal change rate,
   motion-distinct distributions; correct/shuffled/zero z produce
   measurably different actions (above); loaded through exactly the fake-z
   interface.
6. **Visual proof** — 10 labeled videos in `media/` (fake-z and learned-z
   ×: 3 families randomized bodies · 3 families NOMINAL bodies · same
   canonical command m0 to all families on a shared clock · same for m1 —
   two different commands on the same bodies · mid-rollout command switch
   at step 150). Switch discontinuity measured, not eyeballed: fake-z
   policy shows NO spike (0.083 at-switch vs 0.102 baseline); learned-z
   policy switches ABRUPTLY (0.172 vs 0.076, a 2.3× spike). No smooth-
   transition claim is made; the spike is itself causal evidence that the
   learned command drives the actor.

## FSQ design comparison

| design | val MSE | constant | verdict |
|---|---|---|---|
| A quantized [8,5,5,5] | **0.776** | 1.340 | baseline, used everywhere |
| A continuous control (same size) | 0.856 | 1.340 | FSQ WINS |
| B URMA-decoder quantized | 0.822 | 0.874 | weak: only 6% better; h1 0.47 / g1 0.52 fine, **atlas 1.33 — worse than constant** |
| B continuous control | 0.867 | 0.874 | worse still (atlas 1.49) — B's failure is decoder/data (3,200 paired train frames), not quantization |
| C alternating FSQ+PPO | — | — | one-cycle proof passed; longer training not attempted |

**Negative result reported plainly:** design B's embodiment-conditioned
decoding does not transfer to Atlas on a held-out clip at tonight's data
budget, echoing the earlier finding that URMA-style conditioning needs
scale to earn its complexity. Design A is the operative code source.

**RL-side comparison:** at identical 100M budgets, learned-z (35.0) >
fake-z (30.8) mean episode return — temporally meaningful codes help.

## Honest limitations
- Reward scores each family against its STOCK topology reference;
  morphology-conditioned FK target projection (MorphMimicReward-style) was
  NOT ported to the cross-family path. Randomized bodies are therefore
  scored against nominal-body references; leg-scale range kept modest
  partly for this reason and for the stock terminal handler's tolerance.
- 100M steps is a pipeline-proof budget, not convergence; the videos show
  visibly imperfect tracking. Prior work needed 200M+ WITH the goal
  visible; this task (z-only command + morphology) is strictly harder.
- Design C is alternating, not end-to-end; policy gradients never touch
  the encoder.
- The two skipped test files (`test_online_h1.py`, `test_morphology_
  terminal.py`) fail from PRE-EXISTING mismatches between the user's
  committed tests and their uncommitted `online_h1.py`/`embodiment_
  catalog.py` edits (4-dim vs 11-dim morphology spec; negative additive
  bound). Nothing tonight touches those files.

## New/changed code (all ruff-clean, surgical)
| file | change |
|---|---|
| `scripts/scaling/urma_networks.py` | optional `actor_latent` input (projected into actor core only), buffer on `URMAAgentConf`, serialization |
| `scripts/scaling/parallel_env.py` | canonical group-aware `th` (alignment hard-error), goal blanking, reset-obs routing on done, dynamic per-group joint descriptions |
| `scripts/scaling/cross_topology_urma.py` | latent pass-through, routing guard, boundary no-op |
| `scripts/scaling/family_morphology.py` | NEW — generic 4-coord online morphology for h1/g1/atlas + in-graph sampled-model descriptions |
| `scripts/scaling/fsq_motion.py` | NEW — official FSQ (verbatim, arXiv:2309.15505), canonical features, windows, providers |
| `scripts/scaling/train_fsq_motion.py` | NEW — design A build/train/encode |
| `scripts/scaling/train_fsq_urma_decoder.py` | NEW — design B build/train |
| `scripts/scaling/urma_fsq_gates.py` | NEW — runtime gate evidence |
| `scripts/scaling/urma_fsq_alternating.py` | NEW — design C one-cycle proof |
| `scripts/scaling/parallel_cross_humanoid_train.py` | `--clip-windows/--morphology/--blank-goal/--actor-latent-dim/--latent-codes`, latent-aware manifest, getattr-tolerant for old callers |
| `scripts/scaling/render_cross_topology_policy.py` | latent-aware rollout, `--latent-codes/--switch-step/--force-motion/--morphology-override`, switch metrics |
| `scripts/scaling/evaluate_cross_humanoid_policy.py` | manifest field pass-through |
| `scripts/scaling/viper_fsq_motion.sbatch`, `viper_fsq_urma_decoder.sbatch`, `viper_cross_topology.sbatch` | FSQ jobs; EXTRA passthrough; CPU-platform switch |
| `tests/test_urma_motion_latent.py`, `tests/test_fsq_motion.py` | NEW — 15 tests |
| `loco-mujoco` submodule | untouched tonight (latent plumbing pre-existed) |

## Viper job ledger (all resolved except the final run)
| job | what | outcome |
|---|---|---|
| 10908958/59 | design A GPU | failed in 8 s — stale --val-clips default |
| 10908961/62 | design A GPU retry | hung ≥14 min in ROCm init; cancelled (ours) |
| 10909013/14 | design A quantized + continuous, CPU | **done** (0.776 / 0.856) |
| 10909031 | design B, CPU | **done** (0.822) |
| 10910154 | design B continuous, CPU | **done** (0.867) |
| 10909032 | URMA fake-z 2048 envs | compiled 3492 s then HSA_MEMORY_APERTURE_VIOLATION — violated the recorded 768-env ROCm ceiling (my error; ceiling confirmed for this graph). New finding: ROCm compiles this graph ~25× slower than CUDA (4252 s vs 159 s) |
| 10909059 | retry | failed — reserved robot hit the morph wrapper (fixed: reserved robots build plain) |
| 10910264 | URMA fake-z **768 envs**, 150M, 1 epoch | **COMPLETED** (resolved post-handoff, pulled to `checkpoints/viper_urma_fakez4_768/`): 0.58M steps/min on ROCm, final return 33.2 / length 127.2 — the fake-z result REPRODUCES on a second machine, second seed, 1-epoch regime (local 4-epoch 100M gave 30.8/122.6). The 768-env ceiling held for the full run. |

Local GPU (RTX 4060 Ti): both 100M runs at 1.73M steps/min; nothing left
running locally except artifact pulls.
