# Overnight Goal — One Policy, Many Robots (Cross-Embodiment Scale-Up)

**Written 2026-08-02 22:00 CEST. Deliverables due 2026-08-03 09:00 CEST — an
11-hour window.** Compute available: Viper (8 concurrent one-APU jobs, shared
account) **and** one local RTX 4060 Ti, which is measurably faster per run.

---

## 0. The one-sentence mission

Move from *"one policy controls 1,000 bodies of the same robot"* — which is now
**done and evidenced** — to *"one policy controls several different robots, each
with randomised morphology, on one or more motions"*, using an architecture that
can actually represent variable topology rather than a fixed-width MLP.

The claim to earn by 09:00:

> One embodiment-aware policy was trained simultaneously on H1, G1 and Atlas —
> different joint counts, different observation and action dimensions — and every
> robot individually beats its own exact-reset zero-action baseline on held-out
> rollouts across at least two seeds, with no robot hidden behind an aggregate.

The stretch claim:

> …and each family additionally had randomised morphology, so the policy spans
> robots *and* bodies within robots.

---

## 0.1 State at handoff (2026-08-02 22:08 CEST)

- **Viper: completely idle, 0 jobs.** Nothing of ours is queued or running.
- **Local GPU:** a two-stage morphology-bounds curriculum was running, expected to
  finish ~22:51. Check for results before re-running it:

  ```
  experiments/scaling_1000/curriculum/mid/manifest.json
  experiments/scaling_1000/curriculum/wide/manifest.json
  ```

  If either is missing, the machine was restarted mid-run. Re-run with
  `bash scripts/scaling/local_gpu_curriculum.sh` (~45 min) — it is item 1 of the
  local queue in §4.2. It is on the *morphology* axis and is **not** a blocker for
  the cross-topology work, so do not let it delay §3.
- **Environment:** WSL `Ubuntu`, venv `~/jaxgpu` (Python 3.12, jax 0.10.1 + CUDA).
  Survives reboots. Run things as:

  ```
  wsl -d Ubuntu -e bash -lc "cd /mnt/c/Users/smirn/Desktop/robot_learning_ip && \
    export PYTHONPATH=\$PWD/scripts XLA_PYTHON_CLIENT_PREALLOCATE=false && \
    ~/jaxgpu/bin/python <script> ..."
  ```

- **Long local runs must be detached**, or they die with their wrapper:
  `setsid nohup bash <script> > /tmp/<name>.log 2>&1 < /dev/null &`.
  This happened twice tonight — no error, no OOM, process simply gone.

## 1. What is already established — do not redo any of this

Read [`SCALING_1000_RESULTS.md`](SCALING_1000_RESULTS.md) first. Summary of
settled facts:

| Fact | Evidence |
|---|---|
| 1,000-body H1 catalog, balanced 8/8 exposure, deterministic + hashed | `experiments/scaling_1000/catalogs/` |
| 200,000 envs = 1,000 bodies × exactly 200 replicas runs and fits | job 10803153, 1.89 M steps/min, 12.6 GiB |
| One descriptor-MLP beats zero action on **100%** of IID, boundary, OOD and named bodies | `experiments/scaling_1000/evaluations/` |
| **Continuous resampling beats a fixed 1,000-body catalog** by ~19% (t ≈ −6) | §5 of the results file |
| Dance at 1,000 bodies clears its bar (median IID length 530–547) | §6 |
| Online morphology × multi-motion AMP works: 1,000 bodies × 4 clips, **one XLA branch**, leakage 0.0 | §4 |
| The recorded ~140–150 multi-motion "plateau" is **not a matched baseline** | §4 correction |
| Termination, not the policy, capped morphology range — fixed | §5, `morphology_terminal.py` |
| Naive wide-bounds training collapses; warm-started curriculum is the fix | §6c |
| Local GPU ≈ **3.6×** Viper per run (6.89 vs 1.90 M steps/min) | §6b |

**Do not** re-run the 1,000-body walk/dance experiments, the capacity proof, or
the AMP ladder. They are finished.

---

## 2. The actual gap

Everything above is **one topology**. The cross-topology work is a smoke test:
H1/G1/Atlas train together in one padded PPO call for ~1M steps, and the
aggregate held-out return is *below* zero action (H1 −0.435, G1 −1.957, Atlas
+0.492). That is the thing to fix tonight.

There are two candidate architectures and the evidence is genuinely split:

- **Masked MLP** (`scripts/scaling/masked_mlp.py`) — pads observations to the
  largest robot, appends a robot one-hot and a binary action mask, forces unused
  action means to 0 and their std to 1e-3. Works, but the one-hot means it
  *memorises* a fixed robot set; it cannot represent an unseen topology.
- **URMA / URMAv2** (`scripts/scaling/urma_networks.py`) — per-joint describe →
  attend → decode. Can in principle accept any joint set. On fixed-topology H1 it
  is **worse** than the MLP (20% of bodies beat zero action vs 96.9%), but that is
  the case where it has no advantage to exploit.

**The decisive experiment has never been run:** URMA has never been given the
cross-topology input it was designed for. That is roadmap step 7 and it is
tonight's core engineering task.

---

## 3. The mechanism to build (concrete, ~1.5 h)

### 3.1 Generic per-joint descriptions

`OnlineMorphMjxUnitreeH1._urma_joint_descriptions` already builds an
action-ordered 26-dim description per joint, but it is H1-specific and includes
the 4 H1 morphology coordinates.

Create **`scripts/scaling/joint_descriptions.py`** with a family-agnostic
version, computed from any LocoMuJoCo MJX env's model:

```
per joint j (in actuator order):
  body_pos(3), jnt_axis(3), child_count(1), nominal_qpos(1), force_limit(1),
  damping(1), armature(1), stiffness(1), frictionloss(1),
  jnt_range(2), ctrl_range(2), total_mass(1), body_mass(1), body_inertia(3)
  = 22 generic dims
```

Keep the existing scaling constants (`_finite_scaled`) so magnitudes match what
URMA already trains on. **Do not** append family morphology coordinates into the
per-joint block — put those in the global descriptor instead, so the joint block
stays purely structural and transfers across families.

### 3.2 Padded cross-topology joint block

Extend the cross-topology env builder (`build_cross_humanoid_env` in
`scripts/scaling/parallel_cross_humanoid_train.py`, which already does
`append_action_mask=True` and exposes `action_mask_observation_start`) with
`append_joint_features=True`, producing per environment:

```
[ padded base observation | global descriptor | J_max × (22 desc + 3 state + 1 mask) ]
```

with `J_max = 27` (Atlas). Joints beyond a robot's count are zero-filled and
their mask bit is 0. Reuse `URMAInputLayout` from `scaling/online_h1.py` —
it already carries exactly these offsets — and construct one for the padded case.

**Invariants that are already audited and must not regress:**

- the valid-joint mask stays **binary** and **outside** running normalisation;
- padded action means are exactly 0 and their practical std ~1e-3;
- `tests/test_urma_networks.py` and `tests/test_masked_mlp.py` keep passing.

### 3.3 Trainer

Add `CrossTopologyURMAPPO` by the same pattern `MaskedParallelPPO` uses for the
MLP: subclass `URMAPPO`, override `_wrap_env` to accept `ParallelMorphVecEnv`,
and point the network at the padded `URMAInputLayout`. Keep the robot one-hot
**available but optional** (`--robot-one-hot / --no-robot-one-hot`) — the
no-one-hot arm is what tests whether the model generalises structurally rather
than memorising an index.

### 3.4 Required tests before any long run

Add `tests/test_cross_topology_urma.py`:

1. joint descriptions are finite and correctly shaped for H1 (19), G1 (23),
   Atlas (27);
2. the padded joint block's mask has exactly `n_joints` ones per robot;
3. two robots with different joint counts produce different description blocks;
4. padded action means are 0 for invalid joints after one forward pass;
5. one jitted PPO update runs over all three robots without NaNs.

**Do not launch anything long until these pass.**

---

## 4. Experiment plan and schedule

All times CEST. Keep at most **6 of 8** Viper slots busy — the account is shared
and saturating it blocks colleagues (this happened today).

| Window | Action |
|---|---|
| 22:00–23:30 | Build §3, run tests, 2-minute local smoke on all three robots |
| 23:30–00:00 | Submit Viper batch (6 jobs); start local GPU queue |
| 00:00–06:00 | Training. Poll hourly, archive finished runs immediately |
| 06:00–07:30 | Evaluations (local GPU is 3.6× faster — run them here) |
| 07:30–09:00 | Write `CROSS_EMBODIMENT_RESULTS.md`, run tests + ruff, final check |

### 4.1 Viper batch — the matched architecture comparison (6 jobs)

400M steps each ≈ 3.5 h at 1.9 M steps/min, leaving slack. 4,096 envs total
(~1,365 per robot), 64 rollout steps, 32 minibatches, 4 epochs, lr 1e-4.

| Job | Backbone | Robots | One-hot | Seed |
|---|---|---|---|---|
| 1 | URMAv2 | H1+G1+Atlas | no | 1 |
| 2 | URMAv2 | H1+G1+Atlas | no | 2 |
| 3 | masked MLP | H1+G1+Atlas | yes | 1 |
| 4 | masked MLP | H1+G1+Atlas | yes | 2 |
| 5 | URMAv2 | H1 only | no | 1 |
| 6 | masked MLP | H1 only | yes | 1 |

Jobs 5–6 are the single-topology controls: they separate *"URMA is worse in
general"* from *"URMA is worse only where it has no topology variation to
exploit"*. Without them the comparison is uninterpretable.

**ROCm rule:** one optimizer epoch was previously required for URMAv2 on Viper
(4-epoch nested updates segfaulted). Re-verify with a 1-update job **before**
submitting the batch; if it still segfaults, use 1 epoch and 4× the updates and
say so in the manifest.

### 4.2 Local GPU queue (sequential, ~6.9 M steps/min)

1. **Finish the morphology curriculum** already running (`curriculum/mid` →
   `curriculum/wide`), then evaluate on `catalogs_wide/`. This answers whether
   wide morphology is reachable by staging — a prerequisite for combining
   morphology with cross-topology.
2. **URMA vs MLP on H1 + online morphology**, 300M steps, 2 seeds, ~45 min each.
   This is the cheapest clean read on whether URMA's per-joint conditioning helps
   when *bodies* vary but topology does not.
3. If time remains: **H1 + G1 both with online morphology**, one shared policy —
   the first genuine "robots × bodies" run.

### 4.3 Evaluation protocol (non-negotiable)

Reuse `evaluate_cross_humanoid_policy.py`, extended if needed:

- **per robot**, never aggregated — an average hid a real failure today;
- same reset keys and horizon for policy and zero-action arms;
- report fall rate from `absorbing`, **not** from `done` (`done` also fires at the
  env horizon and at reference-clip exhaustion — this produced a metric that read
  0% when the truth was 94%);
- report per-robot: mean/median return, episode length, non-fall, fraction of
  environments beating zero action;
- topology-held-out: evaluate the no-one-hot URMA policy on **ToddlerBot** (30
  joints, already passes a four-topology preflight). Report it even if it fails —
  a negative here is a real result and is expected.

---

## 5. Acceptance criteria

Step 8 of the roadmap is complete only if **all** hold:

1. H1, G1 and Atlas all train in every shared update (verify from the manifest's
   per-robot group sizes, not by assumption);
2. **every** robot beats its exact-reset zero-action baseline on mean return;
3. no robot is hidden by an aggregate — per-robot table published;
4. results hold across at least two seeds;
5. the URMA-vs-MLP verdict is stated with matched budgets and single-topology
   controls, whichever way it falls;
6. topology-held-out behaviour on ToddlerBot is reported, negative or not.

A clean negative — *"URMA does not beat the masked MLP even on cross-topology,
here is the matched evidence"* — is a **successful** outcome and must not be
dressed up. The failure mode to avoid is an unfalsifiable "it trains" claim.

---

## 6. Gotchas that cost real time today — read this section

| Trap | What happens | Do instead |
|---|---|---|
| `cmd \| grep '^\[tag\]'` in pipelines | A crash is silently swallowed; the script marches on and a 60-second failure looks like success | Always include `Error\|Traceback` in the filter **and** check the exit code |
| Standalone `jit(vmap(mjx_reset_with_slot))` | Aborts (`HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`) or hangs at 99% CPU on ROCm | Fuse reset into the surrounding jitted function, as PPO does. `--catalog-check-envs 0` is the default for this reason |
| `build_resume_train_fn` | Returns a **2-arg** function; jitting it directly yields an executable `main()` cannot call | Close over the loaded state: `jax.jit(lambda k: resume_fn(k, state))` |
| Fall rate from `done` | Reads 0% when the true non-fall rate is 94% | Use `absorbing` |
| Two JAX processes, one GPU | Utilisation collapses from 100% to 24% | Serialise local runs; chain with a `while pgrep -f ...; do sleep 20; done` guard |
| JAX preallocation | Takes 13.8 GiB when it needs 1.9 | Export `XLA_PYTHON_CLIENT_PREALLOCATE=false` |
| Saturating all 8 Viper slots | Blocks colleagues on the shared account | Use ≤ 6; **never** cancel another user's job |
| Windows JAX | No CUDA wheels exist | Use the WSL env: `PYTHONPATH=$PWD/scripts ~/jaxgpu/bin/python …` |
| Unmatched historical baselines | Attributed a plateau break to the wrong cause for hours | Re-run the baseline yourself at matched budget before attributing anything |

---

## 7. Deliverables by 09:00

1. `CROSS_EMBODIMENT_RESULTS.md` — per-robot tables, the URMA-vs-MLP verdict with
   confidence intervals across seeds, topology-held-out result, and an explicit
   list of claims **not** supported;
2. all manifests, checkpoints and Slurm outputs archived under
   `experiments/cross_embodiment/`;
3. `tests/test_cross_topology_urma.py` passing, plus the existing suite
   (44 tests currently green) and `ruff check` / `ruff format --check` clean;
4. an updated roadmap status table for steps 7 and 8 in
   [`SCALING_ROADMAP.md`](SCALING_ROADMAP.md);
5. every Viper job resolved — none left running or untracked at handoff.

## 8. If the night goes badly

Priority order when time runs short. Deliver 1–3 complete rather than 1–6 partial:

1. the matched URMA-vs-MLP cross-topology comparison, 1 seed each;
2. per-robot held-out evaluation of whichever arm trained;
3. an honest write-up of what failed and why;
4. the second seed;
5. the ToddlerBot topology-held-out test;
6. the morphology × cross-topology combination.

The single most valuable outcome is a **trustworthy** per-robot answer to
"does one policy control three different humanoids better than doing nothing?" —
not a large number of half-finished runs.
