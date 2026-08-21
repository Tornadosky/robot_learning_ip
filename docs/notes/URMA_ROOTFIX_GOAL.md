# URMA2 RootFix — make the DeepMimic reference contain the motion

**Status:** build LANDED and VERIFIED (steps 1, 2, 4). Gate honoured. See
`logs/danceproof_log.md` Checks 33–52 and
`experiments/urma2_rootfix_20260809/report.md` — **read the SESSION 2 RESULTS
block below before anything else; it overturns several statements in this file.**

---

## ★ SESSION 2 RESULTS (2026-08-08/09) — these override the text below

1. **Root heading tracking WORKS.** Heading error 76° (four controls) → **12.2 ±
   0.7°**, a 6.3× reduction, and it needs **both** the observation and the reward:
   reward alone 79.5° (inert), observation alone 72.3° (marginal). It costs
   **11.9% joint tracking**. The goal file's own prediction about the observation
   is confirmed. *n = 1; three replicates and a weight dose response were in
   flight when cluster access expired.* (Check 50)
2. **The yaw acceptance gate is PASSED**; only the video remains.
3. **The 4-hour gate was declared flat — correctly, on joint tracking, which is
   the wrong statistic for a root build.** `root_heading_error` was logged on
   every arm from Check 34 and went unread for sixteen Checks. **Rule: before
   declaring any term dead, read the quantity that term controls.**
4. **The between-draw spread at fixed seed and identical flags is 2.6% std / 8.8%
   range on `track_err` (n = 12).** Single-arm screens cannot resolve anything
   smaller. **Replication needs no seed change — just re-submit.**
5. **`ep_len` never plateaus**; the plateau statistic is valid for `track_err`
   only. Compare arms at a matched log-point index (`rf_window.sh`).
6. **There is a FALSE PLATEAU from ~point 168 to ~464**, and every arm in this
   log was screened at 448 inside it. Doubling the budget improves `track_err`
   ~10% on the *same* runs. The method converges near **0.0098**, not 0.0145.
7. **Withdrawn on replication:** the command-cap benefit, `velcmd_pure`, the
   4-point pool monotonicity, and Check 32's "0.02% reproducibility".
8. **Body pool:** unlimited beats a finite pool by 5.2% on tracking (n = 6,
   p ≈ 0.019) and 49% on `ep_len`; pool-2 ≈ pool-32, pool-768 ≈ unlimited, so it
   is a threshold between 32 and 768, not a gradient.
9. **Seeds 2/3/4/5 never train from scratch; seed 1 does.** Seed 2 trains
   normally **from a checkpoint** (n = 2), so the environment is sound and the
   failure is entirely in the initial policy. Every from-scratch number in this
   log is one initialization's trajectory. **This is the biggest open bug.**
10. **Amplitude is still unexplained** — the fifth defect, untouched.

**Old status line:** diagnosis complete, build not started
**Written:** 2026-08-08
**Family:** Unitree H1 · **Motion:** `dance2_subject1` · **Stack:** `loco_mjx` / URMA2 / RL-X on MJX
**Resume state:** `logs/danceproof_log.md` (Checks 1–32) — read it before anything else
**Sibling goal, different route:** `H1_MORPHOLOGY_DEEPMIMIC_GOAL.md` runs the stock
LocoMuJoCo + MLP + offline-cache path. **Do not duplicate it.** That agent owns the
baseline; this goal owns the URMA2 online-randomization path. If the two ever need to
be compared, that is a later, matched experiment neither goal should improvise.

---

## Mission

Four measured mechanisms strip the root motion out of the DeepMimic reference before
the policy ever sees it. Fix them, prove the fix with the project's own instruments,
and produce the video that was the original acceptance criterion:

> **One policy, on a randomized H1 body it has never seen, visibly performing
> `dance2_subject1` — travelling and turning — with the reward's own target markers
> drawn on the video.**

This is a **build**, not a search. The diagnosis is finished. If this goal ends with a
list of coefficient values and no root in the reference, it has failed.

---

## ESTABLISHED — do NOT re-derive. Full detail in `logs/danceproof_log.md`.

### The four mechanisms (Checks 22, 23, 28, 30)

| # | mechanism | evidence | on `walk_cycle_s720_n62` |
|---|---|---|---|
| 1 | **root pose never loaded** — `clip_reference.py:184-187` skips the 7 free-base columns | dance discards 159 m of path, **2262° of yaw**, 0.566 m of z | 1.28 m, 8.5°, 3.9 cm — invisible |
| 2 | **root velocity is opt-in and inert** | `VELCMD` on/off/scale moves track_err by **0.02%** (verified, Check 30/32) | costs nothing |
| 3 | **the yaw command is saturated** | in-env: p50 = p95 = max = 1.000, **53.36% of frames** at the cap | **0.00%** saturated |
| 4 | **the pose reward penalises turning** | `rquat` 0.000 → **0.889** at 180° with joints held exactly on the reference | 8.5° total — negligible |

**Why the walk pipeline looked correct for weeks:** all four vanish on the walk clip.
Never validate a reference change on `walk_cycle_s720_n62` alone again.

### Already built and verified

* `ClipArrays.root_height` / `.root_yaw` — loaded, unwrapped, verified against
  independent recomputation (Check 26). **Nothing consumes them yet.**
* `deepmimic_heading_free` (default True) — evaluates the reference at the robot's own
  yaw. Verified: flat in yaw, still sensitive to real pose error, height target
  untouched (Check 28). **Its training benefit is NOT established** — Check 32 reversed
  the sign; three-seed replicates were in flight.
* `tracking_clip_command_cap_scale` (default 1.0) — separate cap for clip-derived
  commands. Wired and verified in-env (Check 30).
* `body_pool_size` — trains on exactly N bodies. Verified 64 envs → exactly 8 distinct
  (Check 21).
* The sbatch now **dumps its resolved flags and exported env** (Check 27). Before that,
  every env-var override was unverifiable and each defaults to a non-empty value.

### Closed negatives — do not re-run

Locomotion reward terms off (two matched pairs, flat); the keep-nominal mask (harmful);
`reference_action_bias` as a training necessity (bias-0 trains fine — it only inflated
the *measurements*); root velocity command on/off/scale; amplitude as the dance fix
(it scales joint offsets and touches none of the four mechanisms).

### Two results that survive

* **Body diversity improves tracking**, monotone across 2 / 8 / 32 / ∞:
  0.016065 → 0.015564 → 0.015129 → **0.014736**, ~5σ end to end (Check 32).
* **DEVRATIO (`termination.tracking_deviation_ratio`) is the one causality lever.**
  Quote **3.323** (bias 0), not 4.959 — the feedforward inflated it (Check 25).

### Facts that will waste a day if rediscovered

* **The H1 model has NO visual meshes** — geom groups `{0: floor, 4: 2, 5: 24}`. Every
  video shows **collision proxy spheres**, which sit below the sole by construction.
  That is what "feet below the ground" was. Use `mj_forward` (not `mj_kinematics`) and
  set `options.geomgroup[:] = 1`, or the frame is black (Check 29).
* `tracking_clip.py` and `default_config.py` are **CRLF**; `clip_reference.py` and
  `tracking.py` are **LF**. A patch written with `\n` patterns silently fails to match
  the CRLF files.
* `ARM_EXTRA` is a **colon-separated** list — a missing leading `:` fuses two flags and
  the job dies in 46 s while **slurm reports ExitCode 0:0**. Only `TRAINING EXITED rc=`
  is honest.
* Submit from **viper11/12/13**, `nr_envs=768`, `MUJOCO_GL=disable`.
* The baseline venv is `/home/smirn/locobase` — **never install into `/home/smirn/locomjx`**;
  doing so upgraded jax and broke every local rollout (Check 17).
* ~~The dance clip has four single-frame yaw glitches (1102, 1104, 4100, 8279),
  each 1 frame. Median-filter them.~~ **FALSE, measured (Checks 35, 52).** Frames
  4100 and 8279 are a genuine smooth ~560 °/s turn and must NOT be filtered.
  Only frame 1102 is a defect (−62.26° in one frame, 2490 °/s) and it is a spike
  on an ordinary turn that never returns to trend — so a median filter is a
  silent no-op and a rate clamp biases the clip's net turn by +1.87%. The fix
  that works is interpolating the yaw **across** the disturbed region with both
  endpoints fixed: 43.46 → 10.19 rad/s, total turn **preserved exactly**, 6 of
  9025 frames touched. Shipped as `_rate_limit_yaw`.
* The clip's total turn is **2161.0°**, not the 2261.8° recorded in Checks 22/26.

---

## ★ THE INSTRUMENT RULE THAT OVERRIDES EVERYTHING (Check 32)

**Never quote an arm's last logged value.** Every result is
**mean ± std over the final 20 log points**, and **no effect may be claimed that is
smaller than the spread printed beside it.**

Reading last values produced a 5.1% "improvement" that reversed sign when re-derived,
and a "4.2% noise floor" that was really 0.02%. `C:/tmp/dp_endpoint_fix.sh` does this;
port it into the repo as a proper script and use it for every table.

Corollaries: the plateau *is* reached by chunk 13, so 13-chunk arms are a valid
endpoint. Run-to-run reproducibility is ~0.02%, so **1% effects are resolvable** —
which means a null really is a null.

---

## THE BUILD — five steps, in this order

Each step ships behind a **config flag defaulting to the current behaviour**, and each
gets an **A/B arm at matched budget**. A silent code change makes every previous arm
incomparable; that already cost a confound (Check 27).

1. **Root heading target.** Publish `internal_state["reference_root_yaw_delta"]` =
   `clip_root_yaw[frame] − clip_root_yaw[rsi_frame]`, interpolated on the same frame
   and blend as the joint target. Requires the robot's yaw at episode start; the reset
   step is already marked by `rsi_phase_pending`.
2. **Root heading reward.** `exp(−wrap(robot_yaw − yaw_origin − reference_yaw_delta)² / T)`.
   Heading is an angle — **body-independent, so it needs no per-body fit.** That is the
   single biggest simplicity win available; take it.
3. **Root height.** `root_height` is already loaded. `reference_root_height` currently
   comes from grounded reference FK; make the clip's own height drive it, scaled per
   body. **`command_axis_mask` is `[1,1,0,1]` — vz is masked to zero**; either un-mask
   it or carry height as a tracking term, not a command. Decide once, write it down.
4. **Observation.** The policy cannot track what it cannot see. Add the heading delta
   (and height target) to the observation. **Without this, steps 1–3 are a reward the
   policy cannot act on** — and the resulting null would be misread as "root tracking
   does not help".
5. **Raise the cap** for clip-derived commands (`tracking_clip_command_cap_scale`), the
   dose-response for which was in flight as `dp_cap4` / `dp_cap8` / `dp_cap4_yawfree`.

**Verification is not optional.** Every step gets a test in the style of Check 28's:
one assertion that the new quantity responds to what it should, and one that it is
**flat in what it must ignore**. A term flat in everything is dead, not fixed.

---

## ★ ARM DISCIPLINE — 8 slots, always full, never idle-waiting

The failure mode this section exists to prevent: submitting 8 long arms and spending
three hours reading a queue.

**Slots 1–6 — the experiment.** Long arms, one named change each, a stated defect and
what would falsify it. Never leave a slot empty because the next hypothesis is not
ready; a control or a replicate of the current best beats an empty GPU.

**Slot 7 — the fast lane. Reserved, permanently.** Arms that answer something in
**under ~10 minutes**: 1–2 chunks from an existing checkpoint, a smoke on a new flag, a
config-dump verification, a deliberately tiny run whose only job is to prove a code path
executes. **Never let slot 7 hold a 40-minute arm.**

**Slot 8 — the long runner.** One continuing run (like `dp_combo_long`, which reached
0.012438 in 4.5 h and is the best dance policy on record). Always have one arm
accumulating steps beyond the 13-chunk table.

**While arms train, the local GPU and your own turns must be producing:**

* videos and contact sheets of whatever checkpoint exists **now** — watch them, and
  write what you saw *before* quoting a number;
* the reference-vs-delivered renderer (`reference_vs_delivered.py`) re-run on any
  reference you change — it is the fastest way to see a broken reference;
* CPU-only audits: clip statistics, FK checks, feasibility, reward-term magnitudes;
* re-deriving old tables under the plateau statistic;
* N1 causality at **both** biases on any checkpoint that lands.

**"Waiting for training" is never an accurate description of a cycle.** If a cycle
produced no measurement, say so plainly in the log and treat it as a process failure.

---

## GATES

**The 4-hour decision point (pre-registered, do not move it).** After the build lands
and the first dose-response returns:

* **Root tracking moves the plateau statistic beyond its spread on the dance** →
  continue to the long run and the demo.
* **Flat** → stop building. Write up the diagnosis with the reference videos, state
  that the root fix did not move training at this budget, and say what the next
  suspect is. **A clean negative delivered on time is the successful outcome here** —
  the anti-goal is discovering it at hour 14.

**Acceptance for the demo:**

| gate | threshold |
|---|---|
| yaw tracked | robot heading follows the clip's, error well inside the ±180° a heading-blind policy would show |
| N1 causality at **bias 0** | scramble ratio ≥ 3.0 **and** scrambled correlation < 0.2 |
| unseen bodies | metrics match the training distribution, not merely "survives" |
| per-body spread | reported — an average over bodies hides the one that folded |
| video | policy and reference side by side, target markers drawn, on an unseen body |

---

## THE OPEN QUESTION NOBODY HAS ANSWERED

**Amplitude.** The policy reproduces **34% of the clip's joint range** and the control
without termination reaches 32% — 4% apart. **None of the four root mechanisms touches
joint amplitude.** There is very likely a fifth defect. When the root work lands or
falls over, this is the next hunt. Do not close this goal claiming the pipeline is
correct while amplitude is unexplained.

Secondary: the dance reference is often **statically infeasible** (median CoM margin
−5.4 cm vs −0.5 cm for walk). Some of this motion may not be executable by H1, and that
would be a legitimate result — but it must be *measured*, not used as an excuse.

---

## DELIVERABLES

`experiments/urma2_rootfix_<timestamp>/` containing `report.md`, `config/`,
`checkpoints/`, `evaluations/`, `videos/`, `logs/`, and the arm table as
**mean ± std**, failing numbers included.

`report.md` states: the claim earned and the claims **not** earned; the four mechanisms
and which are fixed; the arm table with spreads; every closed negative; N1 at both
biases; per-body results with no body hidden in an average; and the videos.

Keep appending numbered Checks to `logs/danceproof_log.md` — it is the resume state and
it is what let this work survive a dozen context resets.

---

## Goal command

```text
Continue the URMA2 RootFix goal autonomously. Read URMA_ROOTFIX_GOAL.md and then
logs/danceproof_log.md (Checks 1-32) FIRST — together they are the resume state and
carry the four mechanisms, the closed negatives, and what each arm would falsify.
Build the five root-tracking steps in order, each behind a default-off flag with an A/B
arm and a verification test that is flat in what it must ignore. Keep all 8 Viper slots
full at all times: slots 1-6 experiment, slot 7 permanently reserved for sub-10-minute
probes from existing checkpoints, slot 8 a continuing long run. While arms train, keep
the local GPU producing videos, contact sheets, CPU audits and re-derived tables — a
cycle that measured nothing is a process failure, and say so in the log. NEVER quote an
arm's last logged value: every result is mean ± std over the final 20 log points, and no
effect may be claimed smaller than its spread. Honour the 4-hour decision point — if
root tracking does not move the plateau statistic beyond its spread, stop and write up
the negative. Do not duplicate H1_MORPHOLOGY_DEEPMIMIC_GOAL.md, which owns the stock
LocoMuJoCo baseline route. Append a numbered Check every cycle with what you measured
and what it falsified, audit every instrument before believing it, and keep failing
numbers in the table.
```
