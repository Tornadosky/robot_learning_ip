# Pipeline v3 — survival at budget, walk transfer, joint-range randomization

**Status:** goal file, not yet started
**Written:** 2026-08-10, immediately after the pipeline_v2 validation session
**Predecessor:** `experiments/pipeline_v2_20260809/STATE.md` (complete) and its
dashboard https://claude.ai/code/artifact/74b58642-7c98-488c-9e6d-cf4c3db5c83c
**Primary robot/motion:** Unitree H1 · gold `dance2_subject4` reference
(`experiments/pipeline_blocks_20260809/gold/nominal_s4_gold.npz`) + `walk1_subject1`

## Mission

Pipeline v2 closed the root-drift mechanism (reward shape, not reference; the
`dance_root_wide` + deviation-termination configuration reaches ratio
1.25 ± 0.21 vs the stand-still floor, best 1.08, end-of-episode error below the
floor in 2/3 seeds). Three things are now the frontier, in priority order:

1. **Survival is the binding constraint, not drift.** Alive fractions were
   0.0–0.4 at 800 eval steps on a 20M budget, and 1 of 3 seeds never learned
   to survive at all. Prove (or refute) that budget fixes survival, and hunt
   the dead-seed failure as a BUG, not noise.
2. **The root fix has only been shown on a motion that travels 1.8 m.** The
   historical catastrophic case is walking: 10 m of travel, best-ever policy
   6.2 m error vs a 3.6 m floor. Transfer the fix to `walk1_subject1`.
3. **Randomization still excludes the feasibility-changing dims.** Add
   joint-range shifts (and then joint-axis rotations) with per-body reference
   clamping in the reward path — the machinery pipeline_v2 deliberately
   deferred.

Plus three riders that reuse existing tooling: cache-trained contact
artifacts, held-out-body evaluation, and GN-vs-FK re-run under the fixed
reward config.

## Doctrine (unchanged from v2 — follow it exactly)

- **Test each part alone, with media, before composing.** A training run is
  not a test. Every phase produces at least one video or image whose content
  a human can judge, plus a verdict JSON.
- **Spheres in every policy/reference video are what the reward actually
  scores**, read from the same arrays at the same phase
  (`p6_policy_video.py` pattern). Never reconstruct approximate targets for
  display.
- **Every drift number is reported against the per-phase stand-still floor**
  (`p2_drift_eval.py`); every survival number against the eval horizon; pin
  or record the start phase.
- **n ≥ 3 seeds for any claim** under continuous morphology or when comparing
  configurations; single seeds are smokes.
- **Screen before training** (`screen_reference.py` gates) and **screen
  rollouts after** (`p5_rollout_screen.py`): feet, self-collision, joint
  limits (magnitude, not just frequency), velocities.
- **Prefer upstream code and configuration**; small additions go in our
  scripts, never silently into vendored `loco-mujoco/`.
- **Quick smokes locally before long runs**; chain long runs detached in WSL
  (`setsid nohup … </dev/null & disown` — background Bash tasks die at
  10 min and kill children; chain-watcher processes have died silently
  before, so verify each chain fired or launch stages directly).
- Keep a running `STATE.md` in the experiment dir; update it every iteration
  so the session survives compaction.

## Phase 1 — survival at budget (CORE)

Question: does the winning configuration actually converge to long episodes,
or is there a plateau/bug the 20M feasibility budget hid?

Arms (all on the gold reference, `dance_root_wide` preset +
`--max-root-deviation 0.5 --goal-type GoalTrajMimicRootErr`):

- `v3_root60_s{0,1,2}` — nominal-pinned, 60M steps (~18 min each).
- `v3_fkmb_root60_s{0,1,2}` — multi-body: wide 11-dim bounds,
  `--reward-type MorphMimicReward --resample-per-episode
  --terminal-handler MorphologyAwareRootPoseTrajTerminalStateHandler`, 60M.
  (The v2 number to beat: ratio 1.63, alive 0.00, n=1 at 20M.)
- If 60M curves are still rising at the end, extend the best seed to 150M
  once — report the curve, do not silently rebrand the budget.

**Dead-seed bug hunt** (this is the part that must NOT degenerate into
hyperparameter tuning): take the v2 non-survivor (`p2_gold_rootwide_s1`,
alive 0.00) and the surviving s0, and diff their training trajectories:
return/length curves from the manifests, action saturation, entropy/std if
recoverable. Then test, one at a time, cheapest first, 20M each:
(a) warm-start from a 2M checkpoint of a surviving seed (rootfix session
found warm-started seeds train normally — verify it here);
(b) `--init-std` 0.1 vs 0.2 vs 0.4;
(c) `--learnable-std`.
Deliverable: a one-paragraph causal statement ("seed death is X, fixed by
Y"), or an honest "not reproduced / not isolated" with the curves shown.

Metrics & media: drift ratio + alive fraction per arm (n=3, sd); learning
curves image (plot return AND episode length vs steps for every seed — build
a small matplotlib script, follow the dataviz skill); one policy video per
arm with reward-target spheres; rollout artifact screens on the best
checkpoint of each arm.

**Gate:** if 60M multi-body still has alive ≈ 0, STOP scaling and diagnose
(termination stats: what fraction of episode ends are deviation vs fall vs
height; the answer changes everything downstream).

## Phase 2 — walk transfer (CORE, cheap)

- Build/verify a gold-conditioned `walk1_subject1` reference: run the offline
  SMPL retarget output (or the cache clip if no SMPL fit exists — note which)
  through `condition_reference.py` semantics with BIDIRECTIONAL grounding
  (`p4_retarget_bodies.py::condition_and_complete`), then
  `screen_reference.py`. Record travel, path length, stance fraction, and the
  stand-still floor. If no reference passes the screens, report that and stop
  this phase — do not train against a failing reference.
- Arms at 20M first (smoke), then 60M n=3 if the smoke shows learning:
  `v3_walk_stock_s0` (control, expected to drift ~2× floor as history says)
  vs `v3_walk_rootwide_s{0,1,2}`.
- Success bar: `rootwide` beats the stand-still floor (ratio < 1.0 would be a
  first for any walking arm in this repo; < 1.3 with end-of-episode error
  below floor is still a strong result — report exactly what is achieved).
- Media: reference playback video with spheres (feet planting visibly);
  policy videos for both arms; drift table row with floor.

## Phase 3 — joint-range randomization with per-body clamping (STRETCH)

Design (agreed in v2): the reference must be clamped to each body's DRAWN
joint ranges, in-graph, or 18% of full-range draws make the knee's start pose
infeasible (pipeline_blocks B2).

1. **Sampler dims:** add `joint_range_shift` (± up to 0.2 rad per bound,
   drawn per joint or as one scalar severity — decide, document) to
   `MORPHOLOGY_SPEC` in `scripts/scaling/online_h1.py`, patching
   `model.jnt_range` in `_apply_morphology`; mirror in the XML generator and
   in `c2_body_equivalence.py`; re-run the c2 equivalence check (expect
   `jnt_range` in the write set, FK sites unchanged for in-range poses).
2. **Clamp in the reward path:** in `MorphMimicReward` (our subclass, not
   vendored code), clip the reference joint targets — both the qpos-term
   targets and the angles fed to the FK site-target pass — to the CURRENT
   model's `jnt_range`. One `jnp.clip` against the patched arrays.
3. **Unit probe before any training** (p6_target_probe pattern): draw bodies
   with shifted ranges, verify (a) targets equal clipped reference exactly,
   (b) unshifted bodies reproduce v2 targets bit-for-bit (regression guard),
   (c) an extreme shift produces visibly clamped targets in a rendered video.
4. **Screens:** extend `screen_reference.py` usage so per-body screening uses
   the DRAWN ranges (the `--stretch`-style hook or an explicit ranges file).
5. **Training:** 20M smoke, then n=3 at the Phase-1-winning budget, multi-body
   with the new dim active at moderate severity; compare drift/survival
   against the 11-dim baseline. Then, if clean, enable joint-axis rotation
   (±0.1 rad) — FK targets absorb axes by construction; verify with the same
   probe.
6. Media: gallery of bodies with shifted ranges (annotate the shifted
   joints), clamped-vs-unclamped target video on one extreme body, training
   comparison table.

**Gate:** if the unit probe cannot show exact clamping, do not train.

## Phase 4 — riders (small, reuse existing tools)

- **Cache contact artifacts:** `p5_rollout_screen.py` on the two
  `p2_cache_ctrl` checkpoints. Question: do policies imitating floating feet
  show worse rollout penetration/sliding than gold-trained ones? One table;
  this decides whether the conditioning stage earns its place (drift and
  survival already tied at 20M).
- **Held-out bodies:** evaluate the best multi-body checkpoint on ≥8 fresh
  seeded bodies (seeds disjoint from anything trained/screened), report
  per-body drift + alive + a 2-body video. No held-out claim exists yet.
- **GN-vs-FK, matched config:** re-run `p6_gn_body04_s0` / `p6_fk_body04_s0`
  arms with `dance_root_wide` + deviation (the v2 comparison used stock
  weights, confounding retarget quality with drift). n=2 minimum, report
  drift + alive + per-term tracking.

## Phase 5 — dashboard

Extend the SAME artifact (republish
`experiments/pipeline_v2_20260809/dashboard.html`'s successor from the new
experiment dir to URL 74b58642-7c98-488c-9e6d-cf4c3db5c83c via the `url`
parameter, or keep one page per goal and cross-link — choose one and say so
on the page). New sections: survival-at-budget curves, walk transfer, the
joint-range stage with its probe media, riders. Same standards: WebP
data-URI videos (cv2→PIL, no ffmpeg), verdict chips, floors printed beside
every drift number, honest "not achieved" cards where applicable. Load
`artifact-design` + `dataviz` skills before building.

## Infrastructure pointers (verified working in v2)

- Trainer: `scripts/scaling/online_h1_train.py` — `--reference-npz`,
  `--reward-weights dance_root_wide`, `--max-root-deviation`, `--goal-type
  GoalTrajMimicRootErr`, `--morph-low/high` (short lists pad to nominal
  pins), presets/bounds in `scripts/scaling/online_h1.py::MORPHOLOGY_SPEC`.
- Eval: `experiments/pipeline_v2_20260809/p2_drift_eval.py` (floor metric),
  `p5_rollout_screen.py`, `p6_policy_video.py`, `p6_target_probe.py`,
  `p7_collect.py` + `p7_build_dashboard.py` (rebuild + republish pattern).
- Retarget/screen: `p4_retarget_bodies.py` (FK + warm-started GN,
  bidirectional grounding), `experiments/pipeline_blocks_20260809/
  screen_reference.py`, `render_reference_video.py`.
- WSL GPU: distro `Ubuntu`, venv `~/dance_env`, RTX 4060 Ti, ~70k steps/s
  single-body / ~49k with MorphMimicReward; 20M ≈ 6 min, 60M ≈ 18 min.
  Parallel small evals during training: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.10`.
- Traps that cost time in v2: pre-expansion checkpoints (obs 438) cannot be
  evaluated under 11-dim code — never mix; `p2_*` glob picked up a stale
  smoke checkpoint; eval scripts write their JSON only at the END (a crash
  loses all arms — prefer per-arm `--arms` invocations); chain watchers can
  die silently (verify or launch directly); the LAFAN1 cache npz is 40 Hz
  while trainer windows are 100 Hz-indexed.
- Cross-session context: the axis-flip memory
  ([[locomujoco-locomjx-axis-flip]]) voids old "broken cache" numbers; the
  cache's real defect is floating feet (68% frames >2 cm, worst +27 cm,
  `screens/cache_window_locomujoco.json`). A concurrent session verified an
  H1+G1 cross-topology pipeline ([[pipeline-hg-h1g1-verified]]) — do not
  duplicate its work; H1-only is this goal's scope.

## Definition of done

1. Survival-at-budget verdict with n=3 at 60M for both single- and
   multi-body full-fix arms, learning-curve media, and a causal (or honestly
   inconclusive) dead-seed statement.
2. Walk transfer measured against its floor with videos, or a screened
   reason why no walk reference is trainable.
3. Joint-range randomization either landed (probe-verified clamp, equivalence
   re-check, n=3 training comparison, media) or blocked with the exact
   failing probe shown.
4. All three riders reported (each is ≤1 hour).
5. Dashboard updated with every new stage, videos embedded, floors printed,
   negative results stated plainly.
6. STATE.md complete; memory updated (extend [[root-drift-mechanism-closed]]
   and [[pipeline-v2-validation]] rather than duplicating).

## Goal command

```text
/goal Read PIPELINE_V3_GOAL.md completely and execute it end-to-end in the pipeline_v2 style: verify each part in isolation with media (videos with true reward-target spheres, galleries, screen verdicts) before composing; quick local smokes before long runs; n>=3 for claims; every drift number against the per-phase stand-still floor; chain long GPU runs detached in WSL and keep working between events; keep STATE.md current. Core = Phase 1 (survival at 60M, n=3, single+multi-body, dead-seed bug hunt) and Phase 2 (walk1_subject1 transfer of the root fix). Stretch = Phase 3 (joint-range randomization with in-graph per-body reference clamping, probe-gated). Riders = cache contact artifacts, held-out bodies, GN-vs-FK at the fixed config. Finish by updating the pipeline dashboard artifact with the new stages and writing the final verdicts to STATE.md and memory. Never lazily wait on training - always have a quick test running or queued.
```
