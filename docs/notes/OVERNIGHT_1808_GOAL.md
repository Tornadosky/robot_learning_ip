# Overnight 18-08 — prove what the blocker is

**Hard deadline 09:00 CEST 2026-08-18.** Conclude by then with one sentence of the
form *"the blocker is X, proven by A vs B"*, or with an explicit statement of what
is still ambiguous and which single arm would settle it. A plausible story is a
failure; a named, controlled comparison is the deliverable.

Prior context, do not re-derive: `BLOCKER_ISOLATION_GOAL.md` (the design) and
`experiments/failure_rootcause_20260817/FINDINGS.md` (everything measured on
2026-08-17, including seven of my own hypotheses that died).

---

## The one question

urma2's pipeline tracks `dance2_subject4` on H1+G1 (965/950 of 1000 steps,
~19.3 s). Ours reaches 4.3 s and its G1 sat below its own zero-action baseline.
**Which difference is responsible?**

**urma2 crossed at 10-13M steps.** That is the whole reason this fits in one
night: 15M is the discriminator, not 300M. A 15M arm is ~8 min locally
(1.93M steps/min) and ~13 min on Viper. Budget dozens, not a handful.

**Crossing criterion, fixed in advance: >= 40% of horizon at 15M, per family.**
urma2 was at 74% by 13.2M; our arms sit at 5-18%. Between 18% and 40% is
"partial" and needs a second seed before it counts.

## Measurement protocol — violating this has already voided results twice today

1. **Never the training log's aggregate `len`.** It mixes families and env counts.
   Per-family, via `scripts/eval_arms.sh` (writes `metrics/fk_<tag>.json`).
2. **Every arm scored against its own zero-action baseline**
   (`scripts/policy_vs_zero.sh`). G1 was below it for the whole project and no
   aggregate ever showed it.
3. **Common currency across pipelines**: seconds survived (`steps / control_hz`),
   fraction of horizon, tracking error **in metres**. urma2's
   `rpos_tracking_error` is body-length-normalised MEAN SQUARED — convert with
   `sqrt(x) * body_length_scale`.
4. Two seeds minimum for anything that decides a branch.
5. No qualitative claim without a video. Stills right after an RSI reset look
   good by construction — that mistake was made today.

---

## Phase 0 — 21:00-22:00. Prerequisites and zero-training checks

Build these first; several arms are blocked on them. All are small.

| # | item | size |
| --- | --- | --- |
| P0.1 | **clip aligner** `scripts/scaling/align_reference.py`: crop + resample a `Trajectory` npz between 40/100 Hz and between windows. Reuse `make_control_references.py` (FK recompute of every derived field, `_slerp` for the root quat). Both pipelines read the SAME schema, so this is the only thing standing between us and a two-way reference swap. | ~40 lines |
| P0.2 | `--n-substeps` passthrough in `parallel_cross_humanoid_train.py` (env param), recorded in the manifest | ~5 lines |
| P0.3 | gravity-based rotation terminal: subclass of the trajectory terminal handler that thresholds tilt from **gravity** instead of angular distance from the clip's quaternion centroid (`loco-mujoco/.../terminal_state_handler/traj.py:94`, centroid is ~90 deg from upright) | ~30 lines |
| P0.4 | urma2 `control_frequency_hz` as a config key (currently a `PDControl.__init__` default) | ~5 lines |
| P0.5 | reference-centred residual PD: put the per-step reference pose into our control function so the action is a residual around it, matching urma2's `position_center = nominal + bias*(reference - nominal)`. The only non-trivial item. If it is not working by 23:00, **drop it** and note it as unported. | moderate |

Zero-training checks, all on **urma2's** clip
`external_data/amass_converted/LAFAN1/{UnitreeH1,UnitreeG1}/dance2_subject4.npz`,
using the batteries already written:

| # | check | script |
| --- | --- | --- |
| W0.1 | ground contact % and penetration | `scripts/scaling/reference_grounding.py` |
| W0.2 | root residual wrench as a fraction of body weight | `scripts/scaling/reference_root_residual.py` |
| W0.3 | inverse-dynamics torque feasibility, per joint | `scripts/scaling/reference_feasibility.py` |
| W0.4 | same-window diff: ours (800 frames @100 Hz) vs theirs resampled from frame 7793 @40 Hz (194.8 s in, 320 frames = the same 8 s) | new, uses P0.1 |

Ours measured: **100% of frames with zero ground contact**, root residual
0.5-3.4x body weight, arm torque saturated 57-96%. Also note the two clips are
**not the same motion**: theirs is 9025 frames @40 Hz (225.6 s, root z down to
0.29 m), ours is 800 @100 Hz (8 s, root z never below 0.92 m).

**Gate:** if urma2's clip is grounded and force-feasible where ours is not, the
reference branch is promoted and gets the Viper majority.

## Phase 1 — 22:00-23:30. The partition (3 arms, decides everything downstream)

| arm | what | where | budget |
| --- | --- | --- | --- |
| X0 | urma2 baseline, unchanged, as tonight's yardstick | local | 15,204,352 |
| X1 | **our trainer + urma2's clip**, aligned to our 8 s window @100 Hz | Viper | 15M |
| X2 | **urma2 + our clip**, aligned to their convention | local | 15,204,352 |

| outcome | conclusion | then |
| --- | --- | --- |
| X1 crosses, X2 fails | **the reference is the blocker** | Phase 2 group R gets 6 Viper slots |
| X1 fails, X2 crosses | **the trainer is the blocker** | Phase 3 is the priority; give it the local GPU all night |
| both cross | interaction, most likely window length x horizon | run A3 and A5 first |
| neither crosses | the alignment is wrong — fix P0.1 before believing anything |

## Phase 2 — 23:30-04:00. Viper additive ladder, 8 slots always full

Our production config at 15M, one thing moved to urma2's convention per arm.
Seeds 3 and 7 for every arm; add seed 11 for anything that crosses.

| group | arm | change |
| --- | --- | --- |
| R | A1 | urma2's clip (= X1) |
| R | A2 | `--reference-grounding per_frame` — retest at 15M; the 60M result may have been past the transition |
| R | A3 | window widened to the full motion range (root z down to ~0.3 m) |
| C | A4 | 50 Hz control (`--n-substeps 10`) |
| C | A5 | horizon 20 s to match urma2 |
| B | A6 | `--pd-action-scale 0.5` on **both** families (H1 trains on torque today; urma2 uses position actuators for both) |
| B | A7 | reference-centred residual action (P0.5) |
| E | A8 | gravity-based rotation terminal (P0.3) |
| E | A9 | no rotation terminal at all — the bracket for A8 |
| O | A10 | `--update-epochs 5` — **local only**, ROCm gate forbids it on Viper |
| O | A11 | wider net `--hidden 512 256` |
| — | A0 | baseline at 15M + zero-action, both families — every arm is read against this |

**Viper discipline** (each of these has cost a run):
`UPDATE_EPOCHS=1` always; `TOTAL_ENVS<=768`, and **256 for single-robot arms** —
512 single-topology died with `ROCM_ERROR_ILLEGAL_ADDRESS`; more than 2
topologies in one graph dies at any env count; never `scancel` another job on the
shared account; `sed -i 's/\r$//'` anything scp'd from Windows; batch arms so the
XLA cache stays warm (~2 min compile warm, ~36 min cold).

## Phase 3 — parallel all night on the local GPU. Subtractive ladder on the WORKING system

Breaking something that works is more informative than fixing something that
does not: there is a clear signal to lose. urma2 at 15M, one setting moved to
ours per arm, ~33 min each. **The first arm to fall below 40% of horizon names
the blocker.**

| arm | change from urma2 | tests |
| --- | --- | --- |
| S1 | `tracking_reference_action_bias=0.0` | actions centred on nominal, not the reference — highest prior |
| S2 | `scaling_factors` x4 (≈ our full-joint-range action) | action scale |
| S3 | `control_frequency_hz=100` (P0.4) | control rate |
| S4 | `reward.nominal_diff_target=nominal` | urma2's own comment: this "rewards the ignore-the-reference baseline" |
| S5 | our clip (= X2) | reference |
| S6 | `deepmimic_enabled=False` | reward composition |

Run order S1, S4, S5, S2, S3, S6 — priors first, so a collapse is found early.

## Phase 4 — the acceptance tests. NOT optional, run regardless of Phases 1-3

The requirement is a working pipeline **with randomization and per-body
retargeting**, so these gate the deliverable:

| arm | where | what |
| --- | --- | --- |
| S7 | local | urma2 + `morphology_coeff=0.3`, fixed mode — does randomization survive? |
| S8 | local | urma2 + `tracking_clip_fit_per_variant=False` — is per-body limit fitting load-bearing? |
| A12 | Viper | our trainer, winning config + `MORPHOLOGY=continuous` — does our randomization survive the fix? |
| A13 | Viper | our trainer, winning config + `catalog2` (box corners) — per-body targets at the extremes |

## Phase 5 — 04:00-08:00. Confirm, do not explore

1. Second and third seeds on every arm that decided something. An unreplicated
   crossing is a screen.
2. **Scale the winner**: 60M on Viper, and 100M if slots allow. The question is
   whether the fix holds past the transition or only accelerates it.
3. **Video of the winning configuration**, both families, with target markers.
   Our side: `render_cross_topology_policy.py --show-targets
   --show-achieved-sites` on Windows with `MUJOCO_GL=wgl`. urma2 side:
   `scripts/render_urma2_g1.sh` as the template — and pass every training setting
   through `--set`, because the renderer defaults
   `tracking_reference_action_bias` to 0.0 and would silently mis-drive a policy
   trained at 1.0.
4. Extract frames and **look at them** before writing any qualitative sentence.

## 08:00-09:00 — write up

Deliverables in `experiments/overnight_1808/`:

* `FINDINGS.md` — the named blocker with the controlled comparison that proves
  it; every hypothesis killed, with the number that killed it; corrections to
  anything in `FINDINGS.md` from 2026-08-17 that this run overturns.
* `STATE.md` — timestamped log, written as you go so the run is resumable.
* `metrics/` — per-family evals and zero-action baselines for every arm.
* `media/` — the winning configuration's video plus the frames you inspected.
* A results table with one row per arm: tag, config delta, seed, per-family
  seconds survived, % of horizon, tracking error in metres, zero-action delta.
* Update `MEMORY.md` + the memory file with the proven blocker.

## Resource rules

* **Both engines busy at all times.** Viper 8 slots full; local GPU running an
  urma2 arm or a render whenever it is not compiling. An idle engine at 03:00 is
  the main failure mode of this plan — if you run out of designed arms, add seeds
  and then a `MORPHOLOGY=continuous` replicate of the current best.
* Local GPU: take `flock /tmp/robot_learning_local_gpu.lock` (another agent may
  use the card). Never two 512-env JAX processes on 16 GB.
* Never inline a `$var` into `wsl -- bash -lc "..."` from git-bash: the outer
  shell eats it and the arg arrives empty. It has silently swallowed a loop
  variable **four** times in this project. Write a script file, `tr -d '\r'` it
  into `/tmp`, run that.
* Never `tail` the urma2 log raw — one 40 KB donated-buffer warning and a 60-line
  box per update. Use `scripts/urma2_progress.sh`.
* If Viper's ssh master socket dies, the gateway needs a password
  (`gssapi-with-mic,password` only, no publickey, no Kerberos ticket) and you
  **cannot** restore it autonomously. Say so in `STATE.md`, move everything to
  the local GPU, and keep going.

## Time discipline

Check the clock before each phase. If behind, cut in this order: A11, A9, S6,
S3, the 100M scale-up. Never cut Phase 4 (the acceptance tests) or Phase 5.3
(the video) — a fix that has not been seen working and does not survive
randomization is not a fix.
