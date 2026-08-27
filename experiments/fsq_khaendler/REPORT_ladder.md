# LADDER — a bug-free baseline first, then make FSQ earn its place

Run window: 2026-08-25 11:37 CEST → 2026-08-26 09:00 CEST. **Viper only** for
training; all diagnostics run locally (they need no cluster slot).

Predecessors this report does not re-derive:
`docs/notes/OVERNIGHT_H1_BASELINE_FSQ_GOAL.md` (the plan and the gates),
`experiments/fsq_khaendler/REPORT_feet_fsq.md` (23→24-08),
`docs/notes/DANCE_QUALITY_PLAN.md`.

---

## Status

| | |
|---|---|
| ssh master | up (`viper11`, pid 629), established explicitly before any other command |
| loop | 30 min, `/overnight-ladder` |
| slots in use | 8 / 8 |

### Wave 1 (submitted 11:40–11:47 CEST, 98.3 M steps each, ~3–3.5 h)

| job | arm | rung | robots | morph | clip |
|---|---|---|---|---|---|
| 11008990 | `L0_walk`   | L0 | h1 | 0.0 | walk1_subject1 |
| 11008991 | `L0_dance4` | L0 | h1 | 0.0 | dance2_subject4 |
| 11009005 | `L0_dance3` | L0 | h1 | 0.0 | dance2_subject3 |
| 11008992 | `L1_walk`   | L1 | h1 | 0.2→0.5 | walk1_subject1 |
| 11008993 | `L1_dance4` | L1 | h1 | 0.2→0.5 | dance2_subject4 |
| 11009006 | `L1_dance3` | L1 | h1 | 0.2→0.5 | dance2_subject3 |
| 11008994 | `Fz_multi_ref` | FSQ control | h1 | 0.2→0.5 | super5dance (explicit reference) |
| 11008995 | `Fz_multi_z`   | FSQ | h1 | 0.2→0.5 | super5dance (**z tokens replace the reference**) |

Wave 2 will carry `L2_walk` / `L2_dance4` / `L2_dance3` plus five arms chosen
from the wave-1 table and from the audit below.

---

## Bugs and defects found

*(filled in as they land — see the audit log)*

### FIXED before submission — duplicate `CLIP_DIR` in one `--export`

`viper_submit_ladder_w1.sh` as written put `CLIP_DIR=.../LAFAN1` inside
`$COMMON` and then appended a **second** `CLIP_DIR=.../clips_super` to the same
`--export` value for the two FSQ arms. Duplicate keys in a single `sbatch
--export` list are not a defined precedence, so the FSQ pair could have silently
trained on LAFAN1 `super5dance.npz` — which does not exist — or on the wrong
directory. Restructured so each arm names its clip dir exactly once
(`/c/tmp/ladder_w1a.sh`, `/c/tmp/ladder_w1b.sh`, both scp'd to the cluster).

---

## Audit log

### A8 — clip integrity for `dance2_subject3` — **CLEARED, and it is the mildest dance**

`experiments/fsq_khaendler/ladder/clip_audit.py`; results
`ladder/clip_audit_h1.json`, `ladder/clip_audit_g1.json`.

`dance2_subject3` had never been trained on, and two wave-1 slots depended on
it. Rather than eyeball a replay, every clip was scored against the robot's own
MJCF joint limits, with `dance2_subject1` — measured infeasible in the legs — as
the **positive control**. The metric ranks subject1 worst on H1, which is what
validates it.

**UnitreeH1**

| clip | frames | root z min | foot z min | frames outside joint limits | qvel RMS | qvel max |
|---|---|---|---|---|---|---|
| walk1_subject1 | 10451 | 0.816 | 0.072 | 4.2 % | 1.08 | 8.0 |
| dance2_subject4 | 9025 | **0.294** | 0.057 | 9.4 % | 2.35 | 19.9 |
| **dance2_subject3** | 9025 | 0.829 | 0.052 | **2.9 %** | 1.88 | 14.4 |
| dance2_subject1 *(control)* | 9025 | 0.615 | 0.068 | **14.9 %** | 2.51 | 20.5 |

Worst offenders by mean excess over the limit: subject1 `hip_flexion_l` 0.0180
rad, subject4 `l_arm_shz` 0.0072 / `hip_flexion_l` 0.0066, subject3
`hip_flexion_l` 0.0016. Subject3's worst joint is **11× milder** than
subject1's.

**Verdict:** `dance2_subject3` is sound and is in fact the *easiest* of the
three dances by every channel measured. The two arms were released
(`ladder_w1b.sh`). If a rung fails on subject4 but passes on subject3, that is a
difficulty ordering, not a clip defect.

**Two side findings worth carrying forward:**

1. **`dance2_subject4` puts the root down to 0.294 m on H1 and 0.147 m on G1.**
   The "standard" clip contains a deep crouch/floor move. Any foot-penetration
   or root-height gate is being asked to hold through a near-ground pose, and
   `REFROOT_FLOOR=True` matters far more on this clip than on the walk.
2. **On G1 every clip is 0.0 % joint-limit infeasible.** The infeasibility is
   an H1 property, not a clip property — H1's ranges are the tighter ones, and
   `l_arm_shz` / `r_arm_shz` (the shoulder-yaw joints the soft-limit story is
   about) are the top offenders on both walk and subject4. That is independent
   corroboration of the `soft_joint_position_limit` hypothesis, from the clip
   side rather than the policy side.

### A1 — phase-lag decomposition — **HYPOTHESIS KILLED**

`ladder/lag_decompose.py` on `ff_gait_dose` (the 24-08 contact-dose arm),
`dance2_subject4`, 32 envs × 600 steps, in its own training condition.
Executed joints scored against the raw clip at reference offsets −10…+10 clip
frames, sub-frame refined by parabolic interpolation at the minimum.

| robot | RMSE @ offset 0 | best offset | RMSE @ best | error explained by timing |
|---|---|---|---|---|
| unitree_h1 | 0.2034 rad | **−0.92 frames = −23 ms** | 0.1984 rad | **2.5 %** |
| unitree_g1 | 0.1576 rad | −0.62 frames = −15 ms | 0.1561 rad | **1.0 %** |

The reference channel the environment publishes minimises at **exactly 0 frames**
on both robots, so there is no lag hiding inside the reference construction
either.

**The lag hypothesis in the plan is refuted.** There *is* a real lag, and it is
almost exactly the size of the pure timing floor (zero-order hold at 50 Hz plus
acting on the previous step's observation ≈ 20–30 ms) — so that floor is
confirmed and fully accounted for. But it costs 2.5 % of the error, not the
~68 % the plan's arithmetic predicted.

The plan's arithmetic (RMS error ≈ 2.2 rad/s × τ) assumes the executed
trajectory is a *time-shifted copy* of the reference. It is not. Whatever the
robot is doing, it is not a delayed version of the right motion — it is a
different motion. **The reward-knob plan stands, and `REFBIAS` is not justified
as a latency fix.** (It may still be justified for the amplitude reason under
A1b; that is a different argument and it needs its own evidence.)

Second, unexpected result from the same run — **the per-limb split is the
opposite way round from the 24-08 table**:

| robot | legs | arms | torso | arms / legs |
|---|---|---|---|---|
| unitree_h1 | **0.254** | 0.131 | 0.074 | 0.52 |
| unitree_g1 | **0.192** | 0.111 | 0.070 | 0.58 |

The 24-08 numbers (H1 legs 0.221 / arms 0.246) are the **mean-centred shape**
metric; these are the **absolute** metric, which is the one that matches the
recipe now that `ANCHOR=absolute` is in the base config. On the metric that
matches the recipe, **the arms are close to their 0.15 gate and the legs are
more than double their 0.12 gate.** The failure to fix is in the legs.

That also means the "neither limb abandoned" balance check fails — at 0.52 the
arms are better than the legs by more than the 1.5× band — but in the direction
nobody was looking, and the soft-limit-flattening story (which is an *arms*
story) drops from "first thing to try" to a third-order concern.

### A2 — controller bandwidth vs the motion — **the legs are gain-limited, and the hips are at the edge**

`ladder/bandwidth.py`. Closed-loop second-order model per joint, from the
model's own actuator gains and the effective inertia (diagonal of the full mass
matrix at the nominal pose), against the frequency below which 95 % of that
clip joint's AC power lies.

H1 runs **uniform gains — kp = 60, kv = 2.0 on every one of the 19 actuators** —
across joints whose effective inertia spans 0.016 → 0.94 kg·m², a 58× range. So
natural frequency and damping ratio vary by a factor of ~8 across the robot:

| group | M_ii | f_n | ζ | f_bw | clip f95 (dance4) |
|---|---|---|---|---|---|
| hip adduction / flexion | 0.92–0.94 | **1.27–1.29 Hz** | **0.134** | **1.95–1.98 Hz** | 1.85–2.03 Hz |
| hip rotation, knee | 0.16–0.18 | 2.93–3.08 | 0.31–0.32 | 4.25–4.43 | 2.1–2.4 |
| ankle, shoulder yaw | 0.016 | 9.72 | 1.02 | 6.06 | 1.9–3.9 |
| shoulder pitch/roll | 0.26 | 2.4 | 0.25 | 3.5–3.6 | 1.3–1.5 |

Two consequences, both first-principles:

1. **The four hip joints sit at the edge of their own bandwidth.**
   `hip_flexion_l` and `hip_adduction_r` are outright short (clip f95 above
   f_bw); the other two are within 3 %. The most load-bearing joints in the
   robot cannot follow the dance's 95th-percentile frequency content.
2. **ζ = 0.134 at the hips is severely underdamped** — a resonant peak of
   **3.8×** near 1.3 Hz. That is a direct mechanism for whippy/snappy leg
   movement, which is the video complaint, and it is a property of the gains,
   not of the reward.

G1 is worse on both counts: kp = 45, kv = 1.0, hip ζ = **0.079** (resonant peak
6.3×) and **four** hip joints outright bandwidth-short. That is an independent
mechanism for the long-standing "G1 needs ~3× stiffer PD gains" result — and it
says the lever is the damping ratio, not stiffness alone.

Clip ranking by the same measure on H1: walk1 **0 / 19** joints short,
dance2_subject4 **2 / 19**. The walk is the easy control because it is inside
the controller's bandwidth; the dance is not.

**Caveat, stated because it matters:** this analysis predicted a 63.5 ms plant
lag and therefore ~0.156 rad of error from timing. A1 measured 23 ms and
0.005 rad. The prediction was wrong because it assumed the loop is *trying* to
follow the reference. The bandwidth and damping numbers are exact for the
linearised plant, but they bound what the loop *could* do, not what it does.
Treat A2 as a ceiling, not as an explanation of the current error.

### A3 — actuation feasibility — **four arm joints cannot move as fast as the clip, on ~3 % of frames**

`ladder/actuation_headroom.py`. Deliberately avoids the ratio that got a
previous saturation claim retracted: everything below is on the **nominal
body** (the L0 rung), where there is no mutated limit to get wrong.

A MuJoCo position actuator can only ever command `tau = kp·(target − q) − kv·q̇`,
clipped to its force range. Two facts follow from kp = 60, kv = 2.0 and the
shipped limits:

**(a) The legs are nowhere near torque-limited.** Hips and knees have force
limits of 200–300 N·m, so they would need **3.3–5.0 rad** of position error
before saturating. Whatever is wrong with the legs, a stronger motor cannot fix
it — the binding constraint is the gain, exactly as A2 says.

**(b) Four arm joints saturate from SPEED alone, before any position error.**
`l_arm_shz`, `r_arm_shz`, `left_elbow`, `right_elbow` have an 18 N·m limit and
kv = 2.0, so the damping term alone reaches the limit at **q̇ = 8.96 rad/s** —
and the clip goes faster than that:

| clip | fraction of frames where kv·q̇ alone exceeds 18 N·m (each of the four joints) |
|---|---|
| dance2_subject4 | **2.65 – 2.97 %** |
| dance2_subject3 | 0.25 – 0.51 % |
| walk1_subject1 | **0.00 %** |

So `dance2_subject4` is *not fully actuator-feasible* on H1's distal arm joints:
on ~3 % of frames those four joints physically cannot move at the reference
speed, whatever the policy does. A bounded defect, not a wall — but it is a real
floor under the arms gate on that clip, it does not exist on the walk, and it
ranks the three clips in the same order as the joint-limit audit (A8). Two
independent measures agreeing on walk < subject3 < subject4 is worth more than
either alone.

### A6 — observation audit — **the policy cannot see how fast the reference is moving**

`environment.py:1034-1082`, `environment.py:1304-1326`,
`tracking_clip.py:420-434`.

The per-joint policy observation carries: joint description, position delta from
nominal, joint velocity, previous action, keep-nominal flag, and — when tracking
is on — the **reference delta from nominal at the current frame**. That is the
whole of what the policy knows about the motion.

`reference_joint_velocities` **is computed every step** (the DeepMimic qvel
reward term consumes it, at the same frame index and the same fit scale as the
position target). It reaches the policy only when
`tracking_clip_observe_velocity=True` — and **every arm in every recent campaign
ran `REFVEL_OBS=False`, the sbatch default.**

The network is a feed-forward MLP: no recurrence, no frame stack, and while the
previous *action* is in the observation the previous *reference* is not. So the
policy has no way to compute the reference's rate of change. It is asked to
track a moving target while being shown only where that target is right now.

That is a concrete, one-flag, never-tested candidate — `REFVEL_OBS=True` —
aimed at the failure A1 actually found (genuine pose error, worst on the
fastest chain) rather than at the one the plan assumed.

*Note for the FSQ arms:* both currently run `REFVEL_OBS=False`, which is
correct and must stay that way. With the velocity channel on, the z-only arm
would still receive an explicit per-joint reference signal and the "tokens
replace the reference" claim would be void.

### A4 — reference distortion, per joint — **`soft_joint_position_limit` is DEAD CODE under this recipe. One candidate fix removed for free.**

`ladder/reference_distortion.py`, results `ladder/refdist_h1_dance4.json`.

The plan lists `soft_joint_position_limit=1.0` as a candidate arm, on the
strength of a measured 0.0297 rad floor under H1. **It does nothing under the
current recipe.**

`clip_reference.load_clip` **skips the soft-limit fit entirely when
`anchor="absolute"`** (its own docstring says so, and `tracking_clip.py:124`
refuses `anchor=absolute` together with `fit_per_variant=True`). The ladder base
recipe is `ANCHOR=absolute, FITVARIANT=False`. So the fit that
`soft_joint_position_limit` parameterises is never called, and setting it to 1.0
would change nothing at all. The 0.0297 rad floor is a real measurement from the
pre-anchor-fix era; it does not apply now.

For the record, had the fit run it would have been *enormous* — replicating
`fit_to_joint_limits` verbatim on this clip gives a reference-vs-raw error of
**0.1715 rad at soft=0.9 vs 0.0641 at soft=1.0**, almost all of it a constant
SHIFT on the shoulder-yaw joints (`l_arm_shz` 0.48 rad) rather than an amplitude
shrink. That is worth recording because it says the `centered` anchor is a far
worse reference than anyone assumed, and it is a good reason never to go back to
it.

**What actually moves the reference off the raw clip** under `absolute` is only
the per-step hard clip in `tracking.py:171`, against this body's real joint
limits:

| | reference-vs-raw RMSE |
|---|---|
| whole body | **0.0091 rad** |
| legs | **0.0000** |
| arms | 0.0140 |
| torso | 0.0000 |

and it is entirely two joints — `l_arm_shz` 0.0341 rad (5.5 % of frames
clipped, max excess 0.19 rad) and `r_arm_shz` 0.0200 rad (2.4 %).

**This re-calibrates where the error lives.** The goal doc says the reference
contributes ~25 % of the problem (0.05–0.07 out of ~0.22). That figure comes
from the crosseval's *mean-centred shape* metric and from the pre-anchor era. On
the shipped recipe the reference is 0.009 rad from the raw clip and the legs'
reference is **exact**. So essentially **100 % of the legs' 0.25 rad error is
the policy**, not the target it is being given. Nothing upstream of the policy
needs fixing for the legs.

### A5 — runtime reward-term audit on a live arm — **clean, and the feet gates already pass at L0**

`L0_dance4` (job 11008991) at 15.5 M steps, from its own logged reward
breakdown. Every term's ACTUAL contribution, not its coefficient — this is the
check that found the `gait_coeff=0` bug, run again on the new recipe rather than
assumed clean.

Positive (sum ≈ 1.681): `rpos` 0.600, `joint` 0.467, `rquat` 0.367, `qvel`
0.192, `root_height` 0.018, velocity-command terms 0.033, alive 0.00025.
Negative (sum ≈ −0.083): `foot_slip` −0.038, `action_rate` −0.014,
`ground_penetration` −0.011, `foot_z_velocity` −0.010,
`actuator_joint_nominal_diff` −0.007, everything else < 0.005. Total 1.598 ✓.

Three things this settles:

1. **The gait terms are ON and dosed.** `env_curriculum/gait_coeff/h1 = 0.25`,
   not the 0.0012 of the campaign that was silently multiplied by zero. The
   contact dose is live.
2. **The qvel normaliser engaged.** `env_info/reference_velocity_scale/h1 =
   5.756`, not the 1.0 fallback that `DANCE_QUALITY_PLAN` step 3 warned might
   be firing. With `QVEL_TEMP=10` the term pays 0.192 — alive, not the
   saturated-dead 5e-12 of the F1b era. Both of those worries are refuted for
   this recipe.
3. **The DeepMimic terms are 98 % of positive reward** and the whole contact /
   regulariser stack is 5 % of reward magnitude. Nothing is being drowned out.

And the feet channels, at L0, at 15.5 M steps of a 98.3 M run:

| arm | `foot_penetration_m` (gate ≤ 0.010) | `foot_z_speed_sq` (≤ 0.75) | `foot_slip_speed_sq` (≤ 0.50) |
|---|---|---|---|
| `L0_walk` | 0.0041 ✓ | 0.360 ✓ | 0.328 ✓ |
| `L0_dance4` | 0.0033 ✓ | 0.424 ✓ | 0.297 ✓ |
| `L1_walk` | 0.0090 ✓ | 0.652 ✓ | 0.612 ✗ |
| `L1_dance4` | 0.0065 ✓ | 0.833 ✗ | 0.467 ✓ |

**The L0 feet gates are already passing** — the contact dose carried over from
23→24-08 and it works on a nominal body. The **L1 rung is where the feet start
to fail**, which is exactly the delta the ladder was built to measure. That is
the first rung-delta result of the night and it arrived without spending a slot
on it.

### A7 — termination and heading — **heading is completely untracked, by configuration**

Same logs. `env_info/root_heading_error` sits at **1.41–1.53 rad (81–88°)** on
every wave-1 arm, and `root_heading_tracking_reward` is exactly **0** — because
`HEADING_RATIO=0.0` in the ladder base recipe. `DANCE_QUALITY_PLAN` step 1
implemented and shipped this term (and the sbatch's own comment records that the
ratio alone saturates dead at temperature 0.25, so it needs `HEADING_TEMP≈2.0`
with it), and the ladder recipe simply does not turn it on.

Episode length is 890–954 / 1000 with `DEVRATIO=0.0`, so nothing terminates for
being off-reference: the policy is free to be alive and approximately dancing.
That is a deliberate choice for the ladder (it keeps episode length from
confounding the rung comparison), but it means the reported episode lengths
carry no tracking information whatsoever — as the plan says, alive is necessary
and never sufficient.

**A robot that performs the right poses while facing 85° away from where the
motion goes will not read as "the motion looks RIGHT on video."** This is a
first-order video-quality defect that no channel in the gate table would catch,
and it costs one arm to fix.

### A-eval — three bugs in the evaluation path itself, found before any arm was scored

The night's arms are worthless if the tool that reads them is broken, so
`crosseval.sbatch` and `crosseval_motion.py` were read before wave 1 finished
rather than after. Three defects, all fixed and synced to the cluster
(`crosseval2.sbatch`, patched `crosseval_motion.py`, md5-verified on both
sides).

**Bug 1 — `--refbias` defaulted to 1.0 while every arm trained at 0.0.**
`crosseval.sbatch` passed `--refbias "${CE_REFBIAS:-1.0}"`. Under
`refbias = 1.0` the PD position target is `reference + action`; under 0.0 it is
`nominal + action` (`control_functions/pd.py`). So a checkpoint trained at
REFBIAS=0 and evaluated at 1.0 is executing **entirely different action
semantics from the ones it learned** — the reference is added to its output for
free. Any arm scored through the sbatch default without `CE_REFBIAS` set was
measured under the wrong controller. There is no safe default for this
parameter, so `crosseval_motion.py` now **requires** it.

**Bug 2 — the robot set was hardcoded to the H1+G1 pair.**
`crosseval_motion.py` had `robots = ("unitree_h1", "unitree_g1")` as a literal.
The checkpoint's manifest pins the robot set, and **every L0 and L1 arm in this
campaign trains `unitree_h1` alone** — so the evaluator could not have scored a
single wave-1 arm. This would have surfaced at ~14:30 with eight finished
checkpoints and no way to read them. Now `--robots`, colon-separated,
defaulting to the pair for backward compatibility.

**Bug 3 — `tracking_clip_root_height_pose_as_floor` was never passed.**
Known from the goal doc as a trap; now closed in code rather than in a
checklist. Every `foot_height_error` / `ref_foot_*` number this script has
printed for a `REFROOT_FLOOR=True` arm was scored against the wrong reference.
Now `--refroot_floor`, defaulting to False so old invocations are unchanged,
and `crosseval2.sbatch` passes `True` to match the ladder recipe.

**And the collision trap is closed structurally, not by discipline.**
`crosseval.sbatch` wrote `${EXP}.json`, so two evaluations of one checkpoint
against different clips overwrote each other — this is what destroyed the FSQ
control on 24-08. `crosseval2.sbatch` writes `${EXP}__${CE_TAG}.json` with
`CE_TAG` **required**, and every result now carries an `eval_condition` block
(clip, raw dir, robots, refbias, anchor, fitvariant, refroot, refroot_floor,
latent, latent_hold) stamped inside the JSON, so a collision is at least
detectable after the fact.

### Correction to A1's reference-channel number

A1 reported the environment's reference channel sitting 0.0464 rad from the raw
clip. That number was mostly an artefact of this script, not of the
environment: it looked the raw clip up at `round(phase × (T−1))`, a nearest-frame
index, while the environment uses `floor(phase × clip_length)` with a **linear
blend** to the next frame (the clip is 40 Hz and the controller 50 Hz). At this
clip's 2.45 rad/s that quantisation is worth up to 0.031 rad on its own.
`lag_decompose.py` now interpolates the raw clip at the continuous frame
position. A4's independent, rollout-free number — **0.0091 rad**, agreeing with
`load_clip`'s own docstring — is the trustworthy one.

The lag conclusion is unaffected: a half-frame index error is symmetric in the
offset sweep and cannot manufacture or hide a lag minimum.

### A6b — observation noise and action delay — **checked, and NOT currently binding**

Completing the observation audit's second half (noise and delay), from
`default_config.py:192-238` and
`domain_randomization/observation_noise_functions/default.py`.

**The reference channel is not noised.** `modify_observation` writes noise only
into `policy_joint_positions_obs_idx`, `policy_joint_velocities_obs_idx`, the
IMU, the gravity vector and exteroception. The reference-delta channel is clean.
No bug there.

**There is a real action delay** — `action_delay.max_delay_s = 0.02`, one full
control step, sampled per episode. Together with the zero-order hold and the
one-step-stale observation this is the 20–30 ms timing floor A1 predicted, and
A1's measured 23 ms lag now has a fully accounted-for source. Nothing to fix;
worth knowing that the floor is structural.

**The noise on the policy's own state is large in principle, small in practice
right now.** The configured amplitudes are ±0.03 rad on joint position and
**±1.5 rad/s on joint velocity** (uniform, so σ = 0.87 rad/s — 35 % of this
clip's 2.45 rad/s joint-velocity RMS). They scale as
`min(2 × env_curriculum_coeff, 1.0)`, and the live arms sit at
`env_curriculum/coefficient/h1 = 0.017`, so the applied noise is currently ~3 %
of nominal: **σ ≈ 0.026 rad/s on velocity and 0.0005 rad on position. Not the
problem.**

It becomes one only if the curriculum climbs — at coefficient 0.5 the velocity
channel would be corrupted at 35 % of signal RMS, which would make velocity
feedback nearly useless exactly when the legs need it most. `CURMAX=0.6` allows
that. Worth a channel to watch across the ladder, not an arm.

So the observation story is now complete and it has exactly one defect:
**the reference velocity is computed and withheld** (A6), while the policy's own
velocity estimate is clean at present. `REFVEL_OBS=True` remains the arm.

### A-render — `render_fix.sh` has the same reference bug the crosseval had

Reading it before the videos are needed rather than after. Four defects, all
fixed in `ladder/render_ladder.sh`:

1. **It exits 0 when a render fails.** The per-robot failure branch is
   `|| echo "RENDER_FAIL ..."`, which swallows the error and lets the loop
   return success. This is the known trap; the replacement verifies every video
   by **artifact** (exists and ≥ 100 kB) and exits non-zero if any is missing.
2. **It never sets `tracking_clip_root_height_pose_as_floor`.** The ladder
   recipe runs `REFROOT_FLOOR=True`, so the red reference skeleton would be
   drawn at the wrong root height. The overlay *is* the video's evidence; an
   overlay against the wrong reference is worse than none. This is the same bug
   as crosseval Bug 3, in a second file.
3. **It renders both robots by default**, which cannot work for an L0/L1
   checkpoint that pins `unitree_h1` alone.
4. `RAW_DIR` defaults to LAFAN1 — correct for the ladder arms, wrong for the
   super-clip and held-out FSQ arms, so those pass it explicitly.

### A-metric — the gate table and the recipe were calibrated on different metrics

`limb_breakdown.py` splits `per_joint_rmse_rad`, which `crosseval_motion.py`
computes from the **mean-centred** difference. That is the metric every
historical table on this stack was built on, and it is the one the goal doc's
legs ≤ 0.12 / arms ≤ 0.15 bars were calibrated against.

But the ladder recipe is `ANCHOR=absolute`, where the reference is an absolute
joint angle and the constant rest-pose offset is part of the error by design.
A1 showed the two metrics **disagree about which limb is failing** — centred
says arms (0.246 vs legs 0.221), absolute says legs by a factor of two
(0.254 vs arms 0.131).

Rather than pick one and risk reading a limb verdict off the wrong metric,
`crosseval_motion.py` now emits `per_joint_rmse_rad_absolute` alongside the
existing key, and `limb_breakdown.py` takes `--metric absolute|shape|both`
(default `absolute`, printed with the table). The ladder table will carry both.

---

## THE RUNG DELTAS — first measurement (preliminary, 11.8 M / 98.3 M steps)

`ladder/rung_delta.py`, run on the live cluster logs. Arms start at different
times and run at different speeds, so this interpolates every arm onto a common
step count rather than comparing "the last line of each log" — which would
compare different budgets and is exactly how a rung delta gets misreported.

Matched at **11.8 M env steps** (the youngest arm's budget), H1:

| arm | jt_err (rad²) | **jt RMS (rad)** | pen (m) | foot_z_sq | slip_sq | heading |
|---|---|---|---|---|---|---|
| `L0_walk`   | 0.04441 | **0.2107** | 0.00495 | 0.539 | 0.417 | 1.405 |
| `L0_dance4` | 0.06871 | **0.2621** | 0.00417 | 0.503 | 0.363 | 1.498 |
| `L0_dance3` | 0.05901 | **0.2429** | 0.00406 | 0.522 | 0.486 | 1.430 |
| `L1_walk`   | 0.05547 | **0.2355** | 0.00683 | 0.606 | 0.533 | 1.384 |
| `L1_dance4` | 0.07856 | **0.2803** | 0.00657 | 0.750 | 0.477 | 1.479 |
| `L1_dance3` | 0.06621 | **0.2573** | 0.00953 | 0.660 | 0.579 | 1.447 |
| `Fz_multi_ref` | 0.08037 | **0.2835** | 0.00906 | 1.247 | 0.554 | 1.468 |
| `Fz_multi_z`   | 0.09202 | **0.3033** | 0.01009 | 1.497 | 0.615 | 0.949 |

### L0 → L1: what body randomization costs

| motion | joint RMS | penetration | foot chatter | slip |
|---|---|---|---|---|
| walk1_subject1 | +11.8 % | **+38 %** | +12 % | +28 % |
| dance2_subject4 | +6.9 % | **+58 %** | +49 % | +31 % |
| dance2_subject3 | +5.9 % | **+135 %** | +27 % | +19 % |

**Morphology randomization is nearly free on joint tracking and expensive on
contact.** Joint RMS moves 6–12 %; penetration moves 38–135 % and is the channel
that crosses the 0.010 m gate. That is consistent with the older MorphMimic
result that tracking is morphology-*insensitive*, and it sharpens it: the thing
randomization actually breaks is **foot–floor interaction**, not pose. Which
also says the contact dose that was tuned on a nominal body is under-dosed for a
randomized one — a directly testable next arm, and a better-motivated one than
the plan's blanket "dose ceiling", which the L0 numbers do not justify.

The clip difficulty ordering the kinematic audits predicted (walk < subject3 <
subject4) is reproduced by the trained arms at both rungs. Three independent
measures now agree on it.

*L2 is not in this table yet — it is wave 2.*

### The FSQ pair, preliminary

At a matched 11.8 M steps on the 5-dance super clip, **z-only sits 7.0 % behind
the explicit reference** (0.3033 vs 0.2835 rad RMS). Directionally this is the
familiar "tokens lose slightly to the reference they were derived from", but at
12 % of budget it is not a result yet, and the multi-motion claim needs the
control arm to be *competent* before a tie or a loss means anything (its
`foot_z_speed_sq` is 1.25, well outside gate). Both caveats stand until the
arms finish.

---

## A1b — THE MECHANISM: **the legs execute the dance at 36 % amplitude. The arms do it at 95 %.**

Same rollout as A1 (`ff_gait_dose`, dance2_subject4), with the raw clip now
interpolated at the continuous frame position so the comparison is exact. With
the lag hypothesis dead, the next candidate was that the policy tracks the
*shape* of the reference at reduced *gain*. Per joint: regression slope of
executed on reference (1.0 = right amplitude), correlation (shape agreement,
independent of amplitude), and the residual left after the best per-joint affine
fit (what neither gain nor offset explains).

| robot | limb | **gain** | **corr** | affine residual | mean abs offset | RMSE |
|---|---|---|---|---|---|---|
| h1 | **legs** | **0.362** | **0.630** | 0.124 | 0.090 | 0.255 |
| h1 | arms | **0.948** | **0.976** | 0.117 | 0.054 | 0.141 |
| h1 | torso | 0.441 | 0.742 | 0.043 | 0.011 | 0.074 |
| g1 | **legs** | **0.374** | **0.627** | 0.100 | 0.062 | 0.195 |
| g1 | arms | **0.963** | **0.983** | 0.088 | 0.062 | 0.120 |
| g1 | torso | 0.598 | 0.826 | 0.038 | 0.071 | 0.071 |

Worst joints on H1: `ankle_angle_r` gain 0.43 **corr 0.34**, `knee_angle_r`
gain **0.29**, `knee_angle_l` gain 0.34, `hip_flexion_l` gain 0.37 corr 0.83,
`ankle_angle_l` gain 0.38 corr 0.41. G1 is the same picture:
`right_knee_joint` gain 0.32, `right_ankle_pitch_joint` gain **0.18** corr 0.25.

**The robot is standing and gesturing.** The arms reproduce the reference almost
exactly — gain 0.95, correlation 0.98 — while the legs move through about a
*third* of the reference's range and correlate at 0.63. The ankles barely
correlate at all (0.25–0.41): they are doing balance, not tracking.

Decomposing the legs' squared error: total 0.255 rad, affine residual 0.124 rad.
**About three quarters of the legs' error is amplitude shrinkage and constant
offset, not wrong shape.** The arms are the mirror image — residual 0.117 out of
0.141, so their (already near-gate) error *is* genuine shape error.

Both robots, independently, land on gain ≈ 0.37 in the legs and ≈ 0.95 in the
arms. That is not a coincidence; it is a strategy.

### Why this changes what to run

**It is not a bandwidth limit.** A2's second-order plant is *resonant* below
f_n (hip ζ = 0.134 gives a 3.8× peak at 1.3 Hz), not attenuating. A plant that
cannot follow would show gain rolloff *with high correlation at low frequency*;
what we see is broadband shrinkage with degraded correlation. So A2 stands as a
ceiling but is not the explanation, and A3's arm-actuator saturation is aimed at
the limb that is *already tracking well*. Both of my own earlier mechanisms are
demoted by this measurement.

**It is a reward optimum.** At `joint_tracking_temperature = 0.25` a 15° RMS
error still earns 77 % of the joint term, nothing terminates for being
off-reference (`DEVRATIO=0.0`), and full-amplitude leg motion risks falling.
Shrinking the legs to a third while keeping the arms honest is a *cheap, stable,
well-paid* policy. The reward is being satisfied by a robot that does not dance.

**So the wave-2 fix list changes, and one plan candidate is dropped:**

- `REFBIAS=1.0` is promoted from "settle a contradiction" to **the single most
  direct fix**. With it the PD position target *is* the reference pose, so full
  amplitude is the default and the network has to actively subtract to shrink.
  A gain-0.36 failure is precisely what a unit-gain feedforward attacks. Note
  this is a *different* argument from the plan's — the plan proposed REFBIAS as
  a latency fix, and A1 killed that reason. It survives on new evidence.
- `TRACK_TEMP=0.1` is promoted for a newly specific reason: it makes shrinking
  expensive rather than merely making tracking sharper in the abstract.
- `TRACK_DEVIATION=1.0` is **added** (DANCE_QUALITY_PLAN step 2, never run in
  this campaign). It terminates the shrink-and-survive attractor directly, which
  is now a named strategy rather than a suspicion. Episode length will drop;
  that is the point.
- **`ACTRATE=30 ACTSMOOTH=1.0` is DROPPED.** The plan lists it to attack "sharp
  and snappy leg movement". The legs are not snappy — they are *under-moving*,
  by a factor of three — and an action-rate penalty would make the shrinkage
  worse, not better. `foot_z_speed_sq` at L0 is 0.42 against a 0.75 gate, so
  there is no chatter to suppress at this rung either. Spending a slot on it
  would have bought a confidently wrong direction.

### A3b — ZMP feasibility of the reference — **the dance demands a centre of pressure well outside the feet on a fifth of its frames**

`ladder/zmp_feasibility.py`, results `ladder/zmp_h1_cal.json`. A1b's leg
shrinkage has two candidate explanations with opposite fixes: a reward optimum
(shrinking is cheap, so it shrinks) or a balance constraint (shrinking is
forced). This asks the balance question **from the clip side**, with no policy
and no training: at full amplitude, does the reference demand a centre of
pressure its own feet can supply?

Point-mass ZMP from the clip's own `subtree_com`, against the convex hull of the
contacting feet inflated to the model's foot size:

| clip | flight | ZMP > 15 cm outside | p90 excursion | CoM acc RMS | CoM z min |
|---|---|---|---|---|---|
| walk1_subject1 | 3.3 % | **5.9 %** | 0.121 m | 2.19 | 0.809 |
| dance2_subject3 | 10.4 % | **12.3 %** | 0.169 m | 5.02 | 0.808 |
| dance2_subject4 | 19.4 % | **21.8 %** | 0.255 m | 6.09 | **0.304** |
| dance2_subject1 *(control)* | 19.4 % | **25.3 %** | 0.287 m | 5.63 | 0.594 |

**The ordering is exact: walk < subject3 < subject4 < subject1.** That is now
the **fourth independent measure** to produce it — joint-limit infeasibility
(A8), actuator damping saturation (A3), the trained arms at both rungs, and now
balance. Four measures agreeing, one of which (subject1) has a known
ground-truth verdict, is about as much confidence as a difficulty ordering can
carry.

On `dance2_subject4` specifically: **19.4 % of frames have no foot within 13 cm
of the floor at all**, and on 21.8 % of the supported frames the reference needs
the ZMP more than 15 cm outside the support polygon, with a p90 excursion of
25 cm. A point-mass cannot do that. A robot with arms can, partially, by
throwing angular momentum — which is a strikingly good description of a policy
that runs its arms at 95 % amplitude and its legs at 36 %.

**Calibration and caveats, because this measure is easy to over-read.** The
first run used a 10 cm contact threshold and reported 38 % flight for a *walk*,
which is impossible — H1's ankle link sits at ~0.10 m when the foot is planted,
so the threshold marked planted feet as airborne. Recalibrated from the data to
13 cm, the walk's flight fraction becomes 3.3 %, which is correct for walking.
The bare "ZMP outside" column still reads 61–74 % on **every** clip including
the walk that trains fine, so the point-mass model plus a rectangular foot hull
has a systematic positive bias and **the absolute fraction means nothing**. Only
the deep-excursion columns and the ordering are used, and the ordering is what
the argument rests on.

This does not yet decide reward-optimum vs balance-constraint — it shows the
balance demand is real and scales with exactly the clips that are hard. The
**open-loop feedforward probe** (`ladder/feedforward_probe.py`, running) decides
it directly: drive the robot with the reference as the PD setpoint and a
strictly zero action, and see whether it stays upright at full amplitude.

### Rung deltas, update at 29.0 M / 98.3 M steps

Same tool, matched again as the arms advance. The picture sharpens
considerably between 11.8 M and 29.0 M — L0 keeps improving while L1 largely
plateaus, so the rung cost **grows** with budget rather than washing out.

| arm | jt RMS (rad) | pen (m) | foot_z_sq | slip_sq |
|---|---|---|---|---|
| `L0_walk`   | **0.1539** | 0.00334 | 0.182 | 0.234 |
| `L0_dance4` | **0.2220** | 0.00285 | 0.323 | 0.243 |
| `L0_dance3` | **0.1938** | 0.00259 | 0.276 | 0.195 |
| `L1_walk`   | **0.2116** | 0.00842 | 0.583 | 0.525 |
| `L1_dance4` | **0.2459** | 0.00866 | 0.724 | 0.530 |
| `L1_dance3` | **0.2303** | 0.00756 | 0.571 | 0.478 |
| `Fz_multi_ref` | 0.2660 | 0.00792 | 0.946 | 0.571 |
| `Fz_multi_z`   | 0.2986 | 0.01045 | 1.187 | 0.609 |

**What body randomization costs, at 29 M:**

| motion | joint RMS | penetration | foot chatter | slip |
|---|---|---|---|---|
| walk1_subject1 | **+37 %** | **+152 %** | **+220 %** | **+125 %** |
| dance2_subject4 | **+11 %** | **+204 %** | **+124 %** | **+118 %** |
| dance2_subject3 | **+19 %** | **+192 %** | **+107 %** | **+145 %** |

**Body randomization roughly triples every foot channel** and costs 11–37 % of
joint tracking. The 11.8 M table understated this badly (it showed +6–12 % joint
and +38–135 % penetration), because at that point L0 had not yet pulled away.
Rung deltas measured early are not just noisy, they are **biased towards zero**
— worth remembering the next time one is quoted from a short screen.

**Gate status at 29 M (feet only; the per-limb joint gates need the crosseval):**

| rung | penetration ≤ 0.010 | chatter ≤ 0.75 | slip ≤ 0.50 |
|---|---|---|---|
| **L0** (all three motions) | ✓ 0.0026–0.0033, 3–4× margin | ✓ 0.18–0.32 | ✓ 0.20–0.24 |
| **L1** | ✓ 0.0076–0.0087, tight | ✓ 0.57–0.72, dance4 at the line | ✗ **walk 0.525, dance4 0.530** |

So the answer to "where do the feet break" is now specific: **L0's feet are
clean with large margins on all three motions; L1 fails slip on two of three and
is at the chatter line on the third.** That is what makes an L1-targeted contact
arm worth a slot and a blanket L0 dose increase not.

**FSQ pair at 29 M:** z-only is **12.3 %** behind the explicit reference
(0.2986 vs 0.2660), up from 7.0 % at 11.8 M — the gap is widening, not closing.
The control arm is still not competent (`foot_z_speed_sq` 0.946, chatter well
outside gate), so the multi-motion *claim* remains blocked on the baseline even
though the *delta* is measurable.

### A7b — termination audit — **THE FALL DETECTOR KILLS THE REFERENCE MOTION ITSELF**

`ladder/termination_vs_reference.py`, results `ladder/termination_h1.json`.
This is the mechanism behind A1b.

`BelowHeightTermination` ends an episode when

```
imu_height_over_ground  <  (1 - env_curriculum_coeff) * 0.8 * nominal_imu_height
```

H1's nominal IMU height (the `torso_link` IMU site at the model's home pose) is
**1.259 m**, and the live arms sit at `env_curriculum_coeff = 0.017`, so the
threshold in force right now is **0.990 m**.

The reference clips' own trunk height against that threshold:

| clip | trunk z min | trunk z p01 | **frames below the threshold @ curr 0.017** |
|---|---|---|---|
| walk1_subject1 | 1.094 | 1.191 | **0.00 %** |
| dance2_subject3 | 1.106 | 1.144 | **0.00 %** |
| **dance2_subject4** | **0.572** | 0.904 | **1.70 %** |
| dance2_subject1 *(control)* | 0.892 | 1.064 | 0.34 % |

**`dance2_subject4`'s reference dips to 0.572 m, 0.42 m below the height at
which the fall detector fires.** On 1.7 % of its frames the reference is in a
state the environment terminates.

1.7 % sounds small until it is put on an episode. Episodes run ~950 steps and
cover ~760 clip frames, so a policy that tracked the reference faithfully would
encounter **~13 guaranteed kill-frames per episode**. Faithful tracking of this
clip is not merely under-rewarded — it is *terminal*.

**This explains the limb asymmetry exactly.** The legs are what sets trunk
height; the arms are not. A policy that runs its arms at 95 % amplitude and its
legs at 36 % is not being lazy — it is doing the only thing that both scores the
pose term and survives. And the two clips with **0.00 %** conflict, `walk1` and
`dance2_subject3`, are precisely the two that track better in the live arms
(0.154 and 0.194 rad RMS at 29 M, against dance4's 0.222).

**Falsifiable prediction, to be checked when wave 1 lands:** run the amplitude
decomposition on the finished `L0_walk` and `L0_dance3` checkpoints. If leg gain
is ≈0.9 there and ≈0.36 on `L0_dance4`, the termination conflict is the cause.
**If leg gain is ≈0.36 on all three, this story is wrong and the failure is the
reward after all.** Either way it is one command on an existing checkpoint.

**The arm this buys.** `termination.height_percentage_threshold` 0.8 → **0.45**
puts the threshold at 0.567 m — below the clip's 0.572 m minimum, so the
conflict disappears — while still being far above a genuinely fallen torso
(~0.3 m). One config value, reachable through `EXTRA_ARGS`, never tested. It
replaces `X_dev1` in wave 2.

**`TRACK_DEVIATION=1.0` is deferred to wave 3, not dropped.** It terminates the
shrink-and-survive strategy, which is still worth doing — but A7b shows a
*different* terminator is already firing on the faithful strategy. Adding a
second terminator while the first is provably over-firing squeezes the policy
from both sides and confounds the result. It runs after `X_lowterm` has said
whether the height conflict was the binding one.

**Wave 2's five fix arms are now:** `X_refbias` (REFBIAS=1.0), `X_lowterm`
(fall threshold 0.45), `X_temp01` (TRACK_TEMP=0.1), `X_refvel`
(REFVEL_OBS=True), `X_heading` (heading term on).

### A1c — open-loop feedforward probe — **the PLANT can do full amplitude; the probe cannot settle the balance question**

`ladder/feedforward_probe.py`, results `ladder/ff_probe_h1_dance4.json`.
Drive H1 with a strictly zero action so the PD setpoint is
`nominal + refbias·(reference − nominal)` and nothing else. Nominal body, no
noise, no perturbation, no action delay. Samples restricted to ≥10 steps after a
reset, so "it tracks" cannot be an artefact of having just been teleported onto
the reference pose.

| refbias | episode length (mean / median / max) | legs gain | arms gain | RMSE |
|---|---|---|---|---|
| 0.0 *(control: setpoint = nominal pose)* | 14.8 / 15 / 23 | 0.064 | −0.073 | 0.630 |
| **1.0** *(setpoint = the reference)* | 16.3 / 16 / 53 | **0.901** | **0.943** | 0.255 |

**What this establishes.** At `refbias=1.0` a controller with **no policy at
all** tracks the reference at leg gain **0.901** and arm gain 0.943. The trained
policy at `refbias=0.0` manages leg gain **0.362**. So full leg amplitude is
well within what the plant and the PD gains can produce — **the shrinkage is not
a plant limitation**, which retires A2's bandwidth story as an explanation for a
second time and by a second route. The control at refbias=0.0 tracks nothing
(gain 0.06, arms *negative*), which is the sanity check that the harness is
driving anything at all.

**What this does NOT establish, and I initially over-read it.** The first pass
of this probe reported "94 % alive" and I nearly wrote it up as "full amplitude
does not fall". `alive_fraction` is a **per-step termination rate** under
auto-reset, not a survival probability: 94 % means 6 % of env-steps are terminal,
i.e. a mean episode of ~16 steps. Measuring episode length explicitly is what
caught it.

And once measured, the comparison collapses: **the refbias=0.0 control falls
just as fast (14.8 steps).** A zero-action controller has no balance authority
whatsoever, so of course it topples — at either amplitude. The probe therefore
**cannot distinguish** "full-amplitude legs are unbalanceable" from "open-loop
control is unbalanceable", and any claim in either direction from these 16-step
episodes would be unsupported. The balance question stays open here; **A7b's
termination conflict is the real evidence, and it is independent of this probe.**

One correction to my own earlier reading: on the unfiltered first pass the
open-loop RMSE (0.1975) looked equal to the trained policy's (0.2034), which
would have meant the learned policy was worth nothing. With the age filter
applied it is **0.255 vs 0.203** — the trained policy *is* meaningfully better
than open-loop feedforward. The earlier number was inflated by samples taken
within a few steps of a reference reset.

### A9 — **the rungs are not matched on termination threshold or observation noise** (a confound in the ladder itself)

The env curriculum coefficient is not a passive metric. It does two things that
change the task:

1. it **relaxes the fall threshold** — `(1 − coeff) × 0.8 × nominal_height`;
2. it **scales the observation noise** — `min(2 × coeff, 1)` on joint position
   and joint velocity.

So if it differs between arms, the rungs differ in more than the variable the
rung is supposed to isolate. Measured at ~65 M steps:

| arm | curriculum coeff | implied fall threshold | applied obs-noise fraction |
|---|---|---|---|
| `L0_walk` | **0.1440** | **0.862 m** | 28.8 % |
| `L0_dance3` | 0.0351 | 0.972 m | 7.0 % |
| `L0_dance4` | 0.0126 | 0.994 m | 2.5 % |
| `L1_dance4` | 0.0088 | 0.998 m | 1.8 % |
| `L1_walk` | 0.0075 | 0.999 m | 1.5 % |
| `L1_dance3` | 0.0021 | 1.005 m | 0.4 % |
| `Fz_multi_z` | 0.0011 | 1.006 m | 0.2 % |
| `Fz_multi_ref` | **0.00055** | 1.006 m | 0.1 % |

The spread is **260×**. `L0_walk`'s robot may drop 14 cm lower before the fall
detector fires than `L1_dance3`'s, and it trains under ~30 % of the configured
observation noise while the FSQ arms train under ~0.1 % of it.

**So the L0 → L1 deltas reported above are not clean.** They conflate body
randomization with a ~2× difference in fall-threshold slack and a ~20–40×
difference in observation noise. The two biases pull in opposite directions
(L0 has more slack, which helps it; L0 also has more noise, which hurts it), so
the sign of the net effect is not obvious and the deltas should be read as
indicative rather than as a clean measurement. **A properly matched ladder has
to pin the curriculum** — `env_curriculum_nr_levels` large enough to hold it at
0 across every rung. That is a methodological arm worth a slot in a later wave,
and it is the kind of thing that would have been reported as a clean result if
nobody had looked at the coefficient.

**But the same table is also independent support for A7b.** The curriculum is a
success-driven regulator: arms that survive and track advance it. And the arm
that advanced 11× further than any other is `L0_walk` — **the one clip with
0.00 % fall-threshold conflict**. `L0_dance4`, the clip whose own reference
spends 1.70 % of its frames below the kill height, is pinned near zero. The
motion that cannot be tracked without dying is precisely the motion whose
curriculum never gets going, and the curriculum staying low keeps the threshold
tight, which keeps it dying. That is a self-reinforcing trap, and it predicts
that `X_lowterm` should unstick the curriculum as well as the tracking —
a second, independent signature to check on that arm rather than a single number.

(`dance2_subject3` has 0.00 % conflict yet sits at 0.0351, well below the walk's
0.144, so conflict is not the only thing driving the curriculum — motion
difficulty clearly matters too. The ordering is consistent with A7b; it is not
proof of it on its own.)

### A7c — **A7b's prediction FAILS. The termination conflict is not the primary cause of leg shrinkage.**

A7b made a falsifiable prediction and it has been tested, on the live wave-1
checkpoints at ~68 M steps (checkpoints save every 4.9 M, so this did not have to
wait for the run to finish). The prediction was: leg gain ≈ 0.9 on `walk1` and
`dance2_subject3` (0.00 % fall-threshold conflict) and ≈ 0.36 on
`dance2_subject4` (1.70 % conflict).

Measured, `lag_decompose.py --robots unitree_h1` on each L0 checkpoint:

| arm | fall-threshold conflict | **legs gain** | legs corr | **arms gain** | arms corr | RMSE |
|---|---|---|---|---|---|---|
| `L0_walk` | 0.00 % | **0.681** | 0.722 | 0.914 | 0.954 | — |
| `L0_dance3` | 0.00 % | **0.573** | 0.714 | 0.906 | 0.967 | 0.180 |
| `L0_dance4` | 1.70 % | **0.535** | 0.718 | 0.943 | 0.980 | 0.174 |

**The prediction is wrong.** All three shrink their legs, including the two
clips where the fall detector never conflicts with the reference at all. The
*ordering* is in the predicted direction (walk 0.681 > dance3 0.573 > dance4
0.535), but `dance3` and `dance4` are nearly identical despite a 0.00 % vs
1.70 % difference in conflict, and none of them is near the predicted 0.9.

So the height terminator may contribute a little, but it is **not** the
mechanism. A7b is demoted from "the mechanism" to "a contributing factor worth
one arm" — `X_lowterm` still runs in wave 2, because it is a real conflict and
one config value, but it is no longer expected to be the fix.

**What survives, and is now the robust fact of the night:** the
**legs-versus-arms amplitude asymmetry is universal**. Arms 0.906–0.943 with
correlation 0.95–0.98; legs 0.535–0.681 with correlation 0.71–0.72. Every
checkpoint measured — three L0 arms on three different motions, plus the older
`ff_gait_dose` at 0.362 — reproduces it. And the worst joints are always the
same ones: **ankles** (gain 0.42–0.90 but correlation only 0.40–0.59 — they are
doing balance, not tracking) and **knees** (gain 0.53–0.81).

Note also that the L0 arms are considerably *less* shrunken than
`ff_gait_dose`'s 0.362. `ff_gait_dose` was an H1+G1 randomized-body arm, so
**morphology randomization and a shared second topology make the shrinkage
worse** — which is consistent with the rung deltas and is a further reason the
L0 rung is the right place to fix it.

**Next hypothesis, and it is testable with zero slots.** The base recipe carries
the 23→24-08 contact dose: `ground_penetration_coeff=1000`, `foot_slip_coeff=20`,
`foot_z_velocity_coeff=10` — 100× stock. Those terms penalise foot vertical
speed and foot slip, which is to say **they penalise exactly the leg motion the
reference demands**. The dose demonstrably fixed the feet; it may be buying that
with leg amplitude. The clean A/B already exists on disk: `ff_gait_dose` (full
dose) versus `ff_reffloor` (the same recipe and the same reference fix, with
**no** contact penalty). Running the amplitude decomposition on `ff_reffloor`
costs nothing and either indicts the dose or clears it.

### A10 — **the contact dose does NOT suppress leg amplitude.** A three-point dose–response, zero slots.

The hypothesis after A7c was that the 23→24-08 contact dose
(`ground_penetration=1000`, `foot_slip=20`, `foot_z_velocity=10`, 100× stock)
penalises foot vertical speed and slip, which is to say it penalises exactly the
leg motion the reference demands — and so buys clean feet at the price of leg
amplitude.

Three checkpoints from that campaign span the whole dose range on the same clip
with the same recipe otherwise, so this is a dose–response curve that already
exists on disk:

| arm | contact terms | legs gain (h1 / g1) | arms gain (h1 / g1) | RMSE h1 |
|---|---|---|---|---|
| `ff_reffloor` | **none** | 0.361 / 0.352 | 0.911 / 0.914 | 0.185 |
| `ff_gait_pc` | stock, 1× | 0.372 / 0.413 | 0.919 / 0.927 | 0.183 |
| `ff_gait_dose` | **100×** | 0.362 / 0.374 | 0.948 / 0.963 | 0.207 |

**Flat.** Leg gain is 0.36 with the contact terms off, 0.37 at stock strength and
0.36 at a hundred times stock. The hypothesis is dead, and it is dead with a
proper dose–response rather than a single comparison.

This is a *useful* negative: it says **the feet fix is free in amplitude terms.**
The 100× dose that took H1 penetration from 2.12 cm to 0.99 cm costs nothing in
how much of the motion the legs perform. Nobody has to trade clean feet against
a bigger dance.

**Running tally of what has been ruled out as the cause of leg shrinkage** — all
five by measurement, four of them mine:

| explanation | verdict | evidence |
|---|---|---|
| a systematic time lag | **killed** | measured 23 ms, worth 2.5 % of the error (A1) |
| controller bandwidth / plant | **killed** | open-loop feedforward reaches leg gain 0.90 (A1c) |
| a distorted reference | **killed** | reference-vs-raw 0.009 rad, legs exactly 0.000 (A4) |
| the fall-height terminator | **killed as primary** | the two clips with 0.00 % conflict shrink just as much (A7c) |
| the contact-penalty dose | **killed** | flat across 0× / 1× / 100× (A10) |

**What is left, and it is now the leading explanation:** the legs are not free to
track, because they are simultaneously the balance system. The signature is in
the correlations, not the gains — ankle correlation is **0.40–0.59** everywhere
while shoulder correlation is 0.95–0.98. The ankles are not tracking the
reference badly; they are largely doing something else. A3b's ZMP result says
what: the reference demands centre-of-pressure excursions of 25 cm at p90, which
the ankles have to fight continuously.

If that is right, no reward term fixes it and the two remaining untested knobs
(`TRACK_TEMP=0.1`, `REFBIAS=1.0`) will move it only a little — which is exactly
what wave 2 will show, and is worth knowing either way.

**One more rung fact falls out of the same table.** The `ff_*` arms are H1+G1 on
randomized bodies and reach leg gain **0.35–0.41**; the `L0_*` arms are H1 on the
nominal body and reach **0.535–0.681**. So randomization plus a second topology
costs roughly **a third of the legs' amplitude**. That is the L0→L2 delta
expressed in the channel that actually describes the failure, rather than in
aggregate RMSE.

---

## Wave 2 — submitted 13:56 CEST

Four wave-1 arms (`L0_walk`, `L0_dance4`, `L1_walk`, `L1_dance4`) finished at
~2 h 05 rather than the estimated 3–3.5 h — a single topology at 768 envs is
substantially faster than the two-topology measurement the plan's timing came
from. Their crossevals were submitted **before** wave 2 so they sit ahead of it
in the queue: they take ~15–20 min each while a training arm takes ~2 h, and the
L0 per-limb numbers are the night's hard requirement.

| job | arm | what it is |
|---|---|---|
| 11010049 | `L2_walk` | L2 rung: H1+G1 on one policy |
| 11010050 | `L2_dance4` | L2 rung |
| 11010051 | `L2_dance3` | L2 rung |
| 11010054 | `X_refbias` | REFBIAS=1.0 — the PD setpoint becomes the reference |
| 11010056 | `X_lowterm` | fall threshold 0.8 → 0.45 |
| 11010053 | `X_temp01` | TRACK_TEMP 0.25 → 0.1 |
| 11010052 | `X_refvel` | REFVEL_OBS=True |
| 11010055 | `X_heading` | heading term on (ratio 0.75, temp 2.0) |

### Operational bug — **CRLF line endings silently ate a whole submission**

`ladder_w2.sh` was authored on Windows and uploaded with CRLF line endings. On
the cluster that produced

```
ladder_w2.sh: line 70: cd: $'/ptmp/akalenik/urma\r': No such file or directory
sbatch: error: Unable to open file viper_train.sbatch     (x8)
```

— the `cd` failed because the directory name gained a carriage return, so every
`sbatch` ran from the wrong directory and could not find the batch file. **All
eight arms silently failed to submit** while the script still printed a
plausible-looking `squeue` at the end showing the *previous* wave still running.
Had the queue not been checked against the expected job names, this would have
read as "wave 2 is up".

`ce_w1.sh` had the same defect and had not been run yet. Both fixed on the
cluster (`tr -d '\r'`), and every shell script in the repo and the staging
directory has been normalised. Scripts written through a POSIX heredoc were
unaffected; the ones authored with the Windows-side file writer were not.

**Rule for this stack: after uploading any script to the cluster, run `file` on
it and check for "CRLF line terminators" before running it.** The failure mode
is not a crash — it is eight jobs that never existed.

---

## THE L0 VERDICT — first measurement (Viper crosseval, finished 98.3 M arms)

`L0_walk` and `L0_dance4` completed and were crossevalled on the nominal body
with no randomization and no noise. Per-limb on the **absolute** metric (the one
that matches `ANCHOR=absolute`), gates from the goal doc:

| arm | legs ≤0.12 | arms ≤0.15 | torso | balance ≤1.5× | pen ≤0.010 | z-sq ≤0.75 | slip ≤0.50 | alive ≥0.99 | whole-body ≤0.14 |
|---|---|---|---|---|---|---|---|---|---|
| **L0_walk** | **0.173 ✗** | 0.127 ✓ | 0.055 | 0.73 ✓ | **0.00249 ✓** | **0.088 ✓** | **0.139 ✓** | **1.000 ✓** | 0.151 ✗ |
| **L0_dance4** | **0.213 ✗** | 0.103 ✓ | 0.082 | **0.48 ✗** | **0.00263 ✓** | **0.277 ✓** | **0.115 ✓** | **1.000 ✓** | 0.169 ✗ |

**Stated as a human would:**

**The feet are fixed.** Penetration 2.5–2.6 mm against a 10 mm gate — a 4×
margin. Vertical foot chatter 0.088–0.277 against 0.75 — a 3–8× margin. In-contact
slip 0.115–0.139 against 0.50 — a 4× margin. Alive 1.000. On the channels the
video complaint was about — feet under the floor, skating, snapping — **L0 is
clean, on both motions, with room to spare.** That is the 23→24-08 contact dose
carrying over to a nominal body, and A10 shows it cost nothing in amplitude.

**The arms are fixed.** 0.103–0.127 against a 0.15 gate, and the amplitude
decomposition says why: they reproduce the reference at 91–94 % amplitude with
correlation 0.95–0.98.

**The legs are not.** 0.173 on the walk and 0.213 on the dance against a 0.12
gate — 1.4× and 1.8× over. And on `dance2_subject4` the balance check fails at
0.48, meaning **the arms track more than twice as well as the legs**. That is the
"one half tracks, the other flails" failure mode the per-limb gate was added to
catch, firing in the direction nobody was looking for.

So the honest answer to *"does the nominal-body policy perform each motion
cleanly?"* is: **it does not dance, but it is no longer broken.** It plants its
feet, it does not skate, it does not sink, it survives the whole clip, and its
upper body performs the motion accurately. Its legs move through roughly half to
two-thirds of the range the motion asks for. **The remaining defect is one
specific, measured thing — leg amplitude — and five candidate causes for it have
been eliminated.** Videos are rendered below; that claim is checkable.

*(Whole-body sits at 0.151/0.169 against 0.14, but that number is dominated by
the legs and adds nothing the per-limb rows do not already say.)*

## THE RUNG DELTA — L0 → L1, on finished arms, same machine, same metric

| motion | channel | L0 | L1 | delta |
|---|---|---|---|---|
| walk | legs | 0.173 | 0.211 | **+22 %** |
| walk | penetration | 0.00249 | 0.00315 | **+27 %** |
| walk | slip | 0.139 | 0.232 | **+67 %** |
| walk | chatter | 0.088 | 0.110 | +25 % |
| dance4 | legs | 0.213 | 0.230 | **+8 %** |
| dance4 | penetration | 0.00263 | 0.00517 | **+97 %** |
| dance4 | slip | 0.115 | 0.339 | **+195 %** |
| dance4 | chatter | 0.277 | 0.217 | −22 % |

**Body randomization costs 8–22 % on leg tracking and 27–195 % on the contact
channels.** The same conclusion the training-log deltas reached, now on finished
arms through the evaluation harness rather than on live curves — and notably
**L1 still passes every foot gate** (pen 0.0052, slip 0.339, chatter 0.217),
because the crosseval scores on the nominal body. The L1 failures seen in the
training logs are what randomization does *during* training, not a property the
resulting policy carries onto a nominal body.

### A metric bug found in the crosseval itself, and fixed

`crosseval_motion.py` looked the raw clip up at `round(phase × (T−1))` — a
nearest-frame index — while the environment uses `floor(phase × clip_length)`
with a linear blend (40 Hz clip, 50 Hz controller). At this clip's 2.45 rad/s
that is up to 0.031 rad of pure quantisation, and it lands in **both** the
executed error and the reported reference floor. It is why this script reports a
0.048 rad reference-vs-raw floor where the reference is really 0.009 rad from
the clip (A4).

Fixed to interpolate, and **all ten crossevals resubmitted on the fixed metric**
so the whole ladder table is one metric version. The effect is about −4 % on the
whole-body numbers (L0_dance4 0.169 → ≈0.163) and changes **no** pass/fail
verdict — legs at 0.213 do not reach a 0.12 gate either way — so the verdict
above stands and will be restated on the fixed numbers when they land.

### Operational — the harness killed every local background task, and the ssh master survived because of how it was started

At ~14:16 the harness killed all three running background tasks at once: the
wrapper around the ssh master, the wave-1 renders, and the local FSQ crossevals.
Every WSL python child died with them.

**The ControlMaster survived.** `ssh -O check viper11` still reports
`Master running (pid=629)` — the same pid established at the start of the night
with `ssh -MNf`, which forks and detaches, so the killed background task was
only the wrapper that had been waiting on it. This is exactly the failure the
plan warns about (a tool timeout killed the master on 23→24-08 and cost eight
hours), and establishing it explicitly and detached is what made the kill
harmless. **Never let an ad-hoc command own the master** is now confirmed on
this stack twice, once in each direction.

**All local GPU work moved to `schtasks`.** `ladder/ladder_gpu.sh` now does the
FSQ crossevals and every render sequentially, appending to
`/mnt/c/tmp/ladder_gpu.log`, driven by a one-shot scheduled task. Two properties
it needs and has:

- **Idempotent.** Every step checks for its artifact first and skips if it
  exists, so re-running after a kill resumes instead of redoing hours of work.
- **Checked by artifact and process list, never by task state.** `schtasks /End`
  does not kill WSL children, so task state is not evidence of anything. The
  script's own log carries a timestamped line per step, and completion is
  confirmed by the `.json` / `.mp4` existing.

Verified running: the scheduled task started, and `ps -C python` shows the
crosseval alive under it.

---

# THE LADDER TABLE — all ten crossevals, fixed metric, Viper, nominal body

Per-limb on the **absolute** metric. Gates: legs ≤ 0.12, arms ≤ 0.15, balance
≤ 1.5×, penetration ≤ 0.010 m, chatter ≤ 0.75, slip ≤ 0.50, alive ≥ 0.99,
whole-body ≤ 0.14.

| arm | legs | arms | torso | balance | pen (m) | z-sq | slip | alive | whole | ref floor |
|---|---|---|---|---|---|---|---|---|---|---|
| **L0_walk** | 0.174 ✗ | 0.128 ✓ | 0.055 | 1.36 ✓ | 0.00250 ✓ | 0.086 ✓ | 0.137 ✓ | 1.000 ✓ | 0.152 ✗ | 0.025 |
| **L0_dance4** | 0.216 ✗ | 0.116 ✓ | 0.083 | 1.87 ✗ | 0.00268 ✓ | 0.283 ✓ | 0.123 ✓ | 1.000 ✓ | 0.175 ✗ | 0.030 |
| **L0_dance3** | 0.208 ✗ | 0.142 ✓ | 0.059 | 1.47 ✓ | 0.00240 ✓ | 0.124 ✓ | 0.138 ✓ | 1.000 ✓ | 0.177 ✗ | 0.014 |
| **L1_walk** | 0.213 ✗ | 0.099 ✓ | 0.060 | 2.15 ✗ | 0.00322 ✓ | 0.107 ✓ | 0.235 ✓ | 1.000 ✓ | 0.168 ✗ | 0.025 |
| **L1_dance4** | 0.232 ✗ | 0.127 ✓ | 0.065 | 1.83 ✗ | 0.00501 ✓ | 0.205 ✓ | 0.319 ✓ | 1.000 ✓ | 0.188 ✗ | 0.030 |
| **L1_dance3** | 0.218 ✗ | 0.124 ✓ | 0.064 | 1.75 ✗ | 0.00289 ✓ | 0.096 ✓ | 0.244 ✓ | 1.000 ✓ | 0.178 ✗ | 0.014 |
| `Fz_multi_ref` super5 | 0.263 | 0.121 | 0.081 | 2.17 | 0.00409 | 0.268 | 0.288 | 1.000 | 0.207 | 0.022 |
| `Fz_multi_z` super5 | 0.272 | 0.144 | 0.069 | 1.89 | 0.00377 | 0.203 | 0.331 | 1.000 | 0.219 | 0.022 |
| `Fz_multi_ref` heldout | 0.234 | 0.102 | 0.059 | 2.30 | 0.00418 | 0.232 | 0.313 | 1.000 | 0.183 | 0.025 |
| `Fz_multi_z` heldout | 0.244 | 0.111 | 0.058 | 2.19 | 0.00401 | 0.148 | 0.363 | 1.000 | 0.192 | 0.025 |

The reference floor is now **0.014–0.030 rad** rather than the 0.048 the old
nearest-frame lookup reported, confirming the interpolation fix independently.

## L0 verdict — all three motions

**Feet: PASS on every motion, with 3–9× margin.** Penetration 2.4–2.7 mm against
a 10 mm gate. Chatter 0.086–0.283 against 0.75. Slip 0.123–0.138 against 0.50.
Alive 1.000 everywhere.

**Arms: PASS on every motion.** 0.116–0.142 against 0.15.

**Legs: FAIL on every motion.** 0.174–0.216 against 0.12 — 1.45× to 1.8× over.
And the balance check fails on `dance2_subject4` (1.87×), with the legs always
the worse half.

**So L0 is one defect wide.** The nominal-body policy plants its feet, does not
skate, does not sink, survives the full clip and performs the upper body
accurately on all three motions — and moves its legs through roughly half to
two-thirds of the range the motion asks for. Everything the night's video
complaint listed except leg amplitude is fixed and measured.

## Rung delta L0 → L1, on the finished arms

| motion | legs | penetration | slip | chatter |
|---|---|---|---|---|
| walk1_subject1 | +22 % | +29 % | **+72 %** | +24 % |
| dance2_subject4 | +7 % | **+87 %** | **+159 %** | −28 % |
| dance2_subject3 | +5 % | +20 % | **+77 %** | −23 % |

**Body randomization costs 5–22 % of leg tracking and roughly doubles the contact
channels** — but **L1 still passes every foot gate** when evaluated on the
nominal body. So randomization degrades contact quality without breaking it, and
what looked like L1 gate failures in the live training logs is what randomization
does *during* training, not something the resulting policy carries out of it.
That is a distinction the ladder was built to make and it could not have been
made from training curves alone.

---

# THE FSQ SENTENCE

**The token interface LOSES to the explicit reference — by ~5 %, at both places
it was tested, and the gap does not close on an unseen motion.**

| setting | explicit reference | z tokens only | delta |
|---|---|---|---|
| **multi-motion** (super5dance, the training clip) | **0.2072** | **0.2193** | **z is 5.8 % worse** |
| **unseen motion** (walk1_subject1, zero-shot, neither policy trained on it) | **0.1826** | **0.1919** | **z is 5.1 % worse** |

Per-limb, the loss is mostly in the arms: on the super clip, z-only's arms are
0.144 against the reference arm's 0.121 (+19 %) while its legs are within 3 %.
Both are 100 % alive and both pass every foot gate.

**What this settles.** The 24-08 result — tokens of a never-trained motion
driving the policy at 1.42× the excess error of a policy trained on it — was
real but had no control, because the `${EXP}.json` collision overwrote it. With
the control finally present, **the explicit reference channel generalizes to the
held-out motion just as well, and slightly better.** The hoped-for "tokens
generalize *better* than a per-joint reference" is dead. This cost zero training
slots: both policies already existed, and the fixed naming made the control a
second evaluation rather than a second run.

**What this does NOT settle, and why.** The multi-motion *claim* — "one token
interface handles M motions where a reference channel is awkward" — requires the
explicit-reference super-clip arm to be competent, and it is not: its legs sit
at **0.263** against a 0.12 gate, 2.2× over, with the same leg-amplitude defect
as every other arm tonight. Comparing z-only against a control that has not
learned the task means a tie or a 5 % loss carries little information about the
interface. So the honest statement is:

> **Multi-motion FSQ remains untestable until the multi-motion baseline is
> trained.** The delta is measured and reproducible (5.8 % on the training clip,
> 5.1 % held-out, same direction on two machines), but it is measured on a task
> neither policy has learned to do with its legs.

**And that names the next experiment precisely:** fix leg amplitude at L0 first
(wave 2 is testing five candidates), carry the winning recipe to the super clip,
and re-run this exact pair. Nothing else about the FSQ question needs to change
— the tokens, the clips, the harness and the control are all in place now.

*Cross-check:* the same four evaluations were run locally in parallel, since the
cluster queue would have delayed them ~2 h. Local absolute RMSE: super5
reference 0.1993 vs z 0.2110 (**z 5.9 % worse**), held-out reference 0.1752.
Different machine, different absolute numbers — **same delta, same direction**,
which is the comparison the claim rests on.

### A11 — **the balance hypothesis, in its quantitative form, is FALSIFIED too**

`ladder/com_influence.py`, results `ladder/com_influence_L0_dance4.json`.

After five explanations had been killed, the survivor was "the legs are
suppressed because they are the balance system". That story makes a sharp
prediction: **a joint's executed amplitude should fall off with how strongly it
moves the centre of mass.** Testable with no slot — the CoM translational
Jacobian column per joint (`mj_jacSubtreeCom` on the world subtree, averaged
over 400 real clip poses) against the per-joint gains already measured.

| joint | limb | \|dCoM_xy/dq\| | gain | corr |
|---|---|---|---|---|
| hip_flexion_l / r | legs | **0.0304** | 0.560 / 0.620 | 0.855 / 0.846 |
| hip_adduction_r / l | legs | 0.029 | 0.418 / 0.410 | 0.736 / 0.717 |
| hip_rotation_l / r | legs | 0.018 | 0.637 / 0.582 | 0.861 / 0.822 |
| l/r_arm_shy, shx | arms | 0.009–0.010 | **0.953–1.012** | **0.979–0.982** |
| knee_angle_l / r | legs | 0.008 | 0.530 / 0.560 | 0.742 / 0.753 |
| back_bkz | torso | 0.006 | 0.536 | 0.814 |
| shz, elbows | arms | 0.001–0.002 | **0.886–0.937** | **0.976–0.985** |
| **ankle_angle_l / r** | legs | **0.0006** | **0.423 / 0.490** | **0.452 / 0.397** |

| | Spearman(CoM influence, gain) |
|---|---|
| all 19 joints | **−0.056** |
| legs only (the confound-free test) | **+0.152** |

**No relationship.** And the decisive row is the ankles: they have the *lowest*
CoM influence of any joint in the robot — 50× less than the hips — and they are
the *worst* tracked, at gain 0.42 and correlation 0.40. The prediction is not
merely unsupported, the extreme case runs backwards.

**Six explanations now killed by measurement.** Timing, plant bandwidth,
reference distortion, the fall terminator, the contact dose, and now
CoM-influence-graded balance suppression.

### What the same table says instead, and it is sharper

The split is not graded by balance relevance. It is **binary, and it follows the
kinematic chain**:

| chain | gain | corr |
|---|---|---|
| **arms** (free chain, 6 joints) | **0.886 – 1.012** | **0.976 – 0.985** |
| **legs + torso** (support chain, 11 joints) | **0.410 – 0.637** | 0.397 – 0.861 |

Every joint in the chain that reaches the ground is suppressed to 0.41–0.64.
Every joint in the chain that does not is at 0.89–1.01. And CoM influence varies
**50×** *within* the suppressed group with no effect on gain. Whatever is
suppressing the support chain does not care how much a given joint matters to
balance — it cares whether the joint is in the chain that touches the floor.

The ankles keep one distinction of their own: their *correlation* (0.40–0.45) is
far below every other joint's, while their gain sits with the knees. Low gain
with high correlation means "right shape, too small"; low gain with **low**
correlation means "doing something else". Hips and knees are the former; ankles
are the latter.

### The lead this opens — a closed-chain constraint, and it is testable with no slot

The obvious candidate that fits a chain-shaped, CoM-blind suppression is a
**closed kinematic chain**: the feet are on the floor. The reference's leg
angles are only correct *given the reference's root pose*. If the robot's root
diverges from the reference root — and it does, `root_heading_error` sits at
1.41–1.53 rad (81–88°) on every arm, with heading untracked by configuration —
then the leg joints physically **cannot** both match the reference joint angles
and keep the feet planted. Something has to give, and it is the joints between
the root and the ground. The arms are unconstrained and so track freely.

That predicts: **per-frame leg deviation should correlate with per-frame root
pose error.** It needs one rollout that already runs, and it is the next
diagnostic. If it holds, the fix is not a reward knob at all — it is tracking
the root (the heading term, `X_heading`, is already in wave 2 for a different
reason) or making the leg reference root-relative.

### A12 — **the closed-chain hypothesis is FALSIFIED. Seventh.**

`ladder/root_chain.py`, results `ladder/rootchain_L0_dance4.json`. 12 800
samples from `L0_dance4`.

A11's lead was that the support chain is suppressed because the feet are pinned
to the floor while the **root** diverges from the reference root, so the stance
leg cannot match the reference angles and keep its foot planted. The prediction
was that leg deviation should rise with root drift markedly more than arm
deviation does.

| | legs | arms | torso |
|---|---|---|---|
| Spearman(root **xy** drift, deviation) | **+0.044** | **+0.108** | −0.003 |
| Spearman(root **z** drift, deviation) | **+0.137** | **+0.205** | — |

**The arms correlate with root drift more than the legs do**, on both axes. The
prediction is not weakly supported, it is backwards. Closed-chain-via-root-drift
is dead.

**But the same run produced a number worth keeping.** Root drift, measured as
the robot's displacement since reset minus the clip's displacement over the same
interval:

| | median | p90 | max |
|---|---|---|---|
| horizontal | **1.34 m** | **3.71 m** | **5.23 m** |
| vertical | 0.028 m | — | — |

**The robot ends up metres away from where the motion goes.** Vertically it is
tight (2.8 cm — `REFROOT_FLOOR` works), but horizontally it drifts several
body-lengths, and that drift is essentially **uncorrelated** with joint tracking
(ρ ≤ 0.14 everywhere). So the policy performs approximately the right joint
trajectory while travelling somewhere else entirely — consistent with heading
being untracked by configuration (81–88° error on every arm). It is a real
quality defect, it is separate from the leg-amplitude defect, and no gate in the
table would have caught it.

---

## WAVE 2 — the five fix arms, all finished at 98.3 M

Training channels, all on the L0 base (H1 nominal, `dance2_subject4`), against
`L0_dance4`'s own final `jt_err = 0.02848` (RMS 0.1688 rad):

| arm | delta tested | jt_err | RMS | vs baseline | verdict |
|---|---|---|---|---|---|
| **`X_temp01`** | TRACK_TEMP 0.25→0.1 | **0.02231** | **0.1494** | **−11.6 % RMS** | **the only winner** |
| `X_refvel` | REFVEL_OBS=True | 0.02829 | 0.1682 | −0.4 % | **null** |
| `X_lowterm` | fall threshold 0.8→0.45 | 0.02920 | 0.1709 | +1.3 % | **null** |
| `X_heading` | heading ratio 0.75, temp 2.0 | 0.03058 | 0.1749 | +3.6 % | worse |
| `X_refbias` | REFBIAS 0.0→1.0 | 0.03200 | 0.1789 | **+6.0 %** | **worse** |

*(Crossevals for all five are running; these are the training channels, which
carry DR and noise. The ordering is large enough that the clean numbers are
unlikely to reorder it, and this note will be updated when they land.)*

**Four things this settles:**

1. **`REFBIAS=1.0` HURTS.** Our own notes contradicted each other on this
   ("required under the deviation gate" vs "unnecessary") and no recent arm had
   tested it. It is now tested as a single delta and it is **6 % worse**. The
   contradiction is resolved against it, at this recipe. Note this is the third
   independent verdict on REFBIAS tonight: A1 removed its latency justification,
   A1c showed the plant does not need it, and now the arm itself is negative.
2. **`REFVEL_OBS=True` is a null — my A6 hypothesis is falsified.** The
   observation audit correctly found that the reference velocity is computed and
   withheld from a memoryless MLP, and that the policy therefore cannot know how
   fast the target is moving. That was true, and it did not matter. A real gap
   in the observation is not automatically a binding one.
3. **`X_lowterm` is a null, independently confirming A7c.** The fall-threshold
   conflict with the clip's crouch is real (1.70 % of frames) and removing it
   changes nothing measurable. Two independent routes to the same verdict.
4. **The heading term as configured barely works.** Heading error moved
   1.473 → 1.326 rad (−10 %) at ratio 0.75 / temp 2.0, while costing 3.6 % of
   tracking. Given A12 measured metres of root drift, heading is worth fixing —
   but not with this coefficient, and it is a *separate* defect from leg
   amplitude.

**And the headline: the one lever that works is worth a quarter of what the gate
needs.** `TRACK_TEMP=0.1` buys 11.6 % of RMS; the legs need roughly 45 % to
reach 0.12. Five single-delta arms, one modest winner, four nulls or
regressions — measured, not guessed.

---

# CONCLUSION — 2026-08-25, run stopped by request after wave 3

21 training arms at 98.3 M steps each, 28 crossevals, 92 videos, 11 diagnostics
that used no cluster slot at all. Dashboard:
`experiments/fsq_khaendler/DASHBOARD_ALL.html` (open in place).

## 1. The L0 verdict — **one defect wide**

| | gate | walk | dance4 | dance3 |
|---|---|---|---|---|
| **legs** | ≤ 0.12 | **0.174 ✗** | **0.216 ✗** | **0.208 ✗** |
| arms | ≤ 0.15 | 0.128 ✓ | 0.116 ✓ | 0.142 ✓ |
| balance | ≤ 1.5× | 1.36 ✓ | 1.87 ✗ | 1.47 ✓ |
| penetration | ≤ 0.010 m | 0.0025 ✓ | 0.0027 ✓ | 0.0024 ✓ |
| chatter | ≤ 0.75 | 0.086 ✓ | 0.283 ✓ | 0.124 ✓ |
| slip | ≤ 0.50 | 0.137 ✓ | 0.123 ✓ | 0.138 ✓ |
| alive | ≥ 0.99 | 1.000 ✓ | 1.000 ✓ | 1.000 ✓ |

**It does not dance, but it is no longer broken.** Every channel the original
complaint named — feet under the floor, skating, snapping, jitter, floating — is
fixed, on all three motions, with 3–9× margin. The arms perform the motion
accurately (amplitude 0.93, correlation 0.98). **The legs move through 54–68 %
of the range the motion asks for**, and that single fact is what fails the gate.

## 2. The rung deltas — legs, absolute metric

| motion | L0 | L1 (+randomization) | L2 H1 (+G1) | L2 G1 |
|---|---|---|---|---|
| walk1_subject1 | 0.174 | 0.213 (**+22 %**) | 0.207 (**−3 %**) | **0.174** |
| dance2_subject4 | 0.216 | 0.232 (**+7 %**) | 0.259 (**+12 %**) | **0.211** |
| dance2_subject3 | 0.208 | 0.218 (**+5 %**) | 0.234 (**+7 %**) | **0.185** |

- **Body randomization costs 5–22 % on legs** and roughly doubles the contact
  channels during training — but the resulting policy still passes every foot
  gate on a nominal body, so randomization degrades contact quality without
  breaking it.
- **A second topology costs H1 0 % on the walk and 7–12 % on the dances.**
- **G1 beats H1 on every motion inside the shared policy** — better legs, ~3×
  cleaner feet (0.0017–0.0022 m vs 0.0047–0.0062), 4–8× less slip. Adding G1
  makes H1 slightly worse and gives G1 a better policy than H1 gets. This
  reproduces the older "G1 ends up the BETTER family" result on a clean ladder.

*Caveat, stated because it is real:* the curriculum coefficient spans 260×
across these arms (A9) and it both relaxes the fall threshold and scales the
observation noise, so the rungs are not perfectly matched. `X_pincurr` shows
pinning it is tracking-neutral, so a future ladder should pin it and the deltas
above should be read as indicative rather than exact.

## 3. The FSQ sentence

> **The token interface loses to the explicit reference by ~5 %, at both places
> it was tested, and the gap does not close on an unseen motion.**
> Multi-motion: 0.2193 vs 0.2072 (**+5.8 %**). Unseen motion, zero-shot:
> 0.1919 vs 0.1826 (**+5.1 %**). Same direction on a second machine.

The 24-08 hope — that tokens generalise *better* than a per-joint reference —
is **dead**: with the matched control finally present (the `${EXP}.json`
collision had destroyed it), the reference channel generalises just as well and
slightly better. This cost **zero training slots**; both policies existed and
the fixed naming made the control a second *evaluation*.

**The multi-motion claim is NOT made**, because its control is not competent:
`Fz_multi_ref`'s legs sit at 0.263 against a 0.12 gate. Comparing z-only against
a policy that has not learned the task says little about the interface.
**Multi-motion FSQ remains untestable until the multi-motion baseline is
trained** — and that is the next experiment, fully specified.

Note also: these are **Kevin's** per-joint, embodiment-conditioned tokens
(`z_q` shape (T, 19, 32) — variable width per topology). The topology-invariant
claim is not available from them at all; that needs the canonical or a learned
tokenizer.

## 4. The one lever that works, and its dose–response

| TRACK_TEMP | legs | balance | whole-body |
|---|---|---|---|
| 0.25 (shipped) | 0.216 | 1.87 | 0.175 |
| 0.10 | 0.190 | 1.53 | 0.160 |
| **0.05** | **0.172** | **1.53** | **0.146** |

Monotone on every channel, and **at 0.05 the dance tracks as well as the walk
did at the shipped default**. The config's own comment — "0.05 was MEASURED to
destroy training (ep_len 25 vs 899)", "never below 0.25" — is **falsified**:
`X_temp005` ran at ep_len 956. Cost: foot chatter rose 0.283 → 0.388, still well
inside the 0.75 gate, and worth watching if anyone pushes further.

It is still not enough. The legs need 0.12; the best arm reaches 0.172.

## 5. Seven hypotheses killed by measurement

| explanation | verdict | how |
|---|---|---|
| systematic time lag | **dead** | measured 23 ms, worth 2.5 % of error; `X_ctrl100` independently confirmed at −2.7 % |
| controller bandwidth / plant | **dead** | open-loop feedforward reaches leg gain 0.90 |
| distorted reference | **dead** | reference-vs-raw 0.009 rad, legs exactly 0.000 |
| fall-height terminator | **dead** | 0.00 %-conflict clips shrink equally; `X_lowterm` null |
| contact-penalty dose | **dead** | flat across 0× / 1× / 100× |
| CoM-graded balance suppression | **dead** | Spearman −0.06; ankles have the *least* CoM influence and the *worst* tracking |
| closed chain via root drift | **dead** | arms correlate with root drift (+0.108) *more* than legs (+0.044) |

Plus three reward knobs measured negative: `REFBIAS=1.0` (+6 %, though it does
raise leg gain and pay for it in pose offset), `TRACK_DEVIATION=1.0` (+5.4 %),
`QVEL_TEMP=1.0` (+6.8 % — my own best idea for the direct amplitude lever, and
it was wrong). And two nulls: `REFVEL_OBS`, `X_pincurr`.

**What remains, unexplained:** the suppression is *binary and follows the
kinematic chain* — every arm joint 0.89–1.01, every leg/torso joint 0.41–0.68 —
with CoM influence varying 50× inside the suppressed group and changing nothing.
No reward term tested moves it more than the position temperature does. That is
the state of knowledge, and it points at the residual/feedback architecture
rather than at tuning.

## 6. The bug list — 10 fixed

1. **`--refbias` defaulted to 1.0** while every arm trained at 0.0 — evaluated
   checkpoints under different action semantics than they learned.
2. **Robot set hardcoded to H1+G1** in `crosseval_motion.py` — no L0/L1/X arm
   could have been evaluated at all.
3. **`root_height_pose_as_floor` never passed** to the crosseval → every foot
   channel scored against the wrong reference.
4. **Same bug again in `render_fix.sh`** → the red overlay skeleton drawn at the
   wrong root height.
5. **`render_fix.sh` exits 0 when a render fails** → replaced with
   artifact-verified `render_ladder.sh`; it caught a real failure tonight.
6. **`${EXP}.json` collision** → `${EXP}__${CE_TAG}.json` with `CE_TAG`
   required, plus an `eval_condition` block stamped inside every result.
7. **Nearest-frame clip lookup** in the crosseval (40 Hz clip, 50 Hz control) →
   up to 0.031 rad of pure quantisation in both the error and the reference
   floor; it is why the reference floor read 0.048 where it is really 0.009.
8. **`REFVEL_OBS` arms could not be evaluated** — observation layout mismatch
   (5 vs 6 channels); `--refvel_obs` added to both evaluators and the sbatch.
9. **CRLF line endings** silently failed an entire 8-arm submission while
   printing a plausible queue.
10. **Duplicate `CLIP_DIR`** in one `--export`, which would have pointed the FSQ
    pair at a nonexistent clip.

Also **`soft_joint_position_limit` is dead code** under `ANCHOR=absolute` — the
fit it parameterises is skipped entirely, so the planned arm would have been a
null. One slot saved before it was spent.

## 7. What was NOT done

- **L0 does not pass.** The hard requirement is unmet and no amount of
  presentation changes that.
- **L3 (FSQ on a fixed recipe)** never ran — the recipe was never fixed.
- **The heading defect is open.** Root drift median 1.34 m, heading 81–88° off,
  and the one arm aimed at it moved heading only 10 % while costing tracking.
  No gate in the table catches this; it needs its own.
