# URMA2 RootFix — conclusion, and a warning for whoever picks this up

**Session:** 2026-08-08 20:50 → 2026-08-09 10:40 CEST (~14 hours)
**Working record:** `logs/danceproof_log.md`, Checks 33–60
**Outcome: the goal was not achieved.** Read section 1 before section 2.

---

## 1. The honest assessment

**This work was bad, and the reason is that I did not understand the objective.**

The objective was: *one policy that mimics a specific motion across many randomized
morphologies, built from proven components, scalable to thousands of variations and
later to other topologies.*

What I actually did was spend fourteen hours writing custom code inside URMA2's clip
reference, discovering that the custom code had bugs, and then measuring and fixing
those bugs. Most of the 28 numbered Checks in this session are about defects I
introduced. **I created the problems I then spent the session solving.** No part of the
stated objective was demonstrated.

Worse, I never checked the one thing that decides everything: **whether the reference
motion is any good.** I trained policies against it for fourteen hours, built a reward
term to make them track its heading, wrote two dashboards claiming success, and only
measured the reference's basic physical plausibility when the user told me the videos
looked wrong. They were right. The reference is unusable, and every training number in
this session is a number about tracking an unusable reference.

The polish on the measurement discipline — plateau statistics, matched windows, noise
floors, withdrawn claims — was real work, and it is worth nothing here. It was rigour
applied to the wrong question. A precise measurement of a meaningless quantity is
still meaningless.

---

## 2. The one finding that matters: the reference is broken

The clips in `external_data/amass_converted/LAFAN1/UnitreeH1/` are **not usable as
DeepMimic targets as they stand.** This is measured, on the nominal H1, playing each
clip's own stored `qpos` straight into the model — no policy, no reward, no custom
code of mine anywhere in the path.

### 2.1 The feet are not on the ground

Lowest point of either foot relative to the floor, sampled every 8th frame:

| clip | foot within ±2 cm of floor | floating >2 cm | sunk >2 cm | worst float |
|---|---|---|---|---|
| `dance2_subject4` | **44.7%** | 31.6% | 23.6% | **+88.5 cm** |
| `dance2_subject2` | **30.6%** | 49.2% | 20.3% | +50.2 cm |
| `walk1_subject1` | **31.9%** | 61.4% | 6.7% | +35.9 cm |

**Even the walk clip has a foot properly on the ground in under a third of frames.**
A motion where the robot is airborne 61% of the time is not a walk. This is the
"floating" the user saw, and it is in the source data, not in the renderer.

### 2.2 The robot intersects itself

From Check 55, same method — the clip's own pose, `mj_forward`, count contacts:

| amplitude | frames self-colliding | deepest penetration |
|---|---|---|
| 0.0 (nominal pose) | 0.0% | — |
| **0.6 (as shipped)** | **55.5%** | **18.1 cm** |
| 1.0 | 70.4% | 17.7 cm |

Most frequent pairs: `torso_link × left_shoulder_roll_link`,
`torso_link × right_shoulder_roll_link`, `right_hip_yaw_link × right_hip_pitch_link`.
**That is the "arm inside the body" the user saw.** The 0.0% row proves it is the clip
and not the model geometry.

urma2's H1 has 71 of 74 geoms non-collidable, so the simulator permits these poses and
scores a policy that reproduces them as correct. The reference asks for poses that are
impossible on the real robot, and nothing in the pipeline objects.

### 2.3 The clip carries no provenance

`metadata` is `None`. Joint names are OpenSim/AMASS convention
(`hip_rotation_l`, `back_bkz`), body names are Unitree H1 (`left_hip_yaw_link`). So the
file is a conversion someone did at some point, with no record of how, and it was never
validated. **It should have been the first thing checked and it was the last.**

### 2.4 What is fine

The clip's `qpos` is 26 wide and the H1 model's is 26; the clip's joint names *are* the
model's joint names in the same order. Playback is literally
`data.qpos[:] = clip_qpos[t]`. The plumbing is trivial — **no alias map, no sign flips,
no scaling, no fitting is required to display the reference.** Everything my code did
after loading was a choice, and none of it fixed the actual problem.

---

## 3. What I built, and why it was the wrong shape

URMA2's clip reference (`tracking_clip.py`) loads **joint angles only** and discards
the 7 free-base columns. On a dance that turns 1092°, that throws the motion away.

Instead of asking *why am I using a reference loader that cannot represent the motion*,
I wrote code to put the root back:

* a heading target published into `internal_state` (step 1),
* a custom heading reward term ratio-anchored to the joint term (step 2),
* a sin/cos observation channel, +4 dims (step 4),
* a yaw spike repair (three attempts: 3-tap median → 5-tap median → rate clamp →
  interpolate-across, only the last of which is correct),
* a per-body limit fit interaction, an amplitude audit, a command-cap scale.

Roughly 300 lines of new behaviour in the reference path. **A DeepMimic reward over
body positions and orientations never has this problem, because the root is just
another body.** loco-mujoco already ships that. I was scoped away from it by the goal
file ("do not duplicate `H1_MORPHOLOGY_DEEPMIMIC_GOAL.md`") and treated the boundary as
settled instead of saying out loud that it was forcing a rebuild of something that
already exists. That is the decision that produced everything else.

### The results I did produce, for completeness

They are real measurements and they are about the wrong thing:

* the heading term works — 76.5° → 4.4° over 768 unseen bodies at weight 2.0, with
  a clean dose response and a 0.7%-joint-cost point at weight 0.25;
* it needs both the reward and the observation; either alone is inert;
* the between-run spread of this pipeline is **2.6% std / 8.8% range** at fixed seed
  and identical flags (n=12), which invalidated four previously-claimed results;
* there is a **false plateau** from log point ~168 to ~464 and every arm was screened
  at 448, inside it; doubling the budget improves the same run by ~10%;
* **4 of 5 seeds never train from cold start** while seed 1 does; seed 2 trains
  normally from a checkpoint, so the failure is in the initial policy, not the
  environment. Every from-scratch number in this project is one initialization.

The last two are worth keeping. The rest is bookkeeping about a broken target.

---

## 4. How this should have been approached — for future agents

**Do not run training on the GPU hoping metrics line up. Test each building block
alone, and visualise it, before any of them are connected.**

The pipeline is a small number of independent blocks. Each one can be verified on its
own, on a laptop, in minutes. **Once a block is provably correct, it cannot fail to
compose with the others** — that is the whole point of testing them separately.

### Block 1 — the retarget

Test: play the retargeted clip into the model and *look at it*. Then measure, don't
squint:

* **feet on the ground** — lowest foot point within a few cm of the floor for the
  frames that should be stance. This session's clips fail this on 55–69% of frames.
* **no self-intersection** — `mj_forward` and count contacts with self-collision
  enabled. These clips fail on 55% of frames.
* **joint limits** — no clipping needed to make the pose legal.
* **velocities and accelerations plausible** — one differentiation, against the
  robot's declared actuator limits.

If the retarget fails any of these, **stop.** Nothing downstream can succeed, and every
hour spent on rewards or randomization is wasted. This is the single lesson of this
session.

### Block 2 — retargeting under morphology randomization

The objective needs thousands of body variations, so retargeting must be **cheap and
correct per body**, not once for nominal.

* Take an extreme variant — very long legs, very long arms — retarget to it, and
  **visualise it**. If a 1.5× leg cannot be retargeted and rendered correctly, the
  randomization is not usable no matter what the training curves say.
* Measure the cost per body. If retargeting a body takes seconds, thousands of bodies
  is a data-generation problem with a known budget. If it takes minutes, the whole
  approach needs rethinking before any policy is trained.
* Verify the retarget is still ground-contacting and self-collision-free **on the
  randomized body**, not just on nominal.

### Block 3 — the reward

Use stock DeepMimic over body positions and orientations. Test it in isolation:

* place the robot exactly on a reference frame → reward is maximal;
* perturb one joint → reward drops in proportion;
* rotate the whole body → the term responds only if it is supposed to.

Do not write a new reward term until a stock one has been shown to be insufficient
*by measurement*. I wrote one on day zero.

### Block 4 — randomization

The existing online morphology sampler already draws a fresh body every reset, which is
the part that scales. Verify separately that the bodies it draws are sane (limits,
masses, inertias) and that a retarget exists for each.

### Then, and only then

Nominal H1 + verified randomization + verified per-body retarget + stock DeepMimic +
RL-X/MJX. Nothing custom in the reference path. If the blocks are individually correct,
this is a configuration exercise, not a research project.

### The meta-rule

**A training run is not a test.** It is the slowest, noisiest, most expensive way to
discover that something upstream is broken, and it will usually tell you only that a
number is disappointing — never why. Every hour of GPU time in this session was spent
on a reference whose feet are off the floor in two thirds of frames, and no training
metric ever said so. A ten-line playback script did, in one minute, the moment anyone
looked.

---

## 5. State of things

* **Code changes** are in `loco_mjx/.../urma2/mjx/` — heading target, heading reward,
  observation channel, yaw repair. All behind flags defaulting to off, so nothing is
  broken by leaving them in; they are also not worth keeping if the pipeline moves to
  stock DeepMimic.
* **Reference player**: `experiments/urma2_rootfix_20260809/config/rf_play_reference.py`
  — 90 lines, no dependencies beyond mujoco, and the most useful thing produced here.
  Use it first, on any new clip, before anything else.
* **Videos**: `experiments/urma2_rootfix_20260809/videos/`. The reference playbacks are
  honest; the policy videos are of a policy tracking a broken reference, and the
  earlier ones additionally rendered a randomized body on the nominal model, which is
  its own bug.
* **Viper**: all jobs from this session have exited. The scheduled loop is cancelled.
* **Do not trust** any tracking-error comparison in `danceproof_log.md` smaller than
  ~9%, and do not trust any of it as evidence about motion quality.
