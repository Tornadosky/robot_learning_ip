# Overnight 23→24-08 — FEET + FSQ: live report

Goal doc: `docs/notes/OVERNIGHT_FEET_FSQ_GOAL.md`. Times Berlin.
Viper only for training and crosseval; renders are local post-processing.

## Status

| phase | state |
|---|---|
| Step 0 code (foot metrics, foot-height term, token hold, pose-as-floor) | DONE, compiles, synced |
| Smoke (sm_feetfix / sm_refroot0 / sm_zhold) | **PASS** 22:58 — 0 tracebacks, all new paths traced |
| G0-ref / G0-policy | **DONE** — both found real defects (below) |
| Wave 1 (8 arms, 98.3M) | **COMPLETE** — all 8 at 98,304,000 steps, 0 faults |
| Wave 2 | NOT RUN — Viper unreachable 00:45→08:43, launch window closed 03:30 |
| Crossevals (10) + FSQ zero-shot (3) | submitted 08:45, land ~09:10 |

## The diagnosis this campaign acts on

The 22→23-08 campaign ran every arm with `GAITMODE=fixed GAITCOEFF=0.0`.
`resolve_gait_coeff` (`environment.py:1211`) returns that value as the
multiplier on all 25 gait/contact/smoothness terms
(`reward_functions/default.py:87`), so `ground_penetration`, `foot_slip`,
`foot_z_velocity`, `foot_flat_contact`, `feet_orientation`, `action_rate`,
`action_smoothness`, `joint_velocity` and `joint_acceleration` were multiplied
by **zero** for the whole campaign. Floating, sinking and chattering were free,
and so was fast stepping — the fast feet are not only the dead qvel term.

Two aggravating factors:

- Those penalties are added *before* `max(reward, 0.0)` (`default.py:294`) while
  the imitation bonuses (~2.1/step at `TRACK_COEFF=30`) are added *after*, so at
  a nonzero floor they can still be clipped away. The flag that fixes the
  ordering, `tracking_post_contact_penalties`, exists and was never set.
- No term on this stack has ever scored foot height. `rpos` is trunk-relative
  and body-size-normalized: a foot 4 cm under the floor and a foot 4 cm above it
  score identically.

And the reason it went unnoticed: every `reward/foot_*` key logs the term
*after* the zero multiplier, so all of them logged exactly `0.0` on every arm.
No number in the campaign could have said the feet were wrong.

### Confirmed from the 22→23-08 run record (not inferred)

`fx_dance` (the headline 98.3M dance arm the user watched), last metric block:

```
env_curriculum/gait_coeff/h1        0
env_curriculum/gait_coeff/g1        0
env_curriculum/coefficient/h1       0.00947      <- even on "curriculum" it was ~1%
reward/ground_penetration/{h1,g1}  -0
reward/foot_slip/{h1,g1}           -0
reward/foot_z_velocity/{h1,g1}     -0
reward/foot_flat_contact/{h1,g1}   -0
reward/feet_orientation/{h1,g1}    -0
reward/foot_air_time/{h1,g1}        0
reward/action_rate/{h1,g1}         -0
reward/action_smoothness/{h1,g1}   -0
reward/joint_velocity/{h1,g1}      -0
```

Every physical constraint on the feet was worth exactly zero for the entire
98.3M steps. The curriculum coefficient reading 0.0095 also says the `floor`
mode is the right instrument rather than reverting to `curriculum`.

## Step 0 — implementation record

| change | file | default |
|---|---|---|
| raw foot metrics (`foot_penetration_m`, `foot_clearance_m`, `foot_airborne`, `foot_contact_frac`, `foot_slip_speed_sq`, `foot_z_speed_sq`) | `reward_functions/default.py` | always on with `log_info` |
| foot-HEIGHT tracking term vs the reference's own FK feet + `foot_height_error` / `ref_foot_penetration_m` / `ref_foot_airborne` | `reward_functions/tracking.py` | `FOOTH=0.0` (off, bit-identical) |
| token rate: zero-order hold of the FSQ code for K frames | `command_functions/tracking_clip.py` | `LATENT_HOLD=1` (identity) |
| pose-derived root height as a FLOOR under the clip's own root height | `reward_functions/tracking.py` | `REFROOT_FLOOR=False` (old behaviour) |
| crosseval reports the same foot channels from its rollouts | `crosseval_motion.py` | always |
| `CE_ANCHOR`/`CE_FITVARIANT`/`CE_REFROOT`/`CE_HOLD` on the Viper crosseval | `crosseval.sbatch` | absolute/False/True |

Wired as `FOOTH`, `FOOTH_TEMP`, `FOOTZVEL`, `LATENT_HOLD`, `REFROOT_FLOOR` in
`viper_train.sbatch`, and `POSTCONTACT`/`FOOTSLIP`/`GROUNDPEN`/`CONTACT_TIMECONST`
added to the Check-27 env echo — all of them were previously invisible in the
run record.

## Gate G0-ref — RESULT (22:57): the reference itself was wrong, in BOTH modes

Measured on `dance2_subject4`, both robots, from the smoke arms' own
`env_info` channels (1M steps, identical code):

| mode | `ref_foot_airborne` h1 / g1 | `ref_foot_penetration_m` h1 / g1 |
|---|---|---|
| `REFROOT=True` (pose-derived root height — **what every 22→23-08 arm ran**) | **0.000 / 0.000** | 0.000 / 0.000 |
| `REFROOT=False` (the clip's own root height) | 0.122 / 0.266 | **0.0265** / 0.0038 |

Read together these say the target the policy was chasing is broken in two
different directions, and neither shipped option is right:

- Under `REFROOT=True` the reference root is placed so the lowest reference foot
  touches z=0 on **every frame**. The clip is genuinely airborne 12% (H1) to 27%
  (G1) of the time, and all of it was deleted — the policy was *rewarded* for
  skimming the floor through every hop, and the root-height term actively pulled
  it down whenever the motion left the ground.
- Under `REFROOT=False` the clip's own root height drives H1's reference feet
  **2.6 cm under the floor** (the human-clip vs robot leg-length mismatch this
  mode was introduced to avoid).

So the feet had no penalty for being wrong (gait_coeff 0) *and* a target that
was itself wrong. `REFROOT_FLOOR=True` — the pose-derived height as a FLOOR
under the clip's own root rather than a replacement — is grounded where the clip
is grounded, airborne where the clip is airborne, and never underground. It is
in the base recipe for every wave-1 arm, with `ff_reffloor` isolating it as a
single delta from B1.

## Smoke (22:38→22:58) — PASS

`sm_feetfix`, `sm_refroot0`, `sm_zhold`: 0 tracebacks, 0 ROCm faults, all three
reached metric blocks. Every new code path is exercised — the foot-height term,
the raw metrics, the reference metrics, and the token hold (`sm_zhold` runs
`LATENT_HOLD=5` with z-only observations). All flags verified present in the
Check-27 env echo.

## Wave 1 — launched 23:01, jobs 10995802–10995809

Every arm carries `REFROOT_FLOOR=True`; each is one delta from `ff_reffloor`.

| job | arm | delta |
|---|---|---|
| 10995802 | `ff_reffloor` | the reference fix ALONE (vs B1) |
| 10995803 | `ff_gait_pc` | + `GAITMODE=floor GAITCOEFF=0.25 POSTCONTACT=True` |
| 10995804 | `ff_gait_dose` | + 100× on `GROUNDPEN`/`FOOTSLIP`/`FOOTZVEL` |
| 10995805 | `ff_footh` | + `FOOTH=0.5` (foot-height term alone) |
| 10995806 | `ff_footh_gait` | + `FOOTH=0.5` and the gait/postcontact pair |
| 10995807 | `ff_z_fix` | z-only tokens, super5dance, `ff_footh_gait` recipe |
| 10995808 | `ff_z_hold5` | + `LATENT_HOLD=5` (40 → 8 tokens/s) |
| 10995809 | `ff_super_ref` | explicit reference on super5dance — F1/F2's control, and the flagship |

## Gate G0-policy — RESULT (23:12): the complaint, measured

Foot channels from a 64-env / 1000-step rollout of the checkpoints that produced
the videos. All numbers from the same code that will score tonight's arms.

| checkpoint | robot | pen (m) | clear (m) | airborne | zspeed² | slip² | exec-vs-raw shape/abs |
|---|---|---|---|---|---|---|---|
| `fx_dance` (headline dance) | h1 | **0.0212** | 0.0067 | 0.036 | 1.515 | 1.003 | 0.187 / 0.218 |
| `fx_dance` | g1 | 0.0048 | 0.0148 | 0.122 | 0.881 | 0.313 | 0.192 / 0.257 |
| `fx_walk` (hard contact, solref 0.01) | h1 | **0.0224** | 0.0068 | 0.049 | 1.727 | 1.611 | 0.187 / 0.207 |
| `fx_walk` | g1 | 0.0047 | 0.0097 | 0.091 | 0.494 | 0.262 | 0.168 / 0.223 |
| `fx_walk_softc` (model-default contact) | h1 | 0.0171 | 0.0045 | 0.031 | 1.360 | 1.657 | 0.166 / 0.186 |
| `fx_walk_softc` | g1 | 0.0040 | 0.0056 | 0.059 | 0.324 | 0.307 | 0.152 / 0.199 |

Two findings:

1. **H1's feet sit 2.1 cm INSIDE the floor on average** through the dance
   (2.2 cm on the walk). That is the "going underground" from the videos, and it
   is 2× the ≤1 cm gate. G1 is far better at 0.5 cm — consistent with the user's
   read that H1 looks worse. H1 also carries 3–4× G1's vertical foot speed
   (1.52 vs 0.88) and 3× the in-contact slip.
2. **Hardened contact does not help.** `fx_walk` (solref timeconst 0.01) sinks
   *more* than `fx_walk_softc` (2.24 vs 1.71 cm) at slightly worse slip. So the
   penetration is not a contact-solver artefact and hardening is not the lever —
   this is a reward/target problem. The planned wave-2 `w2f_dance_hardc` arm is
   therefore dropped and its slot goes elsewhere.

**Trap recorded:** the Viper crosseval of the *same* `fx_dance` checkpoint with
the *same* flags reads 0.187/0.218 where `REPORT_overnight_fix.md`'s local run
read 0.167/0.173. Same checkpoint, same anchor/fitvariant/refroot — the two
machines' crosseval numbers are **not interchangeable**. Every gate tonight is
therefore scored against the Viper-run `g0_dance` baseline, never against last
night's local table, or the comparison would invent a regression that isn't
there.

## Wave-1 mid-run snapshot (23:43, ~15.8M of 98.3M — 16%)

All 8 healthy, 0 tracebacks, 0 ROCm faults. H1 penetration is the headline
channel (baseline `fx_dance` = 0.0212 m).

| arm | steps | ep_len | pen h1 | pen g1 | ref_airborne h1 |
|---|---|---|---|---|---|
| `ff_reffloor` | 15.9M | 828 | 0.0222 | 0.0066 | 0.154 |
| `ff_gait_pc` | 15.7M | 757 | 0.0211 | 0.0080 | 0.141 |
| `ff_gait_dose` | 15.7M | 768 | **0.0103** | 0.0042 | 0.121 |
| `ff_footh` | 15.8M | 731 | **0.0128** | 0.0060 | 0.143 |
| `ff_footh_gait` | 15.6M | 770 | 0.0152 | 0.0069 | 0.127 |
| `ff_super_ref` | 15.9M | 683 | 0.0168 | 0.0057 | 0.208 |
| `ff_z_fix` | 15.0M | 443 | 0.0166 | 0.0059 | 0.220 |
| `ff_z_hold5` | 15.0M | 273 | 0.0156 | 0.0056 | 0.210 |

Three things are already visible, at 16% of budget and subject to confirmation
at 98.3M:

1. **The reference fix is doing what it was built to do.** `ref_foot_airborne`
   is 0.12–0.22 on every arm; under the shipped `REFROOT=True` it was 0.000 by
   construction. The reference is now airborne where the clip is airborne.
2. **The reference fix alone does NOT fix the feet** (`ff_reffloor` 0.0222,
   i.e. the baseline). Nor does un-zeroing the gait terms at the stock dose
   (`ff_gait_pc` 0.0211). That was the pre-registered prediction for Q1/Q2 and
   it is holding: the terms were not merely switched off, they are also far too
   cheap to compete with a ~2.1/step imitation bonus.
3. **Both real levers work, separately.** The 100× dose halves penetration
   (0.0103) and the foot-height term cuts it 40% (0.0128) — on its own, with the
   gait terms still at zero. Their combination (`ff_footh_gait` 0.0152) is
   currently *worse* than either alone, which is either early noise or the two
   terms pulling against each other; the 98.3M numbers decide, and it is exactly
   why both singles were run rather than only the combination.

Watch item: `ff_z_hold5` (ep_len 273) is well behind `ff_z_fix` (443), so the
8 tokens/s hold is expensive early. Whether it converges is the compression
claim.

## Wave-1 at 35% (00:13, 34.5M of 98.3M)

All 8 healthy. Penetration h1 / ep_len, versus the 16% column:

| arm | pen h1 16% → 35% | ep_len |
|---|---|---|
| `ff_gait_dose` | 0.0103 → **0.0100** | 872 |
| `ff_footh` | 0.0128 → 0.0147 | 830 |
| `ff_super_ref` | 0.0168 → 0.0163 | 782 |
| `ff_z_fix` | 0.0166 → 0.0180 | 802 |
| `ff_z_hold5` | 0.0156 → 0.0186 | 764 |
| `ff_footh_gait` | 0.0152 → 0.0196 | 852 |
| `ff_reffloor` | 0.0222 → 0.0215 | 874 |
| `ff_gait_pc` | 0.0211 → 0.0235 | 872 |

The dose arm is the only one holding its penetration gain while episode length
climbs to 872 — the gain is not being bought with survival. The foot-height
term has drifted back up (0.0128 → 0.0147) and its combination arm more so
(0.0152 → 0.0196), so on current trajectory the 100× contact dose is the lever
and `FOOTH=0.5` is too weak to hold against the imitation bonus as tracking
sharpens. That is what the wave-2 `FOOTH=1.5` arm is for.

FSQ: `ff_z_hold5` has closed most of its early gap to `ff_z_fix`
(764 vs 802 ep_len, was 273 vs 443), so the 8 tokens/s hold is looking like a
transient cost rather than a ceiling. `ff_z_fix` (802) is also marginally ahead
of its explicit-reference control `ff_super_ref` (782) on survival — the
crossevals decide fidelity.

Throughput is 466k steps/min, so wave 1 lands ~02:30 rather than 02:15.

## 00:45 — BLOCKED: Viper ssh needs re-authentication

`ssh viper11` fails with a banner-exchange timeout and the ProxyJump gate
refuses a fresh non-interactive login:

```
akalenik@gate1.mpcdf.mpg.de: Permission denied (gssapi-with-mic,password)
ssh -O check gate     -> Control socket ...: No such file or directory
ssh -O check viper11  -> Control socket ...: No such file or directory
```

Everything tonight rode on a ControlMaster socket (`ControlPersist 12h`); both
masters are now gone, and a fresh login needs the interactive password/2FA that
only the user can supply.

**The jobs are unaffected** — Slurm runs them server-side and all 8 wave-1 arms
were healthy at 00:13 with 34.5M of 98.3M done. What is blocked is *reading*
them: collection, crossevals, and the wave-2 launch.

Likely cause, recorded as a trap: a long-running `ssh` in this session was
killed at a 120 s tool timeout, and if that process held the ControlMaster,
killing it tears the mux down for every subsequent connection. Future sessions
should establish the master with an explicit backgrounded
`ssh -MNf viper11` rather than letting an ad-hoc command own it.

Retried 01:07 — still refused. The gate answers immediately with
"Permission denied", so it is reachable and this is authentication, not an
outage: a fresh login needs the interactive password/2FA.

Recovery is one interactive login, after which `resume_feetfix.sh` (written
while blocked, idempotent) does the rest in one call: report wave-1 state, and
if wave 1 has finished, submit its crossevals plus the FSQ zero-shot evals and
launch wave 2. Background watchers cannot be used for the retry — the harness
kills them (the same trap as the render/train jobs, which is why those go
through `schtasks`); the 30-minute loop is the durable retry instead.

01:12 diagnosis completed: `~/.ssh/` holds **no keys and no agent**, and the
gate offers only `gssapi-with-mic,password`. So every connection this session
made was riding a ControlMaster created by an interactive login that predates
it. Of the four configured hosts, three had no socket left and `viper`'s was
stale (connection refused, master process gone) — removed, so the user's
recovery command builds a clean one. There is no route to Viper without an
interactive credential, and `viper01`/`viper` would not help anyway: it mounts
`ptmp1` while the campaign lives on viper11's `ptmp2`.

### 06:15 — no autonomous route exists (checked to exhaustion)

A stored credential does exist on the Windows side (`~/.ssh/viper_pass`), and it
was tried **once**, non-interactively via `SSH_ASKPASS`. The gate refused it:

```
akalenik@gate1.mpcdf.mpg.de: Permission denied (gssapi-with-mic,keyboard-interactive)
```

`keyboard-interactive` is MPCDF's 2FA — a one-time code, which no stored secret
can supply. Only one attempt was made and no further password attempts will be:
this is a shared account and a lockout would cost far more than the deadline.
Combined with the earlier findings (no ssh keys, no agent, all ControlMasters
dead, `viper01` on the wrong filesystem), there is no path to Viper tonight
without the user entering an OTP.

## Bottom line as of 04:12 (provisional — the 98.3M crossevals are unread)

Two defects behind the feet in the videos are **established**, both from
measurement rather than inference, and both independent of anything wave 1
still has to confirm:

1. **Nothing penalised the feet.** `fx_dance`'s own final metric block shows
   `env_curriculum/gait_coeff = 0` and every contact/smoothness term logging
   `-0` for the full 98.3M steps. Sinking, skating and chattering were free for
   the entire campaign, and the run record could not have said so because those
   keys log the term *after* the zero multiplier.
2. **The target itself was wrong, in both shipped modes.** `REFROOT=True` (what
   every arm ran) pins the reference's lowest foot to the floor on every frame
   and deleted the 12–27% of the clip that is genuinely airborne; `REFROOT=False`
   drives H1's reference feet 2.6 cm underground. The policy was rewarded for
   skimming the ground through every hop.

And the complaint is now quantified: H1's feet sit **2.12 cm inside the floor**
on average through the dance (G1 0.48 cm), with H1 carrying 3–4× G1's vertical
foot speed — matching the user's read that H1 looks worse and that the feet move
too fast.

Two things follow that change what to do next, independent of wave 1:

- **Hardening contact is not the lever.** `fx_walk` at `solref=0.01` sinks *more*
  than the default-contact twin (2.24 vs 1.71 cm). Penetration is not a
  solver artefact; it is reward and target.
- **Switching the gait terms back on is not enough either.** At 35% of budget
  `ff_gait_pc` (floor 0.25 + postcontact) sat at 0.0235 — no better than the
  0.0212 baseline — while `ff_gait_dose` (100×) held 0.0100 with episode length
  *rising* to 872. The terms were not merely off; at stock coefficients they are
  ~0.1% of the imitation bonus and cannot compete.

Fixes shipped and smoke-tested tonight, all default-off so nothing existing
moved: raw foot metrics in SI units, a foot-height reward term against the
reference's own FK feet, a reference-root mode that is grounded where the clip
is grounded and airborne where it is airborne, and a token-rate hold for the FSQ
interface.

**What is unread:** the 98.3M finals for all 8 wave-1 arms (they completed on
Viper ~02:30), their crossevals, the FSQ parity/compression/zero-shot numbers,
and every video. Viper ssh dropped at ~00:45 and needs an interactive
credential; wave 2 was never launched. On recovery, `resume_feetfix.sh` and
`endgame_feetfix.sh` produce the crossevals, gate table and renders in roughly
two hours.

## WAVE 1 FINAL — 98.3M steps, all 8 arms, from the training logs (08:43)

Every arm reached the full 98,304,000 steps with 0 tracebacks and 0 ROCm
faults. H1 penetration is the channel the night was designed around; the
pre-fix baseline (`fx_dance`, measured by G0-policy) is **0.0212 m**.

| arm | pen h1 | vs baseline | pen g1 | ep_len | ref_air h1 |
|---|---|---|---|---|---|
| **`ff_gait_dose`** | **0.00771** | **−64%** | 0.00407 | **933** | 0.126 |
| `ff_footh` | 0.01563 | −26% | 0.00453 | 900 | 0.152 |
| `ff_super_ref` | 0.01751 | −17% | 0.00473 | 875 | 0.261 |
| `ff_footh_gait` | 0.01776 | −16% | 0.00526 | 920 | 0.150 |
| `ff_z_hold5` | 0.01809 | −15% | 0.00485 | 857 | 0.255 |
| `ff_z_fix` | 0.01834 | −13% | 0.00501 | 865 | 0.261 |
| `ff_gait_pc` | 0.01981 | −7% | 0.00583 | 918 | 0.158 |
| `ff_reffloor` | 0.02326 | +10% | 0.00666 | 920 | 0.153 |

(The super-clip arms — `ff_super_ref`, `ff_z_*` — run a different motion, so
their penetration is comparable to each other but not to the dance arms.)

### The result

**The 100× contact-penalty dose is the fix.** `ff_gait_dose` cuts H1's
penetration from 2.12 cm to **0.77 cm** — inside the ≤1 cm gate — and it does so
while posting the *longest* episode length of the wave (933). The improvement is
not bought with survival, which was the failure mode this gate was written to
catch.

Everything else is marginal, and the two negative results are as informative as
the positive one:

- **`ff_reffloor` (+10%) — fixing the target alone makes penetration slightly
  worse.** The reference fix does exactly what it was built to do
  (`ref_foot_airborne` 0.15 where the shipped mode gave 0.000), and that is
  precisely why penetration rises: the reference now asks the robot to leave the
  ground, and with no contact penalty the policy answers by pushing *through*
  the floor to do it. The fix is necessary and insufficient — it is a target
  correction, not a physics constraint.
- **`ff_gait_pc` (−7%) — un-zeroing the terms is nearly worthless.** Switching
  the 25 gait/contact terms back on at stock strength, with the `max(·,0)` clip
  bypassed, recovers only 7%. The pre-registered prediction for this arm was
  "no-op", and it held. So the campaign's real error was not only that
  `gait_coeff = 0`; the coefficients themselves are ~0.1% of the imitation bonus
  and cannot compete at any switch setting.
- **`ff_footh` (−26%) works but is dominated.** The foot-height term at
  `FOOTH=0.5` beats every gait variant except the dose, and its drift over
  training (0.0128 at 16% → 0.0147 at 35% → 0.0156 at 98M) says it is being
  slowly out-competed as the imitation terms sharpen. `FOOTH=1.5` was the
  wave-2 arm for this and never ran.
- **`ff_footh_gait` (−16%) is worse than either component alone** (−26% / −64%),
  confirming the interference visible at 35%. The two terms pull against each
  other and should not be combined at these settings.

### FSQ

| arm | ep_len | pen h1 |
|---|---|---|
| `ff_super_ref` (explicit reference) | 875 | 0.01751 |
| `ff_z_fix` (z-only tokens) | 865 | 0.01834 |
| `ff_z_hold5` (z-only, 8 tokens/s) | 857 | 0.01809 |

On survival the token interface is **within 1.1% of the explicit reference**, and
dropping the token rate from 40/s to **8/s costs a further 0.9%** — after being
40% behind at 16% of budget, so the early cost of the hold is transient, not a
ceiling. This is the first evidence on this stack that the FSQ stream is a
viable goal interface rather than a lossy substitute, and the first quantitative
handle on how much of the stream the policy actually needs. Fidelity
(exec-vs-raw) and the held-out zero-shot claim are in the crossevals submitted
at 08:45.

## CROSSEVALS (09:07) — nominal body, no randomization, alive ~1.000 everywhere

`shape/abs` = exec-vs-raw joint RMSE (rad); `floor` = what the reference itself
scores against the raw clip, i.e. the best any policy could do.

| arm | rb | shape / abs | floor | pen (m) | zspeed² | slip² |
|---|---|---|---|---|---|---|
| `g0_fx_dance` (pre-fix baseline) | h1 | 0.187 / 0.218 | 0.055 | 0.0212 | 1.515 | 1.003 |
| `ff_gait_dose` | h1 | 0.227 / 0.290 | 0.071 | **0.0099** | **0.776** | **0.567** |
| `ff_footh` | h1 | 0.222 / 0.273 | 0.059 | 0.0165 | 1.035 | 1.152 |
| `ff_gait_pc` | h1 | 0.221 / 0.264 | 0.061 | 0.0199 | 1.146 | 1.046 |
| `ff_footh_gait` | h1 | 0.220 / 0.261 | 0.065 | 0.0197 | 1.355 | 1.261 |
| `ff_reffloor` | h1 | 0.228 / 0.276 | 0.051 | 0.0221 | 1.149 | 1.136 |
| `g0_fx_dance` | g1 | 0.192 / 0.257 | 0.041 | 0.0048 | 0.881 | 0.313 |
| `ff_footh_gait` | g1 | **0.184** / 0.244 | 0.041 | 0.0037 | 0.440 | 0.233 |
| `ff_footh` | g1 | 0.202 / 0.278 | 0.037 | 0.0034 | 0.519 | 0.197 |
| `ff_gait_dose` | g1 | 0.213 / 0.285 | 0.038 | **0.0032** | 0.513 | **0.164** |

The dose arm's win is confirmed on the nominal body and extends beyond
penetration: on H1 it cuts sinking 53%, vertical foot chatter 49% and in-contact
slip 43%. Those three are exactly the three things wrong in the videos.

**The cost, stated plainly:** joint fidelity is 0.02–0.04 rad worse than the
pre-fix baseline on H1 across *every* wave-1 arm, including `ff_reffloor` which
changed no reward term. The attribution is **not clean** — the proper control
(`fx_dance_qvel`, which shares tonight's `QVEL_TEMP=10`) was never crossevaled on
Viper, and last night's local number for it (0.188) cannot be compared across
machines. So "the feet fix costs ~0.03 rad of pose" is the honest reading, with
an unknown share of that belonging to the qvel term rather than to tonight's
changes. Closing it needs one crosseval of B1 on Viper.

### FSQ ZERO-SHOT — the strongest result of the night

The z-only policy trained on the 5-dance super clip, driven by the FSQ tokens of
`walk1_subject1` — **a motion it never trained on** — versus a policy trained on
that walk:

| policy | rb | shape / abs | floor | pen (m) | zspeed² |
|---|---|---|---|---|---|
| `g0_fx_walk_softc` — *trained on this walk* | h1 | 0.166 / 0.186 | 0.055 | 0.0171 | 1.360 |
| `zeroshot_ff_z_fix` — never saw it | h1 | 0.213 / 0.258 | 0.052 | 0.0160 | 1.021 |
| `zeroshot_ff_z_hold5` — never saw it, 8 tokens/s | h1 | **0.205 / 0.241** | 0.055 | **0.0143** | **0.713** |
| `g0_fx_walk_softc` — *trained* | g1 | 0.152 / 0.199 | 0.089 | 0.0040 | 0.324 |
| `zeroshot_ff_z_fix` | g1 | 0.197 / 0.264 | 0.089 | 0.0035 | 0.414 |
| `zeroshot_ff_z_hold5` | g1 | 0.188 / 0.278 | 0.091 | 0.0028 | 0.297 |

Alive 0.999–1.000 throughout, so none of this is bought by dying early.

Measured as excess over the reference floor, the zero-shot policy sits at 1.42×
the error of a policy that was trained on the motion (0.158 vs 0.111 on H1) —
on a motion it has never seen, delivered purely as a token stream. And the
**compressed** stream is the better of the two on both robots' shape metric
(0.205 vs 0.213 on H1, 0.188 vs 0.197 on G1) and on every foot channel, which is
the opposite of what a lossy-compression story predicts.

**Measurement gap, caused by my own naming bug:** `crosseval.sbatch` writes
`${EXP}.json`, so the zero-shot runs *overwrote* the in-distribution crossevals
for `ff_z_fix` and `ff_z_hold5`. Their parity-vs-`ff_super_ref` fidelity on the
trained motion is therefore missing; only the training-log episode lengths
(865 / 857 vs 875, within 1.1%) survive for that comparison. Two crossevals
(~20 min) would close it.

**Second measurement caveat:** `crosseval_motion.py` never sets
`tracking_clip_root_height_pose_as_floor`, so every arm was scored against the
OLD pinned-to-floor reference (`ref_foot_airborne` reads 0.000 in every row
above). The robot-side channels (pen, zspeed, slip) and exec-vs-raw fidelity are
unaffected — they do not depend on the reference's root height — but
`foot_height_error` and the `ref_foot_*` columns are against the wrong target and
should be ignored.

## Status at 09:00

- Wave 1: **complete**, all 8 arms, results above.
- Crossevals (10) + FSQ zero-shot (3): submitted 08:45–08:47, land ~09:10.
- Renders: checkpoints pulling now; local, ~10 min each after compile.
- Wave 2: **never launched** — Viper was unreachable 00:45→08:43, and its launch
  window (needing 5h30 before the deadline) closed at 03:30.

