# Overnight 22→23-08 — the FIXED baseline (absolute anchor): MORNING REPORT

Goal doc: `C:\Users\smirn\Desktop\robot_learning_ip\docs\notes\OVERNIGHT_BASELINE_FIX_GOAL.md`
All paths below are under `C:\Users\smirn\Desktop\robot_learning_ip\experiments\fsq_khaendler\`.
Times Berlin. Wave-1/2 Viper arms: 98.3M steps each; local arms 30.4M.

## Verdicts against the pre-registered gates

| gate | verdict | evidence |
|---|---|---|
| **Anchor proof** (A4 beats A5 by ≥0.05 rad) | **PASS ×3.6** — A4−A5 = 0.18 rad absolute, both robots | crosseval_fix/fx_dance_final.json vs fx_dance_ctrl_final.json |
| **PASS (walk)** exec-vs-raw ≤0.10, ep_len ≥850, videos | **G1 PASS** (0.090 abs 0.099; ep_len 923–934); **H1 FAIL on metric** (0.142 best, softc) though ep_len 923 and video walks | fx_walk_softc_final.json; media/fx_walk_* |
| **PASS (dance)** ≤0.15, true dance visible | **G1 PASS** (0.134/0.132; visibly the true dance); **H1 near-miss** (0.167/0.173 over a 0.049 floor); videos upright + on-markers both | fx_dance_final.json; media/fx_dance_*, a4final_*_frames.png |
| **Random bodies** (robot-on-green-on-red on randomized body) | **PASS visually** (dance + walk at morph 0.5; bodies visibly re-sample, tracking holds) | media/fx_dance_morph05_*, fx_walk_morph05_*, a4morphFIX_*_frames.png |
| **Stretch: A6 multimotion** passes dance gate | **G1 borderline PASS** (shape 0.142 ≤0.15, abs 0.152 grazes); H1 0.164/0.167 | fx_super_final.json |
| **Stretch: morph-0.7 wave 2** | ran to completion, ep_len walk 860 / dance 791 / super-z 811 at 98M; crossevals + corrected renders PENDING (endgame4, after 08:30) | Viper logs w2_* |
| **B1** (qvel: error ≤0.5× A4, pose not worse by >0.02) | **MIXED** — qvel error 0.54× (narrow miss), pose −0.02 worse (at the boundary); jitter video check pending | fx_dance_qvel_final.json; logs |
| **B2** (deviation 0.5 bootstraps like B100) | **FALSIFIED for the true dance** — B2 2.9, B2r (REFBIAS=1) 10.4 at 98M; mechanism measured (below) | logs + baseline measurements |
| **Deviation gate, corrected (ratio 1.0)** | **WORKS: fx_dance_dev1 FINAL tracking-GATED ep_len 876.8 at 98.3M** — the true dance held for ~17.5 s under termination-enforced fidelity; clears the B2 calibration (400) ×2.2 and even walk's 850 bar (crosseval pending) | Viper log fx_dance_dev1_10989018, sacct COMPLETED |
| **B3** (walk ep_len ≥850 under deviation 0.5) | **PARTIAL** — 599 at the 30M local budget, monotone 46→599; crosseval 0.107–0.152 | fx_walk_conv_final.json |

**Bottom line:** the anchor fix is proven (the single biggest claim of the night),
G1 fully clears every primary gate, H1 clears the videos and survival but sits
0.03–0.05 above the numeric gates with ~0.05 of reference floor beneath it, and
one policy per arm drives BOTH robots plus randomized bodies through the TRUE
motions. The deviation-termination design from the goal doc needed ratio 1.0
instead of 0.5 on the true dance — measured, explained, and fixed within the night.

## Full exec-vs-raw table (98.3M finals; shape / absolute rad; alive all 1.00)

| arm | H1 | G1 | note |
|---|---|---|---|
| A1 fx_walk (hard contact) | 0.152 / 0.149 | 0.090 / 0.099 | |
| A2 fx_walk_softc | **0.142 / 0.142** | 0.095 / 0.104 | soft ≥ hard ⇒ **contact hardening unnecessary**; A2 is the walk pick |
| A3 fx_walk_morph0 | 0.132 / 0.127 | 0.093 / 0.108 | morphology costs H1 ~0.02 |
| A4 fx_dance | 0.167 / 0.173 | **0.134 / 0.132** | headline |
| A5 fx_dance_ctrl (centered) | 0.175 / **0.354** | 0.135 / **0.318** | ref floor 0.320/0.288 = the old distortion |
| B1 fx_dance_qvel | 0.188 / 0.196 | 0.154 / 0.165 | qvel err 0.54× of A4 |
| A6 fx_super (5 dances) | 0.164 / 0.167 | 0.142 / 0.152 | |
| A8 fx_dance_local (30M) | 0.181 / 0.187 | 0.149 / 0.156 | budget still pays |
| B3 fx_walk_conv (dev 0.5) | 0.152 / 0.153 | 0.107 / 0.124 | gated ep_len 599@30M |
| fx_dance_dev1 (dev 1.0, full set) | 0.177 / 0.185 | 0.142 / 0.142 | **gated ep_len 876.8@98M** — ~0.01 pose cost for termination-enforced fidelity + heading + qvel |
| w2_walk07 (morph→0.7) | 0.151 / 0.151 | **0.095 / 0.106** | **G1 passes the walk gate at aggressive morph** |
| w2_dance07 (morph→0.7) | 0.183 / 0.193 | 0.161 / 0.174 | morph 0.7 costs ~0.03 on dance; G1 just over gate |
| w2_superz (z-tokens replace reference) | 0.178 / 0.181 | 0.160 / 0.186 | FSQ thread alive: within 0.01–0.03 of explicit-reference A6 |
| fx_dance_dev1_m07 (dev 1.0 + morph→0.7) | 0.189 / 0.201 | 0.152 / 0.155 | **gated ep_len 803@98M under morph 0.7** (−8% vs morph 0.5); video fx_dance_dev1_m07_* (correct random bodies) |
| A7 fx_cycle (30M) | ep_len ~900 | | walk-cycle sanity |

A5 note: on the mean-centered SHAPE metric A4 and A5 tie — the centered arm
tracks its distorted reference as faithfully as A4 tracks the true one. The
anchor's entire win is at the POSE level, exactly what the videos show and
what "performing the actual motion" means.

## Videos (dual overlay: green = reward targets, red = raw sign-corrected clip)

- `media/fx_dance_unitree_h1.mp4`, `media/fx_dance_unitree_g1.mp4` — A4 final, nominal
- `media/fx_dance_morph05_unitree_h1.mp4`, `..._g1.mp4` — A4 on RANDOMIZED bodies (corrected render; bodies visibly re-sample)
- `media/fx_walk_morph05_unitree_h1.mp4` (+ G1 finishing) — walk pick (A2) on randomized bodies
- `media/fx_dance_ctrl_unitree_{h1,g1}.mp4` — the A5 control for side-by-side
- `media/fx_walk_unitree_{h1,g1}.mp4`, `fx_walk_softc_unitree_{h1,g1}.mp4` — walk arms
- `media/fx_dance_local_unitree_{h1,g1}.mp4` — A8 preview; `media/fx_cycle_unitree_h1.mp4` — A7
- Frame strips for quick reading: `media/a4final_*_frames.png`, `media/a4morphFIX_*_frames.png`, `media/a8_*_frames.png`
- Pending in the render queue (~08:30–09:30): B1 qvel jitter video, fx_super, B3, B2r, dev1, corrected w2 morph-0.7 renders

## The deviation-gate investigation (the night's main negative-turned-positive)

1. B2 (recipe REFBIAS=0.0 + TRACK_DEVIATION=0.5) froze at gated ep_len 2.9.
2. Config-diff vs B100 → REFBIAS suspected; B2r (REFBIAS=1.0) relaunched: 3×
   better (10.4) but still stalled ⇒ REFBIAS partial factor (it is REQUIRED
   under the gate — the goal doc's "drop REFBIAS" only holds for non-gated arms).
3. w2_dance_nohead (heading OFF): identical 10.1 ⇒ heading exonerated.
4. Mechanism measured: termination fires on the INSTANTANEOUS per-step error
   (below_height.py); absolute-anchor baselines are LOOSER than B100's
   (H1 thr 0.141 vs 0.114), but the TRUE dance produces error spikes ~1.7× the
   mean (0.083) on hard segments ⇒ every spike ends the episode at ~10 steps.
   B100 "bootstrapped fine" only because its reference was the shrunk dance.
5. Fix: TRACK_DEVIATION=1.0 (still terminates ignore-level wandering) ⇒
   **fx_dance_dev1 gated ep_len 825 at 43M** — the "actually performing,
   enforced by termination" mechanism now works on the true reference.
   Walk needed no change (B3: 46→599 under 0.5).

## Traps found tonight (each cost time; do not rediscover)

- `tracking_deviation_ratio` was passed TWICE in run.sh and viper_train.sbatch
  (`TRACK_DEVIATION` then a later `DEVRATIO` default) — last-flag-wins silently
  zeroed every deviation export before tonight. Fixed; single line with both fallbacks.
- The env's per-step reference clip uses SOFT limits (0.9): the absolute-anchor
  reference floor is 0.030 (H1) not the 0.009 hard-clip figure; both recorded in
  `g0_anchor*.json`.
- render_policy_video.py forces `domain_randomization.sampling_type="none"`
  before --set overrides — "randomized-body" renders are silently nominal unless
  sampling_type is overridden too. (First morph05 videos were frame-identical
  duplicates; caught by hashing + frame comparison, re-rendered correctly.)
- Harness background tasks get killed: renders AND training must go through
  schtasks. Editing a bash script mid-invocation corrupts the running parse
  (endgame pass-1's one spurious RENDER FAIL).
- The squeue-grep "queue empty" check breaks on a transient ssh failure —
  use sacct job states (endgame pass-1 pulled 93M snapshots early because of this).
- reference_velocity_scale ~5.76/4.65 is DANCE-clip calibration; walk sits near
  1.0 legitimately. The row itself only logs in trees carrying the 08-22 tracking.py.

## Step 0 record (implementation)

`tracking_clip_anchor: centered|absolute` in default_config + load_clip
(+ construction-time guard vs FITVARIANT=True, in both TrackingClipCommands and
RSI); ANCHOR wired through viper_train.sbatch / run.sh (mmtrain+mmsplit) /
run_arm.sh; renderer's red skeleton = absolute anchor for every arm; crosseval
gained --anchor/--fitvariant/--cyclic/--refroot + an ABSOLUTE metric.
G0 (kinematic, through load_clip): H1 0.0091 rad hard-clip (=the goal doc's
0.009), G1 exact; env-level soft floor 0.0297/0.0058. G1 smoke: 1M steps,
no NaN, all flags verified in the process cmdline.

## Wave 3 (morning 23-08, user-approved — close the remaining gate misses)

Jobs 10990890–93: **fx_dance_300** + **fx_walk_300** (resume A4 / A2-softc to
294.9M total — the H1 numeric-gate budget test; A8→A4 improvement and the
B100 150M precedent both predict H1 converges under the gates),
**fx_walk_conv98** (B3's config at full 98.3M — the ≥850 gated walk claim),
**fx_dance_dev1_m07** (dev1 + morph 0.7 — enforced fidelity on aggressive
bodies). ~3 h (fresh) / ~6 h (resumes).

**10:2x — gated-walk bootstrap is SEED-BISTABLE.** fx_walk_conv98 (Viper, no
seed flag → default seed) started at gated ep_len 2.42 — byte-identical to
local B3's first window (2.42, error 0.071) — but NEVER escaped through 45M,
while B3 (seed 1, 1024 envs) escaped within ~2M and reached 599. Under
deviation gating the early phase is bistable and the seed resolves it.
Cancelled; resubmitted as fx_walk_conv98b (job 10991446) with
`--environment.seed=1` via EXTRA_ARGS. Escape is checkable within ~15 min of
its start. Rule for future gated arms: pin the seed, and treat a
first-window ~2.4 that persists past ~5M as a dead run.
UPDATE 10:4x: seed 1 did NOT rescue it (2.49 at 6.7M) — seed hypothesis
falsified; env-config defaults verified identical to run.sh's explicit flags,
leaving batch size (65536 local vs 49152 Viper) and chance. Final attempt
fx_walk_conv98c (job 10991587) applies the night's own rule instead:
REFBIAS=1.0 under the gate. If that stalls too, B3's 599@30M stands as the
gated-walk datapoint and no further slots go to this question.
VERDICT 11:4x: conv98c reached only 5.76 (flat 7M→26M; the REFBIAS 2.3×
factor appeared, the escape did not) — stop rule invoked, arm cancelled.
**B3's 599@30M (local, 1024 envs) stands as the gated-walk result**; the
Viper 768-env batch appears unable to escape the gated-walk bootstrap in
3/3 attempts. Batch size is the leading suspect for the bistability.

**fx_dance_dev1_m07 COMPLETED: gated ep_len 803 at 98.3M under morph→0.7**
(vs 877 at morph 0.5 — aggressive bodies cost only ~8%). The
enforced-fidelity + aggressive-random-bodies claim is closed. Crosseval +
corrected morph-0.7 render queued (endgame5).

Wave-3 mid-run (10:12): fx_dance_300 back at ep_len 908 (143M/295M);
fx_dance_dev1_m07 gated ep_len 718 at 45M — enforced fidelity is surviving
the morph-0.7 ramp. Corrected morph-0.7 videos rendered
(w2_dance07_morph07_*, w2_walk07_morph07_*).

## Wave-3 FINAL (14:35) — the budget question answered

fx_dance_300 (294.9M): H1 0.155/0.159, G1 **0.125/0.125**.
fx_walk_300 (294.9M): H1 0.134/0.126, G1 **0.085/0.091** (best walk number of
the campaign). Tripling the budget improved H1 by only 0.012 (dance) / 0.008
(walk): **G1 passes every gate with margin; H1's remaining 0.03–0.05 gap is
structural, not budget-limited.** Known contributors: ~0.03 of soft-limit
flattening concentrated in the two shoulder-yaw joints + reward weighting.
Closing it is a design change (per-joint weighting, or exempting the
reference targets from the 0.9 soft-limit clip), out of scope for this goal
doc — recorded as the top next lever. H1 dance at 0.155 vs the 0.15 gate is
a documented near-miss, not a pass. Renders (nominal + randomized-body) for
both 300M arms complete automatically (endgame6 tail).

## CAMPAIGN CLOSED (23-08 15:45)

Every run, crosseval, and render is complete. Browse everything at
`experiments/fsq_khaendler/dashboard_overnight_fix.html` (charts + all 45
videos, incl. B100/F1 comparison arms; open in place — videos stream by
relative path). All one-shot Windows scheduled tasks (robot_fix_*) deleted.

## Still running / pending at write time (historical)

- UPDATE 09:4x: endgame3 COMPLETE — every wave-1 + local arm has final
  crossevals and dual-overlay videos in `media/` (incl. B1 qvel jitter check,
  fx_super, B3, B2r). Remaining: endgame4 (dev1 + w2 crossevals) auto-resumes
  on Viper network return; corrected morph-0.7 renders auto-run after it
  (schtasks robot_fix_morph07fixup).

- endgame3 render queue (B1/super/B3/B2r videos), endgame4 (dev1 + w2
  crossevals, corrected morph-0.7 renders) — results land 08:30–09:30;
  json/mp4 paths follow the same naming.
- B2r full-budget record: gated ep_len 10.4 at 98.3M (the ratio-0.5 datapoint).
- ~08:05: Viper ssh unreachable (banner-exchange timeout, network side). dev1's
  last read: gated ep_len 825 at 43M, job running server-side to 98M; endgame4's
  sacct wait auto-resumes when connectivity returns. Its final number and the
  w2 crossevals land in `crosseval_fix/*_final.json` whenever that happens.
