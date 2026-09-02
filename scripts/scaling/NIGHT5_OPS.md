# NIGHT5 ops runbook (2026-08-31 → 09-01, run until 10:00 Berlin)

## RESTART NOTICE (22:30): the user restarts the PC tonight.
The restart kills: the CPU tokenizer task, the WSL orchestrator, and the
Claude session (with its ops cron). After boot the user runs
`bash scripts/scaling/night5_restart.sh` (GPU tokenizer + orchestrator + CPU
multibody tokenizer) and resumes Claude — FIRST ACTION on resume: re-arm the
ops cron with the same prompt, verify all three local processes via their
logs, and pick this runbook back up. Viper is unaffected by the restart.
Updates since first version: T1 CE fix CONFIRMED (nr_envs=64 faults ROCm,
32 works — 16 n4t1 CEs resubmitted at 32); first swing signal EXCELLENT
(n5sw15_ref H1 train-time foot_airborne 0.162 vs ref 0.134 vs historical
0.02-0.03); M-EVAL (32 CEs, tags m3_*/m6_*) queued; tokenizer_mb roster is
5 families (ToddlerBot dropped: 4-bar knees).

Each tick (cron, ~45 min): keep Viper AND the local GPU productive. Never
read a result at n=1; config claims from resolved logs only.

## State at arm time (21:15 Berlin)
- Viper wave 1 IN QUEUE (67 jobs, 11296xxx): `n5sw{015,05,15}_{ref,tok}` ×2
  seeds — swing-match dose grid at hold=1 on the bundle recipe (+4 dumps);
  `ce_t1fix_{a,b,c}` — H1+T1 CE ROCm bisect via crosseval_token3.sbatch
  (CE_XLA_OFF / CE_NRENVS; artifact check makes silent failures loud).
- Local: `b13rkx2dn` Git-Bash background task trains `tokenizer_3t_v2`
  (250 epochs, --foot-channels, walk1 held out) then emits `clips_3t_v2`
  (dance4 + walk1, 3 robots). Log:
  `experiments/fsq_khaendler/_tok_logs/tok3t_v2.log`. WATCH IT: if the log
  is still 0 bytes >20 min after launch or the task died, relaunch under
  Git Bash (NOT WSL — WSL interop wedged it once already tonight).
- Local orchestrator `local_night5.sh` (WSL, nohup) waits for
  `clips_3t_v2/BoosterT1/dance2_subject4_zq.npz`, then trains `ln5_ref` →
  `ln5_tok` (3-robot, swing 0.5, v2 tokens) → 4-seed eval + dumps.
  Heartbeats in `experiments/local_3t/ln5_*.heartbeat`; if no experiment.py
  process, no recent heartbeat, and no `.done`, relaunch it.

## Per tick
1. Viper: `squeue -u akalenik`; failures via sacct since last tick; read
   `ce_t1fix_*` logs when done — record which variant survives (that's the
   T1-eval unblock). Check first SW train logs: `reward/swing_match/h1`
   nonzero and `env_info/ref_airborne_frac/h1` ≈ 0.44.
2. When `clips_3t_v2` exists locally: upload H1/G1 (dance4 + walk1, npz+zq)
   to `viper11:/ptmp/akalenik/urma/clips/tokentest_v2/<robot>/`
   (tr -d '\r' not needed for npz; scp binary). Record v2 held-out walk1
   RMSE from `clips_3t_v2_rec/reconstruction_report.json` — if it is NOT
   clearly better than v1's 0.345 rad, note it and SKIP wave 2's zero-shot
   claims (still submit the arms; they double as A5-token arms).
3. After upload: `bash /ptmp/akalenik/urma/submit_night5b.sh` (guard file
   makes it idempotent).
4. When Viper queue < 40 and wave 2 is in: submit wave 3 fill, ~16 trains:
   s3 seeds for all six SW cells (reuse submit_night5.sh cell definitions
   with seed 3, no chain), plus `n5a4gsw_{ref,tok}` ×2 seeds = super-clip
   (clips_super) + swing 0.5 at hold=1 with dumps (the dance-video upgrade).
   Write submit_night5c.sh modeled on submit_night5.sh.
4b. When the v2 tokenizer task (`b8j2gatdw` or its relaunch) COMPLETES:
   launch `train_tokenizer_multibody.sh` the same Git-Bash background way
   (CPU-only, safe beside GPU training). Record its held-out walk1 RMSE vs
   v2's when it finishes — that is the "does robot diversity help the codec"
   verdict. If it beats v2 clearly, note tokenizer_mb as the B3 candidate.
4c. M-EVAL results (ce_*_mc jobs, tags m3_*/m6_*): aggregate ref-vs-token
   delta at morphology 0.3/0.6 vs 0.0 — "FSQ shines on morphologies" test.
   If the token's h20 advantage HOLDS at mc=0.6, that is a headline.
4d. Wave-3 fill also gets, if capacity remains after the s3 seeds + a4gsw:
   swing-winner dose x hold {5,20} x {ref,tok} x 2 seeds (swing x staleness
   interaction), and n5v2_tok x hold 20 x 2 seeds.
5. After ~08:30 Berlin submit nothing new heavier than CEs; at ~10:00: final
   aggregate (extend /tmp/night3_aggregate.py KEEP to ^n5), update memory
   (night3-campaign-state.md), delete this cron job, post the summary.

## Monitoring, analysis, and debugging methodology (the agent's contract)

**Monitoring (every tick).**
- Viper: queue depth + running count; `sacct --state=FAILED,TIMEOUT,NODE_FAIL`
  since last tick. A CE that "COMPLETED" in under ~5 min is SUSPECT even at
  exit 0 — check its json exists (crosseval_token3 now exits 1 on missing
  artifact; crosseval_token2 does NOT, so old-style CEs still fail silently).
- Local: every managed process has a log; a log that stopped growing for
  >15 min without a completion line = dead. Trains have heartbeat files and
  are resume-safe: relaunch the ORCHESTRATOR, never a bare experiment.py.
- Never scancel other users' jobs; the account also runs foreign jobs
  (v26_*, aipoly_*) — ignore them.

**Analysis (when results land).** Aggregate with /tmp/night3_aggregate.py
(KEEP regex must include the new prefix — extend to ^n5). Rules that bite:
n>=2 always (rollout seed alone moves a crosseval ~5%); aggregate ≠ member
(fetch per-robot numbers); config claims only from RESOLVED logs (the sbatch
echoes every env var; template defaults have lied before); survival confound
(any visited-state metric is episode-length-confounded — check alive_fraction
first, only then read RMSE/airborne).

**Debugging weird results — bug or surprise? Work the checklist in order:**
1. REPRODUCE the number from the raw json (not the aggregate) — is it one
   seed or the cell?
2. RESOLVED CONFIG of the exact arm (grep the train log's "RESOLVED CONFIG
   FLAGS" block) — did the knob actually arrive? (CHECK 2 of the guard:
   train-vs-eval config diff; a CE that does not mirror obs layout returns
   garbage SILENTLY.)
3. MECHANISM METRIC — every reward term logs env_info; if a term is claimed
   active, its logged magnitude must move when its coeff moves (this is how
   foot_air_time was proven inert and swing_match proven live tonight).
4. FLOORS AND CONTROLS — compare against zero-action floor and the known
   baseline cell before believing an improvement; a "gain" without a control
   is a renaming of noise.
5. KNOWN TRAPS in this stack: takeoff-basin (episode length pinned at spawn +
   approx_kl never calming by ~2M = a draw, kill and reseed, don't blame the
   config); ROCm graph faults are shape-specific (fixed by env count, not
   flags); CRLF kills Viper scripts; pgrep -f self-matches from inline ssh;
   WSL-interop wedges Windows-python launches (use Git Bash).
6. VERDICT LANGUAGE — write "bug" only with the mechanism identified;
   otherwise "surprising, mechanism open" + the check that would settle it.
   Log every verdict in the tick summary so the morning report can cite it.

## Analysis crib for the morning
- SW grid verdict: airborne (foot_airborne in CE jsons) vs RMSE per dose;
  success = airborne moves 0.03 → ≥0.08 at ≤5% RMSE cost on any dose.
- ln5 vs ln3: swing term on 3 robots; ln5_tok vs ln5_ref: v2 token value.
- n5v2 walk1 CEs (tag zs*): zero-shot — compare n5v2_tok walk1@h20 vs its
  own dance4@h20 and vs n5sw05_ref-style ref collapse.
