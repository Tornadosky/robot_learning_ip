# Wave 7 campaign — the final matrix before the Sep 5 deadline

Written 2026-09-02 19:40. Machines: **BOX-A** = this Windows/WSL box (RTX 4060 Ti 16 GB),
**BOX-B** = Ubuntu box (RTX 5080 16 GB), **VIPER** = MPCDF MI300A cluster
(2-topology limit: any run with >2 robot models faults on ROCm, fused and split
trainer alike — verified `split_3t` and `b3_split4` logs). Multi-topology runs
therefore live on BOX-A/B only; the two-robot quantitative matrix lives on Viper.

Story: SONIC-style co-training (tokenizer encoder + FSQ inside the policy,
PPO gradients reach it) vs a precomputed token vs reference-only, on **20 LAFAN1
motions** (7 held out as unseen), across **5 topologies** (H1, G1, T1, Atlas,
Apollo) with **randomized bodies**, at full reference rate. Staleness, budget
scaling, transfer and DanceDB zero-shot are the secondary axes.

## 0. Data (what exists / what is being produced)

| item | path | status |
|---|---|---|
| all 40 LAFAN1 motions, 5 robots, our clip format with foot sites | `external_data/amass_converted/LAFAN1_all/{UnitreeH1,UnitreeG1,BoosterT1,Atlas,Apollo}` (7.3 GB) | done |
| feasibility re-issue (stiffened limits + FK sites) | `external_data/amass_converted/LAFAN1_allfix/{BoosterT1,Atlas,Apollo}` | DONE 20:49: 34/34 for each of T1, Atlas, Apollo, 0 failures |
| 20-motion split | `experiments/fsq_khaendler/clips_m20/{train,heldout}_motions.txt` | done (train: dance1 s1-2, dance2 s1-5, walk1 s1-2, walk2 s1, walk3 s1-2, walk4 s1, run1 s2, run2 s1, sprint1 s2, jumps1 s1-2, fight1 s2, fightAndSports1 s1; held out: dance1 s3, walk1 s5, walk3 s5, run2 s4, sprint1 s4, jumps1 s5, fight1 s5) |
| tokenizer on 20 motions (H1+G1, held-out excluded from the codec) | `experiments/fsq_khaendler/tokenizer_m20` | DONE 19:20; joint-angle reconstruction 0.129 rad in-corpus, 0.11–0.14 rad on the 7 unseen motions (v2: 0.15/0.16); the training-log 'eval' is velocity-dominated, ignore it |
| H1/G1 super20 + held-out clips + `_zq`/`_win` sidecars | `experiments/fsq_khaendler/clips_m20` | DONE 19:20 (8 clips × `_zq` + `_win` per robot) |
| 5-robot set (adds T1/Atlas/Apollo super20 + held-out + sidecars) | `experiments/fsq_khaendler/clips_5r` (+ `ROBOTS`, `READY`) | DONE 20:56: all 5 robots, sidecars verified 8/8 per robot |
| handoff zip for BOX-B | `handoff_zips/w7_train_data.zip` (clips_m20, clips_5r, tokenizers, clips_3t_v2) | DONE 21:08, 14.7 GB (window sidecars are float32 on disk) — prefer rsync over LAN (§3) |
| DanceDB (77 dances, AMASS) → H1 → other robots | `external_data/amass_converted/DanceDB/<Robot>` | driver verified (`scripts/scaling/wave7/retarget_dancedb.py`, 50 s per dance on the GPU); the batch is job D1, launched by `after_restart.sh` (it competes with the tokenizer/re-issue for the GPU, so it runs after them) |

Robot screening (`scripts/scaling/wave7/screen_family.py`): Atlas residual 0.00016 (aliases for 4 ankle joints), Apollo 0.000 m positions (2 wrist-yaw joints absent in our model), GR1T2 parked (hip-frame mismatch), Talos/ToddlerBot out.

## 1. Jobs (IDs used below), machine, status

| id | machine | job | status / trigger |
|---|---|---|---|
| J1 | BOX-A | tokenizer_m20 + sidecars for clips_m20 | DONE 19:20 |
| J2 | BOX-A → VIPER | upload clips_m20 + tokenizer; submit wave 7a (`submit_w7a.sh`) | DONE 19:34 (276 jobs; 32 running, 0 failures at 21:00) |
| J3 | BOX-A | 3-robot co-training SMOKE (never run with 3 robots) | PASSED 19:40 (10 updates, recon loss 0.033→0.026) |
| J4 | BOX-A (Windows venv) | T1 → Atlas → Apollo feasibility re-issue, 34 motions each | DONE 20:49 |
| J5 | BOX-A | assemble clips_5r + sidecars + handoff zip | DONE 21:08 |
| J6 | BOX-A | 5-robot co-trained arm, 59M steps, bodies 0.2→0.7 (fallback 4→3 robots on OOM) | starts with `after_restart.sh` after the reboot (never run yet: watch `experiments/local_w7/w7_cot5.log` for the compile and the robot fallback) |
| J7 | BOX-B | 5-robot reference arm, 59M, same recipe | manual, after rsync to BOX-B (§3) |
| J8 | VIPER | wave 7a: 23 trains + 253 cross-evals (see §2) | after J2, ~overnight (each train 35 min–2.3 h, ~30 concurrent) |
| D1 | BOX-A (after reboot) or BOX-B | DanceDB batch retarget → H1 (77 clips, torch CUDA), then re-issue to G1/T1/Atlas/Apollo | manual: `retarget_dancedb.py` then `reissue_clips.py --src-dir external_data/amass_converted/DanceDB --targets UnitreeG1 BoosterT1 Atlas Apollo` |
| D2 | any | DanceDB zero-shot cross-evals on 8–10 dances for every 7a arm | after D1 (sidecars via `emit_sidecars_any.py`) |
| K1 | BOX-B | khaendler-encoder tokenizer (v3b) + one co-trained arm vs J8's | optional, tomorrow |

## 2. The matrix (priority order; all rows are 2 seeds unless noted)

**Tier 1 — the report's core (tonight)**
- VIPER 7a-B1: super20, hold 1, 19.66M: `m20ref`, `m20split`, `m20cot`. Cross-evals: super20 at hold 1 (×2) and hold 20 (×2) + each of the 7 held-out motions at hold 1. → RQ1 (token at full rate on complex motion), RQ3 (unseen motions).
- BOX-A J6 + BOX-B J7: 5 robots × 20 motions, co-trained vs reference, 59M steps, randomized bodies. → RQ6 (one policy, many bodies, many motions). Cross-eval per robot with legs/arms split, zero-action floor, held-out motions.

**Tier 2 — already in the same Viper submission**
- 7a-B2/B4: 2× (all three arms) and 4× (`m20ref4x`, `m20cot4x`, 1 seed). → RQ2 (does the gain survive compute).
- 7a-H20: the three arms trained at hold 20. → RQ4 (staleness on many motions).
- 7a-M7: the three arms under ramp-to-0.7 morphology (1 seed). → RQ5.

**Tier 3 — tomorrow (Sep 3)**
- third seeds for the winning Tier-1 cells (VIPER, ~1 h); 5-robot arms at hold 20 (BOX-B); local morphology eval (perturbed bodies 0.3/0.6) of J6/J7 checkpoints (BOX-A, CUDA only).
- D1 + D2: DanceDB zero-shot (real dances nobody trained on) for every arm.
- K1: khaendler-encoder tokenizer comparison (held-out RMSE, one arm).
- cross-robot token transfer across the five robots (sidecar remap by joint semantics; `build_cross_sidecars.py` pattern).
- renders (co-trained vs reference on dances, unseen motions, all five bodies) + dashboard rebuild (`build_ledger.py`).

**Tier 4 — last day if the winner is clear**
- extend the best 5-robot arm to 120M+ by relaunching with a larger `TOTAL` (checkpoints resume); more seeds at 4× on Viper.

**Research questions → figures**
1. Token vs reference at full rate on 20 motions (bars per arm, legs/arms split, seeds) — 7a-B1.
2. Gain vs budget (lines 1×/2×/4×) — 7a-B2/B4.
3. Unseen motions: 7 LAFAN1 + DanceDB (grouped bars per motion) — 7a zero-shot, D2.
4. Staleness curve (hold 1/20, trained at 1 and 20) — 7a-H20 + wave 6.
5. Randomized bodies (nominal / 0.44 / 0.7) — 6a + 7a-M7 + local M-EVAL.
6. Five topologies, one policy (per-robot RMSE vs floor, leg correlation, videos) — J6/J7.
7. Transfer heatmap (done for 3; extend to 5).

## 3. BOX-B (Ubuntu, RTX 5080): exactly what to do

**BOX-B is set up (2026-09-02 21:40): `melo@192.168.178.41` (hostname melody-pc), repo at
`/home/melo/Projects/ip_project/robot_learning_ip`, key login from BOX-A's WSL.** The password is
not stored anywhere in this repo; ask the user if the key stops working.

Transfer is ssh/rsync over the LAN (~105 MB/s), from BOX-A (WSL), resumable:
```
bash scripts/scaling/wave7/sync_boxb.sh      # clips_m20, clips_5r, tokenizers, clips_3t_v2, LAFAN1_all(+fix), handoff/
```
Setup that already ran on BOX-B: `git clone --depth 1` + `bash handoff/bootstrap_second_machine.sh <repo>`
(submodules from the bundles; venvs `~/jaxgpu` and `~/locomjx` installed with `pip --no-deps` from the
requirement freezes because the working set has known inconsistencies: equinox/lineax/mujoco-warp pins).
Every launcher takes `REPO=/home/melo/Projects/ip_project/robot_learning_ip PY=~/jaxgpu/bin/python`
(those are the defaults in `boxb_night7.sh`).

Overnight orchestrator on BOX-B (one JAX process at a time; done arms skipped; relaunch to resume):
```
ssh melo@192.168.178.41
cd /home/melo/Projects/ip_project/robot_learning_ip && git pull
nohup bash scripts/scaling/wave7/boxb_night7.sh > experiments/local_w7/boxb_night7.log 2>&1 &
#  1. b7smoke_cot3  3-robot co-training smoke (clips_3t_v2, 10 updates)
#  2. J7 w7_ref5    5-robot reference arm, 59M, bodies 0.2->0.7 over 40M   (pair of BOX-A's w7_cot5)
#  3. w7_cot5_h20   5-robot co-trained arm at reference hold 20
#  4. w7_ref5_h20   5-robot reference arm at hold 20
# each arm falls back 5 -> 4 (drop Apollo) -> 3 robots if it cannot start (OOM / compile fault)
tail -f experiments/local_w7/boxb_night7.log      # arms: experiments/local_w7/<name>.log, checkpoints under loco_mjx/experiments/runs/local_w7/
```
Manual single arm (what the orchestrator runs for J7):
```
export REPO=$PWD PY=~/jaxgpu/bin/python
R=$(cat experiments/fsq_khaendler/clips_5r/ROBOTS); N=$(echo $R | tr ':' '
' | wc -l)
NAME=w7_ref5 CLIPDIR=$REPO/experiments/fsq_khaendler/clips_5r CLIP=super20.npz ROBOTS=$R NR_ENVS=$((64*N)) MINIBATCH=$((1024*N))   TOTAL=58982400 SAVE_EVERY=1966080 MORPH_MODE=schedule MORPH_COEFF=0.7 MORPH_START=0.2 MORPH_RAMP=40000000 LATENT=0 PROJECT=local_w7   bash scripts/scaling/wave6/local_train.sh
# co-trained variant: LATENT=1 LATENT_DIM=44 LATENT_DIVISOR=1.0 SIDECAR=_win COTRAIN_ROWS=11 COTRAIN_CH=4
#   COTRAIN_INIT=$REPO/experiments/fsq_khaendler/tokenizer_m20/params.msgpack ; hold-20: add HOLD=20
```
Fixed 2026-09-02 21:45: `local_train.sh` never passed `morphology_coeff_start/ramp_steps`, so every
local `MORPH_MODE=schedule` arm would have died at start with "schedule requires ramp_steps > 0"
(the fallback would then have burnt all three robot sets). Viper's sbatch always passed them.
If the 5-robot graph does not fit in 16 GB, drop `apptronik_apollo` from `ROBOTS` (then `atlas`); `local_night7.sh` does that automatically on BOX-A.

## 4. BOX-A after the reboot (one command)

```
wsl -d Ubuntu -e bash /mnt/c/Users/smirn/Desktop/robot_learning_ip/scripts/scaling/wave7/after_restart.sh
wsl -d Ubuntu -e ssh gate      # then: ssh viper11 hostname   (restores the Viper connection)
```
Then D1: `.venv\Scripts\python.exe scripts/scaling/wave7/retarget_dancedb.py` (Windows venv, torch CUDA, resumable).

## 5. Known problems and workarounds

- 5 topologies on 16 GB: four fused topologies once peaked at 22 GB; sidecars are float16 now, envs 64/robot; automatic fallback 5→4→3.
- Multi-topology arms are 1 seed per machine; BOX-B's reference arm + BOX-A's co-trained arm give the pair, second seeds only if time allows.
- T1's arm retarget has hands 14–23 cm off on dance2_subject4 (legs fine); Atlas/Apollo re-issue verified on one dance (0.2 % out of range).
- Viper cannot run >2 robots (fused and split trainers both fault) — no workaround found; all multi-robot evidence is local.
- Reboot kills WSL processes and the Viper ssh masters; everything local is resumable via `after_restart.sh`.

## 6. Overnight agent brief (2026-09-02 22:00 → morning of 2026-09-03)

Goal for the night: every GPU busy all night, Viper queue never empty, no wrong data. Read §0–§5 first.

### 6.1 Data audit verdict (done 21:55, `scripts/scaling/wave7/verify_clips_5r.py`, report `clips_5r/verify_report.json`)
- Every clip of every robot: no NaN, sidecars present with matching frame counts and joint names
  (G1's clip carries 23 of the model's 29 joints; waist roll/pitch and wrists are untracked, as in all G1 results).
- Joint feasibility in the model's own sign convention (loader alias + sign tables applied): H1 1.8 % of frames
  over a range by ≤0.21 rad (arm yaw), G1 0.1 % / ≤0.02 rad, T1 1.7 % / ≤0.06 rad, Atlas 6.5 % / ≤0.09 rad,
  **Apollo 56 % / ≤0.21 rad (shoulder internal rotation, model range ±0.47 rad)**.
- Fix applied 21:52 (`clamp_clips_to_model.py`, H1 + Atlas + Apollo, then sidecars re-emitted with
  `emit_sidecars_any.py`): references are now inside the model ranges; G1/T1 left untouched (negligible).
  The right-side "violations" a naive audit shows (knee ~1.8 rad) are the loco_mjx right-leg axis negation that
  the loader's sign tables handle — not a data bug.
- Frame counts: H1/G1 clips have 2 more frames per motion than T1/Atlas/Apollo (re-issue trims the ends);
  each robot tracks its own clip, so this does not matter for training; cross-robot token comparisons are
  offset by ≤2 frames (≤0.05 s).
- Motions: 20 train (7 dances, 6 walks, 2 runs, 1 sprint, 2 jumps, 2 fights) + 7 held-out (one per family),
  lists in `clips_m20/{train,heldout}_motions.txt`; held-out never enters super20 or the tokenizer.
- Robots: H1/G1/T1 validated earlier (3-topology baseline 1.7–1.9x vs floor); Atlas/Apollo screened by
  `screen_family.py` (FK residual 1.6e-4 / exact positions). Nothing has been trained on Atlas/Apollo yet —
  the first 5-robot arms ARE the validation; inspect their per-robot tracking early (see 6.4).

### 6.2 Machine loops (what "busy" means)
| machine | orchestrator | next when it finishes |
|---|---|---|
| BOX-A (this box, WSL) | `after_restart.sh` → `local_night7.sh`: w7_cot5 (5-robot co-trained, 59M) → w7_ref5 skipped if BOX-B has it → DanceDB retarget batch | morphology cross-evals of w7_* checkpoints (CUDA only: BOX-A or BOX-B), renders |
| BOX-B (5080) | `boxb_night7.sh`: smoke ✓ → w7_ref5 → w7_cot5_h20 → w7_ref5_h20 | 3rd/4th arms: `w7_cot5 SEED=2`, khaendler-encoder tokenizer (K1), DanceDB re-issue on CPU |
| VIPER | wave 7a (23 trains / 253 CEs) submitted 19:34; 31 running / 90 pending at 22:00 | when pending < 20: submit Tier-2 (`submit_w7b.sh`, to write: seeds 3 of m20ref/split/cot 1x+2x, hold-20 trio seeds 2, m7 trio seeds 2, 4x split) |

Rules that cost a night when broken: ONE JAX process per GPU (a second compile restarts WSL on BOX-A);
torch retargets on the same GPU slow JAX 3x (DanceDB only after the arms, or on BOX-B's CPU); Viper:
never `scancel` other users' jobs, submissions as script FILES, verify arm configs from the RESOLVED log
line (`=== local_train ...` / sbatch echo), never from the template; a claim needs n≥2 seeds.

### 6.3 CPU fillers (32 cores on BOX-B, 16 on BOX-A)
- DanceDB → other robots: `reissue_clips.py --src-dir external_data/amass_converted/DanceDB --out-root
  external_data/amass_converted/DanceDB_fix --targets UnitreeG1 BoosterT1 Atlas Apollo` (mujoco, CPU; needs the
  H1 DanceDB clips from BOX-A's `retarget_dancedb.py` batch first — rsync them over).
- Offline CE scoring / ledger aggregation (`build_ledger.py` + the w7 aggregation still to add).
- Data audits on any new clip set (`verify_clips_5r.py --clip-dir ...`) BEFORE training on it.

### 6.4 Early-warning checks on the 5-robot arms (do at ~2M, ~6M, ~15M steps)
- `grep -o "nr_env_steps[^0-9]*[0-9]*" <arm>.log | tail -1` moves; steps/s ≥ 1.5k on BOX-B, ≥ 1k on BOX-A.
- per-robot `reward/joint_tracking` (log_info=True) rises for ALL five; a flat robot at 6M = its reference or
  sign table is wrong → drop it from `clips_5r/ROBOTS` and relaunch (the orchestrators read ROBOTS at start).
- KL that never calms by ~2M = takeoff-basin failure (memory rule): kill, relaunch with fewer robots.
- checkpoints appear every 1.97M under `loco_mjx/experiments/runs/local_w7/<name>/`.

### 6.5 Morning
- Viper: fetch `/ptmp/akalenik/urma/crosseval/m20*` → `experiments/fsq_khaendler/crosseval_viper/`; add the w7
  aggregation to `build_ledger.py`; republish the ledger.
- rsync BOX-B checkpoints back (`rsync -a melo@192.168.178.41:$D/loco_mjx/experiments/runs/local_w7/ loco_mjx/experiments/runs/local_w7/`),
  run per-robot cross-evals (super20 h1/h20 + held-out) with legs/arms split and the zero-action floor.
