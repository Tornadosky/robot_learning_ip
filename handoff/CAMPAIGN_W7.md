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
| feasibility re-issue (stiffened limits + FK sites) | `external_data/amass_converted/LAFAN1_allfix/{BoosterT1,Atlas,Apollo}` | T1 running (34 motions), Atlas+Apollo auto-start after it; resumable |
| 20-motion split | `experiments/fsq_khaendler/clips_m20/{train,heldout}_motions.txt` | done (train: dance1 s1-2, dance2 s1-5, walk1 s1-2, walk2 s1, walk3 s1-2, walk4 s1, run1 s2, run2 s1, sprint1 s2, jumps1 s1-2, fight1 s2, fightAndSports1 s1; held out: dance1 s3, walk1 s5, walk3 s5, run2 s4, sprint1 s4, jumps1 s5, fight1 s5) |
| tokenizer on 20 motions (H1+G1, held-out excluded from the codec) | `experiments/fsq_khaendler/tokenizer_m20` | done ~19:20; held-out reconstruction 0.035/0.056 rad at epoch 60 (v2 was 0.16/0.20) |
| H1/G1 super20 + held-out clips + `_zq`/`_win` sidecars | `experiments/fsq_khaendler/clips_m20` | produced by the tokenizer script |
| 5-robot set (adds T1/Atlas/Apollo super20 + held-out + sidecars) | `experiments/fsq_khaendler/clips_5r` (+ `ROBOTS`, `READY`) | auto after the re-issue (`build_clips_5r.sh`) |
| handoff zip for BOX-B | `handoff_zips/w7_train_data.zip` (clips_m20, clips_5r, tokenizers, clips_3t_v2; ~3 GB) | auto after clips_5r |
| DanceDB (77 dances, AMASS) → H1 → other robots | `external_data/amass_converted/DanceDB/<Robot>` | driver verified (`scripts/scaling/wave7/retarget_dancedb.py`, 50 s per dance on the GPU); the batch is job D1, launched by `after_restart.sh` (it competes with the tokenizer/re-issue for the GPU, so it runs after them) |

Robot screening (`scripts/scaling/wave7/screen_family.py`): Atlas residual 0.00016 (aliases for 4 ankle joints), Apollo 0.000 m positions (2 wrist-yaw joints absent in our model), GR1T2 parked (hip-frame mismatch), Talos/ToddlerBot out.

## 1. Jobs (IDs used below), machine, status

| id | machine | job | status / trigger |
|---|---|---|---|
| J1 | BOX-A | tokenizer_m20 + sidecars for clips_m20 | running → done ~19:20 |
| J2 | BOX-A → VIPER | upload clips_m20 + tokenizer; submit wave 7a (`submit_w7a.sh`) | auto after J1 (`viper_after_tokenizer.sh`) |
| J3 | BOX-A | 3-robot co-training SMOKE (never run with 3 robots) | auto after J1 (`local_night7.sh`) |
| J4 | BOX-A (Windows venv) | T1 → Atlas → Apollo feasibility re-issue, 34 motions each | running; Atlas+Apollo auto-chained; **resume after reboot with `after_restart.sh`** |
| J5 | BOX-A | assemble clips_5r + sidecars + handoff zip | auto after J4 (+J1) |
| J6 | BOX-A | 5-robot co-trained arm, 59M steps, bodies 0.2→0.7 (fallback 4→3 robots on OOM) | auto after J5 (`local_night7.sh`), overnight |
| J7 | BOX-B | 5-robot reference arm, 59M, same recipe | manual, once BOX-B has the zip (command in §3) |
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

```
git clone --depth 1 https://github.com/Tornadosky/robot_learning_ip.git && cd robot_learning_ip
bash handoff/bootstrap_second_machine.sh $PWD          # venvs from bundles + requirements
# data: download handoff_zips/w7_train_data.zip from the drive, unzip into experiments/fsq_khaendler/
#       (it contains clips_m20, clips_5r, tokenizer_m20, tokenizer_3t_v2, clips_3t_v2)
#       also copy external_data/amass_converted/LAFAN1_all (for evals against raw clips) or set --raw_clip_dir to the clip dir
export REPO=$PWD PY=~/jaxgpu/bin/python
# J7: 5-robot reference arm (robots from clips_5r/ROBOTS; 64 envs per robot; minibatch 1024 per robot)
R=$(cat experiments/fsq_khaendler/clips_5r/ROBOTS); N=$(echo $R | tr ':' '\n' | wc -l)
NAME=w7_ref5 CLIPDIR=$REPO/experiments/fsq_khaendler/clips_5r CLIP=super20.npz ROBOTS=$R NR_ENVS=$((64*N)) MINIBATCH=$((1024*N)) \
  TOTAL=58982400 SAVE_EVERY=1966080 MORPH_MODE=schedule MORPH_COEFF=0.7 MORPH_START=0.2 MORPH_RAMP=40000000 LATENT=0 PROJECT=local_w7 \
  bash scripts/scaling/wave6/local_train.sh
# then (tomorrow) the hold-20 5-robot pair: add HOLD=20; co-trained variant: LATENT=1 LATENT_DIM=44 LATENT_DIVISOR=1.0 SIDECAR=_win
#   COTRAIN_ROWS=11 COTRAIN_CH=4 COTRAIN_INIT=$REPO/experiments/fsq_khaendler/tokenizer_m20/params.msgpack
# D1 needs external_data/smpl (591 MB) + external_data/amass/DanceDB (680 MB) + the Windows-style .venv with torch CUDA; easier on BOX-A after the reboot.
```
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
