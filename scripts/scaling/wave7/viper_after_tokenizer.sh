#!/usr/bin/env bash
# Wait for the m20 tokenizer pipeline, then upload clips_m20 (+ sidecars) and
# tokenizer_m20 to Viper and submit wave 7a. Run from WSL (detached).
set -u
REPO=${REPO:-/mnt/c/Users/smirn/Desktop/robot_learning_ip}
LOG=$REPO/experiments/fsq_khaendler/_tok_logs/tokenizer_m20.log
ROOT=/ptmp/akalenik/urma
log() { echo "[w7a-up $(date '+%m-%d %H:%M:%S')] $*"; }
log "waiting for TOKENIZER PIPELINE DONE"
for i in $(seq 1 720); do grep -q "TOKENIZER PIPELINE DONE" "$LOG" 2>/dev/null && break; sleep 30; done
grep -q "TOKENIZER PIPELINE DONE" "$LOG" || { log "tokenizer did not finish in 6 h; abort"; exit 1; }
log "held-out report:"; grep -h "qpos_rmse_rad_heldout" "$REPO/experiments/fsq_khaendler/clips_m20_rec_tokenizer_m20/reconstruction_report.json" 2>/dev/null | head -4
ls "$REPO/experiments/fsq_khaendler/clips_m20/UnitreeG1/"
log "uploading clips_m20 + tokenizer_m20"
ssh viper11 "mkdir -p $ROOT/clips/clips_m20/UnitreeH1 $ROOT/clips/clips_m20/UnitreeG1 $ROOT/tokenizer_m20"
rsync -a --partial "$REPO/experiments/fsq_khaendler/clips_m20/" viper11:$ROOT/clips/clips_m20/ 2>&1 | tail -2
rsync -a "$REPO/experiments/fsq_khaendler/tokenizer_m20/" viper11:$ROOT/tokenizer_m20/ 2>&1 | tail -1
scp -q "$REPO/scripts/scaling/wave7/submit_w7a.sh" viper11:$ROOT/
ssh viper11 "cd $ROOT && sed -i 's/\r\$//' submit_w7a.sh && DRY=1 bash submit_w7a.sh > /tmp/w7a_dry.txt 2>&1; echo dry trains=\$(grep -c ^TRAIN /tmp/w7a_dry.txt) ces=\$(grep -c '^  CE' /tmp/w7a_dry.txt) abort=\$(grep -c ABORT /tmp/w7a_dry.txt); grep ABORT /tmp/w7a_dry.txt; grep -c ABORT /tmp/w7a_dry.txt | grep -q '^0\$' && bash submit_w7a.sh 2>&1 | tail -2; squeue -u akalenik -h | wc -l"
log "=== W7A SUBMIT STEP DONE ==="
