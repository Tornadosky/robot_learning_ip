#!/usr/bin/env bash
# Wave 7 local orchestrator (one JAX process at a time):
#   1. wait for the m20 tokenizer pipeline (GPU) to finish
#   2. 3-robot co-training SMOKE on clips_3t_v2 (H1+G1+T1, 96 envs, 10 updates)
#      -- co-training has never run with 3 robots
#   3. wait for experiments/fsq_khaendler/clips_5r/READY (5-robot super20 +
#      sidecars, produced after the Atlas/Apollo re-issue)
#   4. 5-robot co-trained arm, then 5-robot reference arm, 60M steps each,
#      morphology schedule on; cross-evals after each
set -u
REPO=${REPO:-/mnt/c/Users/smirn/Desktop/robot_learning_ip}
W6=$REPO/scripts/scaling/wave6
OUT=$REPO/experiments/local_w7; mkdir -p "$OUT"
TOKLOG=$REPO/experiments/fsq_khaendler/_tok_logs/tokenizer_m20.log
C3=$REPO/experiments/fsq_khaendler/clips_3t_v2
C5=$REPO/experiments/fsq_khaendler/clips_5r
TOK=$REPO/experiments/fsq_khaendler/tokenizer_m20/params.msgpack
TOK2=$REPO/experiments/fsq_khaendler/tokenizer_3t_v2/params.msgpack
log() { echo "[night7 $(date '+%m-%d %H:%M:%S')] $*"; }
R5=unitree_h1:unitree_g1:booster_t1:atlas:apptronik_apollo  # overridden by clips_5r/ROBOTS below
COTV="LATENT=1 LATENT_DIM=44 LATENT_DIVISOR=1.0 SIDECAR=_win COTRAIN_ROWS=11 COTRAIN_CH=4"

log "waiting for the tokenizer pipeline"
for i in $(seq 1 720); do grep -q "TOKENIZER PIPELINE DONE" "$TOKLOG" 2>/dev/null && break; sleep 30; done

log "=== 3-robot co-training smoke (clips_3t_v2, tokenizer_3t_v2 init) ==="
env NAME=w7smoke_cot3 CLIPDIR="$C3" ROBOTS=unitree_h1:unitree_g1:booster_t1 NR_ENVS=96 MINIBATCH=2048 TOTAL=61440 SAVE_EVERY=61440 \
    $COTV COTRAIN_INIT="$TOK2" PROJECT=local_w7 bash "$W6/local_train.sh" > "$OUT/smoke_cot3.log" 2>&1
log "smoke rc=$? steps=$(grep -c nr_env_steps "$OUT/smoke_cot3.log")"
grep -m1 -E "Traceback|Error" "$OUT/smoke_cot3.log" | cut -c1-200

log "waiting for $C5/READY"
for i in $(seq 1 1440); do [ -f "$C5/READY" ] && break; sleep 60; done
[ -f "$C5/READY" ] || { log "5-robot data never appeared; exiting"; exit 1; }
R5=$(cat "$C5/ROBOTS"); NR=$(echo "$R5" | tr ":" "
" | wc -l); ENVS=$((64*NR)); MB=$((1024*NR)); log "robots=$R5 envs=$ENVS minibatch=$MB"

arm() {  # <name> <env...>
  local name=$1; shift
  [ -f "$OUT/$name.done" ] && { log "$name done"; return 0; }
  log "TRAIN $name $*"
  env NAME="$name" CLIPDIR="$C5" CLIP=super20.npz ROBOTS=$R5 NR_ENVS=$ENVS MINIBATCH=$MB TOTAL=58982400 SAVE_EVERY=1966080 \
      MORPH_MODE=schedule MORPH_COEFF=0.7 MORPH_START=0.2 MORPH_RAMP=40000000 PROJECT=local_w7 "$@" bash "$W6/local_train.sh" > "$OUT/$name.log" 2>&1
  local steps; steps=$(grep -o "nr_env_steps[^0-9]*[0-9]\+" "$OUT/$name.log" | grep -o "[0-9]\+$" | tail -1)
  log "END $name rc=$? steps=${steps:-0}"; [ "${steps:-0}" -ge 58982400 ] && touch "$OUT/$name.done"
}
# try the full robot list; on a failed start (OOM / compile fault: rc!=0 and no steps)
# fall back to 4 robots (drop Apollo), then 3 (H1+G1+T1)
run_with_fallback() {  # <name> <env...>
  local name=$1; shift
  local full="$R5"
  for cand in "$full" "$(echo "$full" | sed 's/:apptronik_apollo//')" "unitree_h1:unitree_g1:booster_t1"; do
    R5=$cand; NR=$(echo "$R5" | tr ":" "
" | wc -l); ENVS=$((64*NR)); MB=$((1024*NR))
    arm "$name" "$@"
    if [ -f "$OUT/$name.done" ]; then return 0; fi
    local steps; steps=$(grep -o "nr_env_steps[^0-9]*[0-9]\+" "$OUT/$name.log" | grep -o "[0-9]\+$" | tail -1)
    if [ "${steps:-0}" -gt 0 ]; then log "$name stopped mid-run with $R5 (resume by relaunching)"; return 1; fi
    log "$name failed to start with $R5: $(grep -m1 -E 'RESOURCE_EXHAUSTED|OOM|Traceback|Error' "$OUT/$name.log" | cut -c1-120) -> fewer robots"
    mv "$OUT/$name.log" "$OUT/$name.failed_$(echo $cand | tr ':' '_').log"
  done
  return 1
}
run_with_fallback w7_cot5 $COTV COTRAIN_INIT="$TOK"
R5=$(cat "$C5/ROBOTS")
run_with_fallback w7_ref5 LATENT=0
log "=== NIGHT7 DONE ==="
