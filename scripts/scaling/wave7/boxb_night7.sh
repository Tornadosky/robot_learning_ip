#!/usr/bin/env bash
# BOX-B (Ubuntu, RTX 5080) orchestrator for wave 7 -- one JAX process at a time.
#   1. 3-robot co-training smoke (clips_3t_v2, 10 updates) -> proves the stack
#   2. J7  w7_ref5      5-robot reference arm, 59M, bodies 0.2->0.7 (pairs with BOX-A's w7_cot5)
#   3.     w7_cot5_h20  5-robot co-trained arm at reference hold 20
#   4.     w7_ref5_h20  5-robot reference arm at hold 20
# Each arm falls back 5 -> 4 (drop Apollo) -> 3 robots if it cannot start; a run
# that stopped mid-way is resumed by relaunching this script (done arms skipped).
# Launch:  nohup bash scripts/scaling/wave7/boxb_night7.sh > experiments/local_w7/boxb_night7.log 2>&1 &
set -u
REPO=${REPO:-/home/melo/Projects/ip_project/robot_learning_ip}
export REPO PY=${PY:-$HOME/jaxgpu/bin/python}
W6=$REPO/scripts/scaling/wave6
OUT=$REPO/experiments/local_w7; mkdir -p "$OUT"
C3=$REPO/experiments/fsq_khaendler/clips_3t_v2
C5=$REPO/experiments/fsq_khaendler/clips_5r
TOK=$REPO/experiments/fsq_khaendler/tokenizer_m20/params.msgpack
TOK2=$REPO/experiments/fsq_khaendler/tokenizer_3t_v2/params.msgpack
log() { echo "[boxb7 $(date '+%m-%d %H:%M:%S')] $*"; }
COTV="LATENT=1 LATENT_DIM=44 LATENT_DIVISOR=1.0 SIDECAR=_win COTRAIN_ROWS=11 COTRAIN_CH=4"
TOTAL=58982400

if [ ! -f "$OUT/smoke_cot3.done" ]; then
  log "=== 3-robot co-training smoke ==="
  env NAME=b7smoke_cot3 CLIPDIR="$C3" ROBOTS=unitree_h1:unitree_g1:booster_t1 NR_ENVS=96 MINIBATCH=1536 TOTAL=61440 SAVE_EVERY=61440 \
      $COTV COTRAIN_INIT="$TOK2" PROJECT=local_w7 bash "$W6/local_train.sh" > "$OUT/smoke_cot3.log" 2>&1
  n=$(grep -c nr_env_steps "$OUT/smoke_cot3.log"); log "smoke rc=$? steps=$n"
  grep -m1 -E "Traceback|Error" "$OUT/smoke_cot3.log" | cut -c1-200
  [ "$n" -gt 0 ] && touch "$OUT/smoke_cot3.done" || { log "smoke failed; stopping"; exit 1; }
fi

R5=$(cat "$C5/ROBOTS")
arm() {  # <name> <env...>
  local name=$1; shift
  local NR ENVS MB; NR=$(echo "$R5" | tr ':' '\n' | wc -l); ENVS=$((64*NR)); MB=$((1024*NR))
  log "TRAIN $name robots=$R5 envs=$ENVS mb=$MB $*"
  env NAME="$name" CLIPDIR="$C5" CLIP=super20.npz ROBOTS=$R5 NR_ENVS=$ENVS MINIBATCH=$MB TOTAL=$TOTAL SAVE_EVERY=1966080 \
      MORPH_MODE=schedule MORPH_COEFF=0.7 PROJECT=local_w7 "$@" bash "$W6/local_train.sh" >> "$OUT/$name.log" 2>&1
  local rc=$?; local steps; steps=$(grep -o "nr_env_steps[^0-9]*[0-9]\+" "$OUT/$name.log" | grep -o "[0-9]\+$" | tail -1)
  log "END $name rc=$rc steps=${steps:-0}"; [ "${steps:-0}" -ge $TOTAL ] && touch "$OUT/$name.done"
}
run_with_fallback() {  # <name> <env...>
  local name=$1; shift
  [ -f "$OUT/$name.done" ] && { log "$name done"; return 0; }
  local full; full=$(cat "$C5/ROBOTS")
  for cand in "$full" "$(echo "$full" | sed 's/:apptronik_apollo//')" "unitree_h1:unitree_g1:booster_t1"; do
    R5=$cand
    arm "$name" "$@"
    [ -f "$OUT/$name.done" ] && return 0
    local steps; steps=$(grep -o "nr_env_steps[^0-9]*[0-9]\+" "$OUT/$name.log" | grep -o "[0-9]\+$" | tail -1)
    if [ "${steps:-0}" -gt 0 ]; then log "$name stopped mid-run with $R5 (relaunch to resume)"; return 1; fi
    log "$name failed to start with $R5: $(grep -m1 -E 'RESOURCE_EXHAUSTED|OOM|Traceback|Error' "$OUT/$name.log" | cut -c1-120) -> fewer robots"
    mv "$OUT/$name.log" "$OUT/$name.failed_$(echo $cand | tr ':' '_').log"
  done
  return 1
}
run_with_fallback w7_ref5 LATENT=0
run_with_fallback w7_cot5_h20 HOLD=20 $COTV COTRAIN_INIT="$TOK"
run_with_fallback w7_ref5_h20 HOLD=20 LATENT=0
run_with_fallback w7_ref5_s2 LATENT=0 SEED=2
log "=== BOXB NIGHT7 DONE ==="
