#!/usr/bin/env bash
# WAVE 6 LOCAL QUEUE -- strictly sequential (one JAX process at a time; three
# concurrent compiles restarted the WSL VM on 2026-09-02 09:28).
#   0. wait for the smoke chain (launch_smoke2.sh) to finish; abort if any failed
#   1. H1+G1 arms on the bundle recipe, 768 envs, 19.66M, nominal bodies:
#        lw6_ref        reference only (local control)
#        lw6_tok        token, LEGACY shared routing (= every prior "tk4" arm)
#        lw6_split      token, REAL separate Dense(4) projection (first ever)
#        lw6_aux        split + aux next-token head (coeff 0.5, horizon 5)
#        lw6_cot        SONIC co-training, encoder init from tokenizer_3t_v2
#        lw6_cotsc      SONIC co-training from scratch
#        lw6_legw3_ref / lw6_legw3_tok   leg-weighted kernel (x3) controls
#      each followed by CEs: dance4 hold 1 x2 seeds, hold 20 x2, zero floor x1
#   2. the 3-robot hold-20 pair (scripts/scaling/local_h20_3t.sh)
set -u
REPO=/mnt/c/Users/smirn/Desktop/robot_learning_ip
W6=$REPO/scripts/scaling/wave6
OUT=$REPO/experiments/local_w6
EV=$OUT/eval
PY=~/jaxgpu/bin/python
C=$REPO/experiments/fsq_khaendler/clips_3t_v2
RAW=$REPO/external_data/amass_converted/LAFAN1_3t
TOK=$REPO/experiments/fsq_khaendler/tokenizer_3t_v2/params.msgpack
mkdir -p "$EV"
log() { echo "[queue $(date '+%m-%d %H:%M:%S')] $*"; }

log "waiting for smoke chain"
for i in $(seq 1 240); do grep -q "SMOKE_COTAUX_RC" "$OUT/smoke_cotaux.log" 2>/dev/null && break; sleep 30; done
for s in smoke_aux2 smoke_cot smoke_cotaux; do
  rc=$(grep -o "SMOKE_[A-Z0-9]*_RC=[0-9]*" "$OUT/$s.log" 2>/dev/null | tail -1)
  log "$s -> ${rc:-NO RC}"
  case "$rc" in *=0) ;; *) log "SMOKE FAILED ($s) -- queue stops before training arms"; touch "$OUT/QUEUE_BLOCKED_BY_SMOKE"; exit 1;; esac
done
rm -f "$OUT/QUEUE_BLOCKED_BY_SMOKE"

newest_ckpt() { ls -t "$REPO"/loco_mjx/experiments/runs/local_w6/"$1"/*/models/latest.model 2>/dev/null | head -1; }

# arm <name> <env assignments...>
arm() {
  local name=$1; shift
  if [ -f "$OUT/$name.done" ]; then log "$name already done"; return 0; fi
  log "TRAIN $name  $*"
  env NAME="$name" CLIPDIR="$C" PROJECT=local_w6 "$@" bash "$W6/local_train.sh" > "$OUT/$name.log" 2>&1
  local rc=$?
  local steps; steps=$(grep -o "nr_env_steps[^0-9]*[0-9]\+" "$OUT/$name.log" | grep -o "[0-9]\+$" | tail -1)
  log "END $name rc=$rc steps=${steps:-0}"
  if [ "${steps:-0}" -ge 19660800 ]; then touch "$OUT/$name.done"; return 0; fi
  return 1
}
# ce <name> <tag> <seed> <refhold> <zero 0|1> <extra crosseval flags...>
ce() {
  local name=$1 tag=$2 seed=$3 hold=$4 zero=$5; shift 5
  local m; m=$(newest_ckpt "$name"); [ -n "$m" ] || { log "no ckpt for $name"; return 1; }
  local o="$EV/${name}_${tag}_s${seed}.json"; [ -s "$o" ] && return 0
  local args=(--model_path "$m" --clip_dir "$C" --raw_clip_dir "$RAW" --clip dance2_subject4.npz
              --refbias 0.0 --anchor absolute --fitvariant False --refroot True --refroot_floor True
              --refvel_obs False --root_heading_obs True --contact_timeconst 0.004
              --robots unitree_h1:unitree_g1 --nr_envs 64 --steps 1000 --seed "$seed"
              --reference_hold "$hold" --out "$o" "$@")
  [ "$zero" = 1 ] && args+=(--zero_action)
  ( cd "$REPO/experiments/fsq_khaendler" && export PYTHONPATH="$REPO:$REPO/RL-X:$REPO/loco_mjx" MUJOCO_GL=disable \
      XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_command_buffer= \
      JAX_ENABLE_COMPILATION_CACHE=true JAX_COMPILATION_CACHE_DIR=$REPO/.jax_cache_local && \
    $PY crosseval_motion.py "${args[@]}" > "$EV/${name}_${tag}_s${seed}.log" 2>&1 )
  [ -s "$o" ] && log "CE $name $tag s$seed OK" || log "CE $name $tag s$seed MISSING"
}
ce_set() {  # ce_set <name> <flags...>  : h1 x2, h20 x2
  local name=$1; shift
  for s in 0 1; do ce "$name" h1 "$s" 1 0 "$@"; done
  for s in 0 1; do ce "$name" h20 "$s" 20 0 "$@"; done
}
REFF=(--jlat_enc_dim 0)
TOKF=(--latent --latent_dim 32 --latent_replaces False --latent_scope per_joint --latent_divisor 10.0 --jlat_enc_dim 4 --latent_hold 1)
COTF=(--latent --latent_dim 44 --latent_replaces False --latent_scope per_joint --latent_divisor 1.0 --latent_sidecar _win --jlat_enc_dim 4 --latent_hold 1)

arm lw6_ref    LATENT=0                                   && ce_set lw6_ref "${REFF[@]}" && ce lw6_ref zero 0 1 1 "${REFF[@]}"
arm lw6_tok    LATENT=1 JLAT_CH=0                         && ce_set lw6_tok "${TOKF[@]}"
arm lw6_split  LATENT=1 JLAT_CH=-1                        && ce_set lw6_split "${TOKF[@]}"
arm lw6_aux    LATENT=1 JLAT_CH=-1 AUX_COEFF=0.5 AUX_HORIZON=5 && ce_set lw6_aux "${TOKF[@]}"
arm lw6_cot    LATENT=1 LATENT_DIM=44 LATENT_DIVISOR=1.0 SIDECAR=_win COTRAIN_ROWS=11 COTRAIN_CH=4 COTRAIN_INIT="$TOK" && ce_set lw6_cot "${COTF[@]}"
arm lw6_cotsc  LATENT=1 LATENT_DIM=44 LATENT_DIVISOR=1.0 SIDECAR=_win COTRAIN_ROWS=11 COTRAIN_CH=4 && ce_set lw6_cotsc "${COTF[@]}"
arm lw6_legw3_ref LATENT=0 LEGW=3.0                       && ce_set lw6_legw3_ref "${REFF[@]}"
arm lw6_legw3_tok LATENT=1 JLAT_CH=0 LEGW=3.0             && ce_set lw6_legw3_tok "${TOKF[@]}"

log "=== local arms done; starting 3-robot hold-20 pair ==="
bash "$REPO/scripts/scaling/local_h20_3t.sh" > "$REPO/experiments/local_3t/ln6h20_run.log" 2>&1
log "=== QUEUE DONE ==="
