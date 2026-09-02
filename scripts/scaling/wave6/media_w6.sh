#!/usr/bin/env bash
# Wave-6 media: wait for the running local arm to END, pause the queue (it is
# resume-safe: done arms and existing CE JSONs are skipped), dump LONGER
# rollouts with crosseval_motion.py (one JAX process at a time), render, then
# relaunch the queue. Videos land in experiments/local_w6/media/.
set -u
REPO=${REPO:-/mnt/c/Users/smirn/Desktop/robot_learning_ip}
W6=$REPO/scripts/scaling/wave6
OUT=$REPO/experiments/local_w6
DUMPS=$OUT/dumps; MEDIA=$OUT/media; mkdir -p "$DUMPS" "$MEDIA"
PYE=~/jaxgpu/bin/python; PYR=~/locomjx/bin/python
E=$REPO/experiments/fsq_khaendler
MIR=$REPO/viper_mirror/runs/urma2_h1g1
V2=$E/clips_3t_v2; SUPER=$E/clips_super; RAW=$REPO/external_data/amass_converted/LAFAN1_3t
log() { echo "[media $(date '+%m-%d %H:%M:%S')] $*"; }

# 0. wait for the current arm to finish (END line newer than the last TRAIN line)
log "waiting for the running local arm to END"
for i in $(seq 1 360); do
  last=$(grep -E "TRAIN |END " "$OUT/queue.log" | tail -1)
  case "$last" in *"END "*) break;; esac
  sleep 20
done
log "pausing the queue: $last"
pkill -f local_queue_w6.sh; sleep 2; pkill -f crosseval_motion.py; sleep 5
pgrep -fa "experiment.py|crosseval_motion.py" | head -2

ckpt() { ls -t "$MIR/$1"/*/models/latest.model 2>/dev/null | head -1; }
cd "$E"
export PYTHONPATH="$REPO:$REPO/RL-X:$REPO/loco_mjx" MUJOCO_GL=disable XLA_PYTHON_CLIENT_PREALLOCATE=false \
       XLA_FLAGS=--xla_gpu_enable_command_buffer= JAX_ENABLE_COMPILATION_CACHE=true JAX_COMPILATION_CACHE_DIR=$REPO/.jax_cache_local
COMMON=(--refbias 0.0 --anchor absolute --fitvariant False --refroot True --refroot_floor True
        --refvel_obs False --root_heading_obs True --contact_timeconst 0.004
        --robots unitree_h1:unitree_g1 --nr_envs 32 --seed 0 --record_envs 2)
REFF=(--jlat_enc_dim 0)
TOKF=(--latent --latent_dim 32 --latent_replaces False --latent_scope per_joint --latent_divisor 10.0 --jlat_enc_dim 4 --latent_hold 1)
COTF=(--latent --latent_dim 44 --latent_replaces False --latent_scope per_joint --latent_divisor 1.0 --latent_sidecar _win --jlat_enc_dim 4 --latent_hold 1)
# dump <tag> <exp> <clipdir> <clip> <hold> <steps> <flags...>
dump() {
  local tag=$1 exp=$2 cdir=$3 clip=$4 hold=$5 steps=$6; shift 6
  local m; m=$(ckpt "$exp"); [ -n "$m" ] || { log "no ckpt $exp"; return 1; }
  ls "$DUMPS/${tag}__unitree_g1.npz" >/dev/null 2>&1 && { log "dump $tag exists"; return 0; }
  log "DUMP $tag ($exp, $clip, hold $hold, $steps steps)"
  $PYE crosseval_motion.py --model_path "$m" --clip_dir "$cdir" --raw_clip_dir "$cdir" --clip "$clip" \
     "${COMMON[@]}" --reference_hold "$hold" --steps "$steps" --dump_render "$DUMPS/$tag" \
     --out "$DUMPS/$tag.json" "$@" > "$DUMPS/$tag.log" 2>&1
  ls "$DUMPS" | grep -c "^$tag"
}
dump cot_dance4     n6cot_tok_s2    "$V2"    dance2_subject4.npz 1  2400 "${COTF[@]}"
dump ref_dance4     n5sw05_ref_s2   "$V2"    dance2_subject4.npz 1  2400 "${REFF[@]}"
dump cot_walk1_zs   n6cot_tok_s2    "$V2"    walk1_subject1.npz  1  2400 "${COTF[@]}"
dump ref_walk1_zs   n5sw05_ref_s2   "$V2"    walk1_subject1.npz  1  2400 "${REFF[@]}"
dump suph20_tok     n6suph20_tok_s1 "$SUPER" super5dance.npz     20 3600 "${TOKF[@]}"
dump suph20_ref     n6suph20_ref_s1 "$SUPER" super5dance.npz     20 3600 "${REFF[@]}"
dump cot_dance4_h20 n6cot_tok_s2    "$V2"    dance2_subject4.npz 20 2400 "${COTF[@]}"
ls "$DUMPS"

# relaunch the queue FIRST (GPU), then render on CPU/EGL
log "relaunching the queue"
nohup bash "$W6/local_queue_w6.sh" >> "$OUT/queue.log" 2>&1 &
sleep 5

export MUJOCO_GL=egl
render() {  # <npz basename> <robot> <out.mp4> [max_frames]
  local npz="$DUMPS/$1" robot=$2 out=$3 mf=${4:-1100}
  [ -f "$npz" ] || { npz="$REPO/experiments/local_3t/renders_final/$1"; }
  [ -f "$npz" ] || { log "  MISSING $1"; return 1; }
  $PYR rf_render_dance2.py --npz "$npz" --xml "$REPO/loco_mjx/loco_mjx/environments/robots/$robot/data/plane.xml" \
     --out "$MEDIA/$out" --width 800 --height 534 --stride 2 --max_frames "$mf" > "$DUMPS/render_${out%.mp4}.log" 2>&1
  [ -s "$MEDIA/$out" ] && log "  OK $out $(stat -c %s "$MEDIA/$out")" || log "  FAIL $out"
}
render cot_dance4__unitree_g1.npz     unitree_g1 cot_dance4_g1.mp4
render cot_dance4__unitree_h1.npz     unitree_h1 cot_dance4_h1.mp4
render ref_dance4__unitree_g1.npz     unitree_g1 ref_dance4_g1.mp4
render cot_walk1_zs__unitree_g1.npz   unitree_g1 cot_walk1_zeroshot_g1.mp4
render cot_walk1_zs__unitree_h1.npz   unitree_h1 cot_walk1_zeroshot_h1.mp4
render ref_walk1_zs__unitree_h1.npz   unitree_h1 ref_walk1_zeroshot_h1.mp4
render suph20_tok__unitree_g1.npz     unitree_g1 super5_h20_tok_g1.mp4 1700
render suph20_ref__unitree_g1.npz     unitree_g1 super5_h20_ref_g1.mp4 1700
render cot_dance4_h20__unitree_h1.npz unitree_h1 cot_dance4_h20_h1.mp4
render ln5_tok__booster_t1.npz        booster_t1 3robot_tok_t1.mp4 500
render w2h20_tok_s1_xdump__unitree_g1.npz unitree_g1 transfer_g1_from_h1_stream.mp4 500
log "=== MEDIA DONE ==="; ls -la "$MEDIA"
