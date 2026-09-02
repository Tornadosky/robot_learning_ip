#!/usr/bin/env bash
# NIGHT3 local eval -- waits for ln3_tok.done, then crossevals the 3-robot pair:
#   ln3_ref  practical recipe + gb33 gait dose + heading      (roadmap B1)
#   ln3_tok  same + FSQ token, tk4 routing, clips_3t_token    (roadmap B2)
# vs the zero-action floor, 4 seeds each, in the SAME contact world the arms
# trained in (--contact_timeconst 0.004; local crosseval patched for it).
# Also dumps render npz for both arms (seed 0) for the morning video pass.
#
# Launch detached:
#   setsid nohup bash scripts/scaling/ln3_eval.sh \
#     > experiments/local_3t/ln3_eval_run.log 2>&1 < /dev/null &
set -u
REPO=/mnt/c/Users/smirn/Desktop/robot_learning_ip
PY=~/jaxgpu/bin/python
CLIPS=$REPO/external_data/amass_converted/LAFAN1_3t
TOKDIR=$REPO/experiments/fsq_khaendler/clips_3t_token
OUT=$REPO/experiments/local_3t/ln3_eval
RND=$REPO/experiments/local_3t/renders_ln3
mkdir -p "$OUT" "$RND"

# GPU-free precondition, polled: fires whenever the token arm lands.
for i in $(seq 1 240); do
  [ -f "$REPO/experiments/local_3t/ln3_tok.done" ] && break
  sleep 60
done
[ -f "$REPO/experiments/local_3t/ln3_tok.done" ] || { echo "[ln3eval] ln3_tok never finished; giving up"; exit 1; }
sleep 30   # let the trainer process exit fully

MR=$(ls -t "$REPO"/loco_mjx/experiments/runs/local_3t/ln3_ref/*/models/latest.model | head -1)
MT=$(ls -t "$REPO"/loco_mjx/experiments/runs/local_3t/ln3_tok/*/models/latest.model | head -1)
echo "[ln3eval] ref model: $MR"
echo "[ln3eval] tok model: $MT"

cd "$REPO/experiments/fsq_khaendler"
export PYTHONPATH="$REPO:$REPO/RL-X:$REPO/loco_mjx"
export MUJOCO_GL=disable XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
export JAX_ENABLE_COMPILATION_CACHE=true
export JAX_COMPILATION_CACHE_DIR=$REPO/.jax_cache_local

COMMON=(--clip dance2_subject4.npz --refbias 0.0 --anchor absolute
        --fitvariant False --refroot True --refroot_floor True --refvel_obs False
        --root_heading_obs True --contact_timeconst 0.004
        --robots unitree_h1:unitree_g1:booster_t1 --nr_envs 96 --steps 1000)
TOKF=(--latent --latent_dim 32 --latent_replaces False --latent_scope per_joint
      --latent_divisor 10.0 --jlat_enc_dim 4 --latent_hold 1)

run() {  # run <tag> <model> <clipdir> <zero:0|1> <seed> <latent:0|1> [dump]
  local tag="$1" model="$2" cdir="$3" zero="$4" seed="$5" lat="$6" dump="${7:-}"
  local args=(--model_path "$model" --clip_dir "$cdir" --raw_clip_dir "$CLIPS"
              "${COMMON[@]}" --seed "$seed" --out "$OUT/${tag}_s${seed}.json")
  [ "$zero" = "1" ] && args+=(--zero_action)
  [ "$lat" = "1" ] && args+=("${TOKF[@]}")
  [ -n "$dump" ] && args+=(--dump_render "$RND/$dump" --record_envs 3)
  echo "[ln3eval] $tag s$seed $(date -Is)"
  $PY "$REPO/experiments/fsq_khaendler/crosseval_motion.py" "${args[@]}" > "$OUT/${tag}_s${seed}.log" 2>&1
  [ -s "$OUT/${tag}_s${seed}.json" ] && echo "  OK" || echo "  MISSING (see log)"
}

# First reading for every cell at s0 (ref includes the render dump), then the
# remaining seeds. Zero floor once (action-independent; ref env config).
run ln3_ref_policy "$MR" "$CLIPS"  0 0 0 ln3_ref
run ln3_tok_policy "$MT" "$TOKDIR" 0 0 1 ln3_tok
run ln3_zero       "$MR" "$CLIPS"  1 0 0
for s in 1 2 3; do
  run ln3_ref_policy "$MR" "$CLIPS"  0 "$s" 0
  run ln3_tok_policy "$MT" "$TOKDIR" 0 "$s" 1
  run ln3_zero       "$MR" "$CLIPS"  1 "$s" 0
done
echo "[ln3eval] DONE $(date -Is)"
