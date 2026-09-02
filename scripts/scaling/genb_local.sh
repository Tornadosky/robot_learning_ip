#!/usr/bin/env bash
# GEN-B' on the LOCAL CUDA box instead of Viper.
#
# WHY MOVE IT. The Viper probe ran 82 minutes without producing an artifact while
# two H1+G1 crossevals on the same node finished in ~7 minutes each. It was
# CPU-busy the whole time (187 %, `futex_wait_queue`), so it was compiling, not
# deadlocked -- and ROCm compiles on that box run roughly 25x slower than CUDA.
# The cost is specific to a graph that is not already in the Viper cache, and no
# booster_t1 crosseval has ever been run there.
#
# Locally that inverts: CUDA compiles fast AND this box already trains
# booster_t1, so its JAX cache has been built against that robot.
#
# `eval_heldout.py` was considered and rejected: it reports episode length and
# return, not per-joint tracking RMSE against the clip, so it answers "does the
# policy survive on BoosterT1" rather than "does it track". Different question.
#
# The zero-action floor is included, because a held-out score without one is
# uninterpretable -- on robotis_op3 a policy once turned out WORSE than doing
# nothing, and only the floor showed it.
#
# Runs beside the 3-topology training. That job holds ~7 GB of 16 GB and
# preallocation is off, so a 64-env eval fits; it will slow training somewhat.
set -u
REPO=/mnt/c/Users/smirn/Desktop/robot_learning_ip
PY=~/jaxgpu/bin/python
FIX=$REPO/external_data/amass_converted/LAFAN1_fixed
OUT=$REPO/experiments/local_3t/genb
MODEL=$REPO/experiments/local_3t/ckpt/best79_s1.model
mkdir -p "$OUT"
[ -f "$MODEL" ] || { echo "no checkpoint at $MODEL"; exit 1; }

cd "$REPO/experiments/fsq_khaendler"
export PYTHONPATH="$REPO:$REPO/RL-X:$REPO/loco_mjx"
export MUJOCO_GL=disable XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
export JAX_ENABLE_COMPILATION_CACHE=true
export JAX_COMPILATION_CACHE_DIR=$REPO/.jax_cache_local

run() {   # run <tag> <zero:0|1> <seed>
  local tag="$1" zero="$2" sd="$3"
  local args=(
    --model_path "$MODEL"
    --clip_dir "$FIX" --clip dance2_subject4.npz --raw_clip_dir "$FIX"
    --robots booster_t1
    --refbias 0.0 --anchor absolute --fitvariant False
    --refroot True --refroot_floor True --refvel_obs False
    --nr_envs 64 --steps 1000 --seed "$sd"
    --out "$OUT/${tag}_s${sd}.json"
  )
  [ "$zero" = "1" ] && args+=(--zero_action)
  echo "[genb] $tag seed=$sd  $(date -Is)"
  $PY "$REPO/experiments/fsq_khaendler/crosseval_motion.py" "${args[@]}" \
      > "$OUT/${tag}_s${sd}.log" 2>&1
  if [ -s "$OUT/${tag}_s${sd}.json" ]; then
    echo "  ARTIFACT OK"
  else
    echo "  ARTIFACT MISSING -- tail:"; tail -4 "$OUT/${tag}_s${sd}.log"
  fi
}

for s in 0 1 2 3; do
  run t1_policy 0 "$s"
  run t1_zero   1 "$s"
done
echo "[genb] DONE $(date -Is)"
