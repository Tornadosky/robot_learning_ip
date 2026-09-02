#!/usr/bin/env bash
# Crosseval ONE snapshot of the local three-topology policy.
#
# Differs from local_3t_eval.sh only in taking an explicit MODEL (so the
# step-scaling curve can be built from experiments/local_3t/snapshots/) and an
# explicit TAG for the output name. Everything else is byte-identical, so
# snapshot scores stay comparable with ce_local_3t_s0.json.
#
# The 3-robot env compiles for >10 minutes before the rollout starts. Do NOT
# wrap this in a timeout.
set -u
REPO=/mnt/c/Users/smirn/Desktop/robot_learning_ip
PY=~/jaxgpu/bin/python
CLIPS=$REPO/external_data/amass_converted/LAFAN1
OUT=$REPO/experiments/local_3t
MODEL=${MODEL:?set MODEL to a .model path}
TAG=${TAG:?set TAG for the output name}
[ -f "$MODEL" ] || { echo "no such checkpoint: $MODEL"; exit 1; }

cd "$REPO/experiments/fsq_khaendler"
export PYTHONPATH="$REPO:$REPO/RL-X:$REPO/loco_mjx"
export MUJOCO_GL=disable XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
# PERSISTENT COMPILE CACHE. Without this every eval recompiles the
# 3-robot MJX graph from scratch -- measured at 25-55 min EACH, which
# made a 13-eval sweep impossible in one night. Viper's crosseval2.sbatch
# has had this since the LADDER campaign; the local scripts never did.
# The two MIN_ vars are required: the defaults are restrictive enough
# that most entries are silently not cached at all.
export JAX_ENABLE_COMPILATION_CACHE=true
export JAX_COMPILATION_CACHE_DIR="$REPO/.jax_cache_local"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

SEED=${SEED:-0}
STEPS=${STEPS:-1000}
ZERO=${ZERO:-0}
args=(
  --model_path "$MODEL"
  --clip_dir "$CLIPS" --clip dance2_subject4.npz --raw_clip_dir "$CLIPS"
  --robots unitree_h1:unitree_g1:booster_t1
  --refbias 0.0 --anchor absolute --fitvariant False
  --refroot True --refroot_floor True --refvel_obs False
  --nr_envs 96 --steps "$STEPS" --seed "$SEED"
  --out "$OUT/ce_${TAG}_s${SEED}.json"
)
# The zero-action floor. Every score is meaningless without it: an arm that
# "tracks" can be worse than doing nothing, which has happened on this project.
[ "$ZERO" = "1" ] && args+=(--zero_action)

echo "[eval] START tag=$TAG seed=$SEED zero=$ZERO model=$MODEL $(date -Is)"
$PY "$REPO/experiments/fsq_khaendler/crosseval_motion.py" "${args[@]}"
rc=$?
echo "[eval] END rc=$rc $(date -Is)"
[ -s "$OUT/ce_${TAG}_s${SEED}.json" ] && echo "ARTIFACT OK" || echo "ARTIFACT MISSING"
