#!/usr/bin/env bash
# Crosseval + render dump for the local three-topology policy.
#
# Rolls the trained H1+G1+t1 policy against dance2_subject4 and scores executed
# joint angles against the clip each robot was actually retargeted to. Also dumps
# the trajectories so rf_render_dance2.py can draw policy beside reference.
#
# The 3-robot env compiles for >10 minutes before the rollout starts. Do not
# wrap this in a timeout.
set -u
REPO=/mnt/c/Users/smirn/Desktop/robot_learning_ip
PY=~/jaxgpu/bin/python
CLIPS=$REPO/external_data/amass_converted/LAFAN1
OUT=$REPO/experiments/local_3t
MODEL=$(ls -t "$REPO"/loco_mjx/experiments/runs/local_3t/local_3t_dance4/*/models/latest.model 2>/dev/null | head -1)
[ -n "$MODEL" ] || { echo "no checkpoint"; exit 1; }
echo "model: $MODEL"

cd "$REPO/experiments/fsq_khaendler"
export PYTHONPATH="$REPO:$REPO/RL-X:$REPO/loco_mjx"
export MUJOCO_GL=disable XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_enable_command_buffer=

SEED=${SEED:-0}
DUMP=${DUMP:-1}
args=(
  --model_path "$MODEL"
  --clip_dir "$CLIPS" --clip dance2_subject4.npz --raw_clip_dir "$CLIPS"
  --robots unitree_h1:unitree_g1:booster_t1
  --refbias 0.0 --anchor absolute --fitvariant False
  --refroot True --refroot_floor True --refvel_obs False
  # 96, not the usual 64: create_env requires nr_envs divisible by the
  # robot count, and 64 % 3 != 0.
  --nr_envs 96 --steps 1000 --seed "$SEED"
  --out "$OUT/ce_local_3t_s${SEED}.json"
)
[ "$DUMP" = "1" ] && args+=(--dump_render "$OUT/render_3t" --record_envs 2)

echo "[eval] START seed=$SEED  $(date -Is)"
$PY "$REPO/experiments/fsq_khaendler/crosseval_motion.py" "${args[@]}"
rc=$?
echo "[eval] END rc=$rc  $(date -Is)"
[ -s "$OUT/ce_local_3t_s${SEED}.json" ] && echo "ARTIFACT OK" || echo "ARTIFACT MISSING"
