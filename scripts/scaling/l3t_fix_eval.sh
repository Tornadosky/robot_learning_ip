#!/usr/bin/env bash
# Crosseval for the 2026-08-29/30 overnight local arms:
#   L1 l3t_fix          (baseline-fix recipe, heading NOT observed)
#   L2 l3t_fix_head_rx  (rx_p3f0 contact dose + heading 0.20/2.0/observed)
# Both trained on LAFAN1_3t (H1+G1+T1, dance2_subject4) to 19.66M.
#
# Scores policy AND zero-action floor per robot, 4 seeds. --clip_dir and
# --raw_clip_dir are BOTH LAFAN1_3t: score the arm against what it was asked
# to do. L2 sets --root_heading_obs True (checkpoint obs width; a mismatch
# would misalign the obs vector, not error).
#
# GPU-free precondition: l3t_fix_head_rx.done exists (checked below) -- the
# overnight rule is nothing touches the GPU while L1/L2 train.
set -u
REPO=/mnt/c/Users/smirn/Desktop/robot_learning_ip
PY=~/jaxgpu/bin/python
CLIPS=$REPO/external_data/amass_converted/LAFAN1_3t
OUT=$REPO/experiments/local_3t/fixeval
mkdir -p "$OUT"

[ -f "$REPO/experiments/local_3t/l3t_fix_head_rx.done" ] || { echo "L2 not done; refusing GPU"; exit 1; }

M1=$(ls -t "$REPO"/loco_mjx/experiments/runs/local_3t/l3t_fix/*/models/latest.model | head -1)
M2=$(ls -t "$REPO"/loco_mjx/experiments/runs/local_3t/l3t_fix_head_rx/*/models/latest.model | head -1)
echo "L1 model: $M1"
echo "L2 model: $M2"

cd "$REPO/experiments/fsq_khaendler"
export PYTHONPATH="$REPO:$REPO/RL-X:$REPO/loco_mjx"
export MUJOCO_GL=disable XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
export JAX_ENABLE_COMPILATION_CACHE=true
export JAX_COMPILATION_CACHE_DIR=$REPO/.jax_cache_local
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

run() {  # run <tag> <model> <heading True|False> <zero:0|1> <seed>
  local tag="$1" model="$2" heading="$3" zero="$4" seed="$5"
  local args=(
    --model_path "$model"
    --clip_dir "$CLIPS" --clip dance2_subject4.npz --raw_clip_dir "$CLIPS"
    --robots unitree_h1:unitree_g1:booster_t1
    --refbias 0.0 --anchor absolute --fitvariant False
    --refroot True --refroot_floor True --refvel_obs False
    --root_heading_obs "$heading"
    --nr_envs 96 --steps 1000 --seed "$seed"
    --out "$OUT/${tag}_s${seed}.json"
  )
  [ "$zero" = "1" ] && args+=(--zero_action)
  echo "[fixeval] $tag seed=$seed $(date -Is)"
  $PY "$REPO/experiments/fsq_khaendler/crosseval_motion.py" "${args[@]}" > "$OUT/${tag}_s${seed}.log" 2>&1
  [ -s "$OUT/${tag}_s${seed}.json" ] && echo "  ARTIFACT OK" || echo "  ARTIFACT MISSING (see $OUT/${tag}_s${seed}.log)"
}

# Seed-0 for every cell first, so each arm has a first reading ASAP;
# remaining seeds after. Same-graph cells hit the compile cache.
for s in 0 1 2 3; do
  run l1_policy "$M1" False 0 "$s"
  run l1_zero   "$M1" False 1 "$s"
  run l2_policy "$M2" True  0 "$s"
  run l2_zero   "$M2" True  1 "$s"
done
echo "[fixeval] DONE $(date -Is)"
