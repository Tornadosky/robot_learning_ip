#!/usr/bin/env bash
# STEP 0 -- how much of BoosterT1's failure was the IMPOSSIBLE TARGET?
#
# T1 scores 1.12x its zero-action floor against H1's 1.73x and G1's 1.92x, and is
# WORSE than zero action at matched early age. 13.75 % of its reference frames are
# outside jnt_range, so part of that gap is a target the robot cannot reach.
# This decomposes it at ZERO training cost.
#
# The policy is UNCHANGED and so is what it observes: --clip_dir stays on the
# original clip the checkpoint trained against. Only --raw_clip_dir moves, from
# the original (impossible) target to the regenerated feasible one. So this asks:
# how close did the robot get to what it COULD have done?
#
# The zero-action floor is re-scored on both targets too. Without that the
# comparison is worthless -- projecting the clip onto the feasible set moves the
# target closer to everything, the floor included, so only the RATIO is
# interpretable.
#
# Single robot on purpose: the 3-robot graph compiles for >10 minutes and URMA2's
# policy is defined over per-joint descriptions, so it applies unchanged to one
# topology at a time.
set -u
REPO=/mnt/c/Users/smirn/Desktop/robot_learning_ip
PY=~/jaxgpu/bin/python
ORIG=$REPO/external_data/amass_converted/LAFAN1
FIXED=$REPO/external_data/amass_converted/LAFAN1_step0
OUT=$REPO/experiments/local_3t/step0
MODEL=$(ls -t "$REPO"/loco_mjx/experiments/runs/local_3t/local_3t_dance4/*/models/latest.model 2>/dev/null | head -1)
[ -n "$MODEL" ] || { echo "no checkpoint"; exit 1; }
mkdir -p "$OUT"

# A raw dir that is the ORIGINAL everywhere except BoosterT1, which is the
# regenerated feasible clip. Built by copy, not symlink: the npz loader follows
# a path and a broken link would read as a missing clip, not as an error.
mkdir -p "$FIXED"
for r in "$ORIG"/*/; do
  name=$(basename "$r")
  mkdir -p "$FIXED/$name"
  if [ "$name" = "BoosterT1" ]; then
    cp -f "$REPO/external_data/amass_converted/LAFAN1_t1fix/BoosterT1/"*.npz "$FIXED/$name/"
  else
    cp -f "$r"*.npz "$FIXED/$name/" 2>/dev/null || true
  fi
done
echo "merged raw dir: $(ls "$FIXED/BoosterT1")"

cd "$REPO/experiments/fsq_khaendler"
export PYTHONPATH="$REPO:$REPO/RL-X:$REPO/loco_mjx"
export MUJOCO_GL=disable XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
# The compile dominates: the first cell took 11m48s and there are 16 of them.
# crosseval2.sbatch on Viper has always set these; the local eval scripts never
# did, so every local crosseval this project has run recompiled from scratch.
export JAX_ENABLE_COMPILATION_CACHE=true
export JAX_COMPILATION_CACHE_DIR=$REPO/.jax_cache_local
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

run() {  # run <tag> <raw_dir> <zero:0|1> <seed>
  local tag="$1" raw="$2" zero="$3" seed="$4"
  local args=(
    --model_path "$MODEL"
    --clip_dir "$ORIG" --clip dance2_subject4.npz --raw_clip_dir "$raw"
    --robots booster_t1
    --refbias 0.0 --anchor absolute --fitvariant False
    --refroot True --refroot_floor True --refvel_obs False
    --nr_envs 64 --steps 1000 --seed "$seed"
    --out "$OUT/${tag}_s${seed}.json"
  )
  [ "$zero" = "1" ] && args+=(--zero_action)
  echo "[step0] $tag seed=$seed $(date -Is)"
  $PY "$REPO/experiments/fsq_khaendler/crosseval_motion.py" "${args[@]}" > "$OUT/${tag}_s${seed}.log" 2>&1
  [ -s "$OUT/${tag}_s${seed}.json" ] && echo "  ARTIFACT OK" || echo "  ARTIFACT MISSING (see $OUT/${tag}_s${seed}.log)"
}

for s in 0 1 2 3; do
  run orig_policy "$ORIG"  0 "$s"
  run fixed_policy "$FIXED" 0 "$s"
  run orig_zero   "$ORIG"  1 "$s"
  run fixed_zero  "$FIXED" 1 "$s"
done
echo "[step0] DONE $(date -Is)"
