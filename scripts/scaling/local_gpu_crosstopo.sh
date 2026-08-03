#!/bin/bash
# Sequential local-GPU cross-topology queue.
#
# Two JAX processes on one GPU collapse utilisation, so every run here is
# serialised.  Launch detached, or the whole queue dies with its wrapper:
#   setsid nohup bash scripts/scaling/local_gpu_crosstopo.sh > /tmp/crosstopo.log 2>&1 < /dev/null &
set -u

cd /mnt/c/Users/smirn/Desktop/robot_learning_ip
export PYTHONPATH=$PWD/scripts
export XLA_PYTHON_CLIENT_PREALLOCATE=false
PY=~/jaxgpu/bin/python
OUT=experiments/cross_embodiment

# Measured cross-topology throughput on the RTX 4060 Ti is 1.77M steps/min --
# roughly a quarter of the single-topology H1 path, because three MJX branches
# and a 1356-dim padded observation are in every update.  200M steps is ~1.9h,
# which is what fits before the evaluation window.
STEPS=${STEPS:-200000000}
ENVS=${ENVS:-4096}

run() {
  local tag=$1 backbone=$2 onehot=$3 seed=$4 steps=$5
  shift 5
  local dir="$OUT/local_${tag}"
  if [[ -f "$dir/manifest.json" ]]; then
    echo "[queue] SKIP $tag (manifest exists)"
    return 0
  fi
  echo "[queue] START $tag backbone=$backbone onehot=$onehot seed=$seed steps=$steps $(date -Is)"
  $PY scripts/scaling/parallel_cross_humanoid_train.py \
    --backbone "$backbone" \
    --robots h1 g1 atlas \
    --reserve-robots toddlerbot \
    "$onehot" \
    --total-envs "$ENVS" \
    --total-timesteps "$steps" \
    --num-steps 64 \
    --num-minibatches 32 \
    --update-epochs 4 \
    --hidden 256 128 \
    --lr 1e-4 \
    --learnable-std \
    --seed "$seed" \
    --output-dir "$dir" "$@"
  local rc=$?
  echo "[queue] END $tag rc=$rc $(date -Is)"
  return $rc
}

# Seeds 3/4 here, seeds 1/2 on Viper, so the two machines contribute distinct
# seeds to the same matched comparison instead of duplicating each other.
run smoke_urmav2 urmav2 --no-robot-one-hot 99 8388608
run urmav2_s3 urmav2 --no-robot-one-hot 3 "$STEPS"
run maskedmlp_s3 masked_mlp --robot-one-hot 3 "$STEPS"
# Second 4-epoch seed for each arm: acceptance criterion 4 wants the result to
# hold across at least two seeds, and Viper could only supply 1-epoch seeds.
run urmav2_s4 urmav2 --no-robot-one-hot 4 "$STEPS"
run maskedmlp_s4 masked_mlp --robot-one-hot 4 "$STEPS"
# Third seed: URMA's seed spread was 27-37x the MLP's over seeds 3-4, and the
# Atlas ordering rests on two seeds.  This is the cheapest test of both.
run urmav2_s5 urmav2 --no-robot-one-hot 5 "$STEPS"
run maskedmlp_s5 masked_mlp --robot-one-hot 5 "$STEPS"

echo "[queue] ALL DONE $(date -Is)"
