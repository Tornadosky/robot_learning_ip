#!/usr/bin/env bash
# Diagnostic sweep for accelerating G1 nominal DeepMimic dance tracking.
# Runs sequentially on the single GPU. Each writes to a suffixed output cell so the
# baseline (lr 1e-4 / std 0.2) is never clobbered.
set -u
source ~/miniconda3/bin/activate locodm
cd /mnt/c/Users/smirn/Desktop/robot_learning_ip/scripts
export XLA_FLAGS='--xla_gpu_triton_gemm_any=True'
LOGDIR=/mnt/c/Users/smirn/Desktop/robot_learning_ip/logs/g1_experiments
mkdir -p "$LOGDIR"

run() {
  local suffix="$1"; shift
  echo "[${suffix}] start $(date)" | tee "$LOGDIR/${suffix}.log"
  python train_deepmimic_morphology.py \
    --robot g1 --preset nominal --clip dance2_subject4 --duration 30 \
    --num-envs 2048 --total-timesteps 120e6 --num-checkpoints 4 \
    --raw-reference --control pd --out-suffix "$suffix" "$@" \
    >> "$LOGDIR/${suffix}.log" 2>&1
  echo "[${suffix}] done $(date) rc=$?" | tee -a "$LOGDIR/${suffix}.log"
}

run lr3e4_std3 --lr 3e-4 --init-std 0.3
run lr3e4_std2 --lr 3e-4 --init-std 0.2
echo ALL_DONE
