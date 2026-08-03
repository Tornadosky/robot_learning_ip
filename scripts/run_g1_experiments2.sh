#!/usr/bin/env bash
# Second G1 sweep: test PD stiffness as the lever for the balance ceiling.
# Keeps the better optimizer setting from sweep 1 (lr 3e-4, std 0.2).
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
    --raw-reference --control pd --lr 3e-4 --init-std 0.2 \
    --out-suffix "$suffix" "$@" \
    >> "$LOGDIR/${suffix}.log" 2>&1
  echo "[${suffix}] done $(date) rc=$?" | tee -a "$LOGDIR/${suffix}.log"
}

run stiff2_lr3 --pd-gain-scale 2.0
run stiff3_lr3 --pd-gain-scale 3.0
echo ALL_DONE
