#!/usr/bin/env bash
# Best single-shot at a CONVERGED G1 nominal policy: 4x PD gains (wider balance
# basin than 3x -> more reliable crossing) trained in one process to 300M.
# In-process checkpointing only (cross-process resume was found to collapse the policy).
set -u
source ~/miniconda3/bin/activate locodm
cd /mnt/c/Users/smirn/Desktop/robot_learning_ip/scripts
export XLA_FLAGS='--xla_gpu_triton_gemm_any=True'
LOGDIR=/mnt/c/Users/smirn/Desktop/robot_learning_ip/logs/g1_experiments
mkdir -p "$LOGDIR"
echo "[final4x] start $(date)" | tee "$LOGDIR/final4x.log"
python train_deepmimic_morphology.py \
  --robot g1 --preset nominal --clip dance2_subject4 --duration 30 \
  --num-envs 2048 --total-timesteps 300e6 --num-checkpoints 10 \
  --raw-reference --control pd --lr 3e-4 --init-std 0.2 --pd-gain-scale 4.0 \
  --out-suffix final4x \
  >> "$LOGDIR/final4x.log" 2>&1
echo "[final4x] done $(date) rc=$?" | tee -a "$LOGDIR/final4x.log"
echo ALL_DONE
