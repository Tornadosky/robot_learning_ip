#!/usr/bin/env bash
# Final G1 nominal run: the winning recipe from the sweeps.
#   PD control with 3x stiffer gains (the balance-ceiling fix) + lr 3e-4.
# Full 300M budget (same as the H1 nominal baseline) for a fair comparison.
# num-checkpoints 10 -> 30M (73-update) segments, matching the stiff3 diagnostic
# that crossed the balance transition (the 50M-segment run got stuck pre-transition).
set -u
source ~/miniconda3/bin/activate locodm
cd /mnt/c/Users/smirn/Desktop/robot_learning_ip/scripts
export XLA_FLAGS='--xla_gpu_triton_gemm_any=True'
LOGDIR=/mnt/c/Users/smirn/Desktop/robot_learning_ip/logs/g1_experiments
mkdir -p "$LOGDIR"
echo "[final_stiff3_seg30] start $(date)" | tee "$LOGDIR/final_stiff3_seg30.log"
python train_deepmimic_morphology.py \
  --robot g1 --preset nominal --clip dance2_subject4 --duration 30 \
  --num-envs 2048 --total-timesteps 300e6 --num-checkpoints 10 \
  --raw-reference --control pd --lr 3e-4 --init-std 0.2 --pd-gain-scale 3.0 \
  --out-suffix final_stiff3_seg30 \
  >> "$LOGDIR/final_stiff3_seg30.log" 2>&1
echo "[final_stiff3_seg30] done $(date) rc=$?" | tee -a "$LOGDIR/final_stiff3_seg30.log"
echo ALL_DONE
