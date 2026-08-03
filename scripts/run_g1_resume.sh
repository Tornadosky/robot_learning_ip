#!/usr/bin/env bash
# Deterministically extend the known-good G1 policy that crossed the balance
# transition (stiff3_lr3 @120M, len 731) to convergence, instead of re-rolling a
# fresh seed that may not cross. Same recipe: 3x PD gains, lr 3e-4, std 0.2.
set -u
source ~/miniconda3/bin/activate locodm
cd /mnt/c/Users/smirn/Desktop/robot_learning_ip/scripts
export XLA_FLAGS='--xla_gpu_triton_gemm_any=True'
LOGDIR=/mnt/c/Users/smirn/Desktop/robot_learning_ip/logs/g1_experiments
mkdir -p "$LOGDIR"
ROOT=/mnt/c/Users/smirn/Desktop/robot_learning_ip
CKPT="$ROOT/external_data/deepmimic_morphology/g1/dance2_subject4/nominal__stiff3_lr3/checkpoints/ckpt_03_119603200/PPOJax_saved.pkl"
echo "[final_resumed] start $(date)" | tee "$LOGDIR/final_resumed.log"
python train_deepmimic_morphology.py \
  --robot g1 --preset nominal --clip dance2_subject4 --duration 30 \
  --num-envs 2048 --total-timesteps 180e6 --num-checkpoints 6 \
  --raw-reference --control pd --lr 3e-4 --init-std 0.2 --pd-gain-scale 3.0 \
  --resume-from "$CKPT" --resume-steps 119603200 \
  --out-suffix final_resumed \
  >> "$LOGDIR/final_resumed.log" 2>&1
echo "[final_resumed] done $(date) rc=$?" | tee -a "$LOGDIR/final_resumed.log"
echo ALL_DONE
