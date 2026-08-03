#!/usr/bin/env bash
# Reliable converged G1 nominal in ~one run: train several seeds in parallel (vmap)
# with the winning recipe (3x PD gains, lr 3e-4) and keep the best by tracking return.
# 3 seeds @ 1792 envs fits the 16GB GPU (3x2048 OOMs); envs kept close to the proven
# 2048-env crosser. ~70% chance >=1 seed crosses the balance transition in this batch.
set -u
source ~/miniconda3/bin/activate locodm
cd /mnt/c/Users/smirn/Desktop/robot_learning_ip/scripts
export XLA_FLAGS='--xla_gpu_triton_gemm_any=True'
export XLA_PYTHON_CLIENT_PREALLOCATE=false
LOGDIR=/mnt/c/Users/smirn/Desktop/robot_learning_ip/logs/g1_experiments
mkdir -p "$LOGDIR"
SEED="${1:-1}"          # first seed of the batch (override to run a fresh batch)
SUFFIX="${2:-multiseed}"
echo "[$SUFFIX] start $(date) seeds from $SEED" | tee "$LOGDIR/$SUFFIX.log"
python train_deepmimic_multiseed.py \
  --robot g1 --preset nominal --clip dance2_subject4 --duration 30 \
  --num-envs 1792 --n-seeds 3 --total-timesteps 300e6 --seed "$SEED" \
  --raw-reference --control pd --lr 3e-4 --init-std 0.2 --pd-gain-scale 3.0 \
  --out-suffix "$SUFFIX" \
  >> "$LOGDIR/$SUFFIX.log" 2>&1
echo "[$SUFFIX] done $(date) rc=$?" | tee -a "$LOGDIR/$SUFFIX.log"
echo ALL_DONE
