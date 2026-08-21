#!/bin/bash -l
#SBATCH -p apu
#SBATCH -A mage_apu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -c 12
#SBATCH -t 02:00:00
#SBATCH -o /ptmp/akalenik/frontier/fsq_%j.out

# FSQ design-A canonical motion autoencoder (urma_fsq_overnight_20260813).
# Tiny MLP job; the GPU is overkill but keeps the ROCm-safe venv/XLA setup
# identical to every other job on this account.
set -euo pipefail

ROOT=/ptmp/akalenik/frontier
source "$ROOT/venv/bin/activate"
export PYTHONUNBUFFERED=1
# PLATFORM=cpu forces CPU jax (the 2026-08-13 GPU submissions of this tiny
# MLP hung >14 min inside ROCm init with zero output; CPU trains it fine)
if [[ -n "${PLATFORM:-}" ]]; then export JAX_PLATFORMS="$PLATFORM"; fi
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
export JAX_COMPILATION_CACHE_DIR="$ROOT/xla_cache"
export PYTHONPATH="$ROOT/repo/scripts"

DATASET=${DATASET:-"$ROOT/fsq_data/lafan1_h1_100hz_v2.npz"}
VAL_CLIPS=${VAL_CLIPS:-"dance2_subject5 walk3_subject1"}
EPOCHS=${EPOCHS:-400}
LEVELS=${LEVELS:-"8 5 5 5"}
WINDOW=${WINDOW:-10}
STRIDE=${STRIDE:-4}
SEED=${SEED:-0}
EXTRA=${EXTRA:-}
TAG=${TAG:-fsq_a}

cd "$ROOT/repo"
python scripts/scaling/train_fsq_motion.py train \
  --dataset "$DATASET" \
  --output "$ROOT/fsq_${TAG}_${SLURM_JOB_ID}" \
  --val-clips $VAL_CLIPS \
  --window "$WINDOW" \
  --stride "$STRIDE" \
  --levels $LEVELS \
  --epochs "$EPOCHS" \
  --seed "$SEED" \
  $EXTRA

echo "FSQ_DONE tag=${TAG} job=${SLURM_JOB_ID}"
