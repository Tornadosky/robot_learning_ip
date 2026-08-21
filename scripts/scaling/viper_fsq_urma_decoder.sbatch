#!/bin/bash -l
#SBATCH -p apu
#SBATCH -A mage_apu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -c 12
#SBATCH -t 02:00:00
#SBATCH -o /ptmp/akalenik/frontier/fsqb_%j.out

# FSQ design-B (URMA-decoder) — urma_fsq_overnight_20260813.  CPU jax: the
# GPU submissions of the tiny design-A MLP hung in ROCm init on 2026-08-13.
set -euo pipefail

ROOT=/ptmp/akalenik/frontier
source "$ROOT/venv/bin/activate"
export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=${PLATFORM:-cpu}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$ROOT/repo/scripts"

DATASET=${DATASET:-"$ROOT/fsq_data/design_b_paired_v1.npz"}
VAL_WINDOWS=${VAL_WINDOWS:-"dance2_subject5:2000:800"}
EPOCHS=${EPOCHS:-200}
TAG=${TAG:-fsq_b}
EXTRA=${EXTRA:-}

cd "$ROOT/repo"
python scripts/scaling/train_fsq_urma_decoder.py train \
  --dataset "$DATASET" \
  --output "$ROOT/fsq_${TAG}_${SLURM_JOB_ID}" \
  --val-windows $VAL_WINDOWS \
  --epochs "$EPOCHS" \
  $EXTRA

echo "FSQ_B_DONE tag=${TAG} job=${SLURM_JOB_ID}"
