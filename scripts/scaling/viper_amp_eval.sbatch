#!/bin/bash -l
#SBATCH -p apu
#SBATCH -A mage_apu
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH -c 24
#SBATCH -t 04:00:00
#SBATCH -o /ptmp/akalenik/frontier/ampeval_%j.out

# Per-body / per-motion / held-out-motion evaluation of an online AMP policy.

set -euo pipefail

ROOT=/ptmp/akalenik/frontier
source "$ROOT/venv/bin/activate"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
export JAX_COMPILATION_CACHE_DIR="$ROOT/xla_cache"

cd "$ROOT/repo"

CHECKPOINT="${CHECKPOINT:?set CHECKPOINT}"
OUTPUT="${OUTPUT:?set OUTPUT}"
REPLICAS="${REPLICAS:-16}"
HORIZON="${HORIZON:-1000}"
EVAL_SEED="${EVAL_SEED:-20260802}"
HELDOUT_CLIPS="${HELDOUT_CLIPS:-dance2_subject5}"

echo "CHECKPOINT=$CHECKPOINT HELDOUT=$HELDOUT_CLIPS JOB=$SLURM_JOB_ID NODE=$SLURMD_NODENAME"
sha256sum scripts/scaling/evaluate_online_amp_policy.py

python scripts/scaling/evaluate_online_amp_policy.py \
  --checkpoint "$CHECKPOINT" \
  --replicas "$REPLICAS" \
  --horizon "$HORIZON" \
  --seed "$EVAL_SEED" \
  --heldout-clips $HELDOUT_CLIPS \
  --output "$OUTPUT"

echo "AMP_EVAL_DONE job=${SLURM_JOB_ID} out=${OUTPUT}"
