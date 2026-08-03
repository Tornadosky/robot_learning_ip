#!/bin/bash -l
#SBATCH -p apu
#SBATCH -A mage_apu
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH -c 24
#SBATCH -t 04:00:00
#SBATCH -o /ptmp/akalenik/frontier/cateval_%j.out

# Evaluate one checkpoint on every frozen held-out catalog.

set -euo pipefail

ROOT=/ptmp/akalenik/frontier
source "$ROOT/venv/bin/activate"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
export JAX_COMPILATION_CACHE_DIR="$ROOT/xla_cache"

cd "$ROOT/repo"

CHECKPOINT="${CHECKPOINT:?set CHECKPOINT}"
OUTPUT="${OUTPUT:?set OUTPUT}"
REPLICAS="${REPLICAS:-4}"
HORIZON="${HORIZON:-1000}"
EVAL_SEED="${EVAL_SEED:-20260801}"
CATALOG_DIR="${CATALOG_DIR:-experiments/scaling_1000/catalogs}"

echo "CHECKPOINT=$CHECKPOINT REPLICAS=$REPLICAS HORIZON=$HORIZON \
SEED=$EVAL_SEED JOB=$SLURM_JOB_ID NODE=$SLURMD_NODENAME"
sha256sum scripts/scaling/evaluate_catalog_policy.py

python scripts/scaling/evaluate_catalog_policy.py \
  --checkpoint "$CHECKPOINT" \
  --catalogs \
    "$CATALOG_DIR/iid_256.json" \
    "$CATALOG_DIR/boundary_256.json" \
    "$CATALOG_DIR/ood_128.json" \
    "$CATALOG_DIR/validation_128.json" \
  --replicas "$REPLICAS" \
  --horizon "$HORIZON" \
  --seed "$EVAL_SEED" \
  --output "$OUTPUT"

echo "CATALOG_EVAL_DONE job=${SLURM_JOB_ID} out=${OUTPUT}"
