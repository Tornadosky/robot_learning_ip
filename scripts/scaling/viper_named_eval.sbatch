#!/bin/bash -l
#SBATCH -p apu
#SBATCH -A mage_apu
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH -c 24
#SBATCH -t 02:00:00
#SBATCH -o /ptmp/akalenik/frontier/namedeval_%j.out

# Static-model check: evaluate on the named generated_variants H1 morphologies.

set -euo pipefail

ROOT=/ptmp/akalenik/frontier
source "$ROOT/venv/bin/activate"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
export JAX_COMPILATION_CACHE_DIR="$ROOT/xla_cache"

cd "$ROOT/repo"

CHECKPOINT="${CHECKPOINT:?set CHECKPOINT}"
OUTPUT="${OUTPUT:?set OUTPUT}"
REPLICAS="${REPLICAS:-64}"
HORIZON="${HORIZON:-1000}"
EVAL_SEED="${EVAL_SEED:-20260801}"

echo "CHECKPOINT=$CHECKPOINT JOB=$SLURM_JOB_ID NODE=$SLURMD_NODENAME"
sha256sum scripts/scaling/evaluate_catalog_policy.py \
  experiments/scaling_1000/catalogs/named_variants.json

python scripts/scaling/evaluate_catalog_policy.py \
  --checkpoint "$CHECKPOINT" \
  --catalogs experiments/scaling_1000/catalogs/named_variants.json \
  --replicas "$REPLICAS" \
  --horizon "$HORIZON" \
  --seed "$EVAL_SEED" \
  --output "$OUTPUT"

echo "NAMED_EVAL_DONE job=${SLURM_JOB_ID} out=${OUTPUT}"
