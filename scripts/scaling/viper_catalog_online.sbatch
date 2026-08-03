#!/bin/bash -l
#SBATCH -p apu
#SBATCH -A mage_apu
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH -c 24
#SBATCH -t 08:00:00
#SBATCH -o /ptmp/akalenik/frontier/catalog_%j.out

# Catalog-aware online H1 training on one MI300A APU.
#
# Every knob arrives through `sbatch --export=ALL,NAME=value`; inline shell
# variables are not propagated by this cluster setup, which previously made a
# job silently run launcher defaults.

set -euo pipefail

ROOT=/ptmp/akalenik/frontier
source "$ROOT/venv/bin/activate"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
export JAX_COMPILATION_CACHE_DIR="$ROOT/xla_cache"

cd "$ROOT/repo"

RUN_NAME="${RUN_NAME:-catalog}"
CLIP="${CLIP:-walk1_subject1}"
NUM_ENVS="${NUM_ENVS:-8000}"
NUM_STEPS="${NUM_STEPS:-64}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-300000000}"
NUM_MINIBATCHES="${NUM_MINIBATCHES:-32}"
UPDATE_EPOCHS="${UPDATE_EPOCHS:-4}"
HIDDEN_1="${HIDDEN_1:-256}"
HIDDEN_2="${HIDDEN_2:-128}"
SEED="${SEED:-1}"
BACKBONE="${BACKBONE:-mlp}"
CATALOG_MODE="${CATALOG_MODE:-fixed_balanced}"
CATALOG="${CATALOG:-experiments/scaling_1000/catalogs/train_1000.json}"
CAPACITY_ONLY="${CAPACITY_ONLY:-0}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
# 0 keeps the flaky standalone slot-reset graph out of the job; exposure is
# still exact and analytic. Set deliberately to re-test that graph.
CATALOG_CHECK_ENVS="${CATALOG_CHECK_ENVS:-0}"

OUT="$ROOT/${RUN_NAME}_${SLURM_JOB_ID}"

EXTRA=()
if [ "$CATALOG_MODE" = "continuous" ]; then
  # Matched control arm: same budgets, uncontrolled per-reset body sampling.
  EXTRA+=(--resample-per-episode)
else
  EXTRA+=(--catalog "$CATALOG" --catalog-mode "$CATALOG_MODE"
          --catalog-check-envs "$CATALOG_CHECK_ENVS")
fi
if [ "$CAPACITY_ONLY" = "1" ]; then
  EXTRA+=(--capacity-only)
fi
if [ "$PREFLIGHT_ONLY" = "1" ]; then
  EXTRA+=(--preflight-only)
fi

echo "RUN_NAME=$RUN_NAME MODE=$CATALOG_MODE ENVS=$NUM_ENVS STEPS=$NUM_STEPS \
TOTAL=$TOTAL_TIMESTEPS SEED=$SEED JOB=$SLURM_JOB_ID NODE=$SLURMD_NODENAME"
sha256sum scripts/scaling/online_h1.py scripts/scaling/online_h1_train.py \
  scripts/scaling/embodiment_catalog.py scripts/scaling/catalog_vec_env.py
if [ "$CATALOG_MODE" != "continuous" ]; then
  sha256sum "$CATALOG"
fi

python scripts/scaling/online_h1_train.py \
  --clip "$CLIP" \
  --num-envs "$NUM_ENVS" \
  --total-timesteps "$TOTAL_TIMESTEPS" \
  --num-steps "$NUM_STEPS" \
  --num-minibatches "$NUM_MINIBATCHES" \
  --update-epochs "$UPDATE_EPOCHS" \
  --hidden "$HIDDEN_1" "$HIDDEN_2" \
  --backbone "$BACKBONE" \
  --seed "$SEED" \
  --run-tag "${RUN_NAME}_${SLURM_JOB_ID}" \
  --output-dir "$OUT" \
  "${EXTRA[@]}"

echo "CATALOG_ONLINE_DONE job=${SLURM_JOB_ID} out=${OUT}"
