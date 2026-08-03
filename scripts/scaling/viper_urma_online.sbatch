#!/bin/bash -l
#SBATCH -p apu
#SBATCH -A mage_apu
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH -c 24
#SBATCH -t 08:00:00
#SBATCH -o /ptmp/akalenik/frontier/urma_online_%j.out

set -euo pipefail

ROOT=/ptmp/akalenik/frontier
source "$ROOT/venv/bin/activate"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
export JAX_COMPILATION_CACHE_DIR="$ROOT/xla_cache"

BACKBONE=${BACKBONE:-urmav2}
NUM_ENVS=${NUM_ENVS:-256}
TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-32768}
NUM_STEPS=${NUM_STEPS:-32}
NUM_MINIBATCHES=${NUM_MINIBATCHES:-8}
UPDATE_EPOCHS=${UPDATE_EPOCHS:-1}
HIDDEN_1=${HIDDEN_1:-128}
HIDDEN_2=${HIDDEN_2:-64}
RESAMPLE=${RESAMPLE:-0}

EXTRA_ARGS=()
if [[ "$RESAMPLE" == "1" ]]; then
  EXTRA_ARGS+=(--resample-per-episode)
fi

cd "$ROOT/repo"
python scripts/scaling/online_h1_train.py \
  --backbone "$BACKBONE" \
  --num-envs "$NUM_ENVS" \
  --total-timesteps "$TOTAL_TIMESTEPS" \
  --num-steps "$NUM_STEPS" \
  --num-minibatches "$NUM_MINIBATCHES" \
  --update-epochs "$UPDATE_EPOCHS" \
  --hidden "$HIDDEN_1" "$HIDDEN_2" \
  --learnable-std \
  --duration 2 \
  "${EXTRA_ARGS[@]}" \
  --run-tag "viper_${BACKBONE}_${NUM_ENVS}_${SLURM_JOB_ID}" \
  --output-dir "$ROOT/urma_${BACKBONE}_${NUM_ENVS}_${SLURM_JOB_ID}"

echo "URMA_ONLINE_DONE backbone=${BACKBONE} envs=${NUM_ENVS} job=${SLURM_JOB_ID}"
