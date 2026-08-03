#!/bin/bash -l
#SBATCH -p apu
#SBATCH -A mage_apu
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH -c 24
#SBATCH -t 08:00:00
#SBATCH -o /ptmp/akalenik/frontier/crosstopo_%j.out

set -euo pipefail

ROOT=/ptmp/akalenik/frontier
source "$ROOT/venv/bin/activate"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
export JAX_COMPILATION_CACHE_DIR="$ROOT/xla_cache"
export PYTHONPATH="$ROOT/repo/scripts"

BACKBONE=${BACKBONE:-urmav2}
ROBOTS=${ROBOTS:-"h1 g1 atlas"}
ONE_HOT=${ONE_HOT:-0}
SEED=${SEED:-1}
TOTAL_ENVS=${TOTAL_ENVS:-4096}
TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-400000000}
NUM_STEPS=${NUM_STEPS:-64}
NUM_MINIBATCHES=${NUM_MINIBATCHES:-32}
UPDATE_EPOCHS=${UPDATE_EPOCHS:-4}
HIDDEN=${HIDDEN:-"256 128"}
LR=${LR:-1e-4}
TAG=${TAG:-"${BACKBONE}_s${SEED}"}

EXTRA_ARGS=()
if [[ "$ONE_HOT" == "1" ]]; then
  EXTRA_ARGS+=(--robot-one-hot)
else
  EXTRA_ARGS+=(--no-robot-one-hot)
fi

cd "$ROOT/repo"
python scripts/scaling/parallel_cross_humanoid_train.py \
  --backbone "$BACKBONE" \
  --robots $ROBOTS \
  --reserve-robots toddlerbot \
  --total-envs "$TOTAL_ENVS" \
  --total-timesteps "$TOTAL_TIMESTEPS" \
  --num-steps "$NUM_STEPS" \
  --num-minibatches "$NUM_MINIBATCHES" \
  --update-epochs "$UPDATE_EPOCHS" \
  --hidden $HIDDEN \
  --lr "$LR" \
  --learnable-std \
  --seed "$SEED" \
  "${EXTRA_ARGS[@]}" \
  --run-tag "viper_${TAG}_${SLURM_JOB_ID}" \
  --output-dir "$ROOT/crosstopo_${TAG}_${SLURM_JOB_ID}"

echo "CROSSTOPO_DONE backbone=${BACKBONE} tag=${TAG} job=${SLURM_JOB_ID}"
