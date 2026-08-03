#!/bin/bash -l
#SBATCH -p apu
#SBATCH -A mage_apu
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH -c 24
#SBATCH -t 08:00:00
#SBATCH -o /ptmp/akalenik/frontier/online_amp_%j.out

# Online-morphology x multi-motion AMP: one dynamic H1 graph, catalog bodies,
# several LAFAN1 clips, one shared policy and descriptor-blind discriminator.

set -euo pipefail

ROOT=/ptmp/akalenik/frontier
source "$ROOT/venv/bin/activate"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
export JAX_COMPILATION_CACHE_DIR="$ROOT/xla_cache"

cd "$ROOT/repo"

RUN_NAME="${RUN_NAME:-online_amp}"
CATALOG="${CATALOG:-experiments/scaling_1000/catalogs/train_16.json}"
CATALOG_MODE="${CATALOG_MODE:-catalog_resample}"
CLIPS="${CLIPS:-dance2_subject1 dance2_subject2 dance2_subject3 dance2_subject4}"
FRAMES_PER_CLIP="${FRAMES_PER_CLIP:-1000}"
NUM_ENVS="${NUM_ENVS:-1024}"
NUM_STEPS="${NUM_STEPS:-64}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-100000000}"
NUM_MINIBATCHES="${NUM_MINIBATCHES:-32}"
UPDATE_EPOCHS="${UPDATE_EPOCHS:-4}"
N_DISC_EPOCHS="${N_DISC_EPOCHS:-10}"
DISC_LR="${DISC_LR:-5e-5}"
JOINT_SPACE_REWARD="${JOINT_SPACE_REWARD:-1}"
DISC_MINIBATCH="${DISC_MINIBATCH:-2048}"
PROPORTION_ENV_REWARD="${PROPORTION_ENV_REWARD:-0.5}"
HIDDEN_1="${HIDDEN_1:-256}"
HIDDEN_2="${HIDDEN_2:-128}"
SEED="${SEED:-1}"
BLIND="${BLIND:-1}"

OUT="$ROOT/${RUN_NAME}_${SLURM_JOB_ID}"

EXTRA=()
if [ "$JOINT_SPACE_REWARD" = "0" ]; then
  EXTRA+=(--full-mimic-reward)
fi
if [ "$BLIND" = "0" ]; then
  EXTRA+=(--no-blind-discriminator)
fi

echo "RUN_NAME=$RUN_NAME CATALOG=$CATALOG CLIPS=$CLIPS ENVS=$NUM_ENVS \
TOTAL=$TOTAL_TIMESTEPS SEED=$SEED BLIND=$BLIND JOB=$SLURM_JOB_ID NODE=$SLURMD_NODENAME"
sha256sum scripts/scaling/online_amp.py scripts/scaling/online_amp_train.py \
  scripts/scaling/online_h1.py "$CATALOG"

python scripts/scaling/online_amp_train.py \
  --catalog "$CATALOG" \
  --catalog-mode "$CATALOG_MODE" \
  --clips $CLIPS \
  --frames-per-clip "$FRAMES_PER_CLIP" \
  --num-envs "$NUM_ENVS" \
  --total-timesteps "$TOTAL_TIMESTEPS" \
  --num-steps "$NUM_STEPS" \
  --num-minibatches "$NUM_MINIBATCHES" \
  --update-epochs "$UPDATE_EPOCHS" \
  --n-disc-epochs "$N_DISC_EPOCHS" \
  --disc-lr "$DISC_LR" \
  --disc-minibatch-size "$DISC_MINIBATCH" \
  --proportion-env-reward "$PROPORTION_ENV_REWARD" \
  --hidden "$HIDDEN_1" "$HIDDEN_2" \
  --seed "$SEED" \
  --run-tag "${RUN_NAME}_${SLURM_JOB_ID}" \
  --output-dir "$OUT" \
  "${EXTRA[@]}"

echo "ONLINE_AMP_DONE job=${SLURM_JOB_ID} out=${OUT}"
