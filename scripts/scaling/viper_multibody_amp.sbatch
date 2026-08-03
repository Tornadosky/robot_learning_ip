#!/bin/bash -l
#SBATCH -p apu
#SBATCH -A mage_apu
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH -c 24
#SBATCH -t 08:00:00
#SBATCH -o /ptmp/akalenik/frontier/multibody_amp_%j.out

set -euo pipefail

ROOT=/ptmp/akalenik/frontier
source "$ROOT/venv/bin/activate"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
export JAX_COMPILATION_CACHE_DIR="$ROOT/xla_cache"

TOTAL_ENVS=${TOTAL_ENVS:-1024}
TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-262144}

cd "$ROOT/repo"
python scripts/scaling/parallel_multimorph_amp_train.py \
  --variants nominal combined tall_legs heavy_torso \
  --clips dance2_subject1 dance2_subject2 dance2_subject3 dance2_subject4 \
  --frames-per-clip 500 \
  --total-envs "$TOTAL_ENVS" \
  --total-timesteps "$TOTAL_TIMESTEPS" \
  --num-steps 32 \
  --num-minibatches 16 \
  --update-epochs 1 \
  --hidden 256 128 \
  --n-disc-epochs 2 \
  --disc-minibatch-size 2048 \
  --learnable-std \
  --run-tag "viper_amp_4body_4motion_${TOTAL_ENVS}_${SLURM_JOB_ID}" \
  --output-dir "$ROOT/amp_4body_4motion_${TOTAL_ENVS}_${SLURM_JOB_ID}"

echo "MULTIBODY_AMP_DONE envs=${TOTAL_ENVS} job=${SLURM_JOB_ID}"
