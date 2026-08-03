#!/bin/bash -l
#SBATCH -p apu
#SBATCH -A mage_apu
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH -c 24
#SBATCH -t 08:00:00
#SBATCH -o /ptmp/akalenik/frontier/online_h1_%j.out

set -euo pipefail

ROOT=/ptmp/akalenik/frontier
source "$ROOT/venv/bin/activate"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
export JAX_COMPILATION_CACHE_DIR="$ROOT/xla_cache"

cd "$ROOT/repo"
python scripts/scaling/online_h1_train.py \
  --num-envs 8192 \
  --total-timesteps 10485760 \
  --num-steps 64 \
  --num-minibatches 64 \
  --update-epochs 2 \
  --hidden 256 128 \
  --run-tag "viper_online_8192_${SLURM_JOB_ID}" \
  --output-dir "$ROOT/online_8192_${SLURM_JOB_ID}"

echo "ONLINE_H1_DONE job=${SLURM_JOB_ID}"
