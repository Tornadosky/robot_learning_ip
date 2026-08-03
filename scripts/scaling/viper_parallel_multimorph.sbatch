#!/bin/bash -l
#SBATCH -p apu
#SBATCH -A mage_apu
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH -c 24
#SBATCH -t 08:00:00
#SBATCH -o /ptmp/akalenik/frontier/parallel_multimorph_%j.out

set -euo pipefail

ROOT=/ptmp/akalenik/frontier
source "$ROOT/venv/bin/activate"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
export JAX_COMPILATION_CACHE_DIR="$ROOT/xla_cache"

cd "$ROOT/repo"
python scripts/scaling/parallel_multimorph_train.py \
  --variants \
    nominal tall_legs short_legs long_arms broad_shoulders big_feet \
    heavy_torso combined extreme_tall_light extreme_short_heavy \
  --envs-per-morph 200 \
  --total-timesteps 2048000 \
  --num-steps 64 \
  --num-minibatches 32 \
  --update-epochs 2 \
  --hidden 256 128 \
  --run-tag "viper_static_10x200_${SLURM_JOB_ID}" \
  --output-dir "$ROOT/static_10x200_${SLURM_JOB_ID}"

echo "PARALLEL_MULTIMORPH_DONE job=${SLURM_JOB_ID}"
