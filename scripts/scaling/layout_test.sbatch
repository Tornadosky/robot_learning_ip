#!/bin/bash
#SBATCH -J layout_test
#SBATCH -p apu
#SBATCH -A mage_apu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=108000
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH -o /ptmp/akalenik/urma/logs/%x_%j.out
#SBATCH -e /ptmp/akalenik/urma/logs/%x_%j.out
# Validate the FSQ latent observation layout on the deployed code, BEFORE any
# arm trains against it. A layout slip does not crash -- it silently trains the
# policy on the wrong channels and still produces a number.
set -u
module load python-waterboa/2025.06
eval "$(conda shell.bash hook)"
conda activate /ptmp/akalenik/urma/env
ROOT=/ptmp/akalenik/urma
cd "$ROOT"
export PYTHONUNBUFFERED=1 MUJOCO_GL=disable JAX_PLATFORMS=cpu
export PYTHONPATH="$ROOT:$ROOT/RL-X:$ROOT/loco_mjx:${PYTHONPATH:-}"
python tests/test_latent_obs_layout.py
echo "LAYOUT TEST EXIT=$?"
