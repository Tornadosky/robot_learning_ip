#!/bin/bash
#SBATCH -J crosseval
#SBATCH -p apu
#SBATCH -A mage_apu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --ntasks-per-core=1
#SBATCH --mem=108000
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH -o /ptmp/akalenik/urma/logs/%x_%j.out
#SBATCH -e /ptmp/akalenik/urma/logs/%x_%j.out
# Cross-eval one finished arm's checkpoint against the RAW clip, both robots.
# Env vars: EXP (runs dir name), CLIP_DIR (the arm's training clips).
module load python-waterboa/2025.06
eval "$(conda shell.bash hook)"
conda activate /ptmp/akalenik/urma/env

ROOT=/ptmp/akalenik/urma
cd "$ROOT/loco_mjx/experiments"
export PYTHONUNBUFFERED=1 MUJOCO_GL=disable XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
export JAX_ENABLE_COMPILATION_CACHE=true
export JAX_COMPILATION_CACHE_DIR="$ROOT/jax_cache"
mkdir -p "$ROOT/crosseval"

MODEL=$(ls -t "$ROOT/loco_mjx/experiments/runs/urma2_h1g1/$EXP"/*/models/latest.model 2>/dev/null | head -1)
if [ -z "$MODEL" ]; then echo "NO MODEL for $EXP"; exit 1; fi
echo "EXP=$EXP MODEL=$MODEL CLIP_DIR=$CLIP_DIR"

# Optional per-arm overrides: CE_CLIP (clip file), CE_RAW_DIR (clean reference
# dir), CE_REFBIAS, CE_LATENT=1 (z-obs checkpoints).
# CE_ANCHOR/CE_FITVARIANT/CE_REFROOT reproduce the ARM's own reference
# construction. Omitting them silently scored every absolute-anchor arm against
# a centered reference; endgame5 passed them locally but this sbatch did not.
CE_FLAGS=()
[ "${CE_LATENT:-0}" = "1" ] && CE_FLAGS+=(--latent)
CE_FLAGS+=(--anchor "${CE_ANCHOR:-absolute}")
CE_FLAGS+=(--fitvariant "${CE_FITVARIANT:-False}")
CE_FLAGS+=(--refroot "${CE_REFROOT:-True}")
[ -n "${CE_HOLD:-}" ] && CE_FLAGS+=(--latent_hold "$CE_HOLD")
python "$ROOT/crosseval_motion.py" \
  --model_path "$MODEL" \
  --clip_dir "$CLIP_DIR" \
  --clip "${CE_CLIP:-dance2_subject4.npz}" \
  --raw_clip_dir "${CE_RAW_DIR:-$ROOT/clips/LAFAN1}" \
  --refbias "${CE_REFBIAS:-1.0}" \
  --nr_envs 64 --steps 1000 \
  ${CE_FLAGS[@]+"${CE_FLAGS[@]}"} \
  --out "$ROOT/crosseval/${EXP}.json"
echo "CROSSEVAL DONE $EXP"
