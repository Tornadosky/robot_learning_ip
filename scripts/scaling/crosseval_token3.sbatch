#!/bin/bash
#SBATCH -J ce_token3
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
#
# crosseval_token2 + three night5 additions:
#   CE_XLA_OFF=0   drop the --xla_gpu_enable_command_buffer= flag (the H1+T1
#                  eval graph dies with ROCM_ERROR_ILLEGAL_ADDRESS under it --
#                  or under something; this sbatch exists to bisect that).
#   CE_NRENVS      eval env count (was hardcoded 64).
#   ARTIFACT CHECK the n4t1 CEs "COMPLETED" with no json; a missing artifact
#                  now exits 1 so sacct tells the truth.
set -u
module load python-waterboa/2025.06
eval "$(conda shell.bash hook)"
conda activate /ptmp/akalenik/urma/env

ROOT=/ptmp/akalenik/urma
cd "$ROOT/loco_mjx/experiments"
export PYTHONUNBUFFERED=1 MUJOCO_GL=disable XLA_PYTHON_CLIENT_PREALLOCATE=false
if [ "${CE_XLA_OFF:-0}" = "1" ]; then
  export XLA_FLAGS=""
else
  export XLA_FLAGS=--xla_gpu_enable_command_buffer=
fi
export JAX_ENABLE_COMPILATION_CACHE=true
export JAX_COMPILATION_CACHE_DIR="$ROOT/jax_cache"
mkdir -p "$ROOT/crosseval"

MODEL=$(ls -t "$ROOT/loco_mjx/experiments/runs/urma2_h1g1/$EXP"/*/models/latest.model 2>/dev/null | head -1)
if [ -z "$MODEL" ]; then echo "NO MODEL for $EXP"; exit 1; fi

CE_FLAGS=()
[ "${CE_LATENT:-0}" = "1" ] && CE_FLAGS+=(--latent)
CE_FLAGS+=(--latent_dim "${CE_LATENT_DIM:-32}")
CE_FLAGS+=(--latent_hold "${CE_HOLD:-1}")
CE_FLAGS+=(--latent_replaces "${CE_REPLACES:-False}")
CE_FLAGS+=(--latent_scope "${CE_SCOPE:-per_joint}")
CE_FLAGS+=(--latent_divisor "${CE_DIVISOR:-1.0}")
CE_FLAGS+=(--latent_sidecar "${CE_SIDECAR:-_zq}")
CE_FLAGS+=(--reference_hold "${CE_REFHOLD:-1}")
CE_FLAGS+=(--root_heading_obs "${CE_HEADING_OBS:-False}")
CE_FLAGS+=(--contact_timeconst "${CE_CONTACT_TIMECONST:-0.0}")
CE_FLAGS+=(--robots "${CE_ROBOTS:-unitree_h1:unitree_g1}")
# CE_MORPH (night5 M-EVAL): evaluate on a morphology-perturbed body.
CE_FLAGS+=(--morphology_coeff "${CE_MORPH:-0.0}")
CE_FLAGS+=(--anchor "${CE_ANCHOR:-absolute}")
CE_FLAGS+=(--fitvariant "${CE_FITVARIANT:-False}")
CE_FLAGS+=(--refroot "${CE_REFROOT:-True}")
CE_FLAGS+=(--refroot_floor "${CE_REFROOT_FLOOR:-True}")
CE_FLAGS+=(--refvel_obs "${CE_REFVEL_OBS:-False}")
CE_FLAGS+=(--seed "${CE_SEED:-0}")
if [ "${CE_DUMP:-0}" = "1" ]; then
  mkdir -p "$ROOT/renders_night3"
  CE_FLAGS+=(--dump_render "$ROOT/renders_night3/${EXP}_${CE_TAG:-s0}" --record_envs 4)
fi

echo "=== CE CONFIG (token3, XLA_OFF=${CE_XLA_OFF:-0} NRENVS=${CE_NRENVS:-64}) ==="
echo "  EXP=$EXP MODEL=$MODEL CLIP_DIR=$CLIP_DIR CLIP=${CE_CLIP:-dance2_subject4.npz} TAG=${CE_TAG:-s0}"
printf '  %s\n' "${CE_FLAGS[@]}"
echo "=== END CE CONFIG ==="

OUTJ="$ROOT/crosseval/${EXP}_${CE_TAG:-s0}.json"
python "$ROOT/crosseval_motion.py" \
  --model_path "$MODEL" \
  --clip_dir "$CLIP_DIR" \
  --clip "${CE_CLIP:-dance2_subject4.npz}" \
  --raw_clip_dir "${CE_RAW_DIR:-$ROOT/clips/LAFAN1}" \
  --refbias "${CE_REFBIAS:-0.0}" \
  --jlat_enc_dim "${CE_JLAT_ENC_DIM:-4}" \
  --nr_envs "${CE_NRENVS:-64}" --steps 1000 \
  ${CE_FLAGS[@]+"${CE_FLAGS[@]}"} \
  --out "$OUTJ"
rc=$?
if [ ! -s "$OUTJ" ]; then echo "ARTIFACT MISSING ($OUTJ), rc=$rc"; exit 1; fi
echo "CROSSEVAL DONE $EXP ${CE_TAG:-s0} rc=$rc"
