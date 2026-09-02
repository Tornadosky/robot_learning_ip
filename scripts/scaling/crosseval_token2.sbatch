#!/bin/bash
#SBATCH -J ce_token
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
# NIGHT3 crosseval: crosseval_token.sbatch + three additions, defaults chosen so
# an env-var-free submission is bit-identical to the old file:
#   CE_CONTACT_TIMECONST  terrain.contact_solref_timeconst for the EVAL world.
#                         A 0.004 training arm scored in a 0.0 world is a
#                         train-vs-eval physics diff on the exact metrics
#                         (airborne/slip/pen) night3 exists to measure.
#   CE_HOLD               --latent_hold (token staleness at eval; A1 sweep).
#                         Was silently defaulted before; now settable.
#   CE_DUMP=1             also dump render npz-s to renders_night3/ for the
#                         local rf_render pass (the "does it LOOK like dance"
#                         question needs pixels, not RMSE).
set -u
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

CE_FLAGS=()
[ "${CE_LATENT:-0}" = "1" ] && CE_FLAGS+=(--latent)
CE_FLAGS+=(--latent_dim "${CE_LATENT_DIM:-32}")
CE_FLAGS+=(--latent_hold "${CE_HOLD:-1}")
CE_FLAGS+=(--latent_replaces "${CE_REPLACES:-False}")
CE_FLAGS+=(--latent_scope "${CE_SCOPE:-per_joint}")
CE_FLAGS+=(--latent_divisor "${CE_DIVISOR:-1.0}")
CE_FLAGS+=(--reference_hold "${CE_REFHOLD:-1}")
CE_FLAGS+=(--root_heading_obs "${CE_HEADING_OBS:-False}")
CE_FLAGS+=(--contact_timeconst "${CE_CONTACT_TIMECONST:-0.0}")
# CE_ROBOTS (night4 T1 arms): the eval must build the same robot set as the
# checkpoint or the obs layout is wrong. Default = the historical H1+G1 pair.
CE_FLAGS+=(--robots "${CE_ROBOTS:-unitree_h1:unitree_g1}")
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

echo "=== CE CONFIG ==="
echo "  EXP=$EXP MODEL=$MODEL CLIP_DIR=$CLIP_DIR CLIP=${CE_CLIP:-dance2_subject4.npz} TAG=${CE_TAG:-s0}"
printf '  %s\n' "${CE_FLAGS[@]}"
echo "=== END CE CONFIG ==="

python "$ROOT/crosseval_motion.py" \
  --model_path "$MODEL" \
  --clip_dir "$CLIP_DIR" \
  --clip "${CE_CLIP:-dance2_subject4.npz}" \
  --raw_clip_dir "${CE_RAW_DIR:-$ROOT/clips/LAFAN1}" \
  --refbias "${CE_REFBIAS:-0.0}" \
  --jlat_enc_dim "${CE_JLAT_ENC_DIM:-4}" \
  --nr_envs 64 --steps 1000 \
  ${CE_FLAGS[@]+"${CE_FLAGS[@]}"} \
  --out "$ROOT/crosseval/${EXP}_${CE_TAG:-s0}.json"
echo "CROSSEVAL DONE $EXP ${CE_TAG:-s0}"
