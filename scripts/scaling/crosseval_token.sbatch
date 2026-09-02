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
# Crosseval for the TOKEN-ROUTING campaign. A separate file from crosseval2 on
# purpose: the token arms differ in observation LAYOUT, not just in reward, so
# an eval that does not reproduce LATENT_SCOPE / LATENT_DIVISOR / the encoder
# split loads the checkpoint against the wrong channels and still returns a
# number. Every layout knob is passed explicitly here and echoed before the run.
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
# CE_REFHOLD (V1 rate curve): an arm trained against a held reference must be
# scored under the same hold, or the guard's train-vs-eval diff flags it.
CE_FLAGS+=(--reference_hold "${CE_REFHOLD:-1}")
# CE_HEADING_OBS (W1 assembly test): heading obs adds 2 channels to policy AND
# critic -- a mismatch is an observation-width error, not a bad number. Default
# False == crosseval_motion.py's own default, so every older arm is unchanged.
CE_FLAGS+=(--root_heading_obs "${CE_HEADING_OBS:-False}")
CE_FLAGS+=(--anchor "${CE_ANCHOR:-absolute}")
CE_FLAGS+=(--fitvariant "${CE_FITVARIANT:-False}")
CE_FLAGS+=(--refroot "${CE_REFROOT:-True}")
CE_FLAGS+=(--refroot_floor "${CE_REFROOT_FLOOR:-True}")
CE_FLAGS+=(--refvel_obs "${CE_REFVEL_OBS:-False}")
CE_FLAGS+=(--seed "${CE_SEED:-0}")

echo "=== CE CONFIG ==="
echo "  EXP=$EXP MODEL=$MODEL CLIP_DIR=$CLIP_DIR TAG=${CE_TAG:-s0}"
printf '  %s\n' "${CE_FLAGS[@]}"
echo "=== END CE CONFIG ==="

# The separate-encoder arms changed the NETWORK, so the eval has to build the
# same one. joint_latent_encoder_dim reaches the policy through the algorithm
# config, which crosseval_motion.py takes from the algorithm defaults -- pass it
# through EXTRA so a JLAT arm is not evaluated with the shared Dense(4).
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
