#!/bin/bash
#SBATCH -J urma_eval
#SBATCH -p apu
#SBATCH -A mage_apu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --ntasks-per-core=1
#SBATCH --mem=108000
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH -o /ptmp/akalenik/urma/logs/%x_%j.out
#SBATCH -e /ptmp/akalenik/urma/logs/%x_%j.out
#
# Held-out zero-shot eval on MI300A.
#
# Runs on Viper rather than locally for one specific reason: eval_heldout.py
# peaks around 14.5 GB RSS while compiling single-robot envs, which OOM-kills a
# concurrent local trainer on the 24 GB WSL box. Viper has 108 GB and is idle
# while the local GPU extends Phase 0, so the two deliverables proceed at once.
#
# Env vars: MODEL, ROBOTS, OUT, NR_ENVS, STEPS, TERRAIN, CRITIC_EXTERO, TRACKING

module load python-waterboa/2025.06
eval "$(conda shell.bash hook)"
conda activate "/ptmp/akalenik/urma/${VENV:-env}"

ROOT=/ptmp/akalenik/urma
mkdir -p "$ROOT/logs"
cd "$ROOT"

export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Compute nodes have no /dev/dri; EGL cannot init headless and dm_control raises
# at import. eval_heldout.py uses setdefault, so this wins.
export MUJOCO_GL=disable
export JAX_ENABLE_COMPILATION_CACHE=true
export JAX_COMPILATION_CACHE_DIR="$ROOT/jax_cache"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1

MODEL="${MODEL:-$ROOT/phase0fast_100M.model}"
ROBOTS="${ROBOTS:-unitree_h1,unitree_g1,booster_t1,robotis_op3,mit_humanoid,berkeley_humanoid}"
OUT="${OUT:-$ROOT/logs/heldout_eval.json}"
NR_ENVS="${NR_ENVS:-64}"
STEPS="${STEPS:-1000}"
# Must mirror the training run: these two set the observation size, so a
# mismatch silently misaligns the loaded parameters.
TERRAIN="${TERRAIN:-plane}"
CRITIC_EXTERO="${CRITIC_EXTERO:-none}"

echo "NODE=$(hostname) MODEL=$MODEL ROBOTS=$ROBOTS"
python -c "import jax; print('devices:', jax.devices())"

EXTRA=()
[ -n "${TRACKING:-}" ] && EXTRA=(--tracking)

python "$ROOT/eval_heldout.py" \
  --model_path "$MODEL" \
  --robots "$ROBOTS" \
  --nr_envs "$NR_ENVS" \
  --steps "$STEPS" \
  --terrain "$TERRAIN" \
  --critic_extero "$CRITIC_EXTERO" \
  "${EXTRA[@]}" \
  --out "$OUT"

echo "EVAL EXITED rc=$?"
