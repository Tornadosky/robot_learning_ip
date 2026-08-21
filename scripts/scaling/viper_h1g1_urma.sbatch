#!/bin/bash -l
#SBATCH -p apu
#SBATCH -A mage_apu
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH -c 24
#SBATCH -t 08:00:00
#SBATCH -o /ptmp/akalenik/frontier/h1g1_urma_%j.out

# Corrected two-topology Viper launcher.
# Differences from scripts/scaling/viper_cross_topology.sbatch:
#   - fixed H1+G1 only
#   - no forced ToddlerBot reserve
#   - safe MI300A defaults: <=768 envs and one PPO optimizer epoch
#   - no FSQ/latent/goal-blanking flags
#
# Submit a smoke:
#   sbatch --export=ALL,TAG=h1g1_smoke,MORPHOLOGY=catalog2,TOTAL_TIMESTEPS=8000000 \
#     viper_h1g1_urma.sbatch
#
# Submit production:
#   sbatch --export=ALL,TAG=h1g1_300m_s3,SEED=3,TOTAL_TIMESTEPS=300000000 \
#     viper_h1g1_urma.sbatch

set -Eeuo pipefail

ROOT="${ROOT:-/ptmp/akalenik/frontier}"
REPO="${REPO:-$ROOT/repo}"
BACKBONE="${BACKBONE:-urmav2}"
# Set after implementing the shared body-correct no-FSQ goal adapter.
GOAL_TYPE="${GOAL_TYPE:-}"
TERMINAL_HANDLER="${TERMINAL_HANDLER:-}"
MAX_ROOT_DEVIATION="${MAX_ROOT_DEVIATION:-}"
# absolute | episode_start -- see --root-frame in the trainer.
ROOT_FRAME="${ROOT_FRAME:-absolute}"
# Space-separated families to drive with PD instead of torque, e.g. "g1".
PD_CONTROL="${PD_CONTROL:-}"
PD_GAIN_SCALE="${PD_GAIN_SCALE:-1.0}"
# Radians a PD action of 1.0 commands from nominal. Upstream PDControl spans the
# whole joint range (1.92 rad median on G1) and centres on the range midpoint;
# loco_mjx/urma2 uses 0.25-0.75 rad per actuator. Empty keeps upstream.
PD_ACTION_SCALE="${PD_ACTION_SCALE:-}"
# Segmented training: a checkpoint after each segment, so a timeout or node
# fault costs one segment instead of the whole run. INIT_CHECKPOINT resumes.
SEGMENTS="${SEGMENTS:-1}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"
REFERENCE_GROUNDING="${REFERENCE_GROUNDING:-none}"
ROBOTS="${ROBOTS:-h1 g1}"
# Reset rounds per family used to measure body admission into the manifest.
# 0 removes the only evidence that no bodies were rejected.
ADMISSION_CENSUS_RESETS="${ADMISSION_CENSUS_RESETS:-4}"
# Explicit escape hatch for a labelled stock-goal mismatch baseline only.
ALLOW_STOCK_GOAL_BASELINE="${ALLOW_STOCK_GOAL_BASELINE:-0}"
SEED="${SEED:-3}"
TAG="${TAG:-h1g1_${BACKBONE}_s${SEED}}"

CLIP="${CLIP:-dance2_subject4}"
START_FRAME="${START_FRAME:-19482}"
FRAMES="${FRAMES:-800}"
REFERENCE_MODE="${REFERENCE_MODE:-direct}"
REFERENCE_ROOT="${REFERENCE_ROOT:-$REPO/external_data/cross_humanoid}"

MORPHOLOGY="${MORPHOLOGY:-continuous}"
REWARD_TYPE="${REWARD_TYPE:-MorphMimicReward}"
TOTAL_ENVS="${TOTAL_ENVS:-768}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-300000000}"
NUM_STEPS="${NUM_STEPS:-64}"
NUM_MINIBATCHES="${NUM_MINIBATCHES:-16}"
UPDATE_EPOCHS="${UPDATE_EPOCHS:-1}"
HIDDEN_1="${HIDDEN_1:-256}"
HIDDEN_2="${HIDDEN_2:-128}"
LR="${LR:-1e-4}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"

[[ "$BACKBONE" == "urma" || "$BACKBONE" == "urmav2" ]] || {
  echo "BACKBONE must be urma or urmav2" >&2
  exit 2
}
(( TOTAL_ENVS <= 768 )) || {
  echo "Viper safety gate: TOTAL_ENVS=$TOTAL_ENVS exceeds the recorded 768-env ceiling" >&2
  exit 2
}
(( UPDATE_EPOCHS == 1 )) || {
  echo "Viper safety gate: URMA/URMAv2 must use UPDATE_EPOCHS=1 on this ROCm path" >&2
  exit 2
}
if (( TOTAL_TIMESTEPS > 8000000 )) && [[ "$ALLOW_STOCK_GOAL_BASELINE" != "1" ]]; then
  [[ "$GOAL_TYPE" == "MorphGoalTrajMimicRootErr" ]] || {
    echo "Long run blocked: export GOAL_TYPE=MorphGoalTrajMimicRootErr after implementing the shared body-correct goal." >&2
    exit 2
  }
  [[ "$TERMINAL_HANDLER" == "MorphologyAwareRootPoseTrajTerminalStateHandler" || \
     "$TERMINAL_HANDLER" == "GravityRootPoseTrajTerminalStateHandler" ]] || {
    echo "Long run blocked: export TERMINAL_HANDLER=MorphologyAwareRootPoseTrajTerminalStateHandler (or the Gravity subclass)." >&2
    exit 2
  }
  [[ -n "$MAX_ROOT_DEVIATION" ]] || {
    echo "Long run blocked: export MAX_ROOT_DEVIATION (initial experiment value: 0.5)." >&2
    exit 2
  }
fi

source "$ROOT/venv/bin/activate"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
export JAX_COMPILATION_CACHE_DIR="$ROOT/xla_cache"
export PYTHONPATH="$REPO/scripts"
export MUJOCO_GL=disable

cd "$REPO"

for robot in $ROBOTS; do
  ref="$REFERENCE_ROOT/h1_source/$CLIP/$robot/start${START_FRAME}_${FRAMES}f_${REFERENCE_MODE}.npz"
  [[ -f "$ref" ]] || {
    echo "missing reference: $ref" >&2
    exit 3
  }
done

OUT="$ROOT/crosstopo_${TAG}_${SLURM_JOB_ID}"
mkdir -p "$OUT"

{
  echo "timestamp=$(date -Is)"
  echo "job=$SLURM_JOB_ID node=$SLURMD_NODENAME"
  echo "out=$OUT"
  echo "backbone=$BACKBONE robots=h1,g1 morphology=$MORPHOLOGY"
  echo "envs=$TOTAL_ENVS timesteps=$TOTAL_TIMESTEPS update_epochs=$UPDATE_EPOCHS"
  echo "clip=$CLIP start=$START_FRAME frames=$FRAMES"
  echo
  # The Viper copy is an rsync'd tree, not a checkout. Under `set -e` inside a
  # pipefail pipeline a failing `git rev-parse` aborts the whole job before
  # training starts, so provenance is best-effort by design.
  echo "== git revision =="
  git rev-parse HEAD || echo "(not a git checkout; see source hashes below)"
  echo
  echo "== git status --short =="
  git status --short || true
  echo
  echo "== submodules =="
  git submodule status || true
  echo
  echo "== source hashes =="
  sha256sum \
    scripts/scaling/parallel_cross_humanoid_train.py \
    scripts/scaling/parallel_env.py \
    scripts/scaling/family_morphology.py \
    scripts/scaling/urma_networks.py \
    scripts/h1md/morph_mimic_reward.py
} | tee "$OUT/launch_provenance.txt"

ARGS=(
  --backbone "$BACKBONE"
  --robots $ROBOTS
  --source h1
  --clip "$CLIP"
  --start-frame "$START_FRAME"
  --frames "$FRAMES"
  --reference-mode "$REFERENCE_MODE"
  --reference-root "$REFERENCE_ROOT"
  --morphology "$MORPHOLOGY"
  --reward-type "$REWARD_TYPE"
  --no-robot-one-hot
  --append-joint-features
  --learnable-std
  --total-envs "$TOTAL_ENVS"
  --total-timesteps "$TOTAL_TIMESTEPS"
  --num-steps "$NUM_STEPS"
  --num-minibatches "$NUM_MINIBATCHES"
  --update-epochs "$UPDATE_EPOCHS"
  --hidden "$HIDDEN_1" "$HIDDEN_2"
  --lr "$LR"
  --seed "$SEED"
  --run-tag "viper_${TAG}_${SLURM_JOB_ID}"
  --output-dir "$OUT"
  --admission-census-resets "$ADMISSION_CENSUS_RESETS"
  --root-frame "$ROOT_FRAME"
  --segments "$SEGMENTS"
  --reference-grounding "$REFERENCE_GROUNDING"
)
if [[ -n "$INIT_CHECKPOINT" ]]; then
  ARGS+=(--init-checkpoint "$INIT_CHECKPOINT")
fi
if [[ -n "$PD_CONTROL" ]]; then
  # shellcheck disable=SC2206
  ARGS+=(--pd-control $PD_CONTROL --pd-gain-scale "$PD_GAIN_SCALE")
  if [[ -n "$PD_ACTION_SCALE" ]]; then
    ARGS+=(--pd-action-scale "$PD_ACTION_SCALE")
  fi
fi
if [[ -n "$GOAL_TYPE" ]]; then
  ARGS+=(--goal-type "$GOAL_TYPE")
fi
if [[ -n "$TERMINAL_HANDLER" ]]; then
  ARGS+=(--terminal-handler "$TERMINAL_HANDLER")
fi
if [[ -n "$MAX_ROOT_DEVIATION" ]]; then
  ARGS+=(--max-root-deviation "$MAX_ROOT_DEVIATION")
fi

if [[ -n "${N_SUBSTEPS:-}" ]]; then
  ARGS+=(--n-substeps "$N_SUBSTEPS")
fi
if [[ -n "${HORIZON:-}" ]]; then
  ARGS+=(--env-horizon "$HORIZON")
fi
if [[ "${JOINT_TARGET_OBS:-0}" == "1" ]]; then
  ARGS+=(--joint-target-obs)
fi
if [[ -n "${ROOT_ROT_MARGIN:-}" ]]; then
  ARGS+=(--root-rot-margin-degrees "$ROOT_ROT_MARGIN")
fi
if [[ -n "${TILT_DEG:-}" ]]; then
  ARGS+=(--terminal-tilt-degrees "$TILT_DEG")
fi

if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
  ARGS+=(--preflight-only)
fi

printf '[launch]'
printf ' %q' python scripts/scaling/parallel_cross_humanoid_train.py "${ARGS[@]}"
printf '\n'

python scripts/scaling/parallel_cross_humanoid_train.py "${ARGS[@]}"

echo "H1G1_URMA_DONE tag=$TAG job=$SLURM_JOB_ID out=$OUT"
