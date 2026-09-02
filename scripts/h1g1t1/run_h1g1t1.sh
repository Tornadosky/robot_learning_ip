#!/usr/bin/env bash
# One shared URMA2 policy: Unitree H1 + Unitree G1 + Booster T1, one dance clip.
set -Eeuo pipefail

STAGE=${1:-full}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-$(cd "$SCRIPT_DIR/../.." && pwd)}
PY=${PY:-python}
CLIPS=${CLIPS:-$REPO/external_data/amass_converted/LAFAN1}
CLIP_FILE=${CLIP_FILE:-dance2_subject4.npz}
RUN_ID=${RUN_ID:-h1g1t1_dance_$(date +%Y%m%d_%H%M%S)}
PROJECT=${PROJECT:-h1g1t1_debug}
EXP_NAME=${EXP_NAME:-dance2_h1g1t1}
SEED=${SEED:-0}

NR_ENVS=${NR_ENVS:-576}
NR_STEPS=${NR_STEPS:-64}
MINIBATCH=${MINIBATCH:-6144}
NR_EPOCHS=${NR_EPOCHS:-5}
SAVE_EVERY=${SAVE_EVERY:-11796480}
TOTAL=${TOTAL:-117964800}
MORPH_START=${MORPH_START:-0.0}
MORPH_MAX=${MORPH_MAX:-0.3}
MORPH_RAMP=${MORPH_RAMP:-39321600}
TORQUE_EXP=${TORQUE_EXP:-4.0}
EXACT_INERTIA=${EXACT_INERTIA:-True}
BODY_POOL_SIZE=${BODY_POOL_SIZE:-0}
JOINT_TEMP=${JOINT_TEMP:-0.10}
HEADING_RATIO=${HEADING_RATIO:-0.75}
HEADING_TEMP=${HEADING_TEMP:-2.0}

EVAL_ENVS=${EVAL_ENVS:-96}
EVAL_STEPS=${EVAL_STEPS:-1000}
EVAL_BODY_POOL_SIZE=${EVAL_BODY_POOL_SIZE:-128}
INCLUDE_MODEL=${INCLUDE_MODEL:-0}
DUMP_ROLLOUT=${DUMP_ROLLOUT:-1}
RECORD_ENVS=${RECORD_ENVS:-2}

if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "RUN_ID may contain only letters, digits, dot, underscore, and dash" >&2
  exit 2
fi

RESULT_ROOT=$REPO/experiments/h1g1t1_debug
RESULT_DIR=$RESULT_ROOT/$RUN_ID
RUNNER_DIR=$REPO/loco_mjx/experiments/runs/$PROJECT/$EXP_NAME/$RUN_ID
TRAIN_LOG=$RESULT_DIR/train.log
PACKAGE=$RESULT_ROOT/${RUN_ID}_diagnostics.zip

# The monolithic urma2.mjx trainer is a single-device path. Pin one visible
# accelerator by default so multi-GPU workstations cannot change batch geometry.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
if [[ "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
  echo "This urma2.mjx launcher requires exactly one accelerator; CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
  exit 2
fi
export PYTHONPATH="$REPO:$REPO/RL-X:$REPO/loco_mjx${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=${MUJOCO_GL:-disable}
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}
export XLA_FLAGS=${XLA_FLAGS:---xla_gpu_enable_command_buffer=}

WORLD=(
  --runner.mode=train
  --environment.name=locomotion.urma2.mjx
  --algorithm.name=urma2.mjx
  --environment.train_robots="('unitree_h1','unitree_g1','booster_t1')"
  --environment.eval_robots="()"
  --environment.nr_envs_mode=global
  --environment.nr_envs="$NR_ENVS"
  --environment.nr_eval_envs=0
  --environment.seed="$SEED"
  --environment.terrain.type=plane
  --environment.critic_exteroceptive_observation_type=none
  --environment.terrain.contact_solref_timeconst=0.0
  --environment.command.type=tracking_clip
  --environment.reward.type=tracking
  --environment.reward.log_info=True
  --environment.command.tracking_clip_dir="$CLIPS"
  --environment.command.tracking_clip_file="$CLIP_FILE"
  --environment.command.tracking_clip_fit_per_variant=False
  --environment.command.tracking_clip_anchor=absolute
  --environment.command.tracking_clip_velocity_command=True
  --environment.command.tracking_clip_cyclic=False
  --environment.command.tracking_clip_amplitude_scale=1.0
  --environment.command.tracking_clip_velocity_scale=1.0
  --environment.command.tracking_clip_observe_velocity=False
  --environment.command.tracking_clip_observe_root_heading=True
  --environment.command.tracking_clip_root_height_from_pose=True
  --environment.command.tracking_clip_root_height_pose_as_floor=True
  --environment.command.tracking_reference_action_bias=0.0
  --environment.command.tracking_clip_reference_hold=1
  --environment.command.tracking_clip_latent_hold=1
  --environment.reward.nominal_diff_target=reference
  --environment.reward.joint_tracking_coeff=30.0
  --environment.reward.joint_tracking_temperature="$JOINT_TEMP"
  --environment.reward.tracking_curriculum_gated=False
  --environment.reward.gait_coeff_mode=floor
  --environment.reward.gait_coeff_value=0.25
  --environment.reward.action_rate_coeff=3.0
  --environment.reward.action_smoothness_coeff=0.1
  --environment.reward.deepmimic_enabled=True
  --environment.reward.deepmimic_qvel_temperature=10.0
  --environment.reward.deepmimic_foot_height_weight_ratio=1.0
  --environment.reward.deepmimic_foot_height_temperature=0.01
  --environment.reward.foot_z_velocity_coeff=1.0
  --environment.reward.root_heading_tracking_weight_ratio="$HEADING_RATIO"
  --environment.reward.root_heading_tracking_temperature="$HEADING_TEMP"
  --environment.termination.tracking_deviation_ratio=0.0
  --environment.reward.tracking_post_contact_penalties=True
  --environment.reward.foot_slip_coeff=20.0
  --environment.reward.ground_penetration_coeff=1000.0
  --environment.domain_randomization.initial_state.type=reference
  --environment.domain_randomization.sampling_type=step_probability_and_reset
  --environment.domain_randomization.sampling_probability=0.0
  --environment.domain_randomization.seen_robot.morphology_coeff_mode=schedule
  --environment.domain_randomization.seen_robot.morphology_coeff_start="$MORPH_START"
  --environment.domain_randomization.seen_robot.morphology_coeff_value="$MORPH_MAX"
  --environment.domain_randomization.seen_robot.morphology_coeff_ramp_steps="$MORPH_RAMP"
  --environment.domain_randomization.seen_robot.torque_scaling_exponent="$TORQUE_EXP"
  --environment.domain_randomization.seen_robot.exact_inertia_rescale="$EXACT_INERTIA"
  --environment.domain_randomization.seen_robot.body_pool_size="$BODY_POOL_SIZE"
  --environment.env_curriculum_coeff_max=0.6
  --environment.env_curriculum_level_success_tracking_ratio=0.0
  --algorithm.nr_steps="$NR_STEPS"
  --algorithm.minibatch_size="$MINIBATCH"
  --algorithm.nr_epochs="$NR_EPOCHS"
  --algorithm.total_timesteps="$TOTAL"
  --algorithm.evaluation_active=False
  --algorithm.evaluation_and_save_frequency="$SAVE_EVERY"
  --algorithm.save_intermediate_models=True
  --environment.render=False
  --runner.track_console=True
  --runner.track_tb=True
  --runner.track_wandb=False
  --runner.save_model=True
  --runner.project_name="$PROJECT"
  --runner.exp_name="$EXP_NAME"
  --runner.run_name="$RUN_ID"
  --runner.notes="one shared H1+G1+T1 DeepMimic dance policy; heading observable; morphology schedule"
)
TRAIN_COMMAND=("$PY" experiment.py "${WORLD[@]}")

print_command() {
  printf 'cd %q\n' "$REPO/loco_mjx/experiments"
  printf 'CUDA_VISIBLE_DEVICES=%q PYTHONPATH=%q MUJOCO_GL=%q XLA_PYTHON_CLIENT_PREALLOCATE=%q XLA_FLAGS=%q \\\n  ' "$CUDA_VISIBLE_DEVICES" "$PYTHONPATH" "$MUJOCO_GL" "$XLA_PYTHON_CLIENT_PREALLOCATE" "$XLA_FLAGS"
  printf '%q ' "${TRAIN_COMMAND[@]}"
  printf '\n'
}

write_recipe() {
  mkdir -p "$RESULT_DIR/commands" "$RESULT_DIR/environment" "$RESULT_DIR/eval"
  "$PY" - "$RESULT_DIR/recipe.json" <<PY
import json, sys
payload = {
  "run_id": "$RUN_ID", "robots": ["unitree_h1", "unitree_g1", "booster_t1"],
  "clip_dir": "$CLIPS", "clip_file": "$CLIP_FILE", "seed": int("$SEED"),
  "nr_envs": int("$NR_ENVS"), "nr_steps": int("$NR_STEPS"),
  "minibatch_size": int("$MINIBATCH"), "nr_epochs": int("$NR_EPOCHS"),
  "save_every": int("$SAVE_EVERY"), "total_timesteps": int("$TOTAL"),
  "morphology": {"start": float("$MORPH_START"), "maximum": float("$MORPH_MAX"),
                 "ramp_steps": int("$MORPH_RAMP"), "torque_scaling_exponent": float("$TORQUE_EXP"),
                 "exact_inertia_rescale": "$EXACT_INERTIA" == "True", "body_pool_size": int("$BODY_POOL_SIZE")},
  "heading": {"observed": True, "reward_ratio": float("$HEADING_RATIO"),
              "temperature": float("$HEADING_TEMP")},
  "joint_tracking_temperature": float("$JOINT_TEMP"),
  "cuda_visible_devices": "$CUDA_VISIBLE_DEVICES",
  "evaluation": {"nr_envs": int("$EVAL_ENVS"), "steps": int("$EVAL_STEPS"),
                 "body_pool_size": int("$EVAL_BODY_POOL_SIZE"),
                 "dump_nominal_rollout": int("$DUMP_ROLLOUT") == 1,
                 "record_envs": int("$RECORD_ENVS")},
  "runner_dir": "$RUNNER_DIR", "result_dir": "$RESULT_DIR"
}
open(sys.argv[1], "w").write(json.dumps(payload, indent=2))
PY
  print_command > "$RESULT_DIR/commands/train_command.txt"
}

capture_environment() {
  mkdir -p "$RESULT_DIR/environment"
  "$PY" -VV > "$RESULT_DIR/environment/python.txt" 2>&1 || true
  "$PY" -m pip freeze > "$RESULT_DIR/environment/pip_freeze.txt" 2>&1 || true
  nvidia-smi > "$RESULT_DIR/environment/nvidia_smi.txt" 2>&1 || true
  uname -a > "$RESULT_DIR/environment/uname.txt" 2>&1 || true
  if [[ -d "$REPO/.git" ]]; then
    git -C "$REPO" rev-parse HEAD > "$RESULT_DIR/environment/git_commit.txt" 2>&1 || true
    git -C "$REPO" status --short > "$RESULT_DIR/environment/git_status.txt" 2>&1 || true
    git -C "$REPO" diff --binary > "$RESULT_DIR/environment/git_diff.patch" 2>&1 || true
  fi
}

run_preflight() {
  write_recipe
  capture_environment
  "$PY" "$REPO/scripts/h1g1t1/preflight.py" \
    --repo "$REPO" --clip-dir "$CLIPS" --clip-file "$CLIP_FILE" \
    --nr-envs "$NR_ENVS" --nr-steps "$NR_STEPS" --minibatch-size "$MINIBATCH" \
    --save-every "$SAVE_EVERY" --total-timesteps "$TOTAL" \
    --out "$RESULT_DIR/preflight.json" | tee "$RESULT_DIR/preflight.log"
}

verify_training() {
  local rc=$1
  if [[ $rc -ne 0 ]]; then
    echo "Training process exited with status $rc" >&2
    return "$rc"
  fi
  if grep -Eqi 'Uncaught exception|Traceback \(most recent call last\)|CUDA_ERROR|RESOURCE_EXHAUSTED|out of memory' "$TRAIN_LOG"; then
    echo "Training log contains an exception/fatal marker; see $TRAIN_LOG" >&2
    grep -Ein 'Uncaught exception|Traceback \(most recent call last\)|CUDA_ERROR|RESOURCE_EXHAUSTED|out of memory' "$TRAIN_LOG" | tail -30 >&2
    return 1
  fi
  local latest=$RUNNER_DIR/models/latest.model
  if [[ ! -s "$latest" ]]; then
    echo "Missing final checkpoint: $latest" >&2
    return 1
  fi
  local last_step
  last_step=$("$PY" - "$TRAIN_LOG" "$REPO" <<'PY'
import sys
sys.path.insert(0, sys.argv[2])
from scripts.h1g1t1.diagnostics import parse_console_log
frame = parse_console_log(sys.argv[1])
print(int(frame["steps/nr_env_steps"].iloc[-1]))
PY
)
  echo "$last_step" > "$RESULT_DIR/final_logged_step.txt"
  if (( last_step != TOTAL )); then
    echo "Training ended at $last_step steps; expected exactly $TOTAL" >&2
    return 1
  fi
}

run_train() {
  run_preflight
  mkdir -p "$RESULT_DIR"
  echo "[train] run=$RUN_ID result=$RESULT_DIR runner=$RUNNER_DIR" | tee "$RESULT_DIR/status.log"
  cd "$REPO/loco_mjx/experiments"
  set +e
  "${TRAIN_COMMAND[@]}" 2>&1 | tee "$TRAIN_LOG"
  local rc=${PIPESTATUS[0]}
  set -e
  verify_training "$rc"
  echo "[train] verified complete $(date -Is)" | tee -a "$RESULT_DIR/status.log"
}

run_one_eval() {
  local label=$1
  local coeff=$2
  local zero_action=${3:-0}
  local model=$RUNNER_DIR/models/latest.model
  local out=$RESULT_DIR/eval/${label}.json
  local log=$RESULT_DIR/eval/${label}.log
  local args=(
    --model_path "$model"
    --clip_dir "$CLIPS" --raw_clip_dir "$CLIPS" --clip "$CLIP_FILE"
    --robots unitree_h1:unitree_g1:booster_t1
    --refbias 0.0 --anchor absolute --fitvariant False
    --cyclic False --refroot True --refroot_floor True --refvel_obs False
    --root_heading_obs True --reference_hold 1
    --morphology_coeff "$coeff" --torque_scaling_exponent "$TORQUE_EXP"
    --exact_inertia_rescale "$EXACT_INERTIA" --body_pool_size "$EVAL_BODY_POOL_SIZE"
    --nr_envs "$EVAL_ENVS" --steps "$EVAL_STEPS" --seed "$SEED"
    --out "$out"
  )
  if [[ "$zero_action" == 1 ]]; then
    args+=(--zero_action)
  fi
  if [[ "$label" == "policy_nominal" && "$DUMP_ROLLOUT" == 1 ]]; then
    args+=(--dump_render "$RESULT_DIR/eval/policy_nominal_rollout" --record_envs "$RECORD_ENVS")
  fi
  printf '%q ' "$PY" "$REPO/experiments/fsq_khaendler/crosseval_motion.py" "${args[@]}" \
    > "$RESULT_DIR/commands/eval_${label}_command.txt"
  printf '\n' >> "$RESULT_DIR/commands/eval_${label}_command.txt"
  "$PY" "$REPO/experiments/fsq_khaendler/crosseval_motion.py" "${args[@]}" 2>&1 | tee "$log"
  [[ -s "$out" ]] || { echo "Missing crosseval artifact: $out" >&2; return 1; }
  if grep -Eqi 'Uncaught exception|Traceback \(most recent call last\)' "$log"; then
    echo "Crosseval $label logged an exception" >&2
    return 1
  fi
}

run_evaluate() {
  [[ -s "$RUNNER_DIR/models/latest.model" ]] || {
    echo "Checkpoint not found: $RUNNER_DIR/models/latest.model" >&2
    exit 1
  }
  mkdir -p "$RESULT_DIR/eval" "$RESULT_DIR/commands"
  run_one_eval policy_nominal 0.0 0
  run_one_eval policy_morph_015 0.15 0
  run_one_eval policy_morph_030 0.30 0
  run_one_eval zero_action_nominal 0.0 1
}

run_report() {
  [[ -s "$TRAIN_LOG" ]] || { echo "Training log not found: $TRAIN_LOG" >&2; exit 1; }
  "$PY" "$REPO/scripts/h1g1t1/diagnostics.py" report \
    --log "$TRAIN_LOG" --out-dir "$RESULT_DIR/diagnostics" \
    --crosseval "$RESULT_DIR/eval" --recipe-json "$RESULT_DIR/recipe.json"
}

mirror_runner_artifacts() {
  [[ -d "$RUNNER_DIR" ]] || return 0
  rm -rf "$RESULT_DIR/runner"
  mkdir -p "$RESULT_DIR/runner"
  while IFS= read -r -d '' source; do
    local rel=${source#"$RUNNER_DIR"/}
    mkdir -p "$RESULT_DIR/runner/$(dirname "$rel")"
    ln -s "$source" "$RESULT_DIR/runner/$rel"
  done < <(find "$RUNNER_DIR" -type f -print0)
}

run_package() {
  mirror_runner_artifacts
  local flags=()
  [[ "$INCLUDE_MODEL" == 1 ]] && flags+=(--include-model)
  "$PY" "$REPO/scripts/h1g1t1/collect_results.py" \
    --result-dir "$RESULT_DIR" --out "$PACKAGE" "${flags[@]}"
  sha256sum "$PACKAGE" > "$PACKAGE.sha256"
  echo "RESULT_PACKAGE=$PACKAGE"
}

run_smoke() {
  local smoke_id=${RUN_ID}_smoke
  RUN_ID="$smoke_id" NR_ENVS=12 NR_STEPS=16 MINIBATCH=192 \
    SAVE_EVERY=24576 TOTAL=24576 MORPH_RAMP=12288 MORPH_MAX=0.05 \
    bash "$0" train
}

package_failure_artifact() {
  local rc=$?
  trap - EXIT
  if (( rc == 0 )); then
    exit 0
  fi
  if [[ "$STAGE" != "train" && "$STAGE" != "full" ]]; then
    exit "$rc"
  fi

  set +e
  mkdir -p "$RESULT_DIR"
  printf '%s\n' "$rc" > "$RESULT_DIR/failure_exit_code.txt"
  printf '[failure] stage=%s exit_code=%s time=%s\n' \
    "$STAGE" "$rc" "$(date -Is)" | tee -a "$RESULT_DIR/status.log" >&2
  if run_package; then
    echo "FAILURE_RESULT_PACKAGE=$PACKAGE" >&2
  else
    echo "Could not build the partial diagnostic ZIP; preserve $RESULT_DIR directly." >&2
  fi
  exit "$rc"
}

trap package_failure_artifact EXIT

case "$STAGE" in
  dry-run)
    "$PY" -c "import sys; sys.path.insert(0, '$REPO'); from scripts.h1g1t1.diagnostics import validate_recipe; print(validate_recipe($NR_ENVS,$NR_STEPS,$MINIBATCH,$SAVE_EVERY,$TOTAL,3))"
    print_command
    ;;
  preflight) run_preflight ;;
  smoke) run_smoke ;;
  train) run_train ;;
  evaluate) run_evaluate ;;
  report) run_report ;;
  package) run_package ;;
  full)
    run_train
    run_evaluate
    run_report
    run_package
    ;;
  *)
    echo "Usage: $0 {dry-run|preflight|smoke|train|evaluate|report|package|full}" >&2
    exit 2
    ;;
esac
