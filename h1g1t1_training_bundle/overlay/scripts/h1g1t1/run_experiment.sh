#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
STAGE=${1:-all}
PROFILE=${PROFILE:-probe}
REPO=${REPO:-}
PYTHON=${PYTHON:-python}
CLIP=${CLIP:-dance2_subject4.npz}
SEED=${SEED:-0}
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)_${PROFILE}_s${SEED}}
ALLOW_CPU=${ALLOW_CPU:-0}
INCLUDE_MODEL=${INCLUDE_MODEL:-0}
SKIP_EVAL=${SKIP_EVAL:-0}

if [[ -z "$REPO" ]]; then
  echo "REPO is required, e.g. REPO=/path/to/robot_learning_ip" >&2
  exit 2
fi
REPO=$(cd "$REPO" && pwd)
PYTHON=$(realpath -m "${PYTHON/#\~/$HOME}")
CLIP_DIR=${CLIP_DIR:-$REPO/external_data/amass_converted/LAFAN1}
RUN_DIR=${RUN_DIR:-$REPO/experiments/h1g1t1_runs/$RUN_ID}
MODEL_RUN_DIR=$REPO/loco_mjx/experiments/runs/h1g1t1/$PROFILE/$RUN_ID
MODEL=$MODEL_RUN_DIR/models/latest.model
mkdir -p "$RUN_DIR" "$RUN_DIR/config" "$RUN_DIR/logs" "$RUN_DIR/metrics" "$RUN_DIR/plots" "$RUN_DIR/evaluations"

PROFILE_JSON=$RUN_DIR/config/profile.json
"$PYTHON" "$SCRIPT_DIR/profiles.py" "$PROFILE" > "$PROFILE_JSON"
eval "$("$PYTHON" "$SCRIPT_DIR/profiles.py" "$PROFILE" --shell)"

export PYTHONPATH="$REPO:$REPO/RL-X:$REPO/loco_mjx${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=${MUJOCO_GL:-disable}
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}
export XLA_FLAGS=${XLA_FLAGS:---xla_gpu_enable_command_buffer=}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

if [[ "$CUDA_VISIBLE_DEVICES" == *,* && "${ALLOW_MULTI_GPU:-0}" != 1 ]]; then
  echo "This profile is validated for one visible GPU. Set CUDA_VISIBLE_DEVICES to one device, or ALLOW_MULTI_GPU=1 deliberately." >&2
  exit 2
fi

WORLD=(
  --environment.name=locomotion.urma2.mjx
  --algorithm.name=urma2.mjx
  --environment.seed="$SEED"
  --environment.nr_envs_mode=per_device
  --environment.nr_envs="$TOTAL_ENVS"
  --environment.train_robots="('unitree_h1','unitree_g1','booster_t1')"
  --environment.eval_robots="()"
  --environment.terrain.type=plane
  --environment.critic_exteroceptive_observation_type=none
  --environment.policy_exteroceptive_observation_type=none
  --environment.terrain.contact_solref_timeconst=0.0
  --environment.command.type=tracking_clip
  --environment.command.tracking_clip_dir="$CLIP_DIR"
  --environment.command.tracking_clip_file="$CLIP"
  --environment.command.tracking_clip_fit_per_variant=False
  --environment.command.tracking_clip_anchor=absolute
  --environment.command.tracking_clip_velocity_command=True
  --environment.command.tracking_clip_command_cap_scale=8.0
  --environment.command.tracking_clip_cyclic=False
  --environment.command.tracking_clip_amplitude_scale=1.0
  --environment.command.tracking_clip_velocity_scale=1.0
  --environment.command.tracking_clip_observe_velocity=True
  --environment.command.tracking_clip_observe_root_heading=True
  --environment.command.tracking_clip_root_height_from_pose=True
  --environment.command.tracking_clip_root_height_pose_as_floor=True
  --environment.command.tracking_reference_action_bias=1.0
  --environment.command.tracking_clip_latent_hold=1
  --environment.command.tracking_clip_reference_hold=1
  --environment.reward.type=tracking
  --environment.reward.log_info=True
  --environment.reward.nominal_diff_target=reference
  --environment.reward.joint_tracking_coeff=30.0
  --environment.reward.joint_tracking_temperature=0.05
  --environment.reward.tracking_curriculum_gated=False
  --environment.reward.root_height_tracking_coeff=1.0
  --environment.reward.root_height_tracking_temperature=0.01
  --environment.reward.root_heading_tracking_weight_ratio=0.75
  --environment.reward.root_heading_tracking_temperature=0.25
  --environment.reward.root_heading_tracking_kernel=cosine
  --environment.reward.deepmimic_heading_free=True
  --environment.reward.deepmimic_enabled=True
  --environment.reward.deepmimic_qvel_weight_ratio=0.5
  --environment.reward.deepmimic_rpos_weight_ratio=1.25
  --environment.reward.deepmimic_rquat_weight_ratio=0.75
  --environment.reward.deepmimic_qvel_temperature=0.5
  --environment.reward.deepmimic_rpos_temperature=0.01
  --environment.reward.deepmimic_rquat_temperature=0.1
  --environment.reward.deepmimic_foot_height_weight_ratio=0.5
  --environment.reward.deepmimic_foot_height_temperature=0.01
  --environment.reward.gait_coeff_mode=floor
  --environment.reward.gait_coeff_value=0.25
  --environment.reward.action_rate_coeff=3.0
  --environment.reward.action_smoothness_coeff=0.1
  --environment.reward.foot_z_velocity_coeff=1.0
  --environment.reward.tracking_post_contact_penalties=True
  --environment.reward.foot_slip_coeff=20.0
  --environment.reward.ground_penetration_coeff=1000.0
  --environment.termination.tracking_deviation_ratio=1.0
  --environment.domain_randomization.initial_state.type=reference
  --environment.domain_randomization.seen_robot.morphology_coeff_mode=schedule
  --environment.domain_randomization.seen_robot.morphology_coeff_start=0.0
  --environment.domain_randomization.seen_robot.morphology_coeff_value="$MORPHOLOGY_TARGET"
  --environment.domain_randomization.seen_robot.morphology_coeff_ramp_steps="$MORPHOLOGY_RAMP_STEPS"
  --environment.domain_randomization.seen_robot.exact_inertia_rescale=True
  --environment.domain_randomization.seen_robot.torque_scaling_exponent=4.0
  --environment.env_curriculum_coeff_max=0.6
  --environment.env_curriculum_level_success_tracking_ratio=0.75
  --algorithm.nr_steps="$ROLLOUT_STEPS"
  --algorithm.minibatch_size="$MINIBATCH_SIZE"
  --algorithm.nr_epochs="$PPO_EPOCHS"
  --algorithm.total_timesteps="$TOTAL_TIMESTEPS"
  --algorithm.muon_learning_rate_total_timesteps="$TOTAL_TIMESTEPS"
  --algorithm.adam_learning_rate_total_timesteps="$TOTAL_TIMESTEPS"
  --algorithm.evaluation_active=False
  --algorithm.evaluation_and_save_frequency="$SAVE_FREQUENCY"
  --algorithm.save_intermediate_models=True
  --environment.render=False
  --runner.track_console=True
  --runner.track_tb=True
  --runner.track_wandb=False
  --runner.save_model=True
  --runner.project_name=h1g1t1
  --runner.exp_name="$PROFILE"
  --runner.run_name="$RUN_ID"
  --runner.notes="one shared URMA2 policy; H1+G1+T1; $CLIP; DeepMimic; morphology schedule"
)

write_context() {
  {
    printf 'RUN_ID=%q\n' "$RUN_ID"
    printf 'PROFILE=%q\n' "$PROFILE"
    printf 'REPO=%q\n' "$REPO"
    printf 'PYTHON=%q\n' "$PYTHON"
    printf 'CLIP_DIR=%q\n' "$CLIP_DIR"
    printf 'CLIP=%q\n' "$CLIP"
    printf 'SEED=%q\n' "$SEED"
    printf 'MODEL_RUN_DIR=%q\n' "$MODEL_RUN_DIR"
    printf 'MODEL=%q\n' "$MODEL"
  } > "$RUN_DIR/config/run_context.env"
  printf '%q ' "$PYTHON" experiment.py --runner.mode=train "${WORLD[@]}" > "$RUN_DIR/config/train_command.sh"
  printf '\n' >> "$RUN_DIR/config/train_command.sh"
  chmod +x "$RUN_DIR/config/train_command.sh"
  env | sort > "$RUN_DIR/config/environment.txt"
  "$PYTHON" -m pip freeze > "$RUN_DIR/config/pip_freeze.txt" 2>&1 || true
  "$PYTHON" - <<'PY' > "$RUN_DIR/config/runtime_versions.json" 2>&1 || true
import json, platform, sys
mods = {}
for name in ("jax", "jaxlib", "mujoco", "flax", "optax", "numpy"):
    try:
        module = __import__(name)
        mods[name] = getattr(module, "__version__", "unknown")
    except Exception as exc:
        mods[name] = f"unavailable: {type(exc).__name__}: {exc}"
try:
    import jax
    devices = [f"{d.platform}:{d.device_kind}:{d.id}" for d in jax.devices()]
except Exception as exc:
    devices = [f"unavailable: {type(exc).__name__}: {exc}"]
print(json.dumps({"python": sys.version, "platform": platform.platform(), "packages": mods, "jax_devices": devices}, indent=2))
PY
  if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi -q > "$RUN_DIR/config/nvidia_smi_q.txt" 2>&1 || true; fi
  for git_dir in "$REPO" "$REPO/RL-X" "$REPO/loco_mjx"; do
    if git -C "$git_dir" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
      printf '%s %s\n' "$git_dir" "$(git -C "$git_dir" rev-parse HEAD)" >> "$RUN_DIR/config/git_revisions.txt"
      git -C "$git_dir" status --short >> "$RUN_DIR/config/git_status.txt" 2>/dev/null || true
    fi
  done
}

preflight() {
  write_context
  "$PYTHON" "$SCRIPT_DIR/preflight.py" \
    --repo "$REPO" --python "$PYTHON" --clip-dir "$CLIP_DIR" --clip "$CLIP" \
    --profile-json "$PROFILE_JSON" --out "$RUN_DIR/preflight.json" \
    $([[ "$ALLOW_CPU" == 1 ]] && printf '%s' '--allow-cpu')
  (cd "$REPO/loco_mjx/experiments" && "$PYTHON" experiment.py --runner.mode=show_config "${WORLD[@]}") \
    > "$RUN_DIR/config/resolved_config.log" 2>&1
}

train() {
  [[ -s "$RUN_DIR/preflight.json" ]] || preflight
  local monitor_pid=""
  "$SCRIPT_DIR/gpu_monitor.sh" "$RUN_DIR/logs/gpu_telemetry.csv" 30 & monitor_pid=$!
  cleanup_monitor() { if [[ -n "$monitor_pid" ]]; then kill "$monitor_pid" 2>/dev/null || true; wait "$monitor_pid" 2>/dev/null || true; fi; }
  trap cleanup_monitor INT TERM
  echo "[h1g1t1] training start: $(date -Is)" | tee "$RUN_DIR/logs/train.log"
  echo "[h1g1t1] total_envs=$TOTAL_ENVS envs_per_family=$ENVS_PER_FAMILY batch=$BATCH_SIZE minibatches_per_epoch=$MINIBATCHES_PER_EPOCH total=$TOTAL_TIMESTEPS" | tee -a "$RUN_DIR/logs/train.log"
  set +e
  (cd "$REPO/loco_mjx/experiments" && "$PYTHON" experiment.py --runner.mode=train "${WORLD[@]}") \
    > >(tee -a "$RUN_DIR/logs/train.log") 2>&1
  rc=$?
  set -e
  cleanup_monitor
  trap - INT TERM
  echo "[h1g1t1] training process exit=$rc end=$(date -Is)" | tee -a "$RUN_DIR/logs/train.log"
  return "$rc"
}

analyze() {
  "$PYTHON" "$SCRIPT_DIR/parse_training_log.py" "$RUN_DIR/logs/train.log" --out-dir "$RUN_DIR/metrics" \
    > "$RUN_DIR/logs/parse_training.log" 2>&1 || true
  if [[ -s "$RUN_DIR/metrics/training_metrics_wide.csv" ]]; then
    "$PYTHON" "$SCRIPT_DIR/plot_training.py" "$RUN_DIR/metrics/training_metrics_wide.csv" --out-dir "$RUN_DIR/plots" \
      > "$RUN_DIR/logs/plot_training.log" 2>&1 || true
  fi
  "$PYTHON" "$SCRIPT_DIR/verify_run.py" --log "$RUN_DIR/logs/train.log" --expected-step "$TOTAL_TIMESTEPS" \
    --model "$MODEL" --out "$RUN_DIR/run_verification.json"
}

evaluate_one() {
  local seed=$1 morph=$2 zero=$3
  local label="policy"
  [[ "$zero" == 1 ]] && label="reference_feedforward_control"
  local morph_tag
  morph_tag=$(printf '%s' "$morph" | tr '.' 'p')
  local out="$RUN_DIR/evaluations/${label}_morph${morph_tag}_seed${seed}.json"
  local args=(
    --model_path "$MODEL" --clip_dir "$CLIP_DIR" --raw_clip_dir "$CLIP_DIR" --clip "$CLIP"
    --robots unitree_h1:unitree_g1:booster_t1 --refbias 1.0 --anchor absolute --fitvariant False
    --cyclic False --refroot True --refroot_floor True --refvel_obs True --observe_heading True
    --command_cap_scale 8.0 --morphology_coeff "$morph" --exact_inertia_rescale True
    --torque_scaling_exponent 4.0 --joint_temperature 0.05 --heading_kernel cosine
    --nr_envs "$EVAL_ENVS" --steps "$EVAL_STEPS" --seed "$seed" --out "$out"
  )
  [[ "$zero" == 1 ]] && args+=(--zero_action)
  "$PYTHON" "$SCRIPT_DIR/evaluate_policy.py" "${args[@]}" \
    > "$RUN_DIR/logs/eval_${label}_morph${morph_tag}_seed${seed}.log" 2>&1
}

evaluate() {
  [[ "$SKIP_EVAL" == 1 ]] && { echo "[h1g1t1] evaluation skipped by SKIP_EVAL=1"; return 0; }
  [[ -s "$MODEL" ]] || { echo "missing model: $MODEL" >&2; return 1; }
  local seeds
  if [[ -n "${EVAL_SEEDS:-}" ]]; then seeds=$EVAL_SEEDS
  elif [[ "$PROFILE" == full ]]; then seeds="0 1 2 3"
  elif [[ "$PROFILE" == probe ]]; then seeds="0 1"
  else seeds="0"
  fi
  for seed in $seeds; do
    evaluate_one "$seed" 0.0 0
    evaluate_one "$seed" "$MORPHOLOGY_TARGET" 0
  done
  # Same-semantics zero-residual controls. With refbias=1, this is the reference
  # feed-forward controller, not nominal standing.
  evaluate_one 0 0.0 1
  evaluate_one 0 "$MORPHOLOGY_TARGET" 1
  "$PYTHON" "$SCRIPT_DIR/summarize_evaluations.py" "$RUN_DIR/evaluations" --out-dir "$RUN_DIR/evaluations" \
    > "$RUN_DIR/logs/summarize_evaluations.log" 2>&1
}

collect() {
  local out="$RUN_DIR/${RUN_ID}_diagnostics_return.zip"
  local include=()
  [[ "$INCLUDE_MODEL" == 1 ]] && include+=(--include-model)
  "$PYTHON" "$SCRIPT_DIR/collect_diagnostics.py" --run-dir "$RUN_DIR" --model-root "$MODEL_RUN_DIR" \
    --out "$out" "${include[@]}"
  echo "[h1g1t1] return this file: $out"
}

partial_collect() {
  set +e
  analyze >/dev/null 2>&1
  collect >/dev/null 2>&1
  set -e
}
trap 'rc=$?; if [[ $rc -ne 0 ]]; then echo "[h1g1t1] failed with rc=$rc; collecting partial diagnostics" >&2; partial_collect; fi' EXIT

case "$STAGE" in
  preflight) preflight ;;
  train) train ;;
  analyze) analyze ;;
  evaluate) evaluate ;;
  collect) collect ;;
  all) preflight; train; analyze; evaluate; collect ;;
  *) echo "usage: $0 {preflight|train|analyze|evaluate|collect|all}" >&2; exit 2 ;;
esac
trap - EXIT
