#!/usr/bin/env bash
# G1-only reference-vs-token pair on the LOCAL GPU.
#
# Viper is currently aborting every arm inside a fused kernel with
# ROCM_ERROR_ILLEGAL_ADDRESS while a generic JAX matmul+scatter on the SAME node
# completes in 32 s, so the cluster is not the problem and the fault is not yet
# understood. This runs the same experiment on CUDA to get the answer regardless.
#
# The question (REPORT_FSQ_WAVE2.md 2.6d): the token buys -4.0 % on H1 alone and
# ~0 at two topologies. Is that because a policy shared across two bodies cannot
# exploit the channel, or because G1's token is simply weaker than H1's? G1 alone
# has never been run. If it shows ~-4 %, the loss is caused by SHARING.
#
# Local numbers are NOT comparable with Viper numbers -- different device,
# different env count, different budget. Both arms here run identically, so the
# ref-vs-both comparison is internally valid, and that is the only comparison
# this script is for.
#
# Two JAX processes on one GPU collapse utilisation, so the arms are serialised.
# Launch detached:
#   setsid nohup bash scripts/scaling/local_g1_token.sh > /tmp/local_g1.log 2>&1 < /dev/null &
set -u

REPO=/mnt/c/Users/smirn/Desktop/robot_learning_ip
PY=~/jaxgpu/bin/python
CLIPS=$REPO/experiments/fsq_khaendler/clips_super
OUT=$REPO/experiments/local_g1
mkdir -p "$OUT"

cd "$REPO/loco_mjx/experiments"
export PYTHONPATH="$REPO:$REPO/RL-X:$REPO/loco_mjx"
export MUJOCO_GL=disable
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# The Viper aborts all landed in fused kernels; on CUDA keep the default fusion
# behaviour but disable the command buffer, matching the cluster recipe.
export XLA_FLAGS=--xla_gpu_enable_command_buffer=

NR_ENVS=${NR_ENVS:-1024}
NR_STEPS=64                      # batch = 1024*64 = 65536
MINIBATCH=8192                   # divides the batch 8x, divisible by 1 robot
# TOTAL must be a multiple of SAVE_EVERY, and SAVE_EVERY a multiple of the batch
# (nr_envs*nr_steps) -- the trainer asserts both.
SAVE_EVERY=${SAVE_EVERY:-3276800}   # 50 batches at 1024 envs
TOTAL=${TOTAL:-49152000}            # 15 save intervals, 750 batches

WORLD=(
  --environment.name=locomotion.urma2.mjx
  --algorithm.name=urma2.mjx
  --environment.train_robots="('unitree_g1',)"
  --environment.terrain.type=plane
  --environment.critic_exteroceptive_observation_type=none
  --environment.terrain.contact_solref_timeconst=0.0
  --environment.command.type=tracking_clip
  --environment.reward.type=tracking
  --environment.reward.log_info=True
  --environment.command.tracking_clip_dir="$CLIPS"
  --environment.command.tracking_clip_file=super5dance.npz
  --environment.command.tracking_clip_fit_per_variant=False
  --environment.command.tracking_clip_anchor=absolute
  --environment.command.tracking_clip_velocity_command=True
  --environment.command.tracking_clip_cyclic=False
  --environment.command.tracking_clip_amplitude_scale=1.0
  --environment.command.tracking_clip_velocity_scale=1.0
  --environment.command.tracking_clip_observe_velocity=False
  --environment.command.tracking_clip_root_height_from_pose=True
  --environment.command.tracking_clip_root_height_pose_as_floor=True
  --environment.command.tracking_reference_action_bias=0.0
  --environment.reward.nominal_diff_target=reference
  --environment.reward.joint_tracking_coeff=30.0
  --environment.reward.joint_tracking_temperature=0.05
  --environment.reward.tracking_curriculum_gated=False
  --environment.reward.gait_coeff_mode=floor
  --environment.reward.gait_coeff_value=0.25
  --environment.reward.action_rate_coeff=3.0
  --environment.reward.action_smoothness_coeff=0.1
  --environment.reward.deepmimic_enabled=True
  --environment.reward.deepmimic_qvel_temperature=10
  --environment.reward.root_heading_tracking_weight_ratio=0.0
  --environment.reward.root_heading_tracking_temperature=0.25
  --environment.termination.tracking_deviation_ratio=0.0
  --environment.reward.tracking_post_contact_penalties=True
  --environment.reward.foot_slip_coeff=20.0
  --environment.reward.ground_penetration_coeff=1000
  --environment.reward.deepmimic_foot_height_weight_ratio=0.0
  --environment.reward.deepmimic_foot_height_temperature=0.01
  --environment.reward.foot_z_velocity_coeff=10.0
  --environment.command.tracking_clip_latent_hold=1
  --environment.domain_randomization.initial_state.type=reference
  --environment.domain_randomization.seen_robot.morphology_coeff_mode=fixed
  --environment.domain_randomization.seen_robot.morphology_coeff_value=0.0
  --environment.env_curriculum_coeff_max=0.6
  --environment.env_curriculum_level_success_tracking_ratio=0.0
  --environment.nr_envs="$NR_ENVS"
  --algorithm.nr_steps="$NR_STEPS"
  --algorithm.minibatch_size="$MINIBATCH"
  --algorithm.nr_epochs=5
  --algorithm.total_timesteps="$TOTAL"
  --algorithm.evaluation_active=False
  --algorithm.evaluation_and_save_frequency="$SAVE_EVERY"
  --environment.render=False
  --runner.track_console=True
  # TB off: RL-X's writer pulls in torch, which this env does not have and does
  # not need -- every curve in this project is parsed from the console tables.
  --runner.track_tb=False
  --runner.track_wandb=False
  --runner.save_model=True
  --runner.project_name=local_g1
)

run () {
  local name=$1; shift
  local log="$OUT/${name}.log"
  if [[ -f "$OUT/${name}.done" ]]; then echo "[local] SKIP $name (done)"; return 0; fi
  echo "[local] START $name  $(date -Is)  envs=$NR_ENVS total=$TOTAL"
  $PY experiment.py "${WORLD[@]}" "$@" --runner.exp_name="$name" > "$log" 2>&1
  local rc=$?
  # Verify by ARTIFACT, not exit code. RL-X catches an exception, logs
  # "Uncaught exception", and still returns 0 -- so rc=0 alone marked a run that
  # never trained a single step as done. Require training tables in the log.
  local blocks
  blocks=$(grep -c "nr_env_steps" "$log" 2>/dev/null || echo 0)
  echo "[local] END   $name rc=$rc blocks=$blocks  $(date -Is)"
  if [[ $rc -eq 0 && $blocks -gt 10 ]]; then
    touch "$OUT/${name}.done"
  else
    echo "[local] FAILED $name (rc=$rc, only $blocks logged updates)"
    grep -E "Error|error|Traceback|Exception" "$log" | tail -6
    return 1
  fi
}

BOTHZ=(
  --environment.command.tracking_clip_latent_obs=True
  --environment.command.tracking_clip_latent_dim=32
  --environment.command.tracking_clip_latent_replaces_reference=False
)

run local_g1_ref
run local_g1_both "${BOTHZ[@]}"
echo "[local] ALL DONE $(date -Is)"
