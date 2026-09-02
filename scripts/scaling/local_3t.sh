#!/usr/bin/env bash
# ONE POLICY, THREE TOPOLOGIES, ONE MOTION -- on the local CUDA GPU.
#
# H1 + G1 + booster_t1 tracking dance2_subject4, the clip all three have an
# offline retarget for. Every attempt at this on Viper has aborted inside a fused
# kernel with ROCM_ERROR_ILLEGAL_ADDRESS; that fault cannot exist on CUDA, so
# this is the path that does not depend on diagnosing a vendor bug.
#
# Sizing: minibatch must divide the batch AND be divisible by the robot count.
#   192 envs/robot -> batch 192*64 = 12288 = 2^12*3 -> minibatch 6144 (2 per epoch)
# 256 would give batch 16384 = 2^14, which has no factor of 3 and therefore no
# legal minibatch at all for three robots.
#
# Launch detached:
#   setsid nohup bash scripts/scaling/local_3t.sh > /tmp/local_3t.log 2>&1 < /dev/null &
set -u

REPO=/mnt/c/Users/smirn/Desktop/robot_learning_ip
PY=~/jaxgpu/bin/python
CLIPS=$REPO/external_data/amass_converted/LAFAN1
OUT=$REPO/experiments/local_3t
mkdir -p "$OUT"

cd "$REPO/loco_mjx/experiments"
export PYTHONPATH="$REPO:$REPO/RL-X:$REPO/loco_mjx"
export MUJOCO_GL=disable
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_enable_command_buffer=

NR_ENVS=${NR_ENVS:-192}
NR_STEPS=64
MINIBATCH=${MINIBATCH:-6144}
SAVE_EVERY=${SAVE_EVERY:-1966080}     # 160 batches at 192 envs
TOTAL=${TOTAL:-19660800}              # 10 save intervals
NAME=${NAME:-local_3t_dance4}

WORLD=(
  --environment.name=locomotion.urma2.mjx
  --algorithm.name=urma2.mjx
  --environment.train_robots="('unitree_h1','unitree_g1','booster_t1')"
  --environment.terrain.type=plane
  --environment.critic_exteroceptive_observation_type=none
  --environment.terrain.contact_solref_timeconst=0.0
  --environment.command.type=tracking_clip
  --environment.reward.type=tracking
  --environment.reward.log_info=True
  --environment.command.tracking_clip_dir="$CLIPS"
  --environment.command.tracking_clip_file=dance2_subject4.npz
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
  # feet ON: fx_footz showed this lifts the foot-lift ratio 0.30 -> 0.54 and
  # halves penetration. Heading stays off -- three attempts have shown that term
  # does not move heading, and it costs joint accuracy.
  --environment.reward.deepmimic_foot_height_weight_ratio=1.0
  --environment.reward.deepmimic_foot_height_temperature=0.01
  --environment.reward.foot_z_velocity_coeff=1.0
  --environment.reward.root_heading_tracking_weight_ratio=0.0
  --environment.reward.root_heading_tracking_temperature=0.25
  --environment.termination.tracking_deviation_ratio=0.0
  --environment.reward.tracking_post_contact_penalties=True
  # E1 (2026-08-28): the CONTACT dose. ce_corrfinal showed the reference wants
  # the feet airborne 12-28% of the time while this recipe manages 2-3%, and
  # every ankle on all three robots has per-joint corr ~0.00 against a
  # near-perfect ankle reference. These two coefficients are what pins the foot.
  # Defaults are the pre-28-08 values, so omitting them is bit-identical.
  --environment.reward.foot_slip_coeff="${FOOTSLIP:-20.0}"
  --environment.reward.ground_penetration_coeff="${GROUNDPEN:-1000}"
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
  --runner.track_tb=False
  --runner.track_wandb=False
  --runner.save_model=True
  --runner.project_name=local_3t
  --runner.exp_name="$NAME"
)

log="$OUT/${NAME}.log"
echo "[3t] START $NAME  $(date -Is)  envs/robot=$NR_ENVS total=$TOTAL"
$PY experiment.py "${WORLD[@]}" > "$log" 2>&1
rc=$?
# Verify by artifact: RL-X logs an uncaught exception and still exits 0.
blocks=$(grep -c "nr_env_steps" "$log" 2>/dev/null || echo 0)
echo "[3t] END   $NAME rc=$rc blocks=$blocks  $(date -Is)"
if [[ $rc -eq 0 && $blocks -gt 10 ]]; then
  touch "$OUT/${NAME}.done"
  echo "[3t] OK -- three topologies trained locally"
else
  echo "[3t] FAILED"
  grep -E "Error|Traceback|Exception|out of memory|RESOURCE" "$log" | tail -8
fi
