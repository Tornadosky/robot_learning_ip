#!/usr/bin/env bash
# Smoke test the night5 swing-match term: tiny 2-robot run, verify the new
# reward term parses, logs, and reads sane values (ref_airborne_frac ~0.13 on
# H1 dance2_subject4 -- the known reference airborne fraction).
set -u
REPO=/mnt/c/Users/smirn/Desktop/robot_learning_ip
PY=~/jaxgpu/bin/python
cd "$REPO/loco_mjx/experiments"
export PYTHONPATH="$REPO:$REPO/RL-X:$REPO/loco_mjx"
export MUJOCO_GL=disable XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
export JAX_ENABLE_COMPILATION_CACHE=true
export JAX_COMPILATION_CACHE_DIR=$REPO/.jax_cache_local

$PY experiment.py \
  --environment.name=locomotion.urma2.mjx --algorithm.name=urma2.mjx \
  --environment.train_robots="('unitree_h1','unitree_g1')" \
  --environment.terrain.type=plane \
  --environment.critic_exteroceptive_observation_type=none \
  --environment.terrain.contact_solref_timeconst=0.004 \
  --environment.command.type=tracking_clip --environment.reward.type=tracking \
  --environment.reward.log_info=True \
  --environment.command.tracking_clip_dir="$REPO/external_data/amass_converted/LAFAN1" \
  --environment.command.tracking_clip_file=dance2_subject4.npz \
  --environment.command.tracking_clip_fit_per_variant=False \
  --environment.command.tracking_clip_anchor=absolute \
  --environment.command.tracking_clip_root_height_from_pose=True \
  --environment.command.tracking_clip_root_height_pose_as_floor=True \
  --environment.command.tracking_clip_observe_root_heading=True \
  --environment.reward.joint_tracking_coeff=30.0 \
  --environment.reward.joint_tracking_temperature=0.05 \
  --environment.reward.gait_coeff_mode=floor --environment.reward.gait_coeff_value=0.25 \
  --environment.reward.deepmimic_enabled=True --environment.reward.deepmimic_qvel_temperature=10 \
  --environment.reward.deepmimic_foot_height_weight_ratio=0.3333 \
  --environment.reward.deepmimic_foot_height_temperature=0.05 \
  --environment.reward.deepmimic_swing_match_weight_ratio=0.5 \
  --environment.reward.deepmimic_swing_match_ref_threshold_m=0.02 \
  --environment.reward.root_heading_tracking_weight_ratio=0.20 \
  --environment.reward.root_heading_tracking_temperature=2.0 \
  --environment.reward.tracking_post_contact_penalties=True \
  --environment.reward.foot_slip_coeff=6.6667 \
  --environment.reward.ground_penetration_coeff=1000 \
  --environment.domain_randomization.initial_state.type=reference \
  --environment.domain_randomization.seen_robot.morphology_coeff_mode=fixed \
  --environment.domain_randomization.seen_robot.morphology_coeff_value=0.0 \
  --environment.nr_envs=48 --algorithm.nr_steps=64 \
  --algorithm.minibatch_size=1536 --algorithm.nr_epochs=2 \
  --algorithm.total_timesteps=18432 \
  --algorithm.evaluation_active=False --algorithm.evaluation_and_save_frequency=6144 \
  --environment.render=False --runner.track_console=True --runner.track_tb=False \
  --runner.track_wandb=False --runner.save_model=False \
  --runner.project_name=smoke --runner.exp_name=swing_smoke
