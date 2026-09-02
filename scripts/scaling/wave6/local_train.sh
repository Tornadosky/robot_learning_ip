#!/usr/bin/env bash
# WAVE 6 local trainer: one urma2 mmtrain arm on the local GPU (WSL, ~/jaxgpu).
# Bundle recipe (= Viper night-5 base + swing 0.5 + heading), nominal bodies.
#
#   NAME=<exp> CLIPDIR=<dir with UnitreeH1/UnitreeG1[/BoosterT1]> [ROBOTS=h1:g1]
#   [LATENT=0|1] [REPLACES=False] [HOLD=1] [NR_ENVS=768] [MINIBATCH=8192]
#   [TOTAL=19660800] [SAVE_EVERY=1966080] [AUX_COEFF=0] [AUX_HORIZON=1]
#   [AUX_DETACH=False] [LEGW=1.0] [SEED=1] [EXTRA="--flag=v --flag2=v"]
#   [MORPH_MODE=fixed MORPH_COEFF=0.0] [PROJECT=local_w6]
#   bash local_train.sh
set -u
REPO=${REPO:-/mnt/c/Users/smirn/Desktop/robot_learning_ip}
PY=${PY:-~/jaxgpu/bin/python}
NAME=${NAME:?}; CLIPDIR=${CLIPDIR:?}
ROBOTS=${ROBOTS:-unitree_h1:unitree_g1}
LATENT=${LATENT:-0}; REPLACES=${REPLACES:-False}; HOLD=${HOLD:-1}
NR_ENVS=${NR_ENVS:-768}; MINIBATCH=${MINIBATCH:-8192}
TOTAL=${TOTAL:-19660800}; SAVE_EVERY=${SAVE_EVERY:-1966080}
AUX_COEFF=${AUX_COEFF:-0.0}; AUX_HORIZON=${AUX_HORIZON:-1}; AUX_DETACH=${AUX_DETACH:-False}
LEGW=${LEGW:-1.0}; SEED=${SEED:-1}; EXTRA=${EXTRA:-}
JLAT_CH=${JLAT_CH:-0}; LATENT_DIM=${LATENT_DIM:-32}; LATENT_DIVISOR=${LATENT_DIVISOR:-10.0}; SIDECAR=${SIDECAR:-_zq}
COTRAIN_ROWS=${COTRAIN_ROWS:-0}; COTRAIN_CH=${COTRAIN_CH:-4}; COTRAIN_RECON=${COTRAIN_RECON:-1.0}
COTRAIN_FREEZE=${COTRAIN_FREEZE:-False}; COTRAIN_INIT=${COTRAIN_INIT:-}
MORPH_MODE=${MORPH_MODE:-fixed}; MORPH_COEFF=${MORPH_COEFF:-0.0}
MORPH_START=${MORPH_START:-0.2}; MORPH_RAMP=${MORPH_RAMP:-40000000}   # schedule mode only (same defaults as viper_train.sbatch)
PROJECT=${PROJECT:-local_w6}
CLIP=${CLIP:-dance2_subject4.npz}

TR="("; for r in $(echo "$ROBOTS" | tr ':' ' '); do TR="$TR'$r',"; done; TR="$TR)"
LOBS=False; [ "$LATENT" = 1 ] && LOBS=True
JLAT=0; [ "$LATENT" = 1 ] && JLAT=4

cd "$REPO/loco_mjx/experiments"
export PYTHONPATH="$REPO:$REPO/RL-X:$REPO/loco_mjx"
export MUJOCO_GL=disable
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
export JAX_ENABLE_COMPILATION_CACHE=true
export JAX_COMPILATION_CACHE_DIR=$REPO/.jax_cache_local

echo "=== local_train $NAME robots=$TR clipdir=$CLIPDIR latent=$LOBS replaces=$REPLACES hold=$HOLD envs=$NR_ENVS total=$TOTAL aux=$AUX_COEFF/$AUX_HORIZON/$AUX_DETACH legw=$LEGW cotrain=$COTRAIN_ROWS/$COTRAIN_CH/$COTRAIN_RECON/$COTRAIN_FREEZE/[$COTRAIN_INIT] sidecar=$SIDECAR dim=$LATENT_DIM div=$LATENT_DIVISOR seed=$SEED morph=$MORPH_MODE/$MORPH_COEFF/$MORPH_START/$MORPH_RAMP extra=[$EXTRA] $(date -Is)"
exec $PY experiment.py \
  --environment.name=locomotion.urma2.mjx --algorithm.name=urma2.mjx \
  --environment.train_robots="$TR" \
  --environment.seed="$SEED" \
  --environment.terrain.type=plane \
  --environment.critic_exteroceptive_observation_type=none \
  --environment.terrain.contact_solref_timeconst=0.004 \
  --environment.command.type=tracking_clip --environment.reward.type=tracking \
  --environment.reward.log_info=True \
  --environment.command.tracking_clip_dir="$CLIPDIR" \
  --environment.command.tracking_clip_file="$CLIP" \
  --environment.command.tracking_clip_fit_per_variant=False \
  --environment.command.tracking_clip_anchor=absolute \
  --environment.command.tracking_clip_velocity_command=True \
  --environment.command.tracking_clip_cyclic=False \
  --environment.command.tracking_clip_observe_velocity=False \
  --environment.command.tracking_clip_root_height_from_pose=True \
  --environment.command.tracking_clip_root_height_pose_as_floor=True \
  --environment.command.tracking_reference_action_bias=0.0 \
  --environment.command.tracking_clip_observe_root_heading=True \
  --environment.command.tracking_clip_latent_hold=1 \
  --environment.command.tracking_clip_reference_hold="$HOLD" \
  --environment.command.tracking_clip_latent_obs="$LOBS" \
  --environment.command.tracking_clip_latent_dim="$LATENT_DIM" \
  --environment.command.tracking_clip_latent_replaces_reference="$REPLACES" \
  --environment.command.tracking_clip_latent_scope=per_joint \
  --environment.command.tracking_clip_latent_obs_divisor="$LATENT_DIVISOR" \
  --environment.command.tracking_clip_latent_sidecar_suffix="$SIDECAR" \
  --environment.reward.nominal_diff_target=reference \
  --environment.reward.joint_tracking_coeff=30.0 \
  --environment.reward.joint_tracking_temperature=0.05 \
  --environment.reward.joint_tracking_leg_weight="$LEGW" \
  --environment.reward.gait_coeff_mode=floor \
  --environment.reward.gait_coeff_value=0.25 \
  --environment.reward.action_rate_coeff=3.0 \
  --environment.reward.action_smoothness_coeff=0.1 \
  --environment.reward.deepmimic_enabled=True \
  --environment.reward.deepmimic_qvel_temperature=10 \
  --environment.reward.deepmimic_foot_height_weight_ratio=0.3333 \
  --environment.reward.deepmimic_foot_height_temperature=0.05 \
  --environment.reward.deepmimic_swing_match_weight_ratio=0.5 \
  --environment.reward.foot_z_velocity_coeff=1.0 \
  --environment.reward.root_heading_tracking_weight_ratio=0.20 \
  --environment.reward.root_heading_tracking_temperature=2.0 \
  --environment.termination.tracking_deviation_ratio=0.0 \
  --environment.reward.tracking_post_contact_penalties=True \
  --environment.reward.foot_slip_coeff=6.6667 \
  --environment.reward.ground_penetration_coeff=1000 \
  --environment.domain_randomization.initial_state.type=reference \
  --environment.domain_randomization.seen_robot.morphology_coeff_mode="$MORPH_MODE" \
  --environment.domain_randomization.seen_robot.morphology_coeff_value="$MORPH_COEFF" \
  --environment.domain_randomization.seen_robot.morphology_coeff_start="$MORPH_START" \
  --environment.domain_randomization.seen_robot.morphology_coeff_ramp_steps="$MORPH_RAMP" \
  --environment.env_curriculum_coeff_max=0.6 \
  --environment.env_curriculum_level_success_tracking_ratio=0.0 \
  --environment.nr_envs="$NR_ENVS" \
  --algorithm.nr_steps=64 \
  --algorithm.minibatch_size="$MINIBATCH" \
  --algorithm.nr_epochs=5 \
  --algorithm.joint_latent_encoder_dim="$JLAT" \
  --algorithm.joint_latent_channels="$JLAT_CH" \
  --algorithm.aux_token_coeff="$AUX_COEFF" \
  --algorithm.aux_token_horizon="$AUX_HORIZON" \
  --algorithm.aux_token_detach_trunk="$AUX_DETACH" \
  --algorithm.cotrain_window_rows="$COTRAIN_ROWS" \
  --algorithm.cotrain_window_channels="$COTRAIN_CH" \
  --algorithm.cotrain_recon_coeff="$COTRAIN_RECON" \
  --algorithm.cotrain_freeze_encoder="$COTRAIN_FREEZE" \
  --algorithm.cotrain_init_encoder="$COTRAIN_INIT" \
  --algorithm.total_timesteps="$TOTAL" \
  --algorithm.evaluation_active=False \
  --algorithm.evaluation_and_save_frequency="$SAVE_EVERY" \
  --environment.render=False \
  --runner.track_console=True --runner.track_tb=False --runner.track_wandb=False \
  --runner.save_model=True --runner.project_name="$PROJECT" --runner.exp_name="$NAME" \
  $EXTRA
