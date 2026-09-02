#!/usr/bin/env bash
# Bisect the L2 (l3t_fix_head_rx) episode-length pin (~20 steps for 19.66M
# steps, vs L1 growing 20 -> 800 on the same base).
#
# The five knobs L2 changed over L1 split into two groups; each probe applies
# ONE group to the L1 base for 600k steps and the episode_length trend says
# which group pins survival:
#   probe_head : heading trio only  (HRATIO 0.20, HTEMP 2.0, HOBS True)
#   probe_rx   : rx_p3f0 dose only  (SLIP 6.6667, GPEN 333.33, FOOTH 0.0)
# Then L1's fixeval seeds 1-3 (policy+zero) complete the T1-ratio n=4.
set -u
REPO=/mnt/c/Users/smirn/Desktop/robot_learning_ip
PY=~/jaxgpu/bin/python
MERGED=$REPO/external_data/amass_converted/LAFAN1_3t
OUT=$REPO/experiments/local_3t
CLIPS=$MERGED
FEOUT=$OUT/fixeval

# Wait for whatever is on the GPU (the orphaned last eval cell) to exit.
while pgrep -f crosseval_motion.py > /dev/null; do sleep 30; done
echo "[bisect] GPU free $(date -Is)"

cd "$REPO/loco_mjx/experiments"
export PYTHONPATH="$REPO:$REPO/RL-X:$REPO/loco_mjx"
export MUJOCO_GL=disable XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
export JAX_ENABLE_COMPILATION_CACHE=true
export JAX_COMPILATION_CACHE_DIR=$REPO/.jax_cache_local

probe() {  # probe <name> <hratio> <htemp> <hobs> <slip> <gpen> <foothw>
  local NAME="$1" HRATIO="$2" HTEMP="$3" HOBS="$4" SLIP="$5" GPEN="$6" FOOTHW="$7"
  echo "[bisect] START $NAME heading=$HRATIO/$HTEMP/$HOBS dose=$SLIP/$GPEN/$FOOTHW $(date -Is)"
  $PY experiment.py \
    --environment.name=locomotion.urma2.mjx \
    --algorithm.name=urma2.mjx \
    --environment.train_robots="('unitree_h1','unitree_g1','booster_t1')" \
    --environment.terrain.type=plane \
    --environment.critic_exteroceptive_observation_type=none \
    --environment.terrain.contact_solref_timeconst=0.0 \
    --environment.command.type=tracking_clip \
    --environment.reward.type=tracking \
    --environment.reward.log_info=True \
    --environment.command.tracking_clip_dir="$MERGED" \
    --environment.command.tracking_clip_file=dance2_subject4.npz \
    --environment.command.tracking_clip_fit_per_variant=False \
    --environment.command.tracking_clip_anchor=absolute \
    --environment.command.tracking_clip_velocity_command=True \
    --environment.command.tracking_clip_cyclic=False \
    --environment.command.tracking_clip_amplitude_scale=1.0 \
    --environment.command.tracking_clip_velocity_scale=1.0 \
    --environment.command.tracking_clip_observe_velocity=False \
    --environment.command.tracking_clip_root_height_from_pose=True \
    --environment.command.tracking_clip_root_height_pose_as_floor=True \
    --environment.command.tracking_reference_action_bias=0.0 \
    --environment.reward.nominal_diff_target=reference \
    --environment.reward.joint_tracking_coeff=30.0 \
    --environment.reward.joint_tracking_temperature=0.05 \
    --environment.reward.tracking_curriculum_gated=False \
    --environment.reward.gait_coeff_mode=floor \
    --environment.reward.gait_coeff_value=0.25 \
    --environment.reward.action_rate_coeff=3.0 \
    --environment.reward.action_smoothness_coeff=0.1 \
    --environment.reward.deepmimic_enabled=True \
    --environment.reward.deepmimic_qvel_temperature=10 \
    --environment.reward.deepmimic_foot_height_weight_ratio="$FOOTHW" \
    --environment.reward.deepmimic_foot_height_temperature=0.01 \
    --environment.reward.foot_z_velocity_coeff=1.0 \
    --environment.reward.root_heading_tracking_weight_ratio="$HRATIO" \
    --environment.reward.root_heading_tracking_temperature="$HTEMP" \
    --environment.command.tracking_clip_observe_root_heading="$HOBS" \
    --environment.termination.tracking_deviation_ratio=0.0 \
    --environment.reward.tracking_post_contact_penalties=True \
    --environment.reward.foot_slip_coeff="$SLIP" \
    --environment.reward.ground_penetration_coeff="$GPEN" \
    --environment.command.tracking_clip_latent_hold=1 \
    --environment.domain_randomization.initial_state.type=reference \
    --environment.domain_randomization.seen_robot.morphology_coeff_mode=fixed \
    --environment.domain_randomization.seen_robot.morphology_coeff_value=0.0 \
    --environment.env_curriculum_coeff_max=0.6 \
    --environment.env_curriculum_level_success_tracking_ratio=0.0 \
    --environment.nr_envs=192 \
    --algorithm.nr_steps=64 \
    --algorithm.minibatch_size=6144 \
    --algorithm.nr_epochs=5 \
    --algorithm.total_timesteps=614400 \
    --algorithm.evaluation_active=False \
    --algorithm.evaluation_and_save_frequency=614400 \
    --environment.render=False \
    --runner.track_console=True \
    --runner.track_tb=False \
    --runner.track_wandb=False \
    --runner.save_model=False \
    --runner.project_name=local_3t \
    --runner.exp_name="$NAME" > "$OUT/${NAME}.log" 2>&1
  echo "[bisect] END $NAME rc=$? $(date -Is)"
  grep 'episode_length/mean' "$OUT/${NAME}.log" | tail -3
}

probe probe_head 0.20 2.0 True 20.0 1000 1.0
probe probe_rx   0.0  0.25 False 6.6667 333.33 0.0

# L1 fixeval seeds 1-3 for the T1 ratio at n=4.
M1=$(ls -t "$REPO"/loco_mjx/experiments/runs/local_3t/l3t_fix/*/models/latest.model | head -1)
cd "$REPO/experiments/fsq_khaendler"
for s in 1 2 3; do
  for mode in policy zero; do
    args=(
      --model_path "$M1"
      --clip_dir "$CLIPS" --clip dance2_subject4.npz --raw_clip_dir "$CLIPS"
      --robots unitree_h1:unitree_g1:booster_t1
      --refbias 0.0 --anchor absolute --fitvariant False
      --refroot True --refroot_floor True --refvel_obs False
      --root_heading_obs False
      --nr_envs 96 --steps 1000 --seed "$s"
      --out "$FEOUT/l1_${mode}_s${s}.json"
    )
    [ "$mode" = "zero" ] && args+=(--zero_action)
    echo "[bisect] l1_$mode s$s $(date -Is)"
    $PY "$REPO/experiments/fsq_khaendler/crosseval_motion.py" "${args[@]}" > "$FEOUT/l1_${mode}_s${s}.log" 2>&1
    [ -s "$FEOUT/l1_${mode}_s${s}.json" ] && echo "  ARTIFACT OK" || echo "  ARTIFACT MISSING"
  done
done
echo "[bisect] ALL DONE $(date -Is)"
