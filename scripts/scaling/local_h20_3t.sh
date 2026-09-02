#!/usr/bin/env bash
# WAVE6 LOCAL: 3-robot pair TRAINED at reference hold 20 (B4 redo) -- derived from local_night5.sh
# NIGHT5 LOCAL orchestrator -- keeps the local GPU busy all night:
#   0. wait for tokenizer_3t_v2 (+ clips_3t_v2 sidecars) from
#      train_tokenizer_3t_v2.sh
#   1. ln6h20_ref  3-robot, night3 recipe + SWING MATCH 0.5   (swing on 3 robots)
#   2. ln6h20_tok  same + FSQ token from the RETRAINED v2 tokenizer (250 epochs,
#      foot channels) -- fixes ln3_tok's weak-codec confound
#   3. eval both vs zero floor (4 seeds, matched contact world) + render dumps
# Same resume/heartbeat machinery as local_night3.sh.
#
# Launch: setsid nohup bash scripts/scaling/local_night5.sh \
#   > experiments/local_3t/ln6h20_run.log 2>&1 < /dev/null &
set -u
REPO=/mnt/c/Users/smirn/Desktop/robot_learning_ip
PY=~/jaxgpu/bin/python
MERGED=$REPO/external_data/amass_converted/LAFAN1_3t
TOKV2=$REPO/experiments/fsq_khaendler/clips_3t_v2
OUT=$REPO/experiments/local_3t
mkdir -p "$OUT"

[ -f "$TOKV2/BoosterT1/dance2_subject4_zq.npz" ] || { echo "[ln6h20] tokenizer never finished"; exit 1; }
ORIG=$REPO/external_data/amass_converted/LAFAN1
cmp -s "$TOKV2/BoosterT1/dance2_subject4.npz" "$ORIG/BoosterT1/dance2_subject4.npz" \
  && { echo "[ln6h20] ABORT: v2 T1 clip is the INFEASIBLE original"; exit 1; }
echo "[ln6h20] v2 sidecars present and T1 clip is the fixed one"

cd "$REPO/loco_mjx/experiments"
export PYTHONPATH="$REPO:$REPO/RL-X:$REPO/loco_mjx"
export MUJOCO_GL=disable
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
export JAX_ENABLE_COMPILATION_CACHE=true
export JAX_COMPILATION_CACHE_DIR=$REPO/.jax_cache_local

NR_ENVS=192; MINIBATCH=6144; SAVE_EVERY=1966080; TOTAL=19660800

newest_ckpt() {
  ls -t "$REPO"/loco_mjx/experiments/runs/local_3t/"$1"/*/models/latest.model 2>/dev/null | head -1
}
steps_done() {
  grep -o "nr_env_steps[^0-9]*[0-9]\+" "$1" 2>/dev/null | grep -o "[0-9]\+$" | tail -1
}

run_arm() {  # run_arm <name> <clipdir> <latent True|False>
  local NAME="$1" CDIR="$2" LOBS="$3"
  local WORLD=(
    --environment.name=locomotion.urma2.mjx --algorithm.name=urma2.mjx
    --environment.train_robots="('unitree_h1','unitree_g1','booster_t1')"
    --environment.terrain.type=plane
    --environment.critic_exteroceptive_observation_type=none
    --environment.terrain.contact_solref_timeconst=0.004
    --environment.command.type=tracking_clip --environment.reward.type=tracking
    --environment.reward.log_info=True
    --environment.command.tracking_clip_dir="$CDIR"
    --environment.command.tracking_clip_file=dance2_subject4.npz
    --environment.command.tracking_clip_fit_per_variant=False
    --environment.command.tracking_clip_anchor=absolute
    --environment.command.tracking_clip_velocity_command=True
    --environment.command.tracking_clip_cyclic=False
    --environment.command.tracking_clip_observe_velocity=False
    --environment.command.tracking_clip_root_height_from_pose=True
    --environment.command.tracking_clip_root_height_pose_as_floor=True
    --environment.command.tracking_reference_action_bias=0.0
    --environment.command.tracking_clip_observe_root_heading=True
    --environment.command.tracking_clip_latent_hold=1
    --environment.command.tracking_clip_reference_hold=20
    --environment.command.tracking_clip_latent_obs="$LOBS"
    --environment.command.tracking_clip_latent_dim=32
    --environment.command.tracking_clip_latent_replaces_reference=False
    --environment.command.tracking_clip_latent_scope=per_joint
    --environment.command.tracking_clip_latent_obs_divisor=10.0
    --environment.reward.nominal_diff_target=reference
    --environment.reward.joint_tracking_coeff=30.0
    --environment.reward.joint_tracking_temperature=0.05
    --environment.reward.gait_coeff_mode=floor
    --environment.reward.gait_coeff_value=0.25
    --environment.reward.action_rate_coeff=3.0
    --environment.reward.action_smoothness_coeff=0.1
    --environment.reward.deepmimic_enabled=True
    --environment.reward.deepmimic_qvel_temperature=10
    --environment.reward.deepmimic_foot_height_weight_ratio=0.3333
    --environment.reward.deepmimic_foot_height_temperature=0.05
    --environment.reward.deepmimic_swing_match_weight_ratio=0.5
    --environment.reward.foot_z_velocity_coeff=1.0
    --environment.reward.root_heading_tracking_weight_ratio=0.20
    --environment.reward.root_heading_tracking_temperature=2.0
    --environment.termination.tracking_deviation_ratio=0.0
    --environment.reward.tracking_post_contact_penalties=True
    --environment.reward.foot_slip_coeff=6.6667
    --environment.reward.ground_penetration_coeff=1000
    --environment.domain_randomization.initial_state.type=reference
    --environment.domain_randomization.seen_robot.morphology_coeff_mode=fixed
    --environment.domain_randomization.seen_robot.morphology_coeff_value=0.0
    --environment.env_curriculum_coeff_max=0.6
    --environment.env_curriculum_level_success_tracking_ratio=0.0
    --environment.nr_envs="$NR_ENVS"
    --algorithm.nr_steps=64
    --algorithm.minibatch_size="$MINIBATCH"
    --algorithm.nr_epochs=5
    --algorithm.joint_latent_encoder_dim=4
    --algorithm.total_timesteps="$TOTAL"
    --algorithm.evaluation_active=False
    --algorithm.evaluation_and_save_frequency="$SAVE_EVERY"
    --environment.render=False
    --runner.track_console=True --runner.track_tb=False --runner.track_wandb=False
    --runner.save_model=True --runner.project_name=local_3t --runner.exp_name="$NAME"
  )
  local log="$OUT/${NAME}.log"
  local attempt=0 last_steps=-1
  while [ "$attempt" -lt 6 ]; do
    attempt=$((attempt + 1))
    local RESUME=() ckpt
    ckpt=$(newest_ckpt "$NAME")
    [ -n "$ckpt" ] && RESUME=(--runner.load_model="$ckpt")
    echo "[ln6h20] START $NAME attempt=$attempt latent=$LOBS resume=${ckpt:-none}  $(date -Is)"
    date +%s > "$OUT/${NAME}.heartbeat"
    $PY experiment.py "${WORLD[@]}" ${RESUME[@]+"${RESUME[@]}"} >> "$log" 2>&1
    local rc=$?
    local steps; steps=$(steps_done "$log"); steps=${steps:-0}
    echo "[ln6h20] END   $NAME attempt=$attempt rc=$rc steps=$steps  $(date -Is)"
    if [ "$steps" -ge "$TOTAL" ] 2>/dev/null; then
      touch "$OUT/${NAME}.done"; echo "[ln6h20] OK $NAME"; return 0
    fi
    if [ "$steps" -le "$last_steps" ]; then
      echo "[ln6h20] NO PROGRESS -- giving up on $NAME"; return 1
    fi
    last_steps=$steps
    sleep 20
  done
  return 1
}

run_arm ln6h20_ref "$MERGED" False
run_arm ln6h20_tok "$TOKV2" True

echo "[ln6h20] === eval phase ==="
EOUT=$OUT/ln6h20_eval; RND=$OUT/renders_ln6h20
mkdir -p "$EOUT" "$RND"
cd "$REPO/experiments/fsq_khaendler"
MR=$(newest_ckpt ln6h20_ref); MT=$(newest_ckpt ln6h20_tok)
echo "[ln6h20] models: $MR | $MT"
COMMON=(--clip dance2_subject4.npz --refbias 0.0 --anchor absolute
        --fitvariant False --refroot True --refroot_floor True --refvel_obs False
        --root_heading_obs True --contact_timeconst 0.004
        --robots unitree_h1:unitree_g1:booster_t1 --nr_envs 96 --steps 1000 --reference_hold 20)
TOKF=(--latent --latent_dim 32 --latent_replaces False --latent_scope per_joint
      --latent_divisor 10.0 --jlat_enc_dim 4 --latent_hold 1)
evalrun() {  # <tag> <model> <cdir> <zero> <seed> <lat> [dump]
  local args=(--model_path "$2" --clip_dir "$3" --raw_clip_dir "$MERGED"
              "${COMMON[@]}" --seed "$5" --out "$EOUT/${1}_s${5}.json")
  [ "$4" = "1" ] && args+=(--zero_action)
  [ "$6" = "1" ] && args+=("${TOKF[@]}")
  [ -n "${7:-}" ] && args+=(--dump_render "$RND/$7" --record_envs 3)
  echo "[ln6h20eval] $1 s$5 $(date -Is)"
  $PY "$REPO/experiments/fsq_khaendler/crosseval_motion.py" "${args[@]}" > "$EOUT/${1}_s${5}.log" 2>&1
  [ -s "$EOUT/${1}_s${5}.json" ] && echo "  OK" || echo "  MISSING"
}
evalrun ln6h20_ref_h20 "$MR" "$MERGED" 0 0 0 ln6h20_ref
evalrun ln6h20_tok_h20 "$MT" "$TOKV2"  0 0 1 ln6h20_tok
evalrun ln6h20_zero    "$MR" "$MERGED" 1 0 0
for s in 1 2 3; do
  evalrun ln6h20_ref_h20 "$MR" "$MERGED" 0 "$s" 0
  evalrun ln6h20_tok_h20 "$MT" "$TOKV2"  0 "$s" 1
  evalrun ln6h20_zero    "$MR" "$MERGED" 1 "$s" 0
done
echo "[ln6h20] ALL DONE $(date -Is)"
