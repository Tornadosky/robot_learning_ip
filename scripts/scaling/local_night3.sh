#!/usr/bin/env bash
# NIGHT3 LOCAL (2026-08-30 -> 31) -- the two arms Viper cannot run (ROCm wall
# at >2 topologies): the 3-robot story-B pair.
#
#   ln3_ref  H1+G1+T1, practical recipe + the n3gb33 gait dose
#            (FOOTH 0.3333 @ temp 0.05, GROUNDPEN 1000, FOOTSLIP 6.6667,
#            CONTACT_TIMECONST 0.004) + heading (0.20/2.0/observed).
#            The presentable-baseline rerun (roadmap B1) at the L1 budget so
#            the delta vs l3t_fix_head_rx is dose, not compute.
#   ln3_tok  IDENTICAL + the FSQ token at tk4 routing, reading
#            clips_3t_token (fixed-T1 npz + tokenizer_3t per-joint z_q,
#            md5-verified against LAFAN1_3t). The FIRST 3-robot token arm
#            (roadmap B2) -- answers "does FSQ survive 3 topologies".
#
# Same resume/heartbeat machinery as local_3t_fixed.sh (WSL VM deaths are the
# observed failure mode). Launch detached from WSL:
#   setsid nohup bash scripts/scaling/local_night3.sh \
#     > experiments/local_3t/ln3_run.log 2>&1 < /dev/null &
set -u

REPO=/mnt/c/Users/smirn/Desktop/robot_learning_ip
PY=~/jaxgpu/bin/python
MERGED=$REPO/external_data/amass_converted/LAFAN1_3t
TOKDIR=$REPO/experiments/fsq_khaendler/clips_3t_token
OUT=$REPO/experiments/local_3t
mkdir -p "$OUT"

# Same guard as local_3t_fixed.sh: never train T1 against the impossible clip.
ORIG=$REPO/external_data/amass_converted/LAFAN1
for d in "$MERGED" "$TOKDIR"; do
  cmp -s "$d/BoosterT1/dance2_subject4.npz" "$ORIG/BoosterT1/dance2_subject4.npz" \
    && { echo "[ln3] ABORT: BoosterT1 clip in $d is the ORIGINAL (infeasible)"; exit 1; }
done
[ -f "$TOKDIR/BoosterT1/dance2_subject4_zq.npz" ] || { echo "[ln3] ABORT: no T1 zq"; exit 1; }
echo "[ln3] clip guards passed"

cd "$REPO/loco_mjx/experiments"
export PYTHONPATH="$REPO:$REPO/RL-X:$REPO/loco_mjx"
export MUJOCO_GL=disable
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
export JAX_ENABLE_COMPILATION_CACHE=true
export JAX_COMPILATION_CACHE_DIR=$REPO/.jax_cache_local
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

NR_ENVS=${NR_ENVS:-192}
MINIBATCH=${MINIBATCH:-6144}   # 192*64=12288 batch; 8192 % 3 was the 3-topology killer
SAVE_EVERY=${SAVE_EVERY:-1966080}
TOTAL=${TOTAL:-19660800}

newest_ckpt() {
  ls -t "$REPO"/loco_mjx/experiments/runs/local_3t/"$1"/*/models/latest.model 2>/dev/null | head -1
}
steps_done() {
  grep -o "nr_env_steps[^0-9]*[0-9]\+" "$1" 2>/dev/null | grep -o "[0-9]\+$" | tail -1
}

run_arm() {  # run_arm <name> <clipdir> <latent_obs True|False>
  local NAME="$1" CDIR="$2" LOBS="$3"
  local WORLD=(
    --environment.name=locomotion.urma2.mjx
    --algorithm.name=urma2.mjx
    --environment.train_robots="('unitree_h1','unitree_g1','booster_t1')"
    --environment.terrain.type=plane
    --environment.critic_exteroceptive_observation_type=none
    --environment.terrain.contact_solref_timeconst=0.004
    --environment.command.type=tracking_clip
    --environment.reward.type=tracking
    --environment.reward.log_info=True
    --environment.command.tracking_clip_dir="$CDIR"
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
    --environment.command.tracking_clip_observe_root_heading=True
    --environment.command.tracking_clip_latent_hold=1
    --environment.command.tracking_clip_latent_obs="$LOBS"
    --environment.command.tracking_clip_latent_dim=32
    --environment.command.tracking_clip_latent_replaces_reference=False
    --environment.command.tracking_clip_latent_scope=per_joint
    --environment.command.tracking_clip_latent_obs_divisor=10.0
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
    --environment.reward.deepmimic_foot_height_weight_ratio=0.3333
    --environment.reward.deepmimic_foot_height_temperature=0.05
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
    --runner.track_console=True
    --runner.track_tb=False
    --runner.track_wandb=False
    --runner.save_model=True
    --runner.project_name=local_3t
    --runner.exp_name="$NAME"
  )
  local log="$OUT/${NAME}.log"
  local attempt=0 last_steps=-1
  while [ "$attempt" -lt "${MAX_ATTEMPTS:-6}" ]; do
    attempt=$((attempt + 1))
    local RESUME=() ckpt
    ckpt=$(newest_ckpt "$NAME")
    [ -n "$ckpt" ] && RESUME=(--runner.load_model="$ckpt")
    echo "[ln3] START $NAME attempt=$attempt latent=$LOBS resume=${ckpt:-none}  $(date -Is)"
    date +%s > "$OUT/${NAME}.heartbeat"
    $PY experiment.py "${WORLD[@]}" ${RESUME[@]+"${RESUME[@]}"} >> "$log" 2>&1
    local rc=$?
    local blocks steps
    blocks=$(grep -c "nr_env_steps" "$log" 2>/dev/null || echo 0)
    steps=$(steps_done "$log"); steps=${steps:-0}
    echo "[ln3] END   $NAME attempt=$attempt rc=$rc blocks=$blocks steps=$steps  $(date -Is)"
    if [ "$steps" -ge "$TOTAL" ] 2>/dev/null; then
      touch "$OUT/${NAME}.done"; echo "[ln3] OK $NAME ($steps steps)"; return 0
    fi
    if [ "$rc" -eq 0 ] && [ "$blocks" -gt 10 ] && [ "$steps" -ge $((TOTAL * 95 / 100)) ] 2>/dev/null; then
      touch "$OUT/${NAME}.done"; echo "[ln3] OK $NAME ($steps steps, within 5%)"; return 0
    fi
    echo "[ln3] incomplete ($steps/$TOTAL) -- last error lines:"
    grep -E "Error|Traceback|Exception|out of memory|RESOURCE" "$log" | tail -4
    if [ "$steps" -le "$last_steps" ]; then
      echo "[ln3] NO PROGRESS since previous attempt ($steps <= $last_steps) -- giving up on $NAME"
      return 1
    fi
    last_steps=$steps
    sleep 20
  done
  echo "[ln3] EXHAUSTED attempts for $NAME"
  return 1
}

run_arm ln3_ref "$MERGED" False
run_arm ln3_tok "$TOKDIR" True
echo "[ln3] ALL DONE $(date -Is)"
