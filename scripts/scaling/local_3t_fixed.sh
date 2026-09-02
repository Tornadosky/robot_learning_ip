#!/usr/bin/env bash
# ONE POLICY, THREE TOPOLOGIES -- on a reference BoosterT1 can actually reach.
#
# `local_3t_dance4` is the existing baseline: H1 1.73x / G1 1.92x over the
# zero-action floor, and BoosterT1 1.12x -- WORSE than zero action at matched
# early age. §14.1/§14.8 showed why: 13.75 % of T1's reference frames sat outside
# jnt_range, so T1 was scored against a target it cannot reach, and compute did
# nothing for it (-8.8 % over 10x steps against H1's -41 %).
#
# §14.10 re-issued the clips with the joint limits actually enforced. This run is
# the same recipe on the same motion with ONE variable changed: BoosterT1 reads
# the regenerated clip instead of the impossible one. H1 and G1 read exactly the
# clips they always did, so they double as a built-in null control -- if they
# move, something other than the T1 reference changed.
#
# ARM 2 adds heading, which wave 6 CH-L showed is worth 80.7 -> 8.7 deg on H1 for
# +8.1 % joint RMSE, and which matters here because dance2_subject4 accumulates
# 12 058 deg of yaw and spends 74.8 % of its frames more than 90 deg from its
# starting facing.
#
# Launch detached:
#   setsid nohup bash scripts/scaling/local_3t_fixed.sh > .../l3t_fixed_run.log 2>&1 < /dev/null &
#
# Each arm is RESUMABLE and the script writes <arm>.heartbeat, so a Windows-side
# watchdog can tell "still working" from "the VM took it down" -- which is what
# happened to the first attempt. A supervisor living inside WSL cannot survive
# the WSL service dying, which is the failure this run actually hit.
set -u

REPO=/mnt/c/Users/smirn/Desktop/robot_learning_ip
PY=~/jaxgpu/bin/python
ORIG=$REPO/external_data/amass_converted/LAFAN1
FIXED=$REPO/external_data/amass_converted/LAFAN1_fixed
MERGED=$REPO/external_data/amass_converted/LAFAN1_3t
OUT=$REPO/experiments/local_3t
mkdir -p "$OUT"

# The env resolves <tracking_clip_dir>/<robot subdir>/<clip>, so one directory
# has to hold all three families. H1 and G1 come from the ORIGINAL set -- they
# were already feasible (§14.8: G1 clean on all 10 clips, H1 on 10 of 11) and
# re-issuing them would perturb references that trained checkpoints were scored
# against for no reason. Only BoosterT1 is swapped.
mkdir -p "$MERGED"
for r in UnitreeH1 UnitreeG1; do
  mkdir -p "$MERGED/$r"; cp -f "$ORIG/$r/"*.npz "$MERGED/$r/" 2>/dev/null
done
mkdir -p "$MERGED/BoosterT1"
cp -f "$FIXED/BoosterT1/"*.npz "$MERGED/BoosterT1/"
echo "[3t] merged clip dir:"
for r in UnitreeH1 UnitreeG1 BoosterT1; do
  echo "  $r: $(ls "$MERGED/$r" | wc -l) clips"
done
# Fail loudly rather than silently training against the wrong reference.
cmp -s "$MERGED/BoosterT1/dance2_subject4.npz" "$ORIG/BoosterT1/dance2_subject4.npz" \
  && { echo "[3t] ABORT: BoosterT1 clip is still the ORIGINAL"; exit 1; }
echo "[3t] verified: BoosterT1 clip differs from the original"

cd "$REPO/loco_mjx/experiments"
export PYTHONPATH="$REPO:$REPO/RL-X:$REPO/loco_mjx"
export MUJOCO_GL=disable
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
export JAX_ENABLE_COMPILATION_CACHE=true
export JAX_COMPILATION_CACHE_DIR=$REPO/.jax_cache_local
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

NR_ENVS=${NR_ENVS:-192}
MINIBATCH=${MINIBATCH:-6144}
SAVE_EVERY=${SAVE_EVERY:-1966080}
TOTAL=${TOTAL:-19660800}

# Newest checkpoint for an experiment name, across all of RL-X's timestamped run
# directories -- resuming creates a NEW one, so "the run dir" is not stable.
newest_ckpt() {
  ls -t "$REPO"/loco_mjx/experiments/runs/local_3t/"$1"/*/models/latest.model 2>/dev/null | head -1
}

# Steps completed, read from the log rather than trusted: RL-X logs an uncaught
# exception and still exits 0, so exit codes cannot be used to decide "finished".
steps_done() {
  grep -o "nr_env_steps[^0-9]*[0-9]\+" "$1" 2>/dev/null | grep -o "[0-9]\+$" | tail -1
}

run_arm() {  # run_arm <name> <heading_ratio> <heading_temp> <observe_heading> [foot_slip] [ground_pen] [footh_ratio]
  # Contact args 5-7 default to the values ARM 1 has trained on since 18:40, so
  # the resumed checkpoint keeps an identical config. ARM 2 passes the CH-M
  # rx_p3f0 dose (read from rx_p3f0_s2_11174732.out, not from a template).
  local NAME="$1" HRATIO="$2" HTEMP="$3" HOBS="$4"
  local SLIP="${5:-20.0}" GPEN="${6:-1000}" FOOTHW="${7:-1.0}"
  local WORLD=(
    --environment.name=locomotion.urma2.mjx
    --algorithm.name=urma2.mjx
    --environment.train_robots="('unitree_h1','unitree_g1','booster_t1')"
    --environment.terrain.type=plane
    --environment.critic_exteroceptive_observation_type=none
    --environment.terrain.contact_solref_timeconst=0.0
    --environment.command.type=tracking_clip
    --environment.reward.type=tracking
    --environment.reward.log_info=True
    --environment.command.tracking_clip_dir="$MERGED"
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
    --environment.reward.deepmimic_foot_height_weight_ratio="$FOOTHW"
    --environment.reward.deepmimic_foot_height_temperature=0.01
    --environment.reward.foot_z_velocity_coeff=1.0
    --environment.reward.root_heading_tracking_weight_ratio="$HRATIO"
    --environment.reward.root_heading_tracking_temperature="$HTEMP"
    --environment.command.tracking_clip_observe_root_heading="$HOBS"
    --environment.termination.tracking_deviation_ratio=0.0
    --environment.reward.tracking_post_contact_penalties=True
    --environment.reward.foot_slip_coeff="$SLIP"
    --environment.reward.ground_penetration_coeff="$GPEN"
    --environment.command.tracking_clip_latent_hold=1
    --environment.domain_randomization.initial_state.type=reference
    --environment.domain_randomization.seen_robot.morphology_coeff_mode=fixed
    --environment.domain_randomization.seen_robot.morphology_coeff_value=0.0
    --environment.env_curriculum_coeff_max=0.6
    --environment.env_curriculum_level_success_tracking_ratio=0.0
    --environment.nr_envs="$NR_ENVS"
    --algorithm.nr_steps=64
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
  local log="$OUT/${NAME}.log"
  # RESUMABLE. The first attempt at this run died 12 minutes in, before a single
  # training block, when the WSL VM dropped (Wsl/Service/0x8007274c, hit four
  # times in an hour). Nothing was checkpointed, so the whole compile was lost.
  # Each attempt now resumes from the newest checkpoint if one exists, and the
  # loop stops if an attempt fails to ADVANCE -- otherwise a reproducible crash
  # becomes an infinite restart.
  local attempt=0 last_steps=-1
  while [ "$attempt" -lt "${MAX_ATTEMPTS:-6}" ]; do
    attempt=$((attempt + 1))
    local RESUME=() ckpt
    ckpt=$(newest_ckpt "$NAME")
    [ -n "$ckpt" ] && RESUME=(--runner.load_model="$ckpt")
    echo "[3t] START $NAME attempt=$attempt heading=$HRATIO/$HTEMP obs=$HOBS resume=${ckpt:-none}  $(date -Is)"
    date +%s > "$OUT/${NAME}.heartbeat"
    $PY experiment.py "${WORLD[@]}" ${RESUME[@]+"${RESUME[@]}"} >> "$log" 2>&1
    local rc=$?
    local blocks steps
    blocks=$(grep -c "nr_env_steps" "$log" 2>/dev/null || echo 0)
    steps=$(steps_done "$log"); steps=${steps:-0}
    echo "[3t] END   $NAME attempt=$attempt rc=$rc blocks=$blocks steps=$steps  $(date -Is)"

    if [ "$steps" -ge "$TOTAL" ] 2>/dev/null; then
      touch "$OUT/${NAME}.done"; echo "[3t] OK $NAME ($steps steps)"; return 0
    fi
    if [ "$rc" -eq 0 ] && [ "$blocks" -gt 10 ] && [ "$steps" -ge $((TOTAL * 95 / 100)) ] 2>/dev/null; then
      touch "$OUT/${NAME}.done"; echo "[3t] OK $NAME ($steps steps, within 5%)"; return 0
    fi
    echo "[3t] incomplete ($steps/$TOTAL) -- last error lines:"
    grep -E "Error|Traceback|Exception|out of memory|RESOURCE" "$log" | tail -4
    if [ "$steps" -le "$last_steps" ]; then
      echo "[3t] NO PROGRESS since the previous attempt ($steps <= $last_steps) -- giving up on $NAME"
      return 1
    fi
    last_steps=$steps
    sleep 20
  done
  echo "[3t] EXHAUSTED attempts for $NAME"
  return 1
}

# ARM 1: the T1 reference fix ALONE. H1/G1 unchanged = built-in null control.
# Contact args are the values it has always run with -- do NOT change them on a
# resume, that would confound the one variable L1 tests.
run_arm l3t_fix 0.0 0.25 False 20.0 1000 1.0
# ARM 2 (plan L2, `l3t_fix_head_rx`): the paper's final configuration --
# CH-M's rx_p3f0 recipe (POSTCONTACT at the 3.3x-reduced dose, FOOTH off)
# + heading at CH-L's shipped operating point (0.20 / 2.0 / observed).
run_arm l3t_fix_head_rx 0.20 2.0 True 6.6667 333.33 0.0
echo "[3t] ALL DONE $(date -Is)"
