#!/bin/bash
#SBATCH -J urma_h1g1
#SBATCH -p apu
#SBATCH -A mage_apu
#SBATCH --nodes=1
#SBATCH --ntasks=1
# One APU of two also caps cores at half the node's 48 physical cores.
#SBATCH --cpus-per-task=24
#SBATCH --ntasks-per-core=1
# MI300A is an APU with unified CPU/GPU memory, and Slurm rejects a job that
# asks for one of the node's two APUs but more than half its memory.
# Node: RealMemory=220000, 96 CPUs, gpu:2 -> one GPU allows at most 110000.
#SBATCH --mem=108000
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH -o /ptmp/akalenik/urma/logs/%x_%j.out
#SBATCH -e /ptmp/akalenik/urma/logs/%x_%j.out
#
# URMA2 H1+G1 on MI300A.
#
# Submit from viper11/12/13 -- the apu partition has
# AllocNodes=viper[11-13],vipa[1001-1300], so sbatch from viper01 is refused
# with "Access/permission denied" regardless of account.
#
# Env vars honoured: STAGE (fast|train), NR_ENVS, TOTAL_STEPS, SAVE_EVERY, EXP_NAME

module load python-waterboa/2025.06
eval "$(conda shell.bash hook)"
# VENV selects the jax/ROCm combo:
#   env     -> jax 0.7.1 (highest version with a matching rocm7 plugin under
#              RL-X's declared jax<=0.7.2)
#   env102  -> jax 0.10.2, the combo previously verified on MI300A
conda activate "/ptmp/akalenik/urma/${VENV:-env}"

ROOT=/ptmp/akalenik/urma
mkdir -p "$ROOT/logs"
cd "$ROOT/loco_mjx/experiments"

export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Compute nodes have no /dev/dri, so EGL cannot initialize and dm_control raises
# at import. Nothing here renders.
export MUJOCO_GL=disable
# XLA_EXTRA is set by the caller; leave empty to use XLA defaults. A first run
# with --xla_gpu_enable_command_buffer= died in a scatter fusion kernel with
# ROCM_ERROR_ILLEGAL_ADDRESS, so this is now a variable to bisect rather than a
# fixed assumption.
export XLA_FLAGS="${XLA_EXTRA:-}"
# urma2.mjx_split uses its own serialized AOT cache, and experiment.py RAISES if
# that is enabled together with the JAX persistent compilation cache. Branch per
# algorithm rather than exporting the JAX cache unconditionally.
ALGORITHM="${ALGORITHM:-urma2.mjx}"
if [ "$ALGORITHM" = "urma2.mjx_split" ]; then
  export JAX_ENABLE_COMPILATION_CACHE=false
  echo "mjx_split -> JAX persistent cache OFF"
else
  export JAX_ENABLE_COMPILATION_CACHE=true
  export JAX_COMPILATION_CACHE_DIR="$ROOT/jax_cache"
  export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
  export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1
  mkdir -p "$JAX_COMPILATION_CACHE_DIR"
fi

STAGE="${STAGE:-fast}"
NR_ENVS="${NR_ENVS:-2048}"
NR_STEPS="${NR_STEPS:-64}"
BATCH=$((NR_ENVS * NR_STEPS))
SAVE_EVERY="${SAVE_EVERY:-$((BATCH * 16))}"
TOTAL_STEPS="${TOTAL_STEPS:-$((SAVE_EVERY * 48))}"
EXP_NAME="${EXP_NAME:-viper_$STAGE}"

# `fast` matches the local cheap-world config so throughput is comparable;
# `train` is the full task (diverse hfield terrain + critic exteroception).
if [ "$STAGE" = "fast" ]; then
  WORLD=(--environment.terrain.type=plane --environment.critic_exteroceptive_observation_type=none)
elif [ "$STAGE" = "tracktrain" ]; then
  # Phase 1: same cheap world as `fast` (so throughput and the curriculum are
  # comparable to Phase 0) plus the tracking command/reward/observation channel.
  # log_info=True is what records env_info/joint_tracking_error, which is the
  # whole measurement -- without it the run is unreadable.
  WORLD=(--environment.terrain.type=plane --environment.critic_exteroceptive_observation_type=none
         --environment.command.type=tracking --environment.reward.type=tracking
         --environment.reward.log_info=True)
elif [ "$STAGE" = "cliptrain" ]; then
  # Phase 2: same world, but the reference is a REAL retargeted clip. The clip
  # directory is a LOCAL path in default_config.py, so CLIP_DIR must point at
  # the copy on this filesystem or every robot fails to build its reference.
  WORLD=(--environment.terrain.type=plane --environment.critic_exteroceptive_observation_type=none
         --environment.command.type=tracking_clip --environment.reward.type=tracking
         --environment.reward.log_info=True
         --environment.command.tracking_clip_dir="${CLIP_DIR:-/ptmp/akalenik/urma/clips/dance2_subject4}"
         --environment.reward.nominal_diff_target="${NOMINAL_TARGET:-reference}")
elif [ "$STAGE" = "mmtrain" ]; then
  # GaitProof N3: the local `mmtrain` stage, verbatim, on the APU.
  #
  # Two things here are NOT preferences and must not be "simplified":
  #   CLIP_DIR    default_config.py hardcodes a /mnt/c local path, so without
  #               this every robot fails to build its reference.
  #   TRACK_TEMP  0.05 is the shipped default and was MEASURED to destroy
  #               training (ep_len 25 vs 899). Never below 0.25.
  #
  # New in this run (N2): morphology_coeff_mode=schedule ramps the body
  # randomization instead of pinning it, and env_curriculum_coeff_max caps the
  # perturbation/noise regulator that went NaN on g1 at 0.84.
  WORLD=(--environment.terrain.type=plane --environment.critic_exteroceptive_observation_type=none
         # Contact insurance arms only; 0.0 = model default. Probe with
         # eval_contact_quality.py before trusting a value.
         --environment.terrain.contact_solref_timeconst="${CONTACT_TIMECONST:-0.0}"
         --environment.command.type=tracking_clip --environment.reward.type=tracking
         --environment.reward.log_info=True
         --environment.command.tracking_clip_dir="${CLIP_DIR:-/ptmp/akalenik/urma/clips/LAFAN1}"
         --environment.command.tracking_clip_file="${CLIP_FILE:-walk1_subject1.npz}"
         --environment.command.tracking_clip_fit_per_variant="${FITVARIANT:-True}"
         # ANCHOR=absolute makes the target the sign-corrected clip angle
         # (requires FITVARIANT=False); centered is the historical re-centered
         # reference. See clip_reference.load_clip.
         --environment.command.tracking_clip_anchor="${ANCHOR:-centered}"
         --environment.command.tracking_clip_velocity_command="${VELCMD:-True}"
         --environment.command.tracking_clip_cyclic="${CYCLIC:-False}"
         --environment.command.tracking_clip_amplitude_scale="${AMPSCALE:-1.0}"
         --environment.command.tracking_clip_velocity_scale="${VELSCALE:-1.0}"
         --environment.command.tracking_clip_observe_velocity="${REFVEL_OBS:-False}"
         --environment.command.tracking_clip_root_height_from_pose="${REFROOT:-False}"
         --environment.command.tracking_clip_root_height_pose_as_floor="${REFROOT_FLOOR:-False}"
         --environment.command.tracking_reference_action_bias="${REFBIAS:-0.0}"
         --environment.reward.nominal_diff_target="${NOMINAL_TARGET:-reference}"
         --environment.reward.joint_tracking_coeff="${TRACK_COEFF:-30.0}"
         --environment.reward.joint_tracking_temperature="${TRACK_TEMP:-0.25}"
         --environment.reward.joint_tracking_leg_weight="${LEGW:-1.0}"
         --environment.reward.tracking_curriculum_gated="${GATED:-False}"
         # L1-A (RealWalk): a FLOOR under the 25 gait terms, so they cannot be
         # switched off by a velocity-error regulator that never moves. The N3
         # run above ended with env_curriculum_coeff at 0.0012, which put
         # foot_slip at an effective 0.00012 and left nothing penalising ankle
         # chatter or floor clipping. Defaults are the pre-L1-A behaviour.
         --environment.reward.gait_coeff_mode="${GAITMODE:-curriculum}"
         --environment.reward.gait_coeff_value="${GAITCOEFF:-0.0}"
         --environment.reward.action_rate_coeff="${ACTRATE:-3.0}"
         --environment.reward.action_smoothness_coeff="${ACTSMOOTH:-0.1}"
         --environment.reward.deepmimic_enabled="${DEEPMIMIC:-True}"
         --environment.reward.deepmimic_qvel_temperature="${QVEL_TEMP:-0.5}"
         # B100 (2026-08-22) quality knobs. Defaults preserve old behaviour.
         # HEADING_RATIO alone is NOT enough for dance: at temp 0.25 the term
         # saturates dead past ~60 deg — raise HEADING_TEMP with it (B2: 2.0).
         --environment.reward.root_heading_tracking_weight_ratio="${HEADING_RATIO:-0.0}"
         --environment.reward.root_heading_tracking_temperature="${HEADING_TEMP:-0.25}"
         # TRACK_DEVIATION and DEVRATIO are the SAME flag; this line is the only
         # place it is passed (it appeared twice, and with last-flag-wins the
         # DEVRATIO default silently overrode any TRACK_DEVIATION export).
         --environment.termination.tracking_deviation_ratio="${TRACK_DEVIATION:-${DEVRATIO:-0.0}}"
         --environment.reward.tracking_post_contact_penalties="${POSTCONTACT:-False}"
         --environment.reward.foot_slip_coeff="${FOOTSLIP:-0.1}"
         --environment.reward.ground_penetration_coeff="${GROUNDPEN:-10.0}"
         # FEETFIX (23->24-08). FOOTH is the reference foot-HEIGHT term, the one
         # quantity nothing on this stack has ever scored; ratio-anchored to the
         # joint term like qvel/rpos/rquat, 0.0 = off and bit-identical.
         # FOOTZVEL was reachable only through EXTRA_ARGS before.
         --environment.reward.deepmimic_foot_height_weight_ratio="${FOOTH:-0.0}"
         --environment.reward.deepmimic_foot_height_temperature="${FOOTH_TEMP:-0.01}"
         --environment.reward.foot_z_velocity_coeff="${FOOTZVEL:-0.2}"
         --environment.command.tracking_clip_latent_hold="${LATENT_HOLD:-1}"
         # FSQ token routing (2026-08-29). Before this, LATENT_HOLD was the only
         # reachable token knob from here -- turning the token ON at all needed
         # EXTRA_ARGS. LATENT_SCOPE decides WHERE the code enters: "per_joint"
         # is the historical layout, which appends 32 code channels to each
         # joint's 5 and pushes all 37 through the joint encoder's single
         # Dense(4); "global" routes one pooled motion embedding into the
         # general observation, which reaches the 512-wide core unbottlenecked.
         # LATENT_DIVISOR is the per-channel normalisation the latent block
         # never had -- every other channel group has a set_norm entry, and at
         # divisor 1.0 the codes enter ~559x louder than the five real channels
         # beside them.
         --environment.command.tracking_clip_latent_obs="${LATENT_OBS:-False}"
         --environment.command.tracking_clip_latent_dim="${LATENT_DIM:-32}"
         --environment.command.tracking_clip_latent_replaces_reference="${LATENT_REPLACES:-False}"
         --environment.command.tracking_clip_latent_scope="${LATENT_SCOPE:-per_joint}"
         --environment.command.tracking_clip_latent_obs_divisor="${LATENT_DIVISOR:-1.0}"
         --environment.command.tracking_clip_latent_sidecar_suffix="${SIDECAR:-_zq}"
         --environment.domain_randomization.initial_state.type=reference
         --environment.domain_randomization.seen_robot.morphology_coeff_mode="${MORPH_MODE:-schedule}"
         --environment.domain_randomization.seen_robot.morphology_coeff_value="${MORPH_COEFF:-0.7}"
         --environment.domain_randomization.seen_robot.morphology_coeff_start="${MORPH_START:-0.2}"
         --environment.domain_randomization.seen_robot.morphology_coeff_ramp_steps="${MORPH_RAMP:-40000000}"
         --environment.env_curriculum_coeff_max="${CURMAX:-0.6}"
         --environment.env_curriculum_level_success_tracking_ratio="${CURTRACK:-0.0}")
else
  WORLD=()
fi

# Resume from a checkpoint when asked. Same robot set, so the resume guard in
# URMA2.load() is satisfied and the per-robot curriculum coefficients carry over
# instead of restarting from zero.
LOAD=()
if [ -n "${LOAD_MODEL:-}" ]; then
  LOAD=(--runner.load_model="$LOAD_MODEL")
fi

# Free-form extra flags, space separated. Used for the fast-ramp curriculum arm
# (--environment.env_curriculum_nr_levels=10), which reaches high morphology
# randomization inside an overnight budget so tracking error can be measured
# across a WIDE coefficient range rather than the narrow one the default ramp
# reaches. It scales only the step size per success, not the success criterion.
# Separator is a COLON, not a space, for the same reason ROBOTS_LIST is:
# `sbatch --export` splits its value on commas and a SPACE ends the argument
# entirely, so a space-separated list silently loses every flag after the first.
# No flag value here contains a colon.
EXTRA_ARGS_ARR=()
if [ -n "${EXTRA_ARGS:-}" ]; then
  IFS=':' read -r -a EXTRA_ARGS_ARR <<< "$EXTRA_ARGS"
fi
echo "EXTRA_ARGS_ARR=${EXTRA_ARGS_ARR[*]+${EXTRA_ARGS_ARR[*]}}"

# DANCEPROOF Check 27: every flag built from an env var was INVISIBLE in the run
# record. The submit script prints "VERIFY EVERY OVERRIDE before trusting a
# number" and that instruction was unenforceable -- only EXTRA_ARGS was echoed.
# Worse, most of these default to a NON-EMPTY value ("${VELCMD:-True}"), so an
# export that never arrived and an arm deliberately set the other way produce
# identical, indistinguishable logs.
#
# Echo the resolved WORLD flag list, then the env vars themselves. Both, because
# they fail differently: WORLD shows what the trainer was actually told, the env
# dump shows whether the export arrived at all.
echo "=== RESOLVED CONFIG FLAGS (danceproof Check 27) ==="
printf '  %s
' "${WORLD[@]}"
echo "=== ARM ENV AS EXPORTED ==="
for v in CLIP_FILE VELCMD CYCLIC AMPSCALE VELSCALE REFVEL_OBS REFROOT REFBIAS          TRACK_COEFF TRACK_TEMP DEVRATIO TRACK_DEVIATION MORPH_MODE MORPH_COEFF CURMAX          FITVARIANT ANCHOR GAITMODE GAITCOEFF QVEL_TEMP HEADING_RATIO HEADING_TEMP NOMINAL_TARGET ROBOTS_LIST POSTCONTACT FOOTSLIP GROUNDPEN FOOTZVEL FOOTH FOOTH_TEMP JLAT_CH LEGW AUX_COEFF AUX_HORIZON AUX_DETACH SIDECAR COTRAIN_ROWS COTRAIN_CH COTRAIN_RECON COTRAIN_FREEZE COTRAIN_INIT LATENT_HOLD LATENT_OBS LATENT_DIM LATENT_REPLACES LATENT_SCOPE LATENT_DIVISOR JLAT_ENC_DIM CONTACT_TIMECONST REFROOT_FLOOR; do
  echo "  $v=${!v:-<UNSET, sbatch default applies>}"
done
echo "=== END CONFIG ==="

echo "NODE=$(hostname) STAGE=$STAGE NR_ENVS=$NR_ENVS BATCH=$BATCH TOTAL=$TOTAL_STEPS"
python -c "import jax; print('devices:', jax.devices())"

# Which topologies share the policy. Overridable because the interesting
# follow-up to a failed zero-shot result is whether MORE training topologies
# close the gap -- an experiment the 24 GB local box cannot run (4 topologies in
# one jitted graph peaked at 22 GB there) but this 108 GB node can.
#
# Colon separated, not a Python tuple and not spaces: sbatch --export splits its
# value on commas, and a space ends the argument entirely. Colons survive both
# and never occur in robot names. The tuple is rebuilt here.
ROBOTS_LIST="${ROBOTS_LIST:-unitree_h1:unitree_g1}"
TRAIN_ROBOTS="("
for r in $(echo "$ROBOTS_LIST" | tr ':' ' '); do TRAIN_ROBOTS="$TRAIN_ROBOTS'$r',"; done
TRAIN_ROBOTS="$TRAIN_ROBOTS)"
echo "TRAIN_ROBOTS=$TRAIN_ROBOTS"

python experiment.py \
  --environment.name=locomotion.urma2.mjx \
  --algorithm.name="$ALGORITHM" \
  --environment.train_robots="$TRAIN_ROBOTS" \
  "${WORLD[@]}" \
  "${LOAD[@]}" \
  ${EXTRA_ARGS_ARR[@]+"${EXTRA_ARGS_ARR[@]}"} \
  --environment.nr_envs="$NR_ENVS" \
  --algorithm.nr_steps="$NR_STEPS" \
  --algorithm.minibatch_size="${MINIBATCH:-8192}" \
  --algorithm.nr_epochs=5 \
  --algorithm.joint_latent_encoder_dim="${JLAT_ENC_DIM:-4}" \
  --algorithm.joint_latent_channels="${JLAT_CH:-0}" \
  --algorithm.aux_token_coeff="${AUX_COEFF:-0.0}" \
  --algorithm.aux_token_horizon="${AUX_HORIZON:-1}" \
  --algorithm.aux_token_detach_trunk="${AUX_DETACH:-False}" \
  --algorithm.cotrain_window_rows="${COTRAIN_ROWS:-0}" \
  --algorithm.cotrain_window_channels="${COTRAIN_CH:-4}" \
  --algorithm.cotrain_recon_coeff="${COTRAIN_RECON:-1.0}" \
  --algorithm.cotrain_freeze_encoder="${COTRAIN_FREEZE:-False}" \
  --algorithm.cotrain_init_encoder="${COTRAIN_INIT:-}" \
  --algorithm.total_timesteps="$TOTAL_STEPS" \
  --algorithm.evaluation_active=False \
  --algorithm.evaluation_and_save_frequency="$SAVE_EVERY" \
  --environment.render=False \
  --runner.track_console=True \
  --runner.track_tb=True \
  --runner.track_wandb=False \
  --runner.save_model=True \
  --runner.project_name=urma2_h1g1 \
  --runner.exp_name="$EXP_NAME"

echo "TRAINING EXITED rc=$?"
