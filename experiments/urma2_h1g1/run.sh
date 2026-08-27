#!/bin/bash
# URMA2 H1+G1 morphology-randomization runs (WSL local GPU).
#
# Usage:  bash run.sh <stage>
#   smoke   - 1 robot, 4 envs, few steps. Proves the stack imports/compiles/steps on GPU.
#   pair    - h1+g1, small, DR on. Proves multi-robot + morphology randomization.
#   heldout - h1+g1 train, booster_t1/robotis_op3 eval. Proves the zero-shot eval path.
#   scale   - the real Phase 0 run.
set -eo pipefail

STAGE="${1:?usage: run.sh <smoke|pair|heldout|scale>}"
REPO=/mnt/c/Users/smirn/Desktop/robot_learning_ip
VENV=$HOME/locomjx
OUT=$REPO/experiments/urma2_h1g1

source "$VENV/bin/activate"
cd "$REPO/loco_mjx/experiments"

# MJX on a single consumer GPU: don't preallocate the whole card.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl

# Detached runs redirect to a file, where Python would otherwise block-buffer
# stdout and leave the log empty for minutes at a time.
export PYTHONUNBUFFERED=1

# Compile, not simulation, dominates these runs (~230 s per robot topology).
# Persist XLA executables so re-runs with the same robot set skip it.
# Safe here: experiment.py only forbids this for algorithm.name=urma2.mjx_split.
# The persistent compilation cache is a big win for urma2.mjx (identical
# re-runs skip a ~10 min compile). It is INCOMPATIBLE with urma2.mjx_split:
# experiment.py raises if the serialized AOT cache and this cache are both on.
# ALGORITHM is set per stage below; default to the fused variant.
ALGORITHM="${ALGORITHM:-urma2.mjx}"
if [ "$ALGORITHM" = "urma2.mjx_split" ]; then
  export JAX_ENABLE_COMPILATION_CACHE=false
  echo "run.sh: mjx_split -> JAX persistent cache OFF (serialized AOT cache is used instead)"
else
  export JAX_ENABLE_COMPILATION_CACHE=true
  export JAX_COMPILATION_CACHE_DIR="$HOME/.cache/jax_locomjx"
  export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
  export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1
  mkdir -p "$JAX_COMPILATION_CACHE_DIR"
fi

COMMON=(
  --environment.name=locomotion.urma2.mjx
  --runner.track_console=True
  --runner.track_tb=True
  --runner.track_wandb=False
  --runner.project_name=urma2_h1g1
)

# Resume from a checkpoint when LOAD_MODEL is set. The robot set matches, so
# URMA2.load() takes the resume branch and the per-robot curriculum coefficients
# and RNG carry over -- extending a run rather than restarting the curriculum.
LOAD=()
if [ -n "${LOAD_MODEL:-}" ]; then
  LOAD=(--runner.load_model="$LOAD_MODEL")
fi

# Which topologies share the policy. Colon separated, rebuilt into a Python
# tuple. Overridable so the "does more coverage buy zero-shot transfer?" run can
# use 3+ robots -- that experiment has to run HERE, because on Viper's MI300A
# more than two topologies in one graph always dies with
# ROCM_ERROR_ILLEGAL_ADDRESS at any env count (measured 2026-08-04).
#
# NOTE: nr_envs must be divisible by the number of robots, or create_env raises
# before any GPU work happens.
ROBOTS_LIST="${ROBOTS_LIST:-unitree_h1:unitree_g1}"
TRAIN_ROBOTS="("
for r in $(echo "$ROBOTS_LIST" | tr ':' ' '); do TRAIN_ROBOTS="$TRAIN_ROBOTS'$r',"; done
TRAIN_ROBOTS="$TRAIN_ROBOTS)"

case "$STAGE" in
  smoke)
    python experiment.py "${COMMON[@]}" \
      --algorithm.name=urma2.mjx \
      --environment.train_robots="('unitree_h1',)" \
      --environment.nr_envs=4 \
      --algorithm.nr_steps=8 \
      --algorithm.minibatch_size=8 \
      --algorithm.nr_epochs=1 \
      --algorithm.total_timesteps=64 \
      --algorithm.evaluation_active=False \
      --algorithm.evaluation_and_save_frequency=-1 \
      --environment.render=False \
      --runner.save_model=False \
      --runner.exp_name=smoke
    ;;
  pair)
    python experiment.py "${COMMON[@]}" \
      --algorithm.name=urma2.mjx \
      --environment.train_robots="('unitree_h1','unitree_g1')" \
      --environment.nr_envs=64 \
      --algorithm.nr_steps=16 \
      --algorithm.minibatch_size=256 \
      --algorithm.nr_epochs=1 \
      --algorithm.total_timesteps=4096 \
      --algorithm.evaluation_active=False \
      --algorithm.evaluation_and_save_frequency=-1 \
      --environment.render=False \
      --runner.save_model=False \
      --runner.exp_name=pair
    ;;
  heldout)
    python experiment.py "${COMMON[@]}" \
      --algorithm.name=urma2.mjx \
      --environment.train_robots="('unitree_h1','unitree_g1')" \
      --environment.eval_robots="('booster_t1','robotis_op3')" \
      --environment.nr_envs=64 \
      --environment.nr_eval_envs=8 \
      --algorithm.nr_steps=16 \
      --algorithm.minibatch_size=256 \
      --algorithm.nr_epochs=1 \
      --algorithm.total_timesteps=8192 \
      --algorithm.evaluation_active=True \
      --algorithm.evaluation_and_save_frequency=4096 \
      --environment.render=False \
      --runner.save_model=True \
      --runner.exp_name=heldout
    ;;
  train)
    # Phase 0 proper. Two topologies ONLY -- held-out robots are evaluated by a
    # separate process (eval_heldout.py) against the saved checkpoint.
    #
    # Why: every robot topology becomes another branch of one jitted graph, and
    # compiling 4 of them peaked at 22 GB RSS and thrashed a 24 GB box. Training
    # on 2 and evaluating offline keeps peak compile memory at the 2-topology
    # cost and lets us score any number of held-out robots after the fact.
    #
    # RL-X asserts: eval_freq % (nr_envs*nr_steps) == 0, total % eval_freq == 0.
    NR_ENVS="${NR_ENVS:-1024}"
    NR_STEPS="${NR_STEPS:-64}"
    BATCH=$((NR_ENVS * NR_STEPS))
    SAVE_EVERY="${SAVE_EVERY:-$((BATCH * 16))}"
    TOTAL="${TOTAL_STEPS:-$((SAVE_EVERY * 100))}"
    echo "train: nr_envs=$NR_ENVS batch=$BATCH save_every=$SAVE_EVERY total=$TOTAL"
    python experiment.py "${COMMON[@]}" \
      --algorithm.name=urma2.mjx \
      --environment.train_robots="('unitree_h1','unitree_g1')" \
      --environment.nr_envs="$NR_ENVS" \
      --algorithm.nr_steps="$NR_STEPS" \
      --algorithm.minibatch_size="${MINIBATCH:-8192}" \
      --algorithm.nr_epochs=5 \
      --algorithm.total_timesteps="$TOTAL" \
      --algorithm.evaluation_active=False \
      --algorithm.evaluation_and_save_frequency="$SAVE_EVERY" \
      --environment.render=False \
      --runner.save_model=True \
      --runner.exp_name="${EXP_NAME:-train}"
    ;;
  fast)
    # Phase 0 on a cheaper world, to get a result in hours rather than half a day.
    #
    # Two changes vs. `train`, both aimed at the per-step cost:
    #   terrain=plane          - drops per-foot heightfield lookups
    #   critic extero = none   - drops an 80x80 grid sampled per env per step
    #
    # This narrows the claim (flat ground, no terrain awareness) but should also
    # reach curriculum liftoff SOONER: the curriculum advances on velocity
    # tracking, an easier task clears its bar earlier, and morphology
    # randomization only switches on once that bar is cleared.
    NR_ENVS="${NR_ENVS:-1024}"
    NR_STEPS="${NR_STEPS:-64}"
    BATCH=$((NR_ENVS * NR_STEPS))
    SAVE_EVERY="${SAVE_EVERY:-$((BATCH * 32))}"
    TOTAL="${TOTAL_STEPS:-$((SAVE_EVERY * 48))}"
    echo "fast: nr_envs=$NR_ENVS batch=$BATCH save_every=$SAVE_EVERY total=$TOTAL"
    echo "fast: train_robots=$TRAIN_ROBOTS"
    python experiment.py "${COMMON[@]}" "${LOAD[@]}" \
      --algorithm.name=urma2.mjx \
      --environment.train_robots="$TRAIN_ROBOTS" \
      --environment.terrain.type=plane \
      --environment.critic_exteroceptive_observation_type=none \
      --environment.nr_envs="$NR_ENVS" \
      --algorithm.nr_steps="$NR_STEPS" \
      --algorithm.minibatch_size="${MINIBATCH:-8192}" \
      --algorithm.nr_epochs=5 \
      --algorithm.total_timesteps="$TOTAL" \
      --algorithm.evaluation_active=False \
      --algorithm.evaluation_and_save_frequency="$SAVE_EVERY" \
      --environment.render=False \
      --runner.save_model=True \
      --runner.exp_name="${EXP_NAME:-fast}"
    ;;
  tracktrain)
    # Phase 1 proper: the `fast` world plus the tracking command/reward channel.
    # reward.log_info=True is the whole point -- it is what writes
    # env_info/joint_tracking_error, the quantity the morphology-invariance
    # claim is measured on.
    NR_ENVS="${NR_ENVS:-1024}"
    NR_STEPS="${NR_STEPS:-64}"
    BATCH=$((NR_ENVS * NR_STEPS))
    SAVE_EVERY="${SAVE_EVERY:-$((BATCH * 32))}"
    TOTAL="${TOTAL_STEPS:-$((SAVE_EVERY * 48))}"
    echo "tracktrain: nr_envs=$NR_ENVS batch=$BATCH save_every=$SAVE_EVERY total=$TOTAL"
    python experiment.py "${COMMON[@]}" "${LOAD[@]}" \
      --algorithm.name=urma2.mjx \
      --environment.train_robots="('unitree_h1','unitree_g1')" \
      --environment.terrain.type=plane \
      --environment.critic_exteroceptive_observation_type=none \
      --environment.command.type=tracking \
      --environment.reward.type=tracking \
      --environment.reward.log_info=True \
      --environment.nr_envs="$NR_ENVS" \
      --algorithm.nr_steps="$NR_STEPS" \
      --algorithm.minibatch_size="${MINIBATCH:-8192}" \
      --algorithm.nr_epochs=5 \
      --algorithm.total_timesteps="$TOTAL" \
      --algorithm.evaluation_active=False \
      --algorithm.evaluation_and_save_frequency="$SAVE_EVERY" \
      --environment.render=False \
      --runner.save_model=True \
      --runner.exp_name="${EXP_NAME:-tracktrain}"
    ;;
  cliptrain)
    # Phase 2: the tracktrain world, but the reference is a REAL retargeted
    # motion (command.type=tracking_clip) instead of the procedural sinusoid.
    #
    # nominal_diff_target=reference is not optional here. Left at "nominal" the
    # keep-still regularizer pulls every joint toward the rest pose at coeff
    # 100.0 while the tracking term rewards leaving it at coeff 3.0 -- and "sit
    # at nominal" IS the ignore-the-reference baseline, so the default reward
    # pays 33x more for failing the task than for doing it.
    NR_ENVS="${NR_ENVS:-1024}"
    NR_STEPS="${NR_STEPS:-64}"
    BATCH=$((NR_ENVS * NR_STEPS))
    SAVE_EVERY="${SAVE_EVERY:-$((BATCH * 32))}"
    TOTAL="${TOTAL_STEPS:-$((SAVE_EVERY * 48))}"
    echo "cliptrain: nr_envs=$NR_ENVS batch=$BATCH save_every=$SAVE_EVERY total=$TOTAL"
    python experiment.py "${COMMON[@]}" "${LOAD[@]}" \
      --algorithm.name=urma2.mjx \
      --environment.train_robots="$TRAIN_ROBOTS" \
      --environment.terrain.type=plane \
      --environment.critic_exteroceptive_observation_type=none \
      --environment.command.type=tracking_clip \
      --environment.reward.type=tracking \
      --environment.reward.log_info=True \
      --environment.reward.nominal_diff_target="${NOMINAL_TARGET:-reference}" \
      --environment.reward.joint_tracking_coeff="${TRACK_COEFF:-30.0}" \
      --environment.reward.joint_tracking_temperature="${TRACK_TEMP:-0.05}" \
      --environment.nr_envs="$NR_ENVS" \
      --algorithm.nr_steps="$NR_STEPS" \
      --algorithm.minibatch_size="${MINIBATCH:-8192}" \
      --algorithm.nr_epochs=5 \
      --algorithm.total_timesteps="$TOTAL" \
      --algorithm.evaluation_active=False \
      --algorithm.evaluation_and_save_frequency="$SAVE_EVERY" \
      --environment.render=False \
      --runner.save_model=True \
      --runner.exp_name="${EXP_NAME:-cliptrain}"
    ;;
  dmtrain)
    # L4: does the DeepMimic pipeline carry a LEARNING SIGNAL? Direction check
    # only -- not a results run, not a tuning run.
    #
    # Differs from `cliptrain` in four ways, each of which is a fix for something
    # measured on 2026-08-04/05 rather than a preference:
    #
    #   initial_state.type=reference  cliptrain never sets this, so RSI is OFF
    #                                 there despite being the one thing that
    #                                 makes short tracking runs informative
    #                                 (reset error ratio 0.011 vs 1.102).
    #   joint_tracking_temperature    cliptrain defaults to 0.05, which is below
    #                                 the never-go-under-0.25 line and was
    #                                 measured to destroy training outright
    #                                 (ep_len 25 vs 899).
    #   deepmimic_enabled=True        the qvel/rpos/rquat terms this goal exists
    #                                 to add.
    #   tracking_curriculum_gated     keeps the tracking term from starving the
    #                                 velocity command and collapsing the
    #                                 curriculum, which switches off all 25 gait
    #                                 terms and produces a shuffling gait.
    NR_ENVS="${NR_ENVS:-1024}"
    NR_STEPS="${NR_STEPS:-64}"
    BATCH=$((NR_ENVS * NR_STEPS))
    SAVE_EVERY="${SAVE_EVERY:-$((BATCH * 8))}"
    TOTAL="${TOTAL_STEPS:-$((SAVE_EVERY * 20))}"
    echo "dmtrain: nr_envs=$NR_ENVS batch=$BATCH save_every=$SAVE_EVERY total=$TOTAL"
    python experiment.py "${COMMON[@]}" "${LOAD[@]}" \
      --algorithm.name=urma2.mjx \
      --environment.train_robots="$TRAIN_ROBOTS" \
      --environment.terrain.type=plane \
      --environment.critic_exteroceptive_observation_type=none \
      --environment.command.type=tracking_clip \
      --environment.reward.type=tracking \
      --environment.reward.log_info=True \
      --environment.reward.nominal_diff_target="${NOMINAL_TARGET:-reference}" \
      --environment.reward.joint_tracking_coeff="${TRACK_COEFF:-30.0}" \
      --environment.reward.joint_tracking_temperature="${TRACK_TEMP:-0.25}" \
      --environment.reward.tracking_curriculum_gated="${GATED:-True}" \
      --environment.reward.deepmimic_enabled="${DEEPMIMIC:-True}" \
      --environment.command.tracking_clip_velocity_command="${VELCMD:-False}" \
      --environment.domain_randomization.initial_state.type=reference \
      --environment.nr_envs="$NR_ENVS" \
      --algorithm.nr_steps="$NR_STEPS" \
      --algorithm.minibatch_size="${MINIBATCH:-8192}" \
      --algorithm.nr_epochs=5 \
      --algorithm.total_timesteps="$TOTAL" \
      --algorithm.evaluation_active=False \
      --algorithm.evaluation_and_save_frequency="$SAVE_EVERY" \
      --environment.render=False \
      --runner.save_model=True \
      --runner.exp_name="${EXP_NAME:-dmtrain}"
    ;;
  mmtrain)
    # MorphMimic: dmtrain plus the M0-M3 knobs. Everything dmtrain fixed stays
    # fixed; what is new here is that the BODY randomization is now set directly
    # instead of being whatever the velocity-error regulator allowed.
    #
    #   MORPH_MODE   curriculum | fixed. "fixed" pins morphology randomization at
    #                MORPH_COEFF regardless of the curriculum, which is the whole
    #                point of M0 and the only way the M4 sweep can hold a level.
    #   MORPH_COEFF  the level. 0.3 == bodies +-15% in every linear dimension.
    #   FITVARIANT   refit the clip to THIS body's joint limits every step (M1).
    #                Without it 17% of frames on the worst body at morph 0.3 are
    #                silently rewritten by the limit clamp.
    #   CURTRACK     let the curriculum criterion see tracking error (M2), as a
    #                ratio of the clip's ignore baseline. 0 = velocity-only.
    #   DEVRATIO     end the episode once tracking error exceeds this ratio of
    #                the ignore baseline (M3). 0 = off.
    #
    # Note MORPH_MODE=fixed makes the logged env_curriculum coefficient
    # IRRELEVANT as a morphology readout -- read env_curriculum/morphology_coeff
    # instead. Reading the old key here is the "curriculum flat while tracking
    # improves" trap listed in the fix playbook.
    NR_ENVS="${NR_ENVS:-1024}"
    NR_STEPS="${NR_STEPS:-64}"
    BATCH=$((NR_ENVS * NR_STEPS))
    SAVE_EVERY="${SAVE_EVERY:-$((BATCH * 8))}"
    TOTAL="${TOTAL_STEPS:-$((SAVE_EVERY * 20))}"
    echo "mmtrain: robots=$TRAIN_ROBOTS nr_envs=$NR_ENVS batch=$BATCH total=$TOTAL"
    echo "mmtrain: morph_mode=${MORPH_MODE:-fixed} morph_coeff=${MORPH_COEFF:-0.3} fit_variant=${FITVARIANT:-True} anchor=${ANCHOR:-centered} curtrack=${CURTRACK:-0.0} track_deviation=${TRACK_DEVIATION:-${DEVRATIO:-0.0}}"
    # L1-A (RealWalk): GAITMODE/GAITCOEFF put a FLOOR under the 25 gait terms so
    # they cannot be switched off by a stuck velocity-error regulator. Defaults
    # reproduce the old behaviour exactly (mode=curriculum, value ignored).
    echo "mmtrain: gait_mode=${GAITMODE:-curriculum} gait_coeff=${GAITCOEFF:-0.0} track_coeff=${TRACK_COEFF:-30.0} gated=${GATED:-False}"
    python experiment.py "${COMMON[@]}" "${LOAD[@]}" \
      --algorithm.name=urma2.mjx \
      --environment.train_robots="$TRAIN_ROBOTS" \
      --environment.timestep="${TIMESTEP:-0.005}" \
      --environment.terrain.type=plane \
      --environment.terrain.contact_solref_timeconst="${CONTACT_TIMECONST:-0.0}" \
      --environment.critic_exteroceptive_observation_type=none \
      --environment.command.type=tracking_clip \
      --environment.reward.type=tracking \
      --environment.reward.log_info=True \
      --environment.reward.nominal_diff_target="${NOMINAL_TARGET:-reference}" \
      --environment.reward.joint_tracking_coeff="${TRACK_COEFF:-30.0}" \
      --environment.reward.joint_tracking_temperature="${TRACK_TEMP:-0.25}" \
      --environment.reward.tracking_curriculum_gated="${GATED:-False}" \
      --environment.reward.root_heading_tracking_weight_ratio="${HEADING_RATIO:-0.0}" \
      --environment.reward.root_heading_tracking_temperature="${HEADING_TEMP:-0.25}" \
      --environment.termination.tracking_deviation_ratio="${TRACK_DEVIATION:-${DEVRATIO:-0.0}}" \
      --environment.reward.gait_coeff_mode="${GAITMODE:-curriculum}" \
      --environment.reward.gait_coeff_value="${GAITCOEFF:-0.0}" \
      --environment.reward.action_rate_coeff="${ACTRATE:-3.0}" \
      --environment.reward.action_smoothness_coeff="${ACTSMOOTH:-0.1}" \
      --environment.reward.feet_orientation_coeff="${FEET_ORIENT:-1.0}" \
      --environment.reward.deepmimic_enabled="${DEEPMIMIC:-True}" \
      --environment.reward.deepmimic_qvel_temperature="${QVEL_TEMP:-0.5}" \
      --environment.reward.tracking_post_contact_penalties="${POSTCONTACT:-False}" \
      --environment.reward.foot_slip_coeff="${FOOTSLIP:-0.1}" \
      --environment.reward.ground_penetration_coeff="${GROUNDPEN:-10.0}" \
      --environment.command.tracking_clip_dir="${CLIP_DIR:-/mnt/c/Users/smirn/Desktop/robot_learning_ip/external_data/amass_converted/LAFAN1}" \
      --environment.command.tracking_clip_file="${CLIP_FILE:-walk1_subject1.npz}" \
      --environment.command.tracking_clip_velocity_command="${VELCMD:-True}" \
      --environment.command.tracking_clip_fit_per_variant="${FITVARIANT:-True}" \
      --environment.command.tracking_clip_anchor="${ANCHOR:-centered}" \
      --environment.command.tracking_clip_cyclic="${CYCLIC:-False}" \
      --environment.command.tracking_clip_amplitude_scale="${AMPSCALE:-1.0}" \
      --environment.command.tracking_clip_velocity_scale="${VELSCALE:-1.0}" \
      --environment.command.tracking_clip_observe_velocity="${REFVEL_OBS:-False}" \
      --environment.command.tracking_clip_root_height_from_pose="${REFROOT:-False}" \
      --environment.command.tracking_reference_action_bias="${REFBIAS:-0.0}" \
      --environment.control_frequency_hz="${CTRL_HZ:-50}" \
      --environment.scaling_factor_multiplier="${SCALE_MULT:-1.0}" \
      --environment.seed="${ENVSEED:-1}" \
      --environment.domain_randomization.initial_state.type=reference \
      --environment.domain_randomization.seen_robot.morphology_coeff_mode="${MORPH_MODE:-fixed}" \
      --environment.domain_randomization.seen_robot.morphology_coeff_value="${MORPH_COEFF:-0.3}" \
      --environment.domain_randomization.seen_robot.morphology_coeff_start="${MORPH_START:-0.0}" \
      --environment.domain_randomization.seen_robot.morphology_coeff_ramp_steps="${MORPH_RAMP:-0}" \
      --environment.domain_randomization.seen_robot.torque_scaling_exponent="${TORQUE_EXP:-1.0}" \
      --environment.domain_randomization.seen_robot.exact_inertia_rescale="${EXACT_INERTIA:-False}" \
      --environment.env_curriculum_coeff_max="${CURMAX:-1.0}" \
      --environment.env_curriculum_level_success_tracking_ratio="${CURTRACK:-0.0}" \
      --environment.nr_envs="$NR_ENVS" \
      --algorithm.nr_steps="$NR_STEPS" \
      --algorithm.minibatch_size="${MINIBATCH:-8192}" \
      --algorithm.nr_epochs=5 \
      --algorithm.total_timesteps="$TOTAL" \
      --algorithm.evaluation_active=False \
      --algorithm.evaluation_and_save_frequency="$SAVE_EVERY" \
      --environment.render=False \
      --runner.save_model=True \
      --runner.exp_name="${EXP_NAME:-mmtrain}"
    ;;
  tracksmoke)
    # Phase 1 plumbing check: does the tracking command/reward/observation path
    # build, compile and step on BOTH topologies at once?
    python experiment.py "${COMMON[@]}" \
      --algorithm.name=urma2.mjx \
      --environment.train_robots="('unitree_h1','unitree_g1')" \
      --environment.command.type=tracking \
      --environment.reward.type=tracking \
      --environment.reward.log_info=True \
      --environment.nr_envs=32 \
      --algorithm.nr_steps=16 \
      --algorithm.minibatch_size=128 \
      --algorithm.nr_epochs=1 \
      --algorithm.total_timesteps=2048 \
      --algorithm.evaluation_active=False \
      --algorithm.evaluation_and_save_frequency=-1 \
      --environment.render=False \
      --runner.save_model=False \
      --runner.exp_name=tracksmoke
    ;;
  splitsmoke|splitscale)
    # urma2.mjx_split: per-robot network applications and per-robot donated
    # buffers, instead of compiling every topology into ONE jitted graph.
    #
    # This is the whole reason the topology ceiling might move. The fused
    # urma2.mjx builds a single graph with a branch per robot, which is what
    # peaked at 22 GB on 4 topologies here and dies with ROCM_ERROR_ILLEGAL_ADDRESS
    # for >2 topologies on Viper. mjx_split keeps "the per-robot buffers the only
    # big residents" (urma2_split.py:578), so peak memory should grow far more
    # slowly with the robot count.
    #
    # Both hard divisibility constraints are enforced here rather than
    # discovered at runtime: create_env raises if nr_envs % nr_robots, and the
    # split update raises if minibatch_size % nr_robots.
    NR_ROBOTS=$(echo "$ROBOTS_LIST" | tr ':' ' ' | wc -w)
    if [ "$STAGE" = "splitsmoke" ]; then
      NR_ENVS="${NR_ENVS:-$((8 * NR_ROBOTS))}"
      NR_STEPS="${NR_STEPS:-16}"
      MINIBATCH="${MINIBATCH:-$((8 * NR_ROBOTS))}"
      TOTAL="${TOTAL_STEPS:-$((NR_ENVS * NR_STEPS * 2))}"
      SAVE_FLAG=-1
      SAVE_MODEL=False
    else
      # Round nr_envs to a multiple of the robot count, keeping per-robot env
      # count constant so sps is comparable ACROSS topology counts -- the whole
      # point of the scaling sweep.
      PER_ROBOT_ENVS="${PER_ROBOT_ENVS:-128}"
      NR_ENVS="${NR_ENVS:-$((PER_ROBOT_ENVS * NR_ROBOTS))}"
      NR_STEPS="${NR_STEPS:-32}"
      MINIBATCH="${MINIBATCH:-$((512 * NR_ROBOTS))}"
      TOTAL="${TOTAL_STEPS:-$((NR_ENVS * NR_STEPS * 6))}"
      # Default is a throwaway memory/throughput probe, so nothing is saved.
      # Set SAVE_EVERY to turn the same stage into a real training run -- RL-X
      # asserts save_every % (nr_envs*nr_steps) == 0 and total % save_every == 0.
      if [ -n "${SAVE_EVERY:-}" ]; then
        SAVE_FLAG="$SAVE_EVERY"
        SAVE_MODEL=True
      else
        SAVE_FLAG=-1
        SAVE_MODEL=False
      fi
    fi
    echo "$STAGE: robots=$NR_ROBOTS ($ROBOTS_LIST) nr_envs=$NR_ENVS minibatch=$MINIBATCH total=$TOTAL"
    # /usr/bin/time -v gives Maximum resident set size -- the number the
    # topology-ceiling claim rests on. Without it the sweep has no memory axis.
    /usr/bin/time -v python experiment.py "${COMMON[@]}" "${LOAD[@]}" \
      --algorithm.name=urma2.mjx_split \
      --environment.train_robots="$TRAIN_ROBOTS" \
      --environment.terrain.type=plane \
      --environment.critic_exteroceptive_observation_type=none \
      --environment.nr_envs="$NR_ENVS" \
      --algorithm.nr_steps="$NR_STEPS" \
      --algorithm.minibatch_size="$MINIBATCH" \
      --algorithm.nr_epochs=5 \
      --algorithm.total_timesteps="$TOTAL" \
      --algorithm.evaluation_active=False \
      --algorithm.evaluation_and_save_frequency="$SAVE_FLAG" \
      --environment.render=False \
      --runner.save_model="$SAVE_MODEL" \
      --runner.exp_name="${EXP_NAME:-$STAGE}"
    ;;
  mmsplit)
    # M5: does the DeepMimic reward run on 3+ families at once, via mjx_split?
    #
    # This is a SYSTEMS check -- construct, step, train, and report sps and peak
    # RSS against the M4 two-family numbers. It is NOT a tracking-quality run,
    # and the reason is worth stating at the point of use rather than in a
    # report footnote:
    #
    #   Only unitree_h1 and unitree_g1 have a retargeted WALK clip. Atlas and
    #   toddlerbot ship only `dance2_*`, and the dance clip is MEASURED
    #   untrackable (every coefficient tried sat exactly at the ignore
    #   baseline). The one clip all four families share is dance2_subject1, so
    #   that is what this stage uses -- which makes every tracking ratio it
    #   produces meaningless. Read sps and RSS from this stage, nothing else.
    #
    # mjx_split is the many-family path: per-robot network applications and
    # per-robot donated buffers instead of one fused graph with a branch per
    # robot. The fused urma2.mjx peaked at 22 GB on 4 topologies here.
    NR_ROBOTS=$(echo "$ROBOTS_LIST" | tr ':' ' ' | wc -w)
    PER_ROBOT_ENVS="${PER_ROBOT_ENVS:-128}"
    NR_ENVS="${NR_ENVS:-$((PER_ROBOT_ENVS * NR_ROBOTS))}"
    NR_STEPS="${NR_STEPS:-32}"
    MINIBATCH="${MINIBATCH:-$((512 * NR_ROBOTS))}"
    TOTAL="${TOTAL_STEPS:-$((NR_ENVS * NR_STEPS * 6))}"
    CLIP_FILE="${CLIP_FILE:-dance2_subject1.npz}"
    # Keyed by BOTH the directory name and the SHORT name, because
    # resolve_clip_path resolves `robot_config.get("name") or short_name` and
    # most robot_configs have no "name" -- so atlas arrives as "at" and
    # toddlerbot as "tdlb". Mapping only the directory names fails with
    # "No clip mapping for robot 'at'", which is what the first 4-family attempt
    # did. The stock map already carried both forms for h1/g1 for this reason.
    CLIP_MAP="${CLIP_MAP:-{\"unitree_h1\": \"UnitreeH1\", \"unitree_g1\": \"UnitreeG1\", \"h1\": \"UnitreeH1\", \"g1\": \"UnitreeG1\", \"atlas\": \"Atlas\", \"at\": \"Atlas\", \"toddlerbot\": \"ToddlerBot\", \"tdlb\": \"ToddlerBot\", \"booster_t1\": \"BoosterT1\", \"t1\": \"BoosterT1\", \"talos\": \"Talos\", \"tl\": \"Talos\"\}}"
    # SAVE_EVERY turns the throughput probe into a real training run (same
    # pattern as splitscale). RL-X asserts save_every % (nr_envs*nr_steps) == 0
    # and total % save_every == 0.
    if [ -n "${SAVE_EVERY:-}" ]; then
      SAVE_FLAG="$SAVE_EVERY"
      SAVE_MODEL=True
    else
      SAVE_FLAG=-1
      SAVE_MODEL=False
    fi
    echo "mmsplit: robots=$NR_ROBOTS ($ROBOTS_LIST) nr_envs=$NR_ENVS minibatch=$MINIBATCH total=$TOTAL clip=$CLIP_FILE save_every=${SAVE_EVERY:-off}"
    echo "mmsplit: morph=${MORPH_MODE:-fixed}/${MORPH_COEFF:-0.4} refbias=${REFBIAS:-0.0} track_coeff=${TRACK_COEFF:-30.0}"
    # /usr/bin/time -v gives Maximum resident set size -- the number the
    # topology-ceiling claim rests on. Without it the run has no memory axis.
    /usr/bin/time -v python experiment.py "${COMMON[@]}" "${LOAD[@]}" \
      --algorithm.name=urma2.mjx_split \
      --environment.train_robots="$TRAIN_ROBOTS" \
      --environment.terrain.type=plane \
      --environment.critic_exteroceptive_observation_type=none \
      --environment.command.type=tracking_clip \
      --environment.command.tracking_clip_file="$CLIP_FILE" \
      --environment.command.tracking_clip_robot_map="$CLIP_MAP" \
      --environment.command.tracking_clip_fit_per_variant="${FITVARIANT:-True}" \
      --environment.command.tracking_clip_anchor="${ANCHOR:-centered}" \
      --environment.command.tracking_clip_velocity_command="${VELCMD:-True}" \
      --environment.reward.type=tracking \
      --environment.reward.log_info=True \
      --environment.reward.nominal_diff_target="${NOMINAL_TARGET:-reference}" \
      --environment.reward.joint_tracking_coeff="${TRACK_COEFF:-30.0}" \
      --environment.reward.joint_tracking_temperature="${TRACK_TEMP:-0.25}" \
      --environment.reward.deepmimic_enabled="${DEEPMIMIC:-True}" \
      --environment.reward.deepmimic_site_target_mode="${SITE_MODE:-fk}" \
      --environment.reward.tracking_curriculum_gated="${GATED:-False}" \
      --environment.reward.root_heading_tracking_weight_ratio="${HEADING_RATIO:-0.0}" \
      --environment.reward.root_heading_tracking_temperature="${HEADING_TEMP:-0.25}" \
      --environment.reward.deepmimic_qvel_temperature="${QVEL_TEMP:-0.5}" \
      --environment.termination.tracking_deviation_ratio="${TRACK_DEVIATION:-0.0}" \
      --environment.command.tracking_clip_dir="${CLIP_DIR:-/mnt/c/Users/smirn/Desktop/robot_learning_ip/external_data/amass_converted/LAFAN1}" \
      --environment.command.tracking_reference_action_bias="${REFBIAS:-1.0}" \
      --environment.seed="${ENVSEED:-1}" \
      --environment.domain_randomization.initial_state.type=reference \
      --environment.domain_randomization.seen_robot.morphology_coeff_mode="${MORPH_MODE:-fixed}" \
      --environment.domain_randomization.seen_robot.morphology_coeff_value="${MORPH_COEFF:-0.4}" \
      --environment.nr_envs="$NR_ENVS" \
      --algorithm.nr_steps="$NR_STEPS" \
      --algorithm.minibatch_size="$MINIBATCH" \
      --algorithm.nr_epochs=5 \
      --algorithm.total_timesteps="$TOTAL" \
      --algorithm.evaluation_active=False \
      --algorithm.evaluation_and_save_frequency="$SAVE_FLAG" \
      --environment.render=False \
      --runner.save_model="$SAVE_MODEL" \
      --runner.exp_name="${EXP_NAME:-mmsplit}"
    ;;
  *)
    echo "unknown stage: $STAGE" >&2; exit 2 ;;
esac
