#!/usr/bin/env bash
# C14 -- the matched breadth control C13 was missing.
#
# C13 (continuous morphology + spatial rewards) posted a higher return than the
# C9b single-body specialists, but that comparison was not matched: C9b used a
# different trainer, a different terminal handler and per-body references. This
# runs the SAME trainer, handler, reward preset, window, envs, steps and seed as
# C13 and changes exactly one thing -- the morphology bounds collapse to the
# nominal body. Breadth is then the only difference between the two runs.
#
# A second arm pins a mid-range body instead of nominal, so the comparison is not
# hostage to the nominal body happening to be easy.
set -u
source ~/dance_env/bin/activate
cd /mnt/c/Users/smirn/Desktop/robot_learning_ip

OUT=experiments/h1_morphology_deepmimic_20260808

run_arm () {
  local TAG="$1"; shift
  echo "=== ${TAG} ==="
  timeout 2400 python scripts/scaling/online_h1_train.py \
    --clip dance2_subject4 --start-frame 19482 --duration 8.0 \
    --num-envs 512 --num-steps 100 --total-timesteps 20e6 \
    --catalog-mode continuous --resample-per-episode \
    --terminal-handler MorphologyAwareRootPoseTrajTerminalStateHandler \
    --seed 0 --backbone mlp --reward-weights dance \
    --run-tag "${TAG}" --output-dir "${OUT}/checkpoints/${TAG}" "$@"
  echo "=== ${TAG} exit=$? ==="
}

# single body: nominal. Bounds must be a non-empty interval, so use a width the
# model cannot resolve (1e-4 on a scale factor is ~0.02 mm of limb length).
run_arm c14_single_nominal \
  --morph-low 1.0 1.0 1.0 1.0 --morph-high 1.0001 1.0001 1.0001 1.0001

# single body: an off-nominal fixed body, to check the control is not just
# "nominal happens to be easy".
run_arm c14_single_offnominal \
  --morph-low 0.90 1.10 1.10 1.30 --morph-high 0.9001 1.1001 1.1001 1.3001
