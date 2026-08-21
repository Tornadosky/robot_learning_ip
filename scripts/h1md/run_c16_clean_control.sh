#!/usr/bin/env bash
# C16 -- the clean breadth control: one-body catalog + --keep-morph-bounds.
#
# Unlike C14 (collapsed bounds), the descriptor here is normalised by the real
# morphology bounds, so the single-body arms see the same descriptor scaling the
# continuous run does and the four descriptor dimensions are not noise.
# Everything else matches C13 exactly.
set -u
source ~/dance_env/bin/activate
cd /mnt/c/Users/smirn/Desktop/robot_learning_ip

OUT=experiments/h1_morphology_deepmimic_20260808

run_arm () {
  local TAG="$1"; local CAT="$2"
  echo "=== ${TAG} ==="
  timeout 2400 python scripts/scaling/online_h1_train.py \
    --clip dance2_subject4 --start-frame 19482 --duration 8.0 \
    --num-envs 512 --num-steps 100 --total-timesteps 20e6 \
    --catalog "${OUT}/body_catalog/${CAT}" --catalog-mode fixed_balanced \
    --keep-morph-bounds --resample-per-episode \
    --terminal-handler MorphologyAwareRootPoseTrajTerminalStateHandler \
    --seed 0 --backbone mlp --reward-weights dance \
    --run-tag "${TAG}" --output-dir "${OUT}/checkpoints/${TAG}"
  echo "=== ${TAG} exit=$? ==="
}

run_arm c16_clean_nominal    c16_single_nominal.json
run_arm c16_clean_offnominal c16_single_offnominal.json
