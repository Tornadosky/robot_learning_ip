#!/usr/bin/env bash
# C28 -- do the two working levers combine?
#
# Phase-corrected drift, dance, single nominal body:
#   baseline dance @20M            0.82 m
#   dance_root_heavy @20M          0.63 m   (heavier root term, -23%)
#   dance_root       @60M          0.61 m   (more budget, -26%)
# Both levers work independently. If they are additive, dance_root_heavy @60M
# should land near 0.45-0.50 m against a motion that travels 0.66 m -- which
# would be the first genuinely positive spatial-tracking result in this audit,
# and on the least trackable clip in the library.
#
# Third arm carries the winner to CONTINUOUS morphology, the actual scaling
# target, to check the gain is not a single-body artifact.
set -u
source ~/dance_env/bin/activate
cd /mnt/c/Users/smirn/Desktop/robot_learning_ip

OUT=experiments/h1_morphology_deepmimic_20260808

single () {
  local SEED="$1"; local TAG="c28_heavy60m_seed${SEED}"
  echo "=== ${TAG} ==="
  timeout 4200 python scripts/scaling/online_h1_train.py \
    --clip dance2_subject4 --start-frame 19482 --duration 8.0 \
    --num-envs 512 --num-steps 100 --total-timesteps 60e6 \
    --catalog "${OUT}/body_catalog/c16_single_nominal.json" \
    --catalog-mode fixed_balanced --keep-morph-bounds --resample-per-episode \
    --terminal-handler MorphologyAwareRootPoseTrajTerminalStateHandler \
    --seed "${SEED}" --backbone mlp --reward-weights dance_root_heavy \
    --run-tag "${TAG}" --output-dir "${OUT}/checkpoints/${TAG}"
  echo "=== ${TAG} exit=$? ==="
}

continuous () {
  local SEED="$1"; local TAG="c28_heavy60m_continuous_seed${SEED}"
  echo "=== ${TAG} ==="
  timeout 4200 python scripts/scaling/online_h1_train.py \
    --clip dance2_subject4 --start-frame 19482 --duration 8.0 \
    --num-envs 512 --num-steps 100 --total-timesteps 60e6 \
    --catalog-mode continuous --resample-per-episode \
    --terminal-handler MorphologyAwareRootPoseTrajTerminalStateHandler \
    --seed "${SEED}" --backbone mlp --reward-weights dance_root_heavy \
    --run-tag "${TAG}" --output-dir "${OUT}/checkpoints/${TAG}"
  echo "=== ${TAG} exit=$? ==="
}

single 0
single 1
continuous 0
