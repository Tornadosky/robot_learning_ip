#!/usr/bin/env bash
# C20 -- does an undiluted world-frame root term stop the drift?
#
# Arm A (baseline) already exists: c16_clean_nominal + c17_single_seed{1,2},
# i.e. `--reward-weights dance`, n=3 on the single nominal body.
# Arm B is identical except `--reward-weights dance_root`, which restricts
# joints_for_mimic to the root free joint so the qpos term becomes a pure
# world-frame root term instead of 3-of-22 entries in a mean-square.
#
# Deliberately run on the SINGLE-BODY configuration, not continuous morphology:
# C17 measured seed sd of 8.5 there against 61.5 under continuous morphology, so
# a 3-seed comparison is actually readable at this budget.
set -u
source ~/dance_env/bin/activate
cd /mnt/c/Users/smirn/Desktop/robot_learning_ip

OUT=experiments/h1_morphology_deepmimic_20260808

for SEED in 0 1 2; do
  TAG="c20_rootterm_seed${SEED}"
  echo "=== ${TAG} ==="
  timeout 2400 python scripts/scaling/online_h1_train.py \
    --clip dance2_subject4 --start-frame 19482 --duration 8.0 \
    --num-envs 512 --num-steps 100 --total-timesteps 20e6 \
    --catalog "${OUT}/body_catalog/c16_single_nominal.json" \
    --catalog-mode fixed_balanced --keep-morph-bounds --resample-per-episode \
    --terminal-handler MorphologyAwareRootPoseTrajTerminalStateHandler \
    --seed "${SEED}" --backbone mlp --reward-weights dance_root \
    --run-tag "${TAG}" --output-dir "${OUT}/checkpoints/${TAG}"
  echo "=== ${TAG} exit=$? ==="
done
