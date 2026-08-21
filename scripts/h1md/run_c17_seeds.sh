#!/usr/bin/env bash
# C17 -- three seeds each for the breadth comparison.
#
# C14 vs C16 showed the single-body result moving by +/-30 episode length purely
# from changing how the descriptor was normalised, in OPPOSITE directions for the
# two bodies. That is run-to-run variance, not a confound with a sign, so no
# single-seed comparison of breadth can be read. Seed 0 already exists for both
# configurations; this adds seeds 1 and 2 so each has n=3, which is the goal
# document's own bar for a robustness claim.
set -u
source ~/dance_env/bin/activate
cd /mnt/c/Users/smirn/Desktop/robot_learning_ip

OUT=experiments/h1_morphology_deepmimic_20260808

common () {
  echo --clip dance2_subject4 --start-frame 19482 --duration 8.0 \
       --num-envs 512 --num-steps 100 --total-timesteps 20e6 \
       --resample-per-episode \
       --terminal-handler MorphologyAwareRootPoseTrajTerminalStateHandler \
       --backbone mlp --reward-weights dance
}

for SEED in 1 2; do
  TAG="c17_continuous_seed${SEED}"
  echo "=== ${TAG} ==="
  timeout 2400 python scripts/scaling/online_h1_train.py $(common) \
    --catalog-mode continuous --seed "${SEED}" \
    --run-tag "${TAG}" --output-dir "${OUT}/checkpoints/${TAG}"
  echo "=== ${TAG} exit=$? ==="

  TAG="c17_single_seed${SEED}"
  echo "=== ${TAG} ==="
  timeout 2400 python scripts/scaling/online_h1_train.py $(common) \
    --catalog "${OUT}/body_catalog/c16_single_nominal.json" \
    --catalog-mode fixed_balanced --keep-morph-bounds --seed "${SEED}" \
    --run-tag "${TAG}" --output-dir "${OUT}/checkpoints/${TAG}"
  echo "=== ${TAG} exit=$? ==="
done
