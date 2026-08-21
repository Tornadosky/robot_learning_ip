#!/usr/bin/env bash
# C30 -- observable root error ON A HIGH-STANCE CLIP: the last untried combination.
#
# C29 showed that making root position observable does not fix drift on
# dance2_subject4 (0.87 m vs a 0.47 m stand-still floor). The remaining
# explanation is Finding 33/34: that clip has a genuine stance foot in only
# 10.9% of frames, so a policy that KNOWS it is displaced still has nothing to
# push against.
#
# walk1_subject1 has 91.6% stance and a stand-still floor of 3.56 m. This arm
# pairs the observable root error with a motion that has contact to act through.
#
# Pass condition, fixed in advance: drift below 3.56 m. If this also fails, the
# audit's conclusion is that this stack does joint-space imitation well and
# world-space imitation not at all.
set -u
source ~/dance_env/bin/activate
cd /mnt/c/Users/smirn/Desktop/robot_learning_ip

OUT=experiments/h1_morphology_deepmimic_20260808

for SEED in 0 1 2; do
  TAG="c30_walkrooterr_seed${SEED}"
  echo "=== ${TAG} ==="
  timeout 2400 python scripts/scaling/online_h1_train.py \
    --clip walk1_subject1 --duration 8.0 \
    --num-envs 512 --num-steps 100 --total-timesteps 20e6 \
    --catalog "${OUT}/body_catalog/c16_single_nominal.json" \
    --catalog-mode fixed_balanced --keep-morph-bounds --resample-per-episode \
    --terminal-handler MorphologyAwareRootPoseTrajTerminalStateHandler \
    --seed "${SEED}" --backbone mlp --reward-weights dance_root --goal-type GoalTrajMimicRootErr \
    --run-tag "${TAG}" --output-dir "${OUT}/checkpoints/${TAG}"
  echo "=== ${TAG} exit=$? ==="
done
