#!/usr/bin/env bash
# C29 -- THE decisive test: does making root position OBSERVABLE stop the drift?
#
# Finding 41: MimicReward scores world-frame root position, but the robot's own
# observation (FreeJointPosNoXY) and GoalTrajMimic's reference qpos both strip
# root XY, and every site term is upper-body-relative. The policy is charged for
# a displacement it cannot perceive. Finding 40: consequently NO configuration
# beat a stand-still control (dance floor 0.47 m; best policy 0.61 m).
#
# Identical to c20_rootterm except `--goal-type GoalTrajMimicRootErr`, which
# appends the local-frame root position error. The ONLY difference is whether
# the quantity being scored is observable.
#
# Pass condition, fixed in advance: drift below the 0.47 m stand-still floor.
# Above it means observability was necessary but not sufficient.
set -u
source ~/dance_env/bin/activate
cd /mnt/c/Users/smirn/Desktop/robot_learning_ip

OUT=experiments/h1_morphology_deepmimic_20260808

for SEED in 0 1 2; do
  TAG="c29_rooterr_seed${SEED}"
  echo "=== ${TAG} ==="
  timeout 2400 python scripts/scaling/online_h1_train.py \
    --clip dance2_subject4 --start-frame 19482 --duration 8.0 \
    --num-envs 512 --num-steps 100 --total-timesteps 20e6 \
    --catalog "${OUT}/body_catalog/c16_single_nominal.json" \
    --catalog-mode fixed_balanced --keep-morph-bounds --resample-per-episode \
    --terminal-handler MorphologyAwareRootPoseTrajTerminalStateHandler \
    --seed "${SEED}" --backbone mlp --reward-weights dance_root --goal-type GoalTrajMimicRootErr \
    --run-tag "${TAG}" --output-dir "${OUT}/checkpoints/${TAG}"
  echo "=== ${TAG} exit=$? ==="
done
