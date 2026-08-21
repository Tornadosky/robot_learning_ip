#!/usr/bin/env bash
# C34 -- early termination on root deviation, on continuous morphology.
#
# `RootPoseTrajTerminalStateHandler` already computes the root-position deviation
# from the reference; upstream just defaults `max_root_pos_deviation` to 1e6, so
# it never fires. That default is why drifting is free: the policy keeps a full
# episode of reward while wandering metres away. Setting the threshold is the
# entire change -- no new handler, no new reward, no new observation.
#
# Both arms are one policy over CONTINUOUS morphology (a fresh random body every
# episode reset), full dance reward weights, 20M steps. The only difference is
# the deviation threshold. Arm A matches the existing c13_spatial run.
set -u
source ~/dance_env/bin/activate
cd /mnt/c/Users/smirn/Desktop/robot_learning_ip
OUT=experiments/h1_morphology_deepmimic_20260808

run () {
  local TAG="$1"; shift
  echo "=== ${TAG} ==="
  timeout 2400 python scripts/scaling/online_h1_train.py \
    --clip dance2_subject4 --start-frame 19482 --duration 8.0 \
    --num-envs 512 --num-steps 100 --total-timesteps 20e6 \
    --catalog-mode continuous --resample-per-episode \
    --terminal-handler MorphologyAwareRootPoseTrajTerminalStateHandler \
    --backbone mlp --reward-weights dance \
    --run-tag "${TAG}" --output-dir "${OUT}/checkpoints/${TAG}" "$@"
  echo "=== ${TAG} exit=$? ==="
}

run c34_dev05_seed0 --seed 0 --max-root-deviation 0.5
run c34_dev05_seed1 --seed 1 --max-root-deviation 0.5
run c34_nodev_seed1 --seed 1
