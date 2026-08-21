#!/usr/bin/env bash
# C36 -- multi-body training with the reference RETARGETED to each body.
#
# Until now the continuous-morphology trainer scored every randomized body
# against the NOMINAL body's site targets, because MimicReward reads them from
# traj_data.site_xpos, computed once at load on the nominal model. Verified
# consequence: a 1.18x-arm body and a 0.86x-arm body were both told to put their
# hands 1.064 m apart. With MorphMimicReward those targets become 1.190 m and
# 0.967 m respectively -- forward kinematics of the reference joint angles on
# each environment's own morphology arrays, in-graph, no dataset.
set -u
source ~/dance_env/bin/activate
cd /mnt/c/Users/smirn/Desktop/robot_learning_ip
OUT=experiments/h1_morphology_deepmimic_20260808

run () {
  local TAG="$1"; shift
  echo "=== ${TAG} ==="
  timeout 3000 python scripts/scaling/online_h1_train.py \
    --num-envs 512 --num-steps 100 --total-timesteps 20e6 \
    --catalog-mode continuous --resample-per-episode \
    --terminal-handler MorphologyAwareRootPoseTrajTerminalStateHandler \
    --backbone mlp --reward-weights dance --reward-type MorphMimicReward \
    --run-tag "${TAG}" --output-dir "${OUT}/checkpoints/${TAG}" "$@"
  echo "=== ${TAG} exit=$? ==="
}

run c36_morph_seed0 --clip dance2_subject4 --start-frame 19482 --duration 8.0 --seed 0
run c36_morph_seed1 --clip dance2_subject4 --start-frame 19482 --duration 8.0 --seed 1
run c36_morph_walk_seed0 --clip walk1_subject1 --start-frame 10521 --duration 8.0 --seed 0
