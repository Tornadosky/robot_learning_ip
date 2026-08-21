#!/usr/bin/env bash
# C13 -- the C11 continuous-morphology run with the SPATIAL reward terms live.
#
# C11 showed breadth is free but was a joint-space result: online_h1_train.py
# zeroes rpos/rquat/rvel because per-body site targets used to require an
# unaffordable retargeting step. C3b measured that step at ~18 s per 1000 bodies,
# so the justification is gone. This arm turns the site terms back on and scores
# every randomized body against the shared reference's site targets -- i.e. the
# `shared_nominal` construction, which C6 priced at <=7.9% and C10 showed trains
# to full-clip survival on a single body.
#
# Matched to C11 in every other respect: same window, envs, steps, seed.
set -u
source ~/dance_env/bin/activate
cd /mnt/c/Users/smirn/Desktop/robot_learning_ip

OUT=experiments/h1_morphology_deepmimic_20260808

timeout 2400 python scripts/scaling/online_h1_train.py \
  --clip dance2_subject4 --start-frame 19482 --duration 8.0 \
  --num-envs 512 --num-steps 100 --total-timesteps 20e6 \
  --catalog-mode continuous --resample-per-episode \
  --terminal-handler MorphologyAwareRootPoseTrajTerminalStateHandler \
  --seed 0 --backbone mlp --reward-weights dance \
  --run-tag c13_spatial --output-dir "${OUT}/checkpoints/c13_spatial"
echo "=== c13 exit=$? ==="
