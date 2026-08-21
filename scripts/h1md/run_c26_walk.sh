#!/usr/bin/env bash
# C24 -- is the residual drift a BUDGET limit?
#
# C23 concluded the constraint is foot-placement accuracy rather than reward
# specification, but every run in this audit is 20 M steps, which is a
# feasibility budget. If drift falls materially at 60 M and 100 M, the limit is
# training budget and the pipeline is fine. If it plateaus, foot placement is
# structurally hard here and neither more steps nor reward edits will fix it --
# the reference itself is the suspect.
#
# `dance_root` is the arm to scale: it was the best of the four reward variants
# (0.76 m) at no survival cost. Single-body config, where drift seed sd is 0.05,
# so a >0.1 m change is readable.
set -u
source ~/dance_env/bin/activate
cd /mnt/c/Users/smirn/Desktop/robot_learning_ip

OUT=experiments/h1_morphology_deepmimic_20260808

run_budget () {
  local STEPS="$1"; local SEED="$2"; local TAG="c26_walk${3}_seed${SEED}"
  echo "=== ${TAG} (${STEPS} steps) ==="
  timeout 4200 python scripts/scaling/online_h1_train.py \
    --clip walk1_subject1 --duration 8.0 \
    --num-envs 512 --num-steps 100 --total-timesteps "${STEPS}" \
    --catalog "${OUT}/body_catalog/c16_single_nominal.json" \
    --catalog-mode fixed_balanced --keep-morph-bounds --resample-per-episode \
    --terminal-handler MorphologyAwareRootPoseTrajTerminalStateHandler \
    --seed "${SEED}" --backbone mlp --reward-weights dance_root \
    --run-tag "${TAG}" --output-dir "${OUT}/checkpoints/${TAG}"
  echo "=== ${TAG} exit=$? ==="
}

run_budget 20e6  0 20m
run_budget 20e6  1 20m
run_budget 20e6  2 20m
