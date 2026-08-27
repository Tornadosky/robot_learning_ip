#!/usr/bin/env bash
# One arm of the FSQ A/B: urma2-native H1+G1 dance tracking, morph 0.3 fixed.
# Usage: run_arm.sh <exp_name> <clip_dir>
# Config = overnight_1808 X0 baseline + MORPH_COEFF=0.3, 30M steps.
set -uo pipefail
EXP_NAME="${1:?usage: run_arm.sh <exp_name> <clip_dir>}"
CLIP_DIR_ARG="${2:?usage: run_arm.sh <exp_name> <clip_dir>}"
REPO=/mnt/c/Users/smirn/Desktop/robot_learning_ip
E=$REPO/experiments/fsq_khaendler
mkdir -p "$E/logs"

{
  echo "[$(date -Is)] ARM $EXP_NAME start clip_dir=$CLIP_DIR_ARG anchor=${ANCHOR:-centered}"
  # Every value is an overridable default now: the overnight-fix arms export
  # ANCHOR=absolute FITVARIANT=False REFBIAS=0.0 etc. on top; with nothing
  # exported this reproduces the original hardcoded arm exactly.
  export ROBOTS_LIST="${ROBOTS_LIST:-unitree_h1:unitree_g1}"
  export CLIP_FILE="${CLIP_FILE:-dance2_subject4.npz}"
  export CLIP_DIR="$CLIP_DIR_ARG"
  export MORPH_MODE="${MORPH_MODE:-fixed}"
  export MORPH_COEFF="${MORPH_COEFF:-0.3}"
  export REFBIAS="${REFBIAS:-1.0}"
  export NOMINAL_TARGET="${NOMINAL_TARGET:-reference}"
  export FITVARIANT="${FITVARIANT:-True}"
  export ANCHOR="${ANCHOR:-centered}"
  export NR_ENVS="${NR_ENVS:-1024}"
  export TOTAL_STEPS="${TOTAL_STEPS:-30408704}"
  export EXP_NAME
  flock /tmp/robot_learning_local_gpu.lock \
    bash "$REPO/experiments/urma2_h1g1/run.sh" mmtrain
  echo "[$(date -Is)] ARM $EXP_NAME done rc=$?"
} >> "$E/logs/${EXP_NAME}.log" 2>&1
