#!/bin/bash
# Per-robot held-out evaluation of every finished cross-topology run.
#
# Every run gets: (1) a per-robot evaluation over its trained robots against the
# exact-reset zero-action baseline, and (2) for no-one-hot policies, a
# topology-held-out evaluation on ToddlerBot, which was never trained on.
set -u

cd /mnt/c/Users/smirn/Desktop/robot_learning_ip
export PYTHONPATH=$PWD/scripts
export XLA_PYTHON_CLIENT_PREALLOCATE=false
PY=~/jaxgpu/bin/python
OUT=experiments/cross_embodiment
EVAL=$OUT/evaluations
ENVS_PER_ROBOT=${ENVS_PER_ROBOT:-128}
HORIZON=${HORIZON:-200}
mkdir -p "$EVAL"

for run in "$OUT"/*/; do
  tag=$(basename "$run")
  [[ -f "$run/manifest.json" ]] || { echo "[eval] SKIP $tag (no manifest)"; continue; }
  ckpt="$run/checkpoint_final"
  [[ -d "$ckpt" ]] || { echo "[eval] SKIP $tag (no checkpoint)"; continue; }

  target="$EVAL/${tag}_per_robot.json"
  if [[ -f "$target" ]]; then
    echo "[eval] SKIP $tag per-robot (done)"
  else
    echo "[eval] $tag per-robot $(date -Is)"
    $PY scripts/scaling/evaluate_cross_humanoid_policy.py \
      --checkpoint "$ckpt" \
      --envs-per-robot "$ENVS_PER_ROBOT" \
      --horizon "$HORIZON" \
      --output "$target" > /dev/null
    echo "[eval] $tag per-robot rc=$? $(date -Is)"
  fi

  # ToddlerBot is only representable for policies without a positional one-hot.
  if grep -q '"robot_one_hot": false' "$run/manifest.json"; then
    target="$EVAL/${tag}_toddlerbot_heldout.json"
    if [[ -f "$target" ]]; then
      echo "[eval] SKIP $tag toddlerbot (done)"
    else
      echo "[eval] $tag toddlerbot-heldout $(date -Is)"
      $PY scripts/scaling/evaluate_cross_humanoid_policy.py \
        --checkpoint "$ckpt" \
        --robots toddlerbot \
        --envs-per-robot "$ENVS_PER_ROBOT" \
        --horizon "$HORIZON" \
        --output "$target" > /dev/null
      echo "[eval] $tag toddlerbot rc=$? $(date -Is)"
    fi
  fi
done

echo "[eval] ALL DONE $(date -Is)"
