#!/usr/bin/env bash
# Bounded-deep probe: extend extreme_tall_light (legs 1.55x) at its best prior
# recipe (stock gains) from 300M to 1B steps to find where the slow-crawl
# episode-length curve asymptotes (cross ~320+ vs plateau ~1/3 horizon).
set -euo pipefail
cd /mnt/c/Users/smirn/Desktop/robot_learning_ip/scripts
source ~/miniconda3/etc/profile.d/conda.sh
conda activate locodm
mkdir -p logs
LOG=logs/deep1b_extreme_tall_light.log
setsid bash -c 'python -u train_deepmimic_morphology.py \
  --robot g1 --preset extreme_tall_light --clip dance2_subject4 \
  --duration 30 --num-envs 2048 \
  --total-timesteps 1e9 --num-checkpoints 10 \
  --raw-reference --control pd --seed 1 \
  --out-suffix deep1b' > "$LOG" 2>&1 < /dev/null &
echo "PID=$!"
disown || true
