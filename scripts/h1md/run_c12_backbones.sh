#!/usr/bin/env bash
# C12 -- matched MLP / URMA / URMAv2 comparison on continuous morphology.
# Written as a file rather than an inline `wsl bash -lc` one-liner because
# inline $VAR expansion is eaten by the Windows-side quoting.
set -u
source ~/dance_env/bin/activate
cd /mnt/c/Users/smirn/Desktop/robot_learning_ip

OUT=experiments/h1_morphology_deepmimic_20260808

for BB in urma urmav2; do
  echo "=== backbone ${BB} ==="
  timeout 2400 python scripts/scaling/online_h1_train.py \
    --clip dance2_subject4 --start-frame 19482 --duration 8.0 \
    --num-envs 512 --num-steps 100 --total-timesteps 20e6 \
    --catalog-mode continuous --resample-per-episode \
    --terminal-handler MorphologyAwareRootPoseTrajTerminalStateHandler \
    --seed 0 --backbone "${BB}" --run-tag "c12_${BB}" \
    --output-dir "${OUT}/checkpoints/c12_${BB}"
  echo "=== backbone ${BB} exit=$? ==="
done
