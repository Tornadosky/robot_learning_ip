#!/usr/bin/env bash
# Train the two checkpoints needed for the video set that do not exist yet:
#   1. an imitation policy on the WALK clip via the CPU-renderable c9 path
#   2. the plain locomotion baseline, now saving its checkpoint
set -u
source ~/dance_env/bin/activate
cd /mnt/c/Users/smirn/Desktop/robot_learning_ip
OUT=experiments/h1_morphology_deepmimic_20260808

echo "=== walk imitation policy (renderable CPU path) ==="
timeout 2400 python scripts/h1md/c9_shared_policy.py --clip walk1_subject1 --start 10521 \
  --bodies body00_nominal --arms fk --total-timesteps 20e6 \
  --out "${OUT}/evaluations/cvid_walk_policy.json"
echo "=== walk exit=$? ==="

echo "=== locomotion baseline with checkpoint ==="
timeout 2400 python scripts/h1md/c31_locomotion_baseline.py --total-timesteps 20e6 \
  --out "${OUT}/evaluations/c31_locomotion_baseline.json"
echo "=== locomotion exit=$? ==="
