#!/usr/bin/env bash
# tokenizer_3t_v2 on the LOCAL GPU (WSL jaxgpu venv) -- the CPU run paced to
# ~7.5 h (108 s/epoch x 250); CUDA import verified 2026-08-31 22:15.
# Same config as train_tokenizer_3t_v2.sh (250 ep, foot channels, walk1 held
# out), then emits clips_3t_v2 sidecars (dance4 + walk1, 3 robots).
set -eu
REPO=/mnt/c/Users/smirn/Desktop/robot_learning_ip
cd "$REPO"
PY=~/jaxgpu/bin/python
export PYTHONPATH="$REPO:$REPO/loco-mujoco:$REPO/loco_mjx:$REPO/RL-X"
export MUJOCO_GL=disable XLA_PYTHON_CLIENT_PREALLOCATE=false

TOK=experiments/fsq_khaendler/tokenizer_3t_v2
SRC=external_data/amass_converted/LAFAN1_3t
OUT=experiments/fsq_khaendler/clips_3t_v2

$PY scripts/scaling/khaendler_fsq_clip.py train \
  --robots UnitreeH1 UnitreeG1 BoosterT1 \
  --clip dance2_subject1.npz dance2_subject2.npz dance2_subject3.npz \
         dance2_subject4.npz dance2_subject5.npz walk1_subject1.npz \
         walk_cycle_s6764_n46.npz walk_cycle_s720_n62.npz \
         walk_cycle_s7925_n54.npz walk_cycle_s8883_n57.npz \
  --heldout-clips walk1_subject1.npz \
  --clip-dir "$SRC" --out "$TOK" \
  --foot-channels --epochs 250 --batch-size 512 --lr 1.5e-3

echo "=== emit v2 sidecars ==="
$PY scripts/scaling/khaendler_fsq_clip.py reconstruct \
  --robots UnitreeH1 UnitreeG1 BoosterT1 \
  --clip dance2_subject4.npz walk1_subject1.npz \
  --clip-dir "$SRC" --tokenizer "$TOK" --out "${OUT}_rec"

rm -rf "$OUT"
for r in UnitreeH1 UnitreeG1 BoosterT1; do
  mkdir -p "$OUT/$r"
  for c in dance2_subject4 walk1_subject1; do
    cp -f "$SRC/$r/$c.npz" "$OUT/$r/$c.npz"
    cp -f "${OUT}_rec/$r/${c}_zq.npz" "$OUT/$r/${c}_zq.npz"
  done
done
echo "=== v2 DONE ==="
$PY -c "
import json
r = json.load(open('${OUT}_rec/reconstruction_report.json'))
for k, v in r.items():
    print(k, 'heldout' if v['heldout_clip'] else 'tail', round(v['qpos_rmse_rad_heldout'], 4))
"
