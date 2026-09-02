#!/usr/bin/env bash
# Wave 7: tokenizer on the 20-motion LAFAN1 set (H1 + G1), 7 motions held out,
# then sidecars (_zq via reconstruct, _win via emit_window_sidecars) for the
# super20 clip and the held-out clips in experiments/fsq_khaendler/clips_m20.
# GPU (WSL ~/jaxgpu). ENC=old (default, matches wave 6) or ENC=new
# (khaendler's 2026-08-26 encoder/decoder, checked out into loco-mujoco first).
set -eu
REPO=${REPO:-/mnt/c/Users/smirn/Desktop/robot_learning_ip}
cd "$REPO"
PY=${PY:-~/jaxgpu/bin/python}
export PYTHONPATH="$REPO:$REPO/loco-mujoco:$REPO/loco_mjx:$REPO/RL-X"
export MUJOCO_GL=disable XLA_PYTHON_CLIENT_PREALLOCATE=false
NAME=${TOKNAME:-tokenizer_m20}
TOK=experiments/fsq_khaendler/$NAME
SRC=external_data/amass_converted/LAFAN1_all
M20=experiments/fsq_khaendler/clips_m20
TR=$(sed 's/$/.npz/' $M20/train_motions.txt | tr '\n' ' ')
HO=$(sed 's/$/.npz/' $M20/heldout_motions.txt | tr '\n' ' ')
EPOCHS=${EPOCHS:-80}

echo "=== train $NAME: $(echo $TR | wc -w) train + $(echo $HO | wc -w) held-out motions, epochs $EPOCHS  $(date -Is)"
$PY scripts/scaling/khaendler_fsq_clip.py train \
  --robots UnitreeH1 UnitreeG1 \
  --clip $TR $HO --heldout-clips $HO \
  --clip-dir "$SRC" --out "$TOK" \
  --foot-channels --epochs "$EPOCHS" --batch-size 512 --lr 1.5e-3

echo "=== reconstruct (_zq) for super20 + held-out  $(date -Is)"
$PY scripts/scaling/khaendler_fsq_clip.py reconstruct \
  --robots UnitreeH1 UnitreeG1 \
  --clip super20.npz $HO \
  --clip-dir "$M20" --tokenizer "$TOK" --out "${M20}_rec_$NAME"
for r in UnitreeH1 UnitreeG1; do cp -f "${M20}_rec_$NAME/$r/"*_zq.npz "$M20/$r/"; done

echo "=== window sidecars (_win)  $(date -Is)"
JAX_PLATFORMS=cpu $PY scripts/scaling/wave6/emit_window_sidecars.py --tokenizer "$TOK" \
  --clip-dir "$M20" --robots UnitreeH1 UnitreeG1 --clip super20.npz $HO
ls -la "$M20/UnitreeG1/"
echo "=== TOKENIZER PIPELINE DONE  $(date -Is)"
