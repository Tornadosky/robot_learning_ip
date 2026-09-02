#!/usr/bin/env bash
# tokenizer_mb -- the MULTIBODY tokenizer (night5, "more data" axis).
# 4 families x 10 clips (UPDATED 2026-08-31 23:00 after three launch failures):
#   UnitreeH1v2 DROPPED -- it has only 2 dance retargets anywhere (no walk1,
#   so no held-out split: empty test_parts crashed cmd_train), no foot geoms
#   (descriptor crash, now guarded in mulitenvironment.py), and a
#   'floating_base_joint' free-joint name (now allowed). ~16k frames of
#   dance-only data was not worth a fourth relaunch.
#   Atlas/Talos now come from LAFAN1_fixed (feasibility-FIXED retargets, full
#   11-clip roster incl. walk1) via the LAFAN1_mb junction dir -- strictly
#   better than the stale-retarget caveat below.
# Original roster provenance (user asked, 2026-08-31):
#   ToddlerBot DROPPED -- documented 4-bar knees, its retarget was FK-target
#   only; feeding it in risks poisoning the shared per-joint code.
#   Atlas/Talos KEPT WITH A CAVEAT -- their LAFAN1 retargets predate the
#   feasibility fix (Atlas demanded 2.74x joint travel). For a CODEC this is
#   tolerable (compression of trajectories, not physical targets), and the
#   comparison metric is held-out walk1 RMSE on H1/G1 only -- but do NOT use
#   Atlas/Talos mb tokens as RL references without the re-issued clips.
#   T1 excluded: its fixed set lives in LAFAN1_3t (kept for tokenizer_3t_v2).
# Purpose: (a) does robot diversity improve codec generalization? (b) the
# morphology-agnostic candidate for B3.
# CPU-bound (jax-cpu, Windows venv, uses ALL cores via XLA) -- does not touch
# the GPU. Epochs 150/batch 1024 to fit a CPU night (~110-180 s/epoch
# measured on v2's CPU run); NOT epoch-matched to v2(250) -- directional
# verdict tonight, equalize on GPU in daylight if it looks good.
set -eu
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python.exe
[ -x "$PY" ] || PY=python

$PY scripts/scaling/khaendler_fsq_clip.py train \
  --robots UnitreeH1 UnitreeG1 Atlas Talos \
  --clip dance2_subject1.npz dance2_subject2.npz dance2_subject3.npz \
         dance2_subject4.npz dance2_subject5.npz walk1_subject1.npz \
         walk_cycle_s6764_n46.npz walk_cycle_s720_n62.npz \
         walk_cycle_s7925_n54.npz walk_cycle_s8883_n57.npz \
  --heldout-clips walk1_subject1.npz \
  --clip-dir external_data/amass_converted/LAFAN1_mb \
  --out experiments/fsq_khaendler/tokenizer_mb \
  --foot-channels --epochs 60 --batch-size 1024 --lr 1.5e-3

echo "=== held-out walk1 reconstruction (the generalization verdict) ==="
$PY scripts/scaling/khaendler_fsq_clip.py reconstruct \
  --robots UnitreeH1 UnitreeG1 \
  --clip walk1_subject1.npz dance2_subject4.npz \
  --clip-dir external_data/amass_converted/LAFAN1 \
  --tokenizer experiments/fsq_khaendler/tokenizer_mb \
  --out experiments/fsq_khaendler/clips_mb_rec
echo "=== tokenizer_mb DONE ==="
