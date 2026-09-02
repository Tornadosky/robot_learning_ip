#!/usr/bin/env bash
# Re-evaluate the FINAL 3T checkpoint once, purely to emit per_joint_corr.
#
# WHY A SEPARATE SCRIPT: local_3t_sweep.sh is running and bash reads a script
# INCREMENTALLY as it executes, so editing it in place can corrupt the run.
# This waits for it to exit instead. Phase 2 covers snapshots 001-007 only, so
# without this the correlation would never cover the checkpoint the verdict is
# built on.
#
# Distinct TAG so it cannot overwrite ce_final_s0.json.
set -u
REPO=/mnt/c/Users/smirn/Desktop/robot_learning_ip
SNAP=$REPO/experiments/local_3t/snapshots/local_3t_dance4

while pgrep -f "l3tsweep.sh|local_3t_sweep.sh" > /dev/null 2>&1; do
  echo "[corr] sweep still running, waiting 120s $(date -Is)"; sleep 120
done
while pgrep -f "crosseval_motion.py" > /dev/null 2>&1; do
  echo "[corr] an eval is still in flight, waiting 60s $(date -Is)"; sleep 60
done
echo "[corr] GPU free $(date -Is)"

tr -d '\r' < "$REPO/scripts/scaling/local_3t_eval_snap.sh" > /tmp/l3te_corr.sh
LAST=$(ls -1 "$SNAP"/snap_*.model 2>/dev/null | sort | tail -1)
echo "[corr] final checkpoint: $LAST"

MODEL="$LAST" TAG=corrfinal SEED=0 bash /tmp/l3te_corr.sh
MODEL="$LAST" TAG=corrzero  SEED=0 ZERO=1 bash /tmp/l3te_corr.sh
echo "[corr] DONE $(date -Is)"
