#!/usr/bin/env bash
# The measurement the whole campaign is judged on: crosseval the 3-topology
# baseline against the ZERO-ACTION FLOOR, n>=4 seeds.
#
# ORDER MATTERS AND IS DELIBERATE. v1 of this script ran the step-scaling curve
# first and the verdict second. With no compile cache each eval cost 25-55 min,
# so at 13 evals the VERDICT would not have landed by morning while the CONTEXT
# did. Verdict first, curve with whatever time is left.
#
# WHY THE FLOOR: an arm that "tracks" can be worse than doing nothing, and that
# has happened on this project. A score without the floor beside it is not
# interpretable.
#
# The compile cache lives in local_3t_eval_snap.sh; the first eval pays full
# compile and the rest should be far cheaper.
#
# Launch detached:
#   setsid nohup bash scripts/scaling/local_3t_sweep.sh > /tmp/l3t_sweep.log 2>&1 < /dev/null &
set -u
REPO=/mnt/c/Users/smirn/Desktop/robot_learning_ip
SNAP=$REPO/experiments/local_3t/snapshots/local_3t_dance4
EVAL=$REPO/scripts/scaling/local_3t_eval_snap.sh

while pgrep -f "experiment.py" > /dev/null 2>&1; do
  echo "[sweep] training still running, waiting 60s"
  sleep 60
done
echo "[sweep] GPU free at $(date -Is)"

tr -d '\r' < "$EVAL" > /tmp/l3te.sh

LAST=$(ls -1 "$SNAP"/snap_*.model 2>/dev/null | sort | tail -1)
echo "[sweep] final checkpoint: $LAST"

# ---- PHASE 1: THE VERDICT ------------------------------------------------
# Final checkpoint vs the zero-action floor, interleaved so that if the window
# closes early we still hold MATCHED pairs rather than four of one and none of
# the other. Seed 0 of each first, for the same reason.
for SEED in 0 1 2 3; do
  echo "[sweep] === VERDICT policy seed=$SEED $(date -Is)"
  MODEL="$LAST" TAG=final SEED=$SEED bash /tmp/l3te.sh
  echo "[sweep] === VERDICT zero-floor seed=$SEED $(date -Is)"
  MODEL="$LAST" TAG=zero SEED=$SEED ZERO=1 bash /tmp/l3te.sh
done
echo "[sweep] PHASE 1 COMPLETE -- the verdict is on disk $(date -Is)"

# ---- PHASE 2: THE STEP-SCALING CURVE (context) ---------------------------
# Does executed-vs-clip RMSE keep falling with steps, or saturate? Saturation
# means compute is not the blocker. Nice to have, not the verdict.
for S in 001_1966080 003_5898240 005_9830400 007_13762560; do
  M="$SNAP/snap_$S.model"
  [ -f "$M" ] || { echo "[sweep] missing snap_$S, skipping"; continue; }
  echo "[sweep] === curve point snap_$S $(date -Is)"
  MODEL="$M" TAG="snap${S%%_*}" SEED=0 bash /tmp/l3te.sh
done

echo "[sweep] DONE $(date -Is)"
ls -la "$REPO"/experiments/local_3t/ce_*.json
