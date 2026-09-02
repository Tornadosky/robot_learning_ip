#!/usr/bin/env bash
# local_3t.sh + checkpoint snapshots.
#
# WHY HERE AND NOT ON VIPER: evidence collected 2026-08-27 -- every Viper run
# with 3 robots died with HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION at EVERY
# env count tried (192/384/576/1536), while every COMPLETED Viper run had
# NR_ENVS <= 768 and one robot. CUDA has no such fault. The 3-topology baseline
# therefore belongs on the local GPU, permanently.
#
# WHY SNAPSHOTS: the trainer overwrites models/latest.model at each save, so a
# finished run carries no history and the question "does executed-vs-clip RMSE
# keep falling with more steps?" cannot be answered after the fact. This keeps
# one model per save, indexed so that snapshot k == k * SAVE_EVERY steps.
#
# Launch detached:
#   setsid nohup bash scripts/scaling/local_3t_snap.sh > /tmp/local_3t_snap.log 2>&1 < /dev/null &
set -u

REPO=/mnt/c/Users/smirn/Desktop/robot_learning_ip
NAME=${L3T_NAME:-local_3t_dance4}   # NOT ${NAME:-...}: the WSL
                                    # profile exports NAME, which
                                    # silently renamed the run.
SAVE_EVERY=${SAVE_EVERY:-1966080}
SNAPDIR=$REPO/experiments/local_3t/snapshots/$NAME
RUNROOT=$REPO/loco_mjx/experiments/runs/local_3t/$NAME
mkdir -p "$SNAPDIR"
if [ -n "$(ls -A "$SNAPDIR" 2>/dev/null)" ]; then
  echo "[snap] REFUSING to start: $SNAPDIR already has snapshots."
  echo "[snap] Move or delete them, or set L3T_NAME to a fresh run name."
  exit 1
fi

# STALE-MODEL GUARD. Without this the watcher copies whatever latest.model it
# first sees -- including one left behind by an EARLIER run of the same name --
# and labels it "snapshot 1". Observed 12 s after launch on 27-08, which is
# impossible for a real save, and it offsets every later index so the whole
# k -> steps mapping is silently wrong. A mislabelled step curve is worse than
# no step curve, so only models written AFTER this moment count.
START_TS=$(date +%s)
echo "[snap] guard: ignoring any latest.model older than $START_TS"

(
  last=""; k=0
  while true; do
    m=$(ls -t "$RUNROOT"/*/models/latest.model 2>/dev/null | head -1)
    if [ -n "$m" ]; then
      cur=$(stat -c %Y "$m" 2>/dev/null || echo "")
      if [ -n "$cur" ] && [ "$cur" != "$last" ] && [ "$cur" -ge "$START_TS" ]; then
        last="$cur"; k=$((k + 1))
        cp -f "$m" "$SNAPDIR/snap_$(printf '%03d' $k)_$((k * SAVE_EVERY)).model" 2>/dev/null || true
        echo "[snap] $k -> $((k * SAVE_EVERY)) steps"
      fi
    fi
    sleep 30
  done
) &
WATCH=$!
trap 'kill $WATCH 2>/dev/null || true' EXIT

NAME="$NAME" SAVE_EVERY="$SAVE_EVERY" bash "$REPO/scripts/scaling/local_3t.sh"
rc=$?
sleep 35            # let the watcher catch the final save
kill $WATCH 2>/dev/null || true
echo "[3t-snap] rc=$rc snapshots=$(ls "$SNAPDIR" 2>/dev/null | wc -l) in $SNAPDIR"
