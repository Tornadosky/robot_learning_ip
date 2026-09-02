#!/usr/bin/env bash
# NIGHT3 waiter -- runs under nohup on viper11. When wave 1 has mostly drained
# it submits wave 2 (submit_night3b.sh), then wave 3 (submit_night3c.sh).
# Guard files make each wave submit-once even if the waiter is restarted.
# Thresholds keep total queued well under MaxSubmit=300.
set -u
ROOT=/ptmp/akalenik/urma
LOG=$ROOT/logs/night3_waiter.log
exec >>"$LOG" 2>&1
echo "=== waiter start $(date) ==="

wait_below() {  # wait_below <threshold>
  while :; do
    n=$(squeue -u akalenik -h 2>/dev/null | wc -l)
    echo "$(date '+%d %H:%M') queued=$n (waiting for <=$1)"
    [ "$n" -le "$1" ] && return 0
    sleep 300
  done
}

if [ ! -f "$ROOT/.night3b_submitted" ]; then
  wait_below 50
  touch "$ROOT/.night3b_submitted"
  echo "=== submitting WAVE 2 $(date) ==="
  bash "$ROOT/submit_night3b.sh"
else
  echo "wave 2 already submitted, skipping"
fi

sleep 600
if [ ! -f "$ROOT/.night3c_submitted" ]; then
  wait_below 60
  touch "$ROOT/.night3c_submitted"
  echo "=== submitting WAVE 3 $(date) ==="
  bash "$ROOT/submit_night3c.sh"
else
  echo "wave 3 already submitted, skipping"
fi
echo "=== waiter done $(date) ==="
