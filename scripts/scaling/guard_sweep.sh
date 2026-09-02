#!/usr/bin/env bash
# Run pipeline_guard over every arm that has both a training log and crossevals.
# The point is not to pass -- it is to find out whether anything currently
# shipped is lying about its config.
set -u
REPO=/mnt/c/Users/smirn/Desktop/robot_learning_ip
FSQ=$REPO/experiments/fsq_khaendler
LOGS=$FSQ/_guard_logs
mkdir -p "$LOGS"

# One log per arm: the newest .out whose name starts with the arm name.
ssh viper11 "cd /ptmp/akalenik/urma/logs && ls -1t *.out" > /tmp/allogs.txt 2>/dev/null
ARMS=$(sed 's/_[0-9]\{6,\}\.out$//' /tmp/allogs.txt | sort -u)
for arm in $ARMS; do
  case "$arm" in
    hd_*|rx_*|lk_*|w7_*|m9f_*|blw_*|pcx_*|fhd_*|best*|e1_*|e2_*|e8_*|e9_*|e10_*|bl_*|fsqrl_*) ;;
    *) continue ;;
  esac
  [ -f "$LOGS/$arm.out" ] && continue
  f=$(grep -m1 "^${arm}_[0-9]" /tmp/allogs.txt || true)
  [ -z "$f" ] && continue
  scp -q "viper11:/ptmp/akalenik/urma/logs/$f" "$LOGS/$arm.out" 2>/dev/null || true
done
echo "logs fetched: $(ls "$LOGS" | wc -l)"

cd "$FSQ"
PY=$REPO/.venv/Scripts/python.exe
pass=0; fail=0
for f in "$LOGS"/*.out; do
  arm=$(basename "$f" .out)
  out=$("$PY" pipeline_guard.py --log "$f" --eval "wave4_controls/${arm}__*.json" "wave5_artifacts/${arm}__*.json" "wave6_artifacts/${arm}__*.json" 2>&1)
  if echo "$out" | grep -q "^0 problem"; then
    pass=$((pass+1))
  else
    fail=$((fail+1))
    echo "########## $arm"
    echo "$out" | grep -E "^  (X|!|\?) " | head -6
  fi
done
echo "=================================================="
echo "clean: $pass    flagged: $fail"
