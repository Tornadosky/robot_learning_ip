#!/usr/bin/env bash
# Fetch one training log per arm for pipeline_guard. Newest log wins.
set -u
DEST=/mnt/c/Users/smirn/Desktop/robot_learning_ip/experiments/fsq_khaendler/_guard_logs
mkdir -p "$DEST"
ssh viper11 "cd /ptmp/akalenik/urma/logs && ls -1t *.out" > /tmp/allogs.txt
echo "remote logs: $(wc -l < /tmp/allogs.txt)"
seen=""
while read -r f; do
  arm="${f%_*}"                     # strip the trailing _<jobid>
  case " $seen " in *" $arm "*) continue ;; esac
  case "$arm" in
    hd_*|rx_*|lk_*|w7_*|m9f_*|blw_*|pcx_*|fhd_*|best*|e1_*|e2_*|e8_*|e9_*|e10_*|bl_*|fsqrl_*) ;;
    *) continue ;;
  esac
  seen="$seen $arm"
  [ -f "$DEST/$arm.out" ] || scp -q "viper11:/ptmp/akalenik/urma/logs/$f" "$DEST/$arm.out"
done < /tmp/allogs.txt
echo "fetched: $(ls "$DEST" | wc -l)"
