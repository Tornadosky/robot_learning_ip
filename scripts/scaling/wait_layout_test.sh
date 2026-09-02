#!/usr/bin/env bash
# Poll until the layout-validation job leaves the queue, then print its log.
set -u
ROOT=/ptmp/akalenik/urma
JOB="${JOB:?set JOB=<jobid>}"
for i in $(seq 1 60); do
  n=$(ssh -o ConnectTimeout=20 viper11 "squeue -j $JOB -h -o %T 2>/dev/null | wc -l" 2>/dev/null || echo "?")
  if [ "$n" = "0" ]; then break; fi
  sleep 30
done
ssh viper11 "cat $ROOT/logs/layout_test_$JOB.out 2>/dev/null | tail -45"
