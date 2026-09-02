#!/usr/bin/env bash
# ONE wsl session, ONE ssh session, loop on the far side. Spawning a fresh wsl
# client per poll piled up 31 stuck processes today and wedged the WSL service;
# the 3T run's own watchdog documents the same failure mode.
#
# Waits for the arms that actually exercise the new code -- tk1..tk4 -- rather
# than tk0, which is the no-token control and would prove nothing. Reports the
# moment any arm errors, or once every token arm has logged real steps.
ssh -o ConnectTimeout=25 viper11 'bash -s' <<'REMOTE'
ROOT=/ptmp/akalenik/urma
for i in $(seq 1 120); do
  hit=0
  out=""
  for arm in tk1_perj1 tk2_perj10 tk3_glob tk4_perjsep; do
    f=$(ls -t $ROOT/logs/${arm}_*.out 2>/dev/null | head -1)
    if [ -z "$f" ]; then out="$out\n  $arm: not started"; continue; fi
    err=$(grep -m1 -E "Traceback|Error|error:|Killed|out of memory|ValueError|ABORT" "$f" 2>/dev/null | head -1)
    step=$(grep -o "nr_env_steps[^0-9]*[0-9]\+" "$f" 2>/dev/null | tail -1)
    if [ -n "$err" ]; then out="$out\n  $arm: ERROR $err"; hit=1
    elif [ -n "$step" ]; then out="$out\n  $arm: OK $step"; hit=$((hit+1))
    else out="$out\n  $arm: compiling"; fi
  done
  # Stop on any error, or once all four have real steps.
  if echo -e "$out" | grep -q ERROR; then echo -e "STOPPED ON ERROR$out"; exit 0; fi
  ok=$(echo -e "$out" | grep -c " OK ")
  if [ "$ok" = "4" ]; then echo -e "ALL FOUR TOKEN ARMS TRAINING$out"; exit 0; fi
  sleep 60
done
echo -e "WATCH TIMED OUT$out"
REMOTE
