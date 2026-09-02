#!/usr/bin/env bash
# LOCAL WAVE 4 -- 2026-08-28. The three-topology half of the contact-dose test.
#
# WHY LOCAL: Viper has never once survived three topologies (aperture violation
# at every env count ever tried). The baseline question -- "does freeing the feet
# fix the LEGS on the real 3-topology policy?" -- can therefore only be answered
# here. Viper runs the same dose on H1+G1 in parallel (CH-A), so by morning the
# dose has a 2-topology and a 3-topology reading.
#
# WHAT IT TESTS. ce_corrfinal_s0.json, computed on the converged 3T checkpoint:
#   - every ankle on all three robots has per-joint corr ~0.00 against a
#     near-PERFECT ankle reference (ref_vs_raw 0.0023-0.0091 rad)
#   - the reference wants the feet airborne 12.3% (H1) / 25.4% (G1) of the time;
#     the policy manages 3.1% / 4.0%
# Hypothesis: foot_slip_coeff=20 and ground_penetration_coeff=1000 pin the foot,
# and everything distal to a pinned contact is kinematically slaved. Dose them.
#
# ORDER IS DELIBERATE. The control's extra seeds run FIRST and cost 30 minutes:
# ce_corrfinal_s0 is n=1 and every comparison below rests on it, while the
# standing rule on this project is n>=4 because one rollout seed moves a score
# by up to 4.95%.
#
# Launch detached:
#   setsid nohup bash scripts/scaling/local_wave4.sh > /tmp/local_wave4.log 2>&1 < /dev/null &
set -u

REPO=/mnt/c/Users/smirn/Desktop/robot_learning_ip
OUT=$REPO/experiments/local_3t
CTLSNAP=$OUT/snapshots/local_3t_dance4
mkdir -p "$OUT"

# bash reads a script INCREMENTALLY as it runs, so a long-lived launcher must be
# executed from a copy that nothing will edit underneath it (this bit local_3t
# once already). Same for every helper it calls.
tr -d '\r' < "$REPO/scripts/scaling/local_3t_eval_snap.sh" > /tmp/w4_eval.sh
tr -d '\r' < "$REPO/scripts/scaling/local_3t_snap.sh"      > /tmp/w4_snap.sh
tr -d '\r' < "$REPO/scripts/scaling/local_3t.sh"           > /tmp/w4_train.sh

say() { echo "[w4] $(date -Is) $*"; }

# ---------------------------------------------------------------- PHASE 0
# Error bars on the control. ce_corrfinal_s0 is n=1.
say "PHASE 0 -- control seeds 1-3 (policy) and 1 (zero floor)"
CTL=$(ls -1 "$CTLSNAP"/snap_*.model 2>/dev/null | sort | tail -1)
if [ -z "$CTL" ]; then
  say "FATAL: no control checkpoint under $CTLSNAP -- cannot compare anything"
else
  say "control checkpoint: $CTL"
  for S in 1 2 3; do
    MODEL="$CTL" TAG=corrfinal SEED=$S bash /tmp/w4_eval.sh
  done
  MODEL="$CTL" TAG=corrzero SEED=1 bash /tmp/w4_eval.sh
fi

# ---------------------------------------------------------------- PHASE 1..2
# The dose. 10x and 30x on BOTH contact coefficients together. 30x is
# deliberately past the expected optimum: a dose has a CEILING and nothing in
# the dose metric itself flags where "degrade X" silently becomes "remove X".
# foot_penetration_m is the tell, so it is co-reported rather than assumed.
run_dose() {   # run_dose <name> <footslip> <groundpen>
  local name="$1" fs="$2" gp="$3"
  say "TRAIN $name  FOOTSLIP=$fs GROUNDPEN=$gp"
  if [ -n "$(ls -A "$OUT/snapshots/$name" 2>/dev/null)" ]; then
    say "SKIP $name -- snapshot dir already populated"
  else
    L3T_NAME="$name" FOOTSLIP="$fs" GROUNDPEN="$gp" bash /tmp/w4_snap.sh
  fi
  # Verify by ARTIFACT and by logged blocks, never by exit code: RL-X catches
  # exceptions and exits 0.
  local blocks
  blocks=$(grep -c "nr_env_steps" "$OUT/${name}.log" 2>/dev/null || echo 0)
  say "$name blocks=$blocks snapshots=$(ls "$OUT/snapshots/$name" 2>/dev/null | wc -l)"
  if [ "$blocks" -lt 10 ]; then
    say "$name LOOKS DEAD -- skipping its evals so a broken arm produces no numbers"
    grep -E "Error|Traceback|Exception|out of memory" "$OUT/${name}.log" 2>/dev/null | tail -5
    return
  fi
  local last
  last=$(ls -1 "$OUT/snapshots/$name"/snap_*.model 2>/dev/null | sort | tail -1)
  [ -z "$last" ] && { say "$name has no snapshot to evaluate"; return; }
  say "EVAL $name -> $last"
  for S in 0 1 2 3; do
    MODEL="$last" TAG="${name}_fin" SEED=$S bash /tmp/w4_eval.sh
  done
}

run_dose l3t_c10 2.0     100
run_dose l3t_c30 0.66667 33.333

say "LOCAL WAVE 4 COMPLETE"
ls -la "$OUT"/ce_*.json | tail -20
