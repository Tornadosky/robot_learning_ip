#!/usr/bin/env bash
# Assemble experiments/fsq_khaendler/clips_5r: super20 (20 train motions) +
# 7 held-out motions per robot, with _win/_zq sidecars from tokenizer_m20.
#   H1, G1  : copied from clips_m20 (already built + sidecars from the tokenizer run)
#   T1, Atlas, Apollo : from LAFAN1_allfix (feasibility re-issue), sidecars via
#                       emit_sidecars_any.py (encoder-only, alias tables)
# Robots whose re-issued clips are incomplete are skipped; the resulting robot
# list is written to clips_5r/ROBOTS and clips_5r/READY marks completion.
# Waits for the Atlas/Apollo re-issue and the tokenizer if asked (WAIT=1).
set -u
REPO=${REPO:-/mnt/c/Users/smirn/Desktop/robot_learning_ip}
cd "$REPO"
PY=${PY:-~/jaxgpu/bin/python}
L=experiments/fsq_khaendler/_tok_logs
M20=experiments/fsq_khaendler/clips_m20
FIX=external_data/amass_converted/LAFAN1_allfix
OUT=experiments/fsq_khaendler/clips_5r
TOK=experiments/fsq_khaendler/tokenizer_m20
log() { echo "[5r $(date '+%m-%d %H:%M:%S')] $*"; }
if [ "${WAIT:-1}" = 1 ]; then
  log "waiting for tokenizer + Atlas/Apollo re-issue"
  for i in $(seq 1 1440); do
    t=$(grep -c "TOKENIZER PIPELINE DONE" $L/tokenizer_m20.log 2>/dev/null); a=$(grep -c "^OK\|^!!" $L/reissue_aa_all.log 2>/dev/null)
    [ "${t:-0}" -ge 1 ] && [ "${a:-0}" -ge 68 ] && break; sleep 60
  done
fi
TR=$(sed 's/$/.npz/' $M20/train_motions.txt | tr '\n' ' ')
HO=$(sed 's/$/.npz/' $M20/heldout_motions.txt | tr '\n' ' ')
mkdir -p $OUT
ROBOTS=""
for r in UnitreeH1 UnitreeG1; do mkdir -p $OUT/$r; cp -f $M20/$r/*.npz $OUT/$r/; ROBOTS="$ROBOTS $r"; done
for r in BoosterT1 Atlas Apollo; do
  n=0; for c in $TR $HO; do [ -f $FIX/$r/$c ] && n=$((n+1)); done
  if [ "$n" -lt 27 ]; then log "$r: only $n/27 re-issued clips -> SKIPPED"; continue; fi
  mkdir -p $OUT/$r
  JAX_PLATFORMS=cpu PYTHONPATH=$REPO $PY experiments/fsq_khaendler/build_superclip.py --clip-dir $FIX --out-dir $OUT --name super20.npz --robots $r --clips $TR 2>&1 | grep -v pygame | tail -1
  for c in $HO; do cp -f $FIX/$r/$c $OUT/$r/; done
  ROBOTS="$ROBOTS $r"
done
declare -A MAP=([UnitreeH1]=unitree_h1 [UnitreeG1]=unitree_g1 [BoosterT1]=booster_t1 [Atlas]=atlas [Apollo]=apptronik_apollo)
NEW=""; for r in $ROBOTS; do case $r in BoosterT1|Atlas|Apollo) NEW="$NEW ${MAP[$r]}";; esac; done
if [ -n "$NEW" ]; then
  log "sidecars for:$NEW"
  JAX_PLATFORMS=cpu PYTHONPATH=$REPO:$REPO/loco_mjx $PY scripts/scaling/wave7/emit_sidecars_any.py --tokenizer $TOK --clip-dir $OUT --robots $NEW --clip super20.npz $HO 2>&1 | grep -v pygame
fi
echo $(for r in $ROBOTS; do echo -n "${MAP[$r]}:"; done | sed 's/:$//') > $OUT/ROBOTS
ls $OUT/*/ | head -40; du -sh $OUT
touch $OUT/READY; log "READY robots=$(cat $OUT/ROBOTS)"
