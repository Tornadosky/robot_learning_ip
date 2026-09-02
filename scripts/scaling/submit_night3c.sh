#!/usr/bin/env bash
# NIGHT3 WAVE 3 -- fired by night3b_waiter.sh after wave 2 has mostly drained.
#
#   n3a2{rep,ref}4x   extends the A2 budget axis to 4x (78.6M steps): does the
#                     token-only gap keep closing, and does ref keep pace?
#   n3a4h2            the crossover point on the multi-clip staleness curve
#                     (W2 put the single-clip crossover between h1 and h5).
#   w2h3              sharpens the single-clip crossover location (h2 was
#                     token-positive, h5 firmly negative on the practical base).
#   s4 seeds          n=4 for the cells wave 2 left at n=3: A4 h1/h5, w2h10,
#                     gb33h1, and the three A1 LATENT_HOLD cells.
#
# 23 trains + 92 CEs = 115 jobs.
#   DRY=1  print, submit nothing
set -u
ROOT=/ptmp/akalenik/urma
cd "$ROOT"
DRY="${DRY:-0}"

TOKMC=$ROOT/clips/tokentest_mc
SUPER=$ROOT/clips/clips_super
LAF=$ROOT/clips/LAFAN1
XLA="--xla_gpu_enable_command_buffer="
CE2=$ROOT/crosseval_token2.sbatch

base() {  # base <clipdir> <clipfile> <total_steps> <footh> <ftemp> <gpen> <tc>
  local b="STAGE=mmtrain,NR_ENVS=768,TOTAL_STEPS=$3,SAVE_EVERY=1966080"
  b="$b,CLIP_FILE=$2,CLIP_DIR=$1"
  b="$b,REFVEL_OBS=False,REFROOT=True,REFROOT_FLOOR=True,REFBIAS=0.0"
  b="$b,TRACK_COEFF=30.0,TRACK_TEMP=0.05,FITVARIANT=False,ANCHOR=absolute"
  b="$b,GAITMODE=floor,GAITCOEFF=0.25,QVEL_TEMP=10"
  b="$b,FOOTZVEL=1.0,FOOTH_TEMP=$5,FOOTH=$4"
  b="$b,FOOTSLIP=6.6667,GROUNDPEN=$6,POSTCONTACT=True"
  b="$b,CONTACT_TIMECONST=$7"
  b="$b,HEADING_RATIO=0.20,HEADING_TEMP=2.0"
  b="$b,XLA_EXTRA=$XLA"
  echo "$b"
}

REF="LATENT_OBS=False,JLAT_ENC_DIM=0"
T4="LATENT_OBS=True,LATENT_DIM=32,LATENT_REPLACES=False,LATENT_SCOPE=per_joint,LATENT_DIVISOR=10.0,JLAT_ENC_DIM=4"
TREP="LATENT_OBS=True,LATENT_DIM=32,LATENT_REPLACES=True,LATENT_SCOPE=per_joint,LATENT_DIVISOR=10.0,JLAT_ENC_DIM=4"
HOBS="--environment.command.tracking_clip_observe_root_heading=True"

sub() {  local name="$1" dep="$2" exp="$3"
  local d=(); [ -n "$dep" ] && [ "$dep" != "FAKE" ] && d=(--dependency="afterany:$dep")
  if [ "$DRY" = "1" ]; then echo "TRAIN $name dep=${dep:-none}" >&2; echo "  $exp" >&2; echo "FAKE"; return; fi
  sbatch --parsable -J "$name" ${d[@]+"${d[@]}"} --export=ALL,"EXP_NAME=$name,$exp" "$ROOT/viper_train.sbatch"
}
ce() {
  local arm="$1" jid="$2" sd="$3" cdir="$4" rdir="$5" cf="$6" extra="$7"
  local d=(); [ -n "$jid" ] && [ "$jid" != "FAKE" ] && d=(--dependency="afterok:$jid")
  [ "$DRY" = "1" ] && { echo "  CE $arm s$sd clip=$cf $extra" >&2; return; }
  sbatch --parsable -J "ce_$arm" ${d[@]+"${d[@]}"} \
    --export=ALL,"EXP=$arm,CLIP_DIR=$cdir,CE_RAW_DIR=$rdir,CE_CLIP=$cf,CE_TAG=s$sd,CE_SEED=$sd,CE_REFBIAS=0.0,CE_ANCHOR=absolute,CE_FITVARIANT=False,CE_REFROOT=True,CE_REFROOT_FLOOR=True,CE_REFVEL_OBS=False,$extra" \
    "$CE2" > /dev/null
}
ce4() { for s in 0 1 2 3; do ce "$1" "$2" "$s" "$3" "$4" "$5" "$6"; done; }

ce_ref()  { echo "CE_LATENT=0,CE_JLAT_ENC_DIM=0,CE_REFHOLD=$1,CE_HEADING_OBS=True"; }
ce_t4()   { echo "CE_LATENT=1,CE_SCOPE=per_joint,CE_DIVISOR=10.0,CE_REPLACES=False,CE_JLAT_ENC_DIM=4,CE_REFHOLD=$1,CE_HEADING_OBS=True"; }
ce_trep() { echo "CE_LATENT=1,CE_SCOPE=per_joint,CE_DIVISOR=10.0,CE_REPLACES=True,CE_JLAT_ENC_DIM=4,CE_REFHOLD=$1,CE_HEADING_OBS=True"; }

ea() {  local v="--environment.command.tracking_clip_reference_hold=$1"
  [ "$2" != "1" ] && v="$v:--environment.seed=$2"
  echo "EXTRA_ARGS=$v:$HOBS"
}

seedrun() {  # <cell> <base> <expv> <ce_extra> <refhold> <cdir> <rdir> <clip> <seed> <dep> [xx] -> jid
  local name="${1}_s${9}"
  local exports="$2,$3,$(ea "$5" "$9")"
  [ -n "${11:-}" ] && exports="$exports,${11}"
  local J; J=$(sub "$name" "${10}" "$exports")
  echo "$name -> $J" >&2
  ce4 "$name" "$J" "$6" "$7" "$8" "$4"
  echo "$J"
}
chain2() {
  local cell="$1" b="$2" expv="$3" cex="$4" hold="$5" cdir="$6" rdir="$7" cf="$8" xx="${9:-}" PREV="" s J
  for s in 1 2; do
    J=$(seedrun "$cell" "$b" "$expv" "$cex" "$hold" "$cdir" "$rdir" "$cf" "$s" "$PREV" "$xx")
    PREV="$J"
  done
}

DANCE=dance2_subject4.npz
SUPERC=super5dance.npz
B1X=$(base "$TOKMC" "$DANCE" 19660800 0.0 0.01 333.33 0.0)
B4X=$(base "$TOKMC" "$DANCE" 78643200 0.0 0.01 333.33 0.0)
BA4=$(base "$SUPER" "$SUPERC" 19660800 0.0 0.01 333.33 0.0)
BGB=$(base "$TOKMC" "$DANCE" 19660800 0.3333 0.05 1000 0.004)

echo "########## A2 4x budget (token-only + ref control) ##########"
chain2 n3a2rep4x "$B4X" "$TREP" "$(ce_trep 1)" 1 "$TOKMC" "$LAF" "$DANCE"
chain2 n3a2ref4x "$B4X" "$REF"  "$(ce_ref 1)"  1 "$TOKMC" "$LAF" "$DANCE"

echo "########## multi-clip crossover h2 + single-clip crossover h3 ##########"
chain2 n3a4h2_ref "$BA4" "$REF" "$(ce_ref 2)" 2 "$SUPER" "$SUPER" "$SUPERC"
chain2 n3a4h2_tok "$BA4" "$T4"  "$(ce_t4 2)"  2 "$SUPER" "$SUPER" "$SUPERC"
chain2 w2h3_ref   "$B1X" "$REF" "$(ce_ref 3)" 3 "$TOKMC" "$LAF" "$DANCE"
chain2 w2h3_tok   "$B1X" "$T4"  "$(ce_t4 3)"  3 "$TOKMC" "$LAF" "$DANCE"

echo "########## s4 for the n=3 cells ##########"
seedrun n3a4h1_ref "$BA4" "$REF" "$(ce_ref 1)" 1 "$SUPER" "$SUPER" "$SUPERC" 4 "" > /dev/null
seedrun n3a4h1_tok "$BA4" "$T4"  "$(ce_t4 1)"  1 "$SUPER" "$SUPER" "$SUPERC" 4 "" > /dev/null
seedrun n3a4h5_ref "$BA4" "$REF" "$(ce_ref 5)" 5 "$SUPER" "$SUPER" "$SUPERC" 4 "" > /dev/null
seedrun n3a4h5_tok "$BA4" "$T4"  "$(ce_t4 5)"  5 "$SUPER" "$SUPER" "$SUPERC" 4 "" > /dev/null
seedrun w2h10_ref  "$B1X" "$REF" "$(ce_ref 10)" 10 "$TOKMC" "$LAF" "$DANCE" 4 "" > /dev/null
seedrun w2h10_tok  "$B1X" "$T4"  "$(ce_t4 10)"  10 "$TOKMC" "$LAF" "$DANCE" 4 "" > /dev/null
seedrun n3gb33h1_ref "$BGB" "$REF" "$(ce_ref 1),CE_CONTACT_TIMECONST=0.004" 1 "$TOKMC" "$LAF" "$DANCE" 4 "" > /dev/null
seedrun n3gb33h1_tok "$BGB" "$T4"  "$(ce_t4 1),CE_CONTACT_TIMECONST=0.004" 1 "$TOKMC" "$LAF" "$DANCE" 4 "" > /dev/null
seedrun n3a1rep_lh5  "$B1X" "$TREP" "$(ce_trep 1),CE_HOLD=5"  1 "$TOKMC" "$LAF" "$DANCE" 4 "" "LATENT_HOLD=5"  > /dev/null
seedrun n3a1rep_lh20 "$B1X" "$TREP" "$(ce_trep 1),CE_HOLD=20" 1 "$TOKMC" "$LAF" "$DANCE" 4 "" "LATENT_HOLD=20" > /dev/null
seedrun n3a1tok_lh5  "$B1X" "$T4"   "$(ce_t4 1),CE_HOLD=5"    1 "$TOKMC" "$LAF" "$DANCE" 4 "" "LATENT_HOLD=5"  > /dev/null

echo "########## WAVE 3 SUBMITTED ##########"
squeue -u akalenik -h | wc -l
