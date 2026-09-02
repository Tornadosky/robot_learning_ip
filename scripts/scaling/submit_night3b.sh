#!/usr/bin/env bash
# NIGHT3 WAVE 2 -- fired by night3b_waiter.sh once wave 1 has mostly drained.
# Wave 1 ran ~26-wide (the assoc MaxJobs=8 is not what actually gates the apu
# partition), so it drains by ~midnight; this wave keeps the night productive:
#
#   s3 for every wave-1 cell      seed noise has bitten this project repeatedly
#                                 ("rollout seed alone moves a crosseval 4.95%");
#                                 n=3 trains x 4 CE seeds on all 21 cells.
#   s4 for the G dose grid + A4G  the dose decision is THE morning decision;
#                                 n=4 there.
#   w2h10                         fills the practical-recipe rate curve
#                                 (1/2/5/10/20 -- h10 was V1/B4-base only).
#   n3a4h20                       staleness horizon on the multi-clip base.
#   n3gb33h1 (+dumps)             the video arm: single dance2_subject4 at
#                                 hold=1 WITH the mid gait dose.
#   n3a2ref{2x,3x}                THE MISSING CONTROL for A2: if ref also gains
#                                 from 2x/3x budget, token-only's gap closing
#                                 is generic compute, not token information.
#
# 49 trains + 196 CEs + 2 dumps = 247 jobs.
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
cedump() {
  local arm="$1" jid="$2" cdir="$3" rdir="$4" cf="$5" extra="$6"
  local d=(); [ -n "$jid" ] && [ "$jid" != "FAKE" ] && d=(--dependency="afterok:$jid")
  [ "$DRY" = "1" ] && { echo "  CEDUMP $arm $extra" >&2; return; }
  sbatch --parsable -J "ced_$arm" ${d[@]+"${d[@]}"} \
    --export=ALL,"EXP=$arm,CLIP_DIR=$cdir,CE_RAW_DIR=$rdir,CE_CLIP=$cf,CE_TAG=dump,CE_SEED=0,CE_DUMP=1,CE_REFBIAS=0.0,CE_ANCHOR=absolute,CE_FITVARIANT=False,CE_REFROOT=True,CE_REFROOT_FLOOR=True,CE_REFVEL_OBS=False,$extra" \
    "$CE2" > /dev/null
}

ce_ref()  { echo "CE_LATENT=0,CE_JLAT_ENC_DIM=0,CE_REFHOLD=$1,CE_HEADING_OBS=True"; }
ce_t4()   { echo "CE_LATENT=1,CE_SCOPE=per_joint,CE_DIVISOR=10.0,CE_REPLACES=False,CE_JLAT_ENC_DIM=4,CE_REFHOLD=$1,CE_HEADING_OBS=True"; }
ce_trep() { echo "CE_LATENT=1,CE_SCOPE=per_joint,CE_DIVISOR=10.0,CE_REPLACES=True,CE_JLAT_ENC_DIM=4,CE_REFHOLD=$1,CE_HEADING_OBS=True"; }

ea() {  local v="--environment.command.tracking_clip_reference_hold=$1"
  [ "$2" != "1" ] && v="$v:--environment.seed=$2"
  echo "EXTRA_ARGS=$v:$HOBS"
}

# seedrun <cell> <base> <expv> <ce_extra> <refhold> <cdir> <rdir> <clip> <seed> <dep> [xx] -> jid
seedrun() {
  local name="${1}_s${9}"
  local exports="$2,$3,$(ea "$5" "$9")"
  [ -n "${11:-}" ] && exports="$exports,${11}"
  local J; J=$(sub "$name" "${10}" "$exports")
  echo "$name -> $J" >&2
  ce4 "$name" "$J" "$6" "$7" "$8" "$4"
  echo "$J"
}
# chain2 for the brand-new cells (s1 -> s2)
chain2() {
  local cell="$1" b="$2" expv="$3" cex="$4" hold="$5" cdir="$6" rdir="$7" cf="$8" xx="${9:-}" PREV="" s J J1=""
  for s in 1 2; do
    J=$(seedrun "$cell" "$b" "$expv" "$cex" "$hold" "$cdir" "$rdir" "$cf" "$s" "$PREV" "$xx")
    [ "$s" = "1" ] && J1="$J"
    PREV="$J"
  done
  echo "$J1"
}

DANCE=dance2_subject4.npz
SUPERC=super5dance.npz
B1X=$(base "$TOKMC" "$DANCE" 19660800 0.0 0.01 333.33 0.0)

echo "########## s3 (all wave-1 cells) + s4 (G grid + A4G) ##########"
# gseed <cell> <expv> <cefn> <footh> <ftemp> <gpen> <tc>   (hold=5, s3 then s4)
gseed() {
  local cell="$1" expv="$2" cefn="$3" b cex J
  b=$(base "$TOKMC" "$DANCE" 19660800 "$4" "$5" "$6" "$7")
  cex="$($cefn 5),CE_CONTACT_TIMECONST=$7"
  J=$(seedrun "$cell" "$b" "$expv" "$cex" 5 "$TOKMC" "$LAF" "$DANCE" 3 "")
  seedrun "$cell" "$b" "$expv" "$cex" 5 "$TOKMC" "$LAF" "$DANCE" 4 "$J" > /dev/null
}
gseed n3gct_ref  "$REF" ce_ref 0.0    0.01 1000   0.004
gseed n3gct_tok  "$T4"  ce_t4  0.0    0.01 1000   0.004
gseed n3gf33_ref "$REF" ce_ref 0.3333 0.05 333.33 0.0
gseed n3gf33_tok "$T4"  ce_t4  0.3333 0.05 333.33 0.0
gseed n3gf50_ref "$REF" ce_ref 0.5    0.05 333.33 0.0
gseed n3gf50_tok "$T4"  ce_t4  0.5    0.05 333.33 0.0
gseed n3gb33_ref "$REF" ce_ref 0.3333 0.05 1000   0.004
gseed n3gb33_tok "$T4"  ce_t4  0.3333 0.05 1000   0.004
gseed n3gb50_ref "$REF" ce_ref 0.5    0.05 1000   0.004
gseed n3gb50_tok "$T4"  ce_t4  0.5    0.05 1000   0.004

BA4G=$(base "$SUPER" "$SUPERC" 19660800 0.3333 0.05 1000 0.004)
J=$(seedrun n3a4g_ref "$BA4G" "$REF" "$(ce_ref 1),CE_CONTACT_TIMECONST=0.004" 1 "$SUPER" "$SUPER" "$SUPERC" 3 "")
seedrun n3a4g_ref "$BA4G" "$REF" "$(ce_ref 1),CE_CONTACT_TIMECONST=0.004" 1 "$SUPER" "$SUPER" "$SUPERC" 4 "$J" > /dev/null
J=$(seedrun n3a4g_tok "$BA4G" "$T4" "$(ce_t4 1),CE_CONTACT_TIMECONST=0.004" 1 "$SUPER" "$SUPER" "$SUPERC" 3 "")
seedrun n3a4g_tok "$BA4G" "$T4" "$(ce_t4 1),CE_CONTACT_TIMECONST=0.004" 1 "$SUPER" "$SUPER" "$SUPERC" 4 "$J" > /dev/null

BA4=$(base "$SUPER" "$SUPERC" 19660800 0.0 0.01 333.33 0.0)
seedrun n3a4h1_ref "$BA4" "$REF" "$(ce_ref 1)" 1 "$SUPER" "$SUPER" "$SUPERC" 3 "" > /dev/null
seedrun n3a4h1_tok "$BA4" "$T4"  "$(ce_t4 1)"  1 "$SUPER" "$SUPER" "$SUPERC" 3 "" > /dev/null
seedrun n3a4h5_ref "$BA4" "$REF" "$(ce_ref 5)" 5 "$SUPER" "$SUPER" "$SUPERC" 3 "" > /dev/null
seedrun n3a4h5_tok "$BA4" "$T4"  "$(ce_t4 5)"  5 "$SUPER" "$SUPER" "$SUPERC" 3 "" > /dev/null

B2X=$(base "$TOKMC" "$DANCE" 39321600 0.0 0.01 333.33 0.0)
B3X=$(base "$TOKMC" "$DANCE" 58982400 0.0 0.01 333.33 0.0)
seedrun n3a2rep2x "$B2X" "$TREP" "$(ce_trep 1)" 1 "$TOKMC" "$LAF" "$DANCE" 3 "" > /dev/null
seedrun n3a2rep3x "$B3X" "$TREP" "$(ce_trep 1)" 1 "$TOKMC" "$LAF" "$DANCE" 3 "" > /dev/null

seedrun n3a1rep_lh5  "$B1X" "$TREP" "$(ce_trep 1),CE_HOLD=5"  1 "$TOKMC" "$LAF" "$DANCE" 3 "" "LATENT_HOLD=5"  > /dev/null
seedrun n3a1rep_lh20 "$B1X" "$TREP" "$(ce_trep 1),CE_HOLD=20" 1 "$TOKMC" "$LAF" "$DANCE" 3 "" "LATENT_HOLD=20" > /dev/null
seedrun n3a1tok_lh5  "$B1X" "$T4"   "$(ce_t4 1),CE_HOLD=5"    1 "$TOKMC" "$LAF" "$DANCE" 3 "" "LATENT_HOLD=5"  > /dev/null

echo "########## new cells: w2h10, a4h20, gb33h1(+dumps), a2ref budget control ##########"
chain2 w2h10_ref "$B1X" "$REF" "$(ce_ref 10)" 10 "$TOKMC" "$LAF" "$DANCE" > /dev/null
chain2 w2h10_tok "$B1X" "$T4"  "$(ce_t4 10)"  10 "$TOKMC" "$LAF" "$DANCE" > /dev/null

chain2 n3a4h20_ref "$BA4" "$REF" "$(ce_ref 20)" 20 "$SUPER" "$SUPER" "$SUPERC" > /dev/null
chain2 n3a4h20_tok "$BA4" "$T4"  "$(ce_t4 20)"  20 "$SUPER" "$SUPER" "$SUPERC" > /dev/null

BGB=$(base "$TOKMC" "$DANCE" 19660800 0.3333 0.05 1000 0.004)
J=$(chain2 n3gb33h1_ref "$BGB" "$REF" "$(ce_ref 1),CE_CONTACT_TIMECONST=0.004" 1 "$TOKMC" "$LAF" "$DANCE")
cedump n3gb33h1_ref_s1 "$J" "$TOKMC" "$LAF" "$DANCE" "$(ce_ref 1),CE_CONTACT_TIMECONST=0.004"
J=$(chain2 n3gb33h1_tok "$BGB" "$T4" "$(ce_t4 1),CE_CONTACT_TIMECONST=0.004" 1 "$TOKMC" "$LAF" "$DANCE")
cedump n3gb33h1_tok_s1 "$J" "$TOKMC" "$LAF" "$DANCE" "$(ce_t4 1),CE_CONTACT_TIMECONST=0.004"

chain2 n3a2ref2x "$B2X" "$REF" "$(ce_ref 1)" 1 "$TOKMC" "$LAF" "$DANCE" > /dev/null
chain2 n3a2ref3x "$B3X" "$REF" "$(ce_ref 1)" 1 "$TOKMC" "$LAF" "$DANCE" > /dev/null

echo "########## WAVE 2 SUBMITTED ##########"
squeue -u akalenik -h | wc -l
