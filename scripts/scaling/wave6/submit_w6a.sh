#!/usr/bin/env bash
# WAVE 6a (2026-09-02) -- "does the FSQ token pay BESIDES staleness?"  No new code.
#
#   A  BUDGET      token-only (LATENT_REPLACES) vs reference-only at 2x and 4x the
#                  19.66M budget: is token-only's +11/+19% an information gap or a
#                  compute gap?  Zero-shot walk1 at hold 1 on both.
#   B  MULTI-MOTION 5-dance super-clip: {ref,tok} at 2x budget (was a tie at 1x),
#                  and {ref,tok} TRAINED at hold 20 (multi-motion staleness).
#   C  MORPHOLOGY  every Viper arm so far trained under the sbatch default
#                  morphology schedule (0.2 -> 0.44 at 19.66M).  Two controls:
#                  m0 = nominal body only (fixed 0.0); m7 = ramp to 0.7 by 15M.
#                  Token x randomization interaction at hold 1 (and h20 CEs).
#   D  GAIT        token at swing dose 25 (ref exists: n5sw25_ref).
#   E  extra CEs   zero-shot walk1 at HOLD 1 for the existing n5v2_tok seeds and
#                  n5sw015_ref seeds (token vs ref on an unseen clip, fresh ref).
#
# 26 trains + ~150 CEs.  DRY=1 to print.  Guard: .w6a_submitted
set -u
ROOT=/ptmp/akalenik/urma
cd "$ROOT"
DRY="${DRY:-0}"
[ -f "$ROOT/.w6a_submitted" ] && [ "$DRY" != "1" ] && { echo "already submitted"; exit 0; }

V2=$ROOT/clips/tokentest_v2
TOKMC=$ROOT/clips/tokentest_mc
SUPER=$ROOT/clips/clips_super
SUPERC=super5dance.npz
LAF=$ROOT/clips/LAFAN1
XLA="--xla_gpu_enable_command_buffer="
CE3=$ROOT/crosseval_token3.sbatch
for f in "$V2/UnitreeH1/walk1_subject1_zq.npz" "$SUPER/UnitreeG1/super5dance_zq.npz" "$TOKMC/UnitreeH1/dance2_subject4_zq.npz" "$CE3" "$ROOT/viper_train.sbatch"; do
  [ -f "$f" ] || { echo "ABORT: missing $f"; exit 1; }
done

base() {  # <clipdir> <clipfile> <total_steps>
  local b="STAGE=mmtrain,NR_ENVS=768,TOTAL_STEPS=$3,SAVE_EVERY=1966080"
  b="$b,CLIP_FILE=$2,CLIP_DIR=$1"
  b="$b,REFVEL_OBS=False,REFROOT=True,REFROOT_FLOOR=True,REFBIAS=0.0"
  b="$b,TRACK_COEFF=30.0,TRACK_TEMP=0.05,FITVARIANT=False,ANCHOR=absolute"
  b="$b,GAITMODE=floor,GAITCOEFF=0.25,QVEL_TEMP=10"
  b="$b,FOOTZVEL=1.0,FOOTH_TEMP=0.05,FOOTH=0.3333"
  b="$b,FOOTSLIP=6.6667,GROUNDPEN=1000,POSTCONTACT=True"
  b="$b,CONTACT_TIMECONST=0.004,HEADING_RATIO=0.20,HEADING_TEMP=2.0"
  b="$b,XLA_EXTRA=$XLA"
  echo "$b"
}
S1=19660800; S2=39321600; S4=78643200
REF="LATENT_OBS=False,JLAT_ENC_DIM=0"
T4="LATENT_OBS=True,LATENT_DIM=32,LATENT_REPLACES=False,LATENT_SCOPE=per_joint,LATENT_DIVISOR=10.0,JLAT_ENC_DIM=4"
TREP="LATENT_OBS=True,LATENT_DIM=32,LATENT_REPLACES=True,LATENT_SCOPE=per_joint,LATENT_DIVISOR=10.0,JLAT_ENC_DIM=4"
M0="MORPH_MODE=fixed,MORPH_COEFF=0.0"
M7="MORPH_MODE=schedule,MORPH_COEFF=0.7,MORPH_START=0.2,MORPH_RAMP=15000000"
HOBS="--environment.command.tracking_clip_observe_root_heading=True"
SW() { echo "--environment.reward.deepmimic_swing_match_weight_ratio=$1"; }
CEREF() { echo "CE_LATENT=0,CE_JLAT_ENC_DIM=0,CE_HEADING_OBS=True,CE_REFHOLD=$1"; }
CET4()  { echo "CE_LATENT=1,CE_SCOPE=per_joint,CE_DIVISOR=10.0,CE_REPLACES=False,CE_JLAT_ENC_DIM=4,CE_HEADING_OBS=True,CE_REFHOLD=$1"; }
CETR()  { echo "CE_LATENT=1,CE_SCOPE=per_joint,CE_DIVISOR=10.0,CE_REPLACES=True,CE_JLAT_ENC_DIM=4,CE_HEADING_OBS=True,CE_REFHOLD=$1"; }

ea() {  # <refhold> <seed> [extra colon-joined flags]
  local v="--environment.command.tracking_clip_reference_hold=$1"
  [ "$2" != "1" ] && v="$v:--environment.seed=$2"
  [ -n "${3:-}" ] && v="$v:$3"
  echo "EXTRA_ARGS=$v:$HOBS"
}
sub() {  # <name> <dep|-> <export-string> -> jid
  local d=(); [ -n "$2" ] && [ "$2" != "-" ] && [ "$2" != "FAKE" ] && d=(--dependency="afterany:$2")
  if [ "$DRY" = 1 ]; then echo "TRAIN $1 dep=$2" >&2; echo "   $3" >&2; echo FAKE; return; fi
  sbatch --parsable -J "$1" ${d[@]+"${d[@]}"} --export=ALL,"EXP_NAME=$1,$3" "$ROOT/viper_train.sbatch"
}
ce() {  # <exp> <dep|-> <tag> <ceseed> <extra> <clipdir> <rawdir> <clip>
  local exp=$1 jid=$2 tag=$3 sd=$4 extra=$5 cdir=$6 rdir=$7 clip=$8
  local d=(); [ -n "$jid" ] && [ "$jid" != "-" ] && [ "$jid" != "FAKE" ] && d=(--dependency="afterok:$jid")
  [ "$DRY" = 1 ] && { echo "  CE $exp $tag clip=$clip $extra" >&2; return; }
  sbatch --parsable -J "ce_$exp" ${d[@]+"${d[@]}"} \
    --export=ALL,"EXP=$exp,CLIP_DIR=$cdir,CE_RAW_DIR=$rdir,CE_CLIP=$clip,CE_TAG=$tag,CE_SEED=$sd,CE_REFBIAS=0.0,CE_ANCHOR=absolute,CE_FITVARIANT=False,CE_REFROOT=True,CE_REFROOT_FLOOR=True,CE_REFVEL_OBS=False,CE_CONTACT_TIMECONST=0.004,CE_NRENVS=64,$extra" \
    "$CE3" > /dev/null
}
# cell <name> <base> <expv> <refhold> <extra_flags> <cefun> <cedir> <cerawdir> <ceclip> <ce_holds:"1,20"> <zs 0|1>
cell() {
  local name=$1 b=$2 expv=$3 rh=$4 xf=$5 cef=$6 cdir=$7 rdir=$8 clip=$9 holds=${10} zs=${11}
  local PREV="-" s J
  for s in 1 2; do
    J=$(sub "${name}_s$s" "$PREV" "$b,$expv,$(ea "$rh" "$s" "$xf")")
    echo "${name}_s$s -> $J" >&2
    for h in ${holds//,/ }; do
      local n=4; [ "$h" != "$rh" ] && n=2
      for ((cs=0; cs<n; cs++)); do ce "${name}_s$s" "$J" "h${h}_s$cs" "$cs" "$($cef "$h")" "$cdir" "$rdir" "$clip"; done
    done
    if [ "$zs" = 1 ]; then
      for cs in 0 1; do ce "${name}_s$s" "$J" "zsh1_$cs" "$cs" "$($cef 1)" "$V2" "$V2" walk1_subject1.npz; done
    fi
    PREV="$J"
  done
}
SW05="$(SW 0.5)"

echo "##### A: budget -- token-only vs ref at 2x / 4x #####"
cell n6rep2x_tok "$(base "$V2" dance2_subject4.npz $S2)" "$TREP" 1 "$SW05" CETR "$V2" "$V2" dance2_subject4.npz "1" 1
cell n6ref2x     "$(base "$V2" dance2_subject4.npz $S2)" "$REF"  1 "$SW05" CEREF "$V2" "$V2" dance2_subject4.npz "1" 1
cell n6rep4x_tok "$(base "$V2" dance2_subject4.npz $S4)" "$TREP" 1 "$SW05" CETR "$V2" "$V2" dance2_subject4.npz "1" 1
cell n6ref4x     "$(base "$V2" dance2_subject4.npz $S4)" "$REF"  1 "$SW05" CEREF "$V2" "$V2" dance2_subject4.npz "1" 1

echo "##### B: multi-motion super-clip -- 2x budget at hold 1, and trained at hold 20 #####"
cell n6sup2x_ref  "$(base "$SUPER" $SUPERC $S2)" "$REF" 1  "$SW05" CEREF "$SUPER" "$SUPER" $SUPERC "1" 0
cell n6sup2x_tok  "$(base "$SUPER" $SUPERC $S2)" "$T4"  1  "$SW05" CET4  "$SUPER" "$SUPER" $SUPERC "1" 0
cell n6suph20_ref "$(base "$SUPER" $SUPERC $S1)" "$REF" 20 "$SW05" CEREF "$SUPER" "$SUPER" $SUPERC "20,1" 0
cell n6suph20_tok "$(base "$SUPER" $SUPERC $S1)" "$T4"  20 "$SW05" CET4  "$SUPER" "$SUPER" $SUPERC "20,1" 0

echo "##### C: morphology -- nominal-only control (m0) and ramp-to-0.7 (m7) #####"
cell n6m0_ref "$(base "$V2" dance2_subject4.npz $S1),$M0" "$REF" 1 "$SW05" CEREF "$V2" "$V2" dance2_subject4.npz "1,20" 1
cell n6m0_tok "$(base "$V2" dance2_subject4.npz $S1),$M0" "$T4"  1 "$SW05" CET4  "$V2" "$V2" dance2_subject4.npz "1,20" 1
cell n6m7_ref "$(base "$V2" dance2_subject4.npz $S1),$M7" "$REF" 1 "$SW05" CEREF "$V2" "$V2" dance2_subject4.npz "1,20" 1
cell n6m7_tok "$(base "$V2" dance2_subject4.npz $S1),$M7" "$T4"  1 "$SW05" CET4  "$V2" "$V2" dance2_subject4.npz "1,20" 1

echo "##### D: token at swing dose 25 (mc codec, pairs with n5sw25_ref) #####"
cell n6sw25_tok "$(base "$TOKMC" dance2_subject4.npz $S1)" "$T4" 1 "$(SW 2.5)" CET4 "$TOKMC" "$LAF" dance2_subject4.npz "1" 0

echo "##### E: zero-shot walk1 at HOLD 1 on existing checkpoints #####"
for s in 1 2 3; do for cs in 0 1; do ce "n5v2_tok_s$s" "-" "zsh1_$cs" "$cs" "$(CET4 1)" "$V2" "$V2" walk1_subject1.npz; done; done
for s in 1 2; do for cs in 0 1; do ce "n5sw015_ref_s$s" "-" "zsh1_$cs" "$cs" "$(CEREF 1)" "$V2" "$V2" walk1_subject1.npz; done; done

[ "$DRY" = 1 ] || touch "$ROOT/.w6a_submitted"
echo "##### WAVE 6a SUBMITTED #####"
squeue -u akalenik -h | wc -l
