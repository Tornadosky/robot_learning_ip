#!/usr/bin/env bash
# NIGHT4 (2026-08-31 morning, ~4h) -- attack what night3 left open.
#
# Diagnosis (see night3 aggregate): penetration/heading are SOLVED by the
# contact bundle + heading obs; the ONE open gait gap is airborne (policies
# 0.01-0.08 vs ref demand 0.06-0.14). The reward's explicit air-time term
# (foot_air_time_coeff, default 3.0, x0.25 gait floor, vs TRACK_COEFF=30) has
# been running at ~2.5% of the tracking term -- historically "i.e. off".
#
#   AIR   foot_air_time_coeff {15,45} on the bundle recipe, hold=1;
#         +FOOTZVEL 0.2 variant (stop penalizing lift velocity);
#         +GAITCOEFF 0.75 variant (lift all gait terms together).
#         {ref,tok} x 2 seeds each + render dumps on air15/air45.
#   RC    bundle-recipe rate curve completion: h2/h10/h20 x {ref,tok} x 2
#         (h1=n3gb33h1, h5=n3gb33 already exist) -> headline curve without
#         H1's 13-17mm underground defect.
#   T1V   first T1 token cell on Viper: H1+T1 (2 topologies, under the ROCm
#         wall), clips_3t_token sidecars (tokenizer_3t, self-consistent),
#         {ref,tok} x 2 seeds at hold=1. CE via CE_ROBOTS.
#
# 32 trains + 128 CEs + 4 dumps = 164 jobs.  DRY=1 to print only.
set -u
ROOT=/ptmp/akalenik/urma
cd "$ROOT"
DRY="${DRY:-0}"

TOKMC=$ROOT/clips/tokentest_mc
TOK3T=$ROOT/clips/clips_3t_token
LAF=$ROOT/clips/LAFAN1
XLA="--xla_gpu_enable_command_buffer="
CE2=$ROOT/crosseval_token2.sbatch

# The night4 base: practical recipe + contact bundle + gb33 FOOTH dose.
# Parametrized: clipdir, refhold handled via ea(); dose knobs via overrides.
base() {  # base <clipdir> <footzvel> <gaitcoeff>
  local b="STAGE=mmtrain,NR_ENVS=768,TOTAL_STEPS=19660800,SAVE_EVERY=1966080"
  b="$b,CLIP_FILE=dance2_subject4.npz,CLIP_DIR=$1"
  b="$b,REFVEL_OBS=False,REFROOT=True,REFROOT_FLOOR=True,REFBIAS=0.0"
  b="$b,TRACK_COEFF=30.0,TRACK_TEMP=0.05,FITVARIANT=False,ANCHOR=absolute"
  b="$b,GAITMODE=floor,GAITCOEFF=$3,QVEL_TEMP=10"
  b="$b,FOOTZVEL=$2,FOOTH_TEMP=0.05,FOOTH=0.3333"
  b="$b,FOOTSLIP=6.6667,GROUNDPEN=1000,POSTCONTACT=True"
  b="$b,CONTACT_TIMECONST=0.004"
  b="$b,HEADING_RATIO=0.20,HEADING_TEMP=2.0"
  b="$b,XLA_EXTRA=$XLA"
  echo "$b"
}

REF="LATENT_OBS=False,JLAT_ENC_DIM=0"
T4="LATENT_OBS=True,LATENT_DIM=32,LATENT_REPLACES=False,LATENT_SCOPE=per_joint,LATENT_DIVISOR=10.0,JLAT_ENC_DIM=4"
HOBS="--environment.command.tracking_clip_observe_root_heading=True"

sub() {  local name="$1" dep="$2" exp="$3"
  local d=(); [ -n "$dep" ] && [ "$dep" != "FAKE" ] && d=(--dependency="afterany:$dep")
  if [ "$DRY" = "1" ]; then echo "TRAIN $name dep=${dep:-none}" >&2; echo "  $exp" >&2; echo "FAKE"; return; fi
  sbatch --parsable -J "$name" ${d[@]+"${d[@]}"} --export=ALL,"EXP_NAME=$name,$exp" "$ROOT/viper_train.sbatch"
}
ce() {
  local arm="$1" jid="$2" sd="$3" cdir="$4" extra="$5"
  local d=(); [ -n "$jid" ] && [ "$jid" != "FAKE" ] && d=(--dependency="afterok:$jid")
  [ "$DRY" = "1" ] && { echo "  CE $arm s$sd $extra" >&2; return; }
  sbatch --parsable -J "ce_$arm" ${d[@]+"${d[@]}"} \
    --export=ALL,"EXP=$arm,CLIP_DIR=$cdir,CE_RAW_DIR=$RAWDIR,CE_CLIP=dance2_subject4.npz,CE_TAG=s$sd,CE_SEED=$sd,CE_REFBIAS=0.0,CE_ANCHOR=absolute,CE_FITVARIANT=False,CE_REFROOT=True,CE_REFROOT_FLOOR=True,CE_REFVEL_OBS=False,CE_CONTACT_TIMECONST=0.004,$extra" \
    "$CE2" > /dev/null
}
ce4() { for s in 0 1 2 3; do ce "$1" "$2" "$s" "$3" "$4"; done; }
cedump() {
  local arm="$1" jid="$2" cdir="$3" extra="$4"
  local d=(); [ -n "$jid" ] && [ "$jid" != "FAKE" ] && d=(--dependency="afterok:$jid")
  [ "$DRY" = "1" ] && { echo "  CEDUMP $arm" >&2; return; }
  sbatch --parsable -J "ced_$arm" ${d[@]+"${d[@]}"} \
    --export=ALL,"EXP=$arm,CLIP_DIR=$cdir,CE_RAW_DIR=$RAWDIR,CE_CLIP=dance2_subject4.npz,CE_TAG=dump,CE_SEED=0,CE_DUMP=1,CE_REFBIAS=0.0,CE_ANCHOR=absolute,CE_FITVARIANT=False,CE_REFROOT=True,CE_REFROOT_FLOOR=True,CE_REFVEL_OBS=False,CE_CONTACT_TIMECONST=0.004,$extra" \
    "$CE2" > /dev/null
}

ce_ref()  { echo "CE_LATENT=0,CE_JLAT_ENC_DIM=0,CE_REFHOLD=$1,CE_HEADING_OBS=True"; }
ce_t4()   { echo "CE_LATENT=1,CE_SCOPE=per_joint,CE_DIVISOR=10.0,CE_REPLACES=False,CE_JLAT_ENC_DIM=4,CE_REFHOLD=$1,CE_HEADING_OBS=True"; }

ea() {  # ea <refhold> <seed> [extra-colon-flags]
  local v="--environment.command.tracking_clip_reference_hold=$1"
  [ "$2" != "1" ] && v="$v:--environment.seed=$2"
  [ -n "${3:-}" ] && v="$v:$3"
  echo "EXTRA_ARGS=$v:$HOBS"
}

# seedrun/chain2: the WAVE-2 implementation (its jid capture worked; wave-1's
# gcell-style indirection is what broke the first dump jobs).
seedrun() {  # <cell> <base> <expv> <ce_extra> <refhold> <cdir> <seed> <dep> [eargs] [exports] -> jid
  local name="${1}_s${7}"
  local exports="$2,$3,$(ea "$5" "$7" "${9:-}")"
  [ -n "${10:-}" ] && exports="$exports,${10}"
  local J; J=$(sub "$name" "$8" "$exports")
  echo "$name -> $J" >&2
  ce4 "$name" "$J" "$6" "$4"
  echo "$J"
}
chain2() {  # <cell> <base> <expv> <ce_extra> <refhold> <cdir> [eargs] [exports] -> s1 jid
  local PREV="" s J J1=""
  for s in 1 2; do
    J=$(seedrun "$1" "$2" "$3" "$4" "$5" "$6" "$s" "$PREV" "${7:-}" "${8:-}")
    [ "$s" = "1" ] && J1="$J"
    PREV="$J"
  done
  echo "$J1"
}

# RAWDIR: what CEs score against. LAFAN1 for H1+G1 arms; the fixed-T1 design-B
# dir for the T1 pair (LAFAN1/BoosterT1 is the INFEASIBLE original -- scoring
# T1 against it re-creates the pre-fix defect). Subshells inherit it.
RAWDIR=$LAF

BSTD=$(base "$TOKMC" 1.0 0.25)
BZ02=$(base "$TOKMC" 0.2 0.25)
BG75=$(base "$TOKMC" 1.0 0.75)
AIR15="--environment.reward.foot_air_time_coeff=15.0"
AIR45="--environment.reward.foot_air_time_coeff=45.0"

echo "########## AIR -- dose the air-time term (bundle recipe, hold=1) ##########"
J=$(chain2 n4air15_ref "$BSTD" "$REF" "$(ce_ref 1)" 1 "$TOKMC" "$AIR15")
cedump n4air15_ref_s1 "$J" "$TOKMC" "$(ce_ref 1)"
J=$(chain2 n4air15_tok "$BSTD" "$T4" "$(ce_t4 1)" 1 "$TOKMC" "$AIR15")
cedump n4air15_tok_s1 "$J" "$TOKMC" "$(ce_t4 1)"
J=$(chain2 n4air45_ref "$BSTD" "$REF" "$(ce_ref 1)" 1 "$TOKMC" "$AIR45")
cedump n4air45_ref_s1 "$J" "$TOKMC" "$(ce_ref 1)"
J=$(chain2 n4air45_tok "$BSTD" "$T4" "$(ce_t4 1)" 1 "$TOKMC" "$AIR45")
cedump n4air45_tok_s1 "$J" "$TOKMC" "$(ce_t4 1)"
chain2 n4air15z_ref "$BZ02" "$REF" "$(ce_ref 1)" 1 "$TOKMC" "$AIR15" > /dev/null
chain2 n4air15z_tok "$BZ02" "$T4"  "$(ce_t4 1)"  1 "$TOKMC" "$AIR15" > /dev/null
chain2 n4gait75_ref "$BG75" "$REF" "$(ce_ref 1)" 1 "$TOKMC" > /dev/null
chain2 n4gait75_tok "$BG75" "$T4"  "$(ce_t4 1)"  1 "$TOKMC" > /dev/null

echo "########## RC -- bundle-recipe rate curve h2/h10/h20 ##########"
for h in 2 10 20; do
  chain2 "n4rc${h}_ref" "$BSTD" "$REF" "$(ce_ref "$h")" "$h" "$TOKMC" > /dev/null
  chain2 "n4rc${h}_tok" "$BSTD" "$T4"  "$(ce_t4 "$h")"  "$h" "$TOKMC" > /dev/null
done

echo "########## T1V -- H1+T1 token pair on Viper (bundle recipe, hold=1) ##########"
BT1=$(base "$TOK3T" 1.0 0.25)
RAWDIR=$TOK3T
RL="ROBOTS_LIST=unitree_h1:booster_t1"
CET1="CE_ROBOTS=unitree_h1:booster_t1"
chain2 n4t1_ref "$BT1" "$REF" "$(ce_ref 1),$CET1" 1 "$TOK3T" "" "$RL" > /dev/null
chain2 n4t1_tok "$BT1" "$T4"  "$(ce_t4 1),$CET1"  1 "$TOK3T" "" "$RL" > /dev/null

echo "########## NIGHT4 SUBMITTED ##########"
squeue -u akalenik -h | wc -l
