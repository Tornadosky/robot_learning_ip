#!/usr/bin/env bash
# W2 -- three questions W1 opened, all on the PRACTICAL recipe (B4 + rx_p3f0
# dose + heading), submitted 2026-08-30 ~13:15 after W1/W3 drained clean.
#
#   1. WHERE IS THE CROSSOVER? W1: token +2.3/+4.4 % at hold=1 but
#      -21.5/-22.1 % at hold=5 on this base. hold=2 pair locates the boundary
#      of "condition on the token when...".
#   2. HOW FAR DOES THE FLATNESS EXTEND? V1's token arm was ~flat to hold=10
#      (B4). hold=20 pair on the modern base: does ref keep collapsing while
#      the token holds, or does the token's 250 ms window finally run out
#      (10 steps ~ the window; 20 exceeds it -- registered expectation: the
#      token arm DEGRADES at 20, the open question is whether it still wins)?
#   3. CAN THE TOKEN REPLACE THE REFERENCE? LATENT_REPLACES=True at hold=1:
#      reference leaves the OBSERVATION entirely (reward unchanged). If this
#      tracks near the ref baseline, the token is a sufficient interface;
#      if it fails badly, the reference obs carries unique alignment info.
#
# 10 trains x 2-seed chains + 4 CE seeds each = 50 jobs.
#   DRY=1  print, submit nothing
set -u
ROOT=/ptmp/akalenik/urma
cd "$ROOT"
DRY="${DRY:-0}"

TOKMC=$ROOT/clips/tokentest_mc
LAF=$ROOT/clips/LAFAN1
XLA="--xla_gpu_enable_command_buffer="

# BIT-IDENTICAL to submit_w1.sh's BM (the practical recipe).
BM="STAGE=mmtrain,NR_ENVS=768,TOTAL_STEPS=19660800,SAVE_EVERY=1966080"
BM="$BM,CLIP_FILE=dance2_subject4.npz,CLIP_DIR=$TOKMC"
BM="$BM,REFVEL_OBS=False,REFROOT=True,REFROOT_FLOOR=True,REFBIAS=0.0"
BM="$BM,TRACK_COEFF=30.0,TRACK_TEMP=0.05,FITVARIANT=False,ANCHOR=absolute"
BM="$BM,GAITMODE=floor,GAITCOEFF=0.25,QVEL_TEMP=10"
BM="$BM,FOOTZVEL=1.0,FOOTH_TEMP=0.01,FOOTH=0.0"
BM="$BM,FOOTSLIP=6.6667,GROUNDPEN=333.33,POSTCONTACT=True"
BM="$BM,HEADING_RATIO=0.20,HEADING_TEMP=2.0"
BM="$BM,XLA_EXTRA=$XLA"

REF="LATENT_OBS=False,JLAT_ENC_DIM=0"
T4="LATENT_OBS=True,LATENT_DIM=32,LATENT_REPLACES=False,LATENT_SCOPE=per_joint,LATENT_DIVISOR=10.0,JLAT_ENC_DIM=4"
TREP="LATENT_OBS=True,LATENT_DIM=32,LATENT_REPLACES=True,LATENT_SCOPE=per_joint,LATENT_DIVISOR=10.0,JLAT_ENC_DIM=4"
HOBS="--environment.command.tracking_clip_observe_root_heading=True"

sub() {  local name="$1" dep="$2" exp="$3"
  local d=(); [ -n "$dep" ] && [ "$dep" != "FAKE" ] && d=(--dependency="afterany:$dep")
  if [ "$DRY" = "1" ]; then echo "TRAIN $name dep=${dep:-none}" >&2; echo "  $exp" >&2; echo "FAKE"; return; fi
  sbatch --parsable -J "$name" ${d[@]+"${d[@]}"} --export=ALL,"EXP_NAME=$name,$exp" "$ROOT/viper_train.sbatch"
}
ce() {   local arm="$1" jid="$2" sd="$3" extra="$4"
  local d=(); [ -n "$jid" ] && [ "$jid" != "FAKE" ] && d=(--dependency="afterok:$jid")
  [ "$DRY" = "1" ] && { echo "  CE $arm s$sd $extra" >&2; return; }
  sbatch --parsable -J "ce_$arm" ${d[@]+"${d[@]}"} \
    --export=ALL,"EXP=$arm,CLIP_DIR=$TOKMC,CE_RAW_DIR=$LAF,CE_TAG=s$sd,CE_SEED=$sd,CE_REFBIAS=0.0,CE_ANCHOR=absolute,CE_FITVARIANT=False,CE_REFROOT=True,CE_REFROOT_FLOOR=True,CE_REFVEL_OBS=False,$extra" \
    "$ROOT/crosseval_token.sbatch" > /dev/null
}
ce4() { for s in 0 1 2 3; do ce "$1" "$2" "$s" "$3"; done; }

ce_ref()  { echo "CE_LATENT=0,CE_JLAT_ENC_DIM=0,CE_REFHOLD=$1,CE_HEADING_OBS=True"; }
ce_t4()   { echo "CE_LATENT=1,CE_SCOPE=per_joint,CE_DIVISOR=10.0,CE_REPLACES=False,CE_JLAT_ENC_DIM=4,CE_REFHOLD=$1,CE_HEADING_OBS=True"; }
ce_trep() { echo "CE_LATENT=1,CE_SCOPE=per_joint,CE_DIVISOR=10.0,CE_REPLACES=True,CE_JLAT_ENC_DIM=4,CE_REFHOLD=$1,CE_HEADING_OBS=True"; }

ea() {  local v="--environment.command.tracking_clip_reference_hold=$1"
  [ "$2" = "2" ] && v="$v:--environment.seed=2"
  echo "EXTRA_ARGS=$v:$HOBS"
}

chain() {  # chain <basename> <EXPVAR-name> <cefn> <hold>
  local base="$1" rt="$2" cefn="$3" hold="$4" PREV="" s EXPV
  case "$rt" in REF) EXPV="$REF";; T4) EXPV="$T4";; TREP) EXPV="$TREP";; esac
  for s in 1 2; do
    local name="${base}_s${s}"
    local J=$(sub "$name" "$PREV" "$BM,$EXPV,$(ea "$hold" "$s")")
    echo "$name -> $J"
    ce4 "$name" "$J" "$($cefn "$hold")"
    PREV="$J"
  done
}

echo "########## W2 -- crossover / horizon / token-only ##########"
chain w2h2_ref   REF  ce_ref  2
chain w2h2_tok   T4   ce_t4   2
chain w2h20_ref  REF  ce_ref  20
chain w2h20_tok  T4   ce_t4   20
chain w2rep_h1   TREP ce_trep 1
echo "########## SUBMITTED ##########"
