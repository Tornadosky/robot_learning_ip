#!/usr/bin/env bash
# NIGHT3 WAVE 4 (A3) -- future-vs-past window ablation, token arms only.
# Sidecars are TIME-SHIFTS of the mc emission (make_a3_shifted_sidecars.py):
#   tokentest_mc       future-only [t..t+10]   (already measured: W1/W2/V1MC)
#   tokentest_mc_cent  centered    [t-5..t+5]
#   tokentest_mc_past  past-only   [t-10..t]
# If the token's value is PREDICTION, past-only should collapse toward the ref
# arm at stale holds; if it is SMOOTHING, the three should tie.
# {cent,past} x hold {1,5} x 2 seeds = 8 trains + 32 CEs.
#   DRY=1  print, submit nothing
set -u
ROOT=/ptmp/akalenik/urma
cd "$ROOT"
DRY="${DRY:-0}"

LAF=$ROOT/clips/LAFAN1
XLA="--xla_gpu_enable_command_buffer="
CE2=$ROOT/crosseval_token2.sbatch

base() {  # base <clipdir>
  local b="STAGE=mmtrain,NR_ENVS=768,TOTAL_STEPS=19660800,SAVE_EVERY=1966080"
  b="$b,CLIP_FILE=dance2_subject4.npz,CLIP_DIR=$1"
  b="$b,REFVEL_OBS=False,REFROOT=True,REFROOT_FLOOR=True,REFBIAS=0.0"
  b="$b,TRACK_COEFF=30.0,TRACK_TEMP=0.05,FITVARIANT=False,ANCHOR=absolute"
  b="$b,GAITMODE=floor,GAITCOEFF=0.25,QVEL_TEMP=10"
  b="$b,FOOTZVEL=1.0,FOOTH_TEMP=0.01,FOOTH=0.0"
  b="$b,FOOTSLIP=6.6667,GROUNDPEN=333.33,POSTCONTACT=True"
  b="$b,CONTACT_TIMECONST=0.0"
  b="$b,HEADING_RATIO=0.20,HEADING_TEMP=2.0"
  b="$b,XLA_EXTRA=$XLA"
  echo "$b"
}

T4="LATENT_OBS=True,LATENT_DIM=32,LATENT_REPLACES=False,LATENT_SCOPE=per_joint,LATENT_DIVISOR=10.0,JLAT_ENC_DIM=4"
HOBS="--environment.command.tracking_clip_observe_root_heading=True"

sub() {  local name="$1" dep="$2" exp="$3"
  local d=(); [ -n "$dep" ] && [ "$dep" != "FAKE" ] && d=(--dependency="afterany:$dep")
  if [ "$DRY" = "1" ]; then echo "TRAIN $name dep=${dep:-none}" >&2; echo "  $exp" >&2; echo "FAKE"; return; fi
  sbatch --parsable -J "$name" ${d[@]+"${d[@]}"} --export=ALL,"EXP_NAME=$name,$exp" "$ROOT/viper_train.sbatch"
}
ce() {
  local arm="$1" jid="$2" sd="$3" cdir="$4" hold="$5"
  local d=(); [ -n "$jid" ] && [ "$jid" != "FAKE" ] && d=(--dependency="afterok:$jid")
  [ "$DRY" = "1" ] && { echo "  CE $arm s$sd hold=$hold dir=$cdir" >&2; return; }
  sbatch --parsable -J "ce_$arm" ${d[@]+"${d[@]}"} \
    --export=ALL,"EXP=$arm,CLIP_DIR=$cdir,CE_RAW_DIR=$LAF,CE_CLIP=dance2_subject4.npz,CE_TAG=s$sd,CE_SEED=$sd,CE_REFBIAS=0.0,CE_ANCHOR=absolute,CE_FITVARIANT=False,CE_REFROOT=True,CE_REFROOT_FLOOR=True,CE_REFVEL_OBS=False,CE_LATENT=1,CE_SCOPE=per_joint,CE_DIVISOR=10.0,CE_REPLACES=False,CE_JLAT_ENC_DIM=4,CE_REFHOLD=$hold,CE_HEADING_OBS=True" \
    "$CE2" > /dev/null
}
ea() {  local v="--environment.command.tracking_clip_reference_hold=$1"
  [ "$2" != "1" ] && v="$v:--environment.seed=$2"
  echo "EXTRA_ARGS=$v:$HOBS"
}

arm() {  # arm <cell> <clipdir> <hold>
  local cell="$1" cdir="$2" hold="$3" PREV="" s J
  local b; b=$(base "$cdir")
  for s in 1 2; do
    local name="${cell}_s${s}"
    J=$(sub "$name" "$PREV" "$b,$T4,$(ea "$hold" "$s")")
    echo "$name -> $J"
    for cs in 0 1 2 3; do ce "$name" "$J" "$cs" "$cdir" "$hold"; done
    PREV="$J"
  done
}

CENT=$ROOT/clips/tokentest_mc_cent
PAST=$ROOT/clips/tokentest_mc_past
[ "$DRY" = "1" ] || { [ -d "$CENT/UnitreeH1" ] && [ -d "$PAST/UnitreeH1" ] || { echo "ABORT: shifted clip dirs missing"; exit 1; }; }

echo "########## A3 window ablation ##########"
arm n3a3cent_h1 "$CENT" 1
arm n3a3cent_h5 "$CENT" 5
arm n3a3past_h1 "$PAST" 1
arm n3a3past_h5 "$PAST" 5
echo "########## WAVE 4 (A3) SUBMITTED ##########"
squeue -u akalenik -h | wc -l
