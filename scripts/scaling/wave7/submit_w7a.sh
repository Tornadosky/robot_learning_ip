#!/usr/bin/env bash
# WAVE 7a (2026-09-02 night) -- the many-motion H1+G1 matrix on Viper.
# Clip set: clips_m20 = 20 LAFAN1 motions concatenated (super20.npz) + 7
# held-out motions as separate clips (zero-shot), sidecars from tokenizer_m20
# (H1+G1, 7 motions held out of the codec too).
#
#   arms   REF (reference only) | SPLIT (precomputed code, own projection) |
#          COT (SONIC co-training, encoder init from tokenizer_m20)
#   B1     hold 1, 1x budget, seeds 1-2        (6)
#   B2     hold 1, 2x budget, seeds 1-2        (6)
#   B4     hold 1, 4x budget, REF + COT, seed 1 (2)
#   H20    trained at hold 20, 1x, seeds 1-2   (6)
#   M7     ramp-to-0.7 morphology, 1x, seed 1  (3)
#   CEs per train: super20 h1 x2, h20 x2, 7 held-out motions h1 x1  (11)
# 23 trains + ~253 CEs. DRY=1 to print. Guard: .w7a_submitted
set -u
ROOT=/ptmp/akalenik/urma
cd "$ROOT"
DRY="${DRY:-0}"
[ -f "$ROOT/.w7a_submitted" ] && [ "$DRY" != "1" ] && { echo "already submitted"; exit 0; }
M20=$ROOT/clips/clips_m20
TOKP=$ROOT/tokenizer_m20/params.msgpack
XLA="--xla_gpu_enable_command_buffer="
CE3=$ROOT/crosseval_token3.sbatch
HELD="dance1_subject3 walk1_subject5 walk3_subject5 run2_subject4 sprint1_subject4 jumps1_subject5 fight1_subject5"
for f in "$M20/UnitreeH1/super20_win.npz" "$M20/UnitreeG1/super20_zq.npz" "$M20/UnitreeG1/fight1_subject5_win.npz" "$TOKP"; do
  [ -f "$f" ] || { echo "ABORT: missing $f"; exit 1; }
done

base() {  # <total_steps>
  local b="STAGE=mmtrain,NR_ENVS=768,TOTAL_STEPS=$1,SAVE_EVERY=1966080"
  b="$b,CLIP_FILE=super20.npz,CLIP_DIR=$M20"
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
SPLIT="LATENT_OBS=True,LATENT_DIM=32,LATENT_REPLACES=False,LATENT_SCOPE=per_joint,LATENT_DIVISOR=10.0,JLAT_ENC_DIM=4,JLAT_CH=-1"
COT="LATENT_OBS=True,LATENT_DIM=44,LATENT_REPLACES=False,LATENT_SCOPE=per_joint,LATENT_DIVISOR=1.0,SIDECAR=_win,JLAT_ENC_DIM=4,COTRAIN_ROWS=11,COTRAIN_CH=4,COTRAIN_RECON=1.0,COTRAIN_INIT=$TOKP"
M7="MORPH_MODE=schedule,MORPH_COEFF=0.7,MORPH_START=0.2,MORPH_RAMP=15000000"
HOBS="--environment.command.tracking_clip_observe_root_heading=True"
SW05="--environment.reward.deepmimic_swing_match_weight_ratio=0.5"
CEREF() { echo "CE_LATENT=0,CE_JLAT_ENC_DIM=0,CE_HEADING_OBS=True,CE_REFHOLD=$1"; }
CESPL() { echo "CE_LATENT=1,CE_SCOPE=per_joint,CE_DIVISOR=10.0,CE_REPLACES=False,CE_JLAT_ENC_DIM=4,CE_HEADING_OBS=True,CE_REFHOLD=$1"; }
CECOT() { echo "CE_LATENT=1,CE_LATENT_DIM=44,CE_SIDECAR=_win,CE_SCOPE=per_joint,CE_DIVISOR=1.0,CE_REPLACES=False,CE_JLAT_ENC_DIM=4,CE_HEADING_OBS=True,CE_REFHOLD=$1"; }
ea() { local v="--environment.command.tracking_clip_reference_hold=$1:$SW05"; [ "$2" != "1" ] && v="$v:--environment.seed=$2"; echo "EXTRA_ARGS=$v:$HOBS"; }
sub() {
  local d=(); [ -n "$2" ] && [ "$2" != "-" ] && [ "$2" != "FAKE" ] && d=(--dependency="afterany:$2")
  if [ "$DRY" = 1 ]; then echo "TRAIN $1 dep=$2" >&2; echo "   $3" >&2; echo FAKE; return; fi
  sbatch --parsable -J "$1" ${d[@]+"${d[@]}"} --export=ALL,"EXP_NAME=$1,$3" "$ROOT/viper_train.sbatch"
}
ce() {  # <exp> <dep|-> <tag> <seed> <extra> <clip>
  local d=(); [ -n "$2" ] && [ "$2" != "-" ] && [ "$2" != "FAKE" ] && d=(--dependency="afterok:$2")
  [ "$DRY" = 1 ] && { echo "  CE $1 $3 clip=$6" >&2; return; }
  sbatch --parsable -J "ce_$1" ${d[@]+"${d[@]}"} \
    --export=ALL,"EXP=$1,CLIP_DIR=$M20,CE_RAW_DIR=$M20,CE_CLIP=$6,CE_TAG=$3,CE_SEED=$4,CE_REFBIAS=0.0,CE_ANCHOR=absolute,CE_FITVARIANT=False,CE_REFROOT=True,CE_REFROOT_FLOOR=True,CE_REFVEL_OBS=False,CE_CONTACT_TIMECONST=0.004,CE_NRENVS=64,$5" \
    "$CE3" > /dev/null
}
# cell <name> <steps> <expv> <trainhold> <cefun> <seeds> [extra]
cell() {
  local name=$1 steps=$2 expv=$3 rh=$4 cef=$5 seeds=$6 extra=${7:-}
  local PREV="-" s J
  for s in $seeds; do
    J=$(sub "${name}_s$s" "$PREV" "$(base $steps),$expv${extra:+,$extra},$(ea "$rh" "$s")")
    echo "${name}_s$s -> $J" >&2
    for cs in 0 1; do ce "${name}_s$s" "$J" "h1_s$cs" "$cs" "$($cef 1)" super20.npz; done
    for cs in 0 1; do ce "${name}_s$s" "$J" "h20_s$cs" "$cs" "$($cef 20)" super20.npz; done
    for m in $HELD; do ce "${name}_s$s" "$J" "zs_${m}" 0 "$($cef 1)" "$m.npz"; done
    PREV="$J"
  done
}
echo "##### B1 #####"
cell m20ref   $S1 "$REF"   1 CEREF "1 2"
cell m20split $S1 "$SPLIT" 1 CESPL "1 2"
cell m20cot   $S1 "$COT"   1 CECOT "1 2"
echo "##### B2 #####"
cell m20ref2x   $S2 "$REF"   1 CEREF "1 2"
cell m20split2x $S2 "$SPLIT" 1 CESPL "1 2"
cell m20cot2x   $S2 "$COT"   1 CECOT "1 2"
echo "##### B4 #####"
cell m20ref4x $S4 "$REF" 1 CEREF "1"
cell m20cot4x $S4 "$COT" 1 CECOT "1"
echo "##### H20 #####"
cell m20refh20   $S1 "$REF"   20 CEREF "1 2"
cell m20splith20 $S1 "$SPLIT" 20 CESPL "1 2"
cell m20coth20   $S1 "$COT"   20 CECOT "1 2"
echo "##### M7 #####"
cell m20refm7   $S1 "$REF"   1 CEREF "1" "$M7"
cell m20splitm7 $S1 "$SPLIT" 1 CESPL "1" "$M7"
cell m20cotm7   $S1 "$COT"   1 CECOT "1" "$M7"
[ "$DRY" = 1 ] || touch "$ROOT/.w7a_submitted"
echo "##### WAVE 7a SUBMITTED #####"
squeue -u akalenik -h | wc -l
