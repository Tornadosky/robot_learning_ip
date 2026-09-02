#!/usr/bin/env bash
# WAVE 6b (2026-09-02) -- the NEW-CODE arms (needs deploy_wave6_viper.sh first).
# Base = n5v2_tok / n5sw05_ref recipe (bundle + swing 0.5 + heading, v2 clips,
# Viper default morphology schedule), hold 1, 19.66M, 2 seeds unless noted.
#
#   SPLIT   n6split_tok    token through a REAL separate Dense(4) (JLAT_CH=-1);
#                          every earlier "tk4" arm was the shared routing.
#   AUX     n6aux_tok      split + aux next-token head coeff 0.5 horizon 5
#           n6aux1_tok     coeff 1.0 horizon 5
#           n6auxd_tok     detach-trunk probe (1 seed): is next-token info already there?
#   LEGS    n6legw3_{ref,tok}   leg joints weighted x3 in the tracking kernel
#           n6legw5_{ref,tok}   x5
#   COTRAIN n6cot_tok      SONIC co-training, encoder init from tokenizer_3t_v2
#           n6cotsc_tok    from scratch
#           n6cotfr_tok    frozen encoder (1 seed; = online frozen tokenizer control)
#           n6cotaux_tok   co-training + aux head
# CEs: dance4 h1 x4, h20 x2; walk1 zero-shot h1 x2 for token arms.
# 22 trains + ~150 CEs.  DRY=1 to print.  Guard: .w6b_submitted
set -u
ROOT=/ptmp/akalenik/urma
cd "$ROOT"
DRY="${DRY:-0}"
[ -f "$ROOT/.w6b_submitted" ] && [ "$DRY" != "1" ] && { echo "already submitted"; exit 0; }
V2=$ROOT/clips/tokentest_v2
TOKP=$ROOT/tokenizer_3t_v2/params.msgpack
XLA="--xla_gpu_enable_command_buffer="
CE3=$ROOT/crosseval_token3.sbatch
for f in "$V2/UnitreeH1/dance2_subject4_win.npz" "$V2/UnitreeG1/walk1_subject1_win.npz" "$TOKP" "$ROOT/loco_mjx/loco_mjx/algorithms/urma2/mjx/fsq_cotrain.py"; do
  [ -f "$f" ] || { echo "ABORT: missing $f (run deploy_wave6_viper.sh)"; exit 1; }
done
grep -q "CE_SIDECAR" "$CE3" || { echo "ABORT: crosseval_token3.sbatch lacks CE_SIDECAR"; exit 1; }
grep -q "COTRAIN_ROWS" "$ROOT/viper_train.sbatch" || { echo "ABORT: viper_train.sbatch lacks COTRAIN_ROWS"; exit 1; }

B="STAGE=mmtrain,NR_ENVS=768,TOTAL_STEPS=19660800,SAVE_EVERY=1966080"
B="$B,CLIP_FILE=dance2_subject4.npz,CLIP_DIR=$V2"
B="$B,REFVEL_OBS=False,REFROOT=True,REFROOT_FLOOR=True,REFBIAS=0.0"
B="$B,TRACK_COEFF=30.0,TRACK_TEMP=0.05,FITVARIANT=False,ANCHOR=absolute"
B="$B,GAITMODE=floor,GAITCOEFF=0.25,QVEL_TEMP=10"
B="$B,FOOTZVEL=1.0,FOOTH_TEMP=0.05,FOOTH=0.3333"
B="$B,FOOTSLIP=6.6667,GROUNDPEN=1000,POSTCONTACT=True"
B="$B,CONTACT_TIMECONST=0.004,HEADING_RATIO=0.20,HEADING_TEMP=2.0"
B="$B,XLA_EXTRA=$XLA"
REF="LATENT_OBS=False,JLAT_ENC_DIM=0"
TOK="LATENT_OBS=True,LATENT_DIM=32,LATENT_REPLACES=False,LATENT_SCOPE=per_joint,LATENT_DIVISOR=10.0,JLAT_ENC_DIM=4"
SPLIT="$TOK,JLAT_CH=-1"
COT="LATENT_OBS=True,LATENT_DIM=44,LATENT_REPLACES=False,LATENT_SCOPE=per_joint,LATENT_DIVISOR=1.0,SIDECAR=_win,JLAT_ENC_DIM=4,COTRAIN_ROWS=11,COTRAIN_CH=4,COTRAIN_RECON=1.0"
HOBS="--environment.command.tracking_clip_observe_root_heading=True"
SW05="--environment.reward.deepmimic_swing_match_weight_ratio=0.5"
CEREF() { echo "CE_LATENT=0,CE_JLAT_ENC_DIM=0,CE_HEADING_OBS=True,CE_REFHOLD=$1"; }
CETOK() { echo "CE_LATENT=1,CE_SCOPE=per_joint,CE_DIVISOR=10.0,CE_REPLACES=False,CE_JLAT_ENC_DIM=4,CE_HEADING_OBS=True,CE_REFHOLD=$1"; }
CECOT() { echo "CE_LATENT=1,CE_LATENT_DIM=44,CE_SIDECAR=_win,CE_SCOPE=per_joint,CE_DIVISOR=1.0,CE_REPLACES=False,CE_JLAT_ENC_DIM=4,CE_HEADING_OBS=True,CE_REFHOLD=$1"; }

ea() { local v="--environment.command.tracking_clip_reference_hold=1:$SW05"; [ "$1" != "1" ] && v="$v:--environment.seed=$1"; echo "EXTRA_ARGS=$v:$HOBS"; }
sub() {  # <name> <dep|-> <export>
  local d=(); [ -n "$2" ] && [ "$2" != "-" ] && [ "$2" != "FAKE" ] && d=(--dependency="afterany:$2")
  if [ "$DRY" = 1 ]; then echo "TRAIN $1 dep=$2" >&2; echo "   $3" >&2; echo FAKE; return; fi
  sbatch --parsable -J "$1" ${d[@]+"${d[@]}"} --export=ALL,"EXP_NAME=$1,$3" "$ROOT/viper_train.sbatch"
}
ce() {  # <exp> <dep|-> <tag> <seed> <extra> <clip>
  local d=(); [ -n "$2" ] && [ "$2" != "-" ] && [ "$2" != "FAKE" ] && d=(--dependency="afterok:$2")
  [ "$DRY" = 1 ] && { echo "  CE $1 $3 clip=$6 $5" >&2; return; }
  sbatch --parsable -J "ce_$1" ${d[@]+"${d[@]}"} \
    --export=ALL,"EXP=$1,CLIP_DIR=$V2,CE_RAW_DIR=$V2,CE_CLIP=$6,CE_TAG=$3,CE_SEED=$4,CE_REFBIAS=0.0,CE_ANCHOR=absolute,CE_FITVARIANT=False,CE_REFROOT=True,CE_REFROOT_FLOOR=True,CE_REFVEL_OBS=False,CE_CONTACT_TIMECONST=0.004,CE_NRENVS=64,$5" \
    "$CE3" > /dev/null
}
# cell <name> <expv> <cefun> <nseeds> <zs 0|1> [extra export]
cell() {
  local name=$1 expv=$2 cef=$3 ns=$4 zs=$5 extra=${6:-}
  local PREV="-" s J
  for ((s=1; s<=ns; s++)); do
    J=$(sub "${name}_s$s" "$PREV" "$B,$expv${extra:+,$extra},$(ea "$s")")
    echo "${name}_s$s -> $J" >&2
    for cs in 0 1 2 3; do ce "${name}_s$s" "$J" "h1_s$cs" "$cs" "$($cef 1)" dance2_subject4.npz; done
    for cs in 0 1; do ce "${name}_s$s" "$J" "h20_s$cs" "$cs" "$($cef 20)" dance2_subject4.npz; done
    [ "$zs" = 1 ] && for cs in 0 1; do ce "${name}_s$s" "$J" "zsh1_$cs" "$cs" "$($cef 1)" walk1_subject1.npz; done
    PREV="$J"
  done
}

echo "##### SPLIT #####"
cell n6split_tok "$SPLIT" CETOK 2 1
echo "##### AUX #####"
cell n6aux_tok  "$SPLIT,AUX_COEFF=0.5,AUX_HORIZON=5" CETOK 2 1
cell n6aux1_tok "$SPLIT,AUX_COEFF=1.0,AUX_HORIZON=5" CETOK 2 0
cell n6auxd_tok "$SPLIT,AUX_COEFF=0.5,AUX_HORIZON=5,AUX_DETACH=True" CETOK 1 0
echo "##### LEGS #####"
cell n6legw3_ref "$REF,LEGW=3.0" CEREF 2 0
cell n6legw3_tok "$TOK,LEGW=3.0" CETOK 2 0
cell n6legw5_ref "$REF,LEGW=5.0" CEREF 2 0
cell n6legw5_tok "$TOK,LEGW=5.0" CETOK 2 0
echo "##### COTRAIN #####"
cell n6cot_tok    "$COT,COTRAIN_INIT=$TOKP" CECOT 2 1
cell n6cotsc_tok  "$COT" CECOT 2 1
cell n6cotfr_tok  "$COT,COTRAIN_INIT=$TOKP,COTRAIN_FREEZE=True" CECOT 1 0
cell n6cotaux_tok "$COT,COTRAIN_INIT=$TOKP,AUX_COEFF=0.5,AUX_HORIZON=5" CECOT 2 0

[ "$DRY" = 1 ] || touch "$ROOT/.w6b_submitted"
echo "##### WAVE 6b SUBMITTED #####"
squeue -u akalenik -h | wc -l
