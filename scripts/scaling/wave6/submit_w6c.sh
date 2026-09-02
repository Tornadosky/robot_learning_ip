#!/usr/bin/env bash
# WAVE 6c (2026-09-02 afternoon) -- follow-ups on the wave-6b headline:
# the co-trained encoder (n6cot_tok, v2 init) is the best hold-1 policy
# (G1 0.149 / H1 0.179 vs ref 0.163 / 0.195, both seeds), with lower LEG error.
#
#   SEEDS   n6cot_tok s3,s4; n6split_tok s3; n6aux_tok s3
#   H20-3WAY trained at hold 20 on v2/swing-0.5: n6h20_ref, n6h20_tok (legacy),
#            n6coth20_tok -- does co-training also help staleness?
#   RECON   n6cot01_tok  recon coeff 0.1 (is the recon term load-bearing?)
#   REPL    n6cotrep_tok co-training WITHOUT the explicit reference channel
#            (reference-free tracking through the online encoder)
#   MORPH   n6cotm0_tok (nominal only), n6cotm7_tok (ramp to 0.7 by 15M)
#   LEGS    n6cotlegw5_tok  co-training + leg kernel x5
#   MULTI   n6cotsup_tok    co-training on the 5-dance super-clip (2x budget,
#            pairs with n6sup2x_{ref,tok})
# CEs: dance4 h1 x4 + h20 x2 (+ walk1 zero-shot h1 x2 for cot arms).
# 21 trains + ~130 CEs.  DRY=1 to print.  Guard: .w6c_submitted
set -u
ROOT=/ptmp/akalenik/urma
cd "$ROOT"
DRY="${DRY:-0}"
[ -f "$ROOT/.w6c_submitted" ] && [ "$DRY" != "1" ] && { echo "already submitted"; exit 0; }
V2=$ROOT/clips/tokentest_v2
SUPER=$ROOT/clips/clips_super
SUPERC=super5dance.npz
TOKP=$ROOT/tokenizer_3t_v2/params.msgpack
XLA="--xla_gpu_enable_command_buffer="
CE3=$ROOT/crosseval_token3.sbatch
for f in "$V2/UnitreeH1/dance2_subject4_win.npz" "$SUPER/UnitreeG1/super5dance_win.npz" "$SUPER/UnitreeH1/super5dance_win.npz" "$TOKP"; do
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
S1=19660800; S2=39321600
B=$(base "$V2" dance2_subject4.npz $S1)
BSUP=$(base "$SUPER" $SUPERC $S2)
REF="LATENT_OBS=False,JLAT_ENC_DIM=0"
TOK="LATENT_OBS=True,LATENT_DIM=32,LATENT_REPLACES=False,LATENT_SCOPE=per_joint,LATENT_DIVISOR=10.0,JLAT_ENC_DIM=4"
SPLIT="$TOK,JLAT_CH=-1"
COT="LATENT_OBS=True,LATENT_DIM=44,LATENT_REPLACES=False,LATENT_SCOPE=per_joint,LATENT_DIVISOR=1.0,SIDECAR=_win,JLAT_ENC_DIM=4,COTRAIN_ROWS=11,COTRAIN_CH=4,COTRAIN_RECON=1.0,COTRAIN_INIT=$TOKP"
COTREP="LATENT_OBS=True,LATENT_DIM=44,LATENT_REPLACES=True,LATENT_SCOPE=per_joint,LATENT_DIVISOR=1.0,SIDECAR=_win,JLAT_ENC_DIM=4,COTRAIN_ROWS=11,COTRAIN_CH=4,COTRAIN_RECON=1.0,COTRAIN_INIT=$TOKP"
M0="MORPH_MODE=fixed,MORPH_COEFF=0.0"
M7="MORPH_MODE=schedule,MORPH_COEFF=0.7,MORPH_START=0.2,MORPH_RAMP=15000000"
HOBS="--environment.command.tracking_clip_observe_root_heading=True"
SW05="--environment.reward.deepmimic_swing_match_weight_ratio=0.5"
CEREF() { echo "CE_LATENT=0,CE_JLAT_ENC_DIM=0,CE_HEADING_OBS=True,CE_REFHOLD=$1"; }
CETOK() { echo "CE_LATENT=1,CE_SCOPE=per_joint,CE_DIVISOR=10.0,CE_REPLACES=False,CE_JLAT_ENC_DIM=4,CE_HEADING_OBS=True,CE_REFHOLD=$1"; }
CECOT() { echo "CE_LATENT=1,CE_LATENT_DIM=44,CE_SIDECAR=_win,CE_SCOPE=per_joint,CE_DIVISOR=1.0,CE_REPLACES=False,CE_JLAT_ENC_DIM=4,CE_HEADING_OBS=True,CE_REFHOLD=$1"; }
CECOTR() { echo "CE_LATENT=1,CE_LATENT_DIM=44,CE_SIDECAR=_win,CE_SCOPE=per_joint,CE_DIVISOR=1.0,CE_REPLACES=True,CE_JLAT_ENC_DIM=4,CE_HEADING_OBS=True,CE_REFHOLD=$1"; }

ea() { local v="--environment.command.tracking_clip_reference_hold=$1:$SW05"; [ "$2" != "1" ] && v="$v:--environment.seed=$2"; echo "EXTRA_ARGS=$v:$HOBS"; }
sub() {
  local d=(); [ -n "$2" ] && [ "$2" != "-" ] && [ "$2" != "FAKE" ] && d=(--dependency="afterany:$2")
  if [ "$DRY" = 1 ]; then echo "TRAIN $1 dep=$2" >&2; echo "   $3" >&2; echo FAKE; return; fi
  sbatch --parsable -J "$1" ${d[@]+"${d[@]}"} --export=ALL,"EXP_NAME=$1,$3" "$ROOT/viper_train.sbatch"
}
ce() {  # <exp> <dep|-> <tag> <seed> <extra> <clipdir> <clip>
  local d=(); [ -n "$2" ] && [ "$2" != "-" ] && [ "$2" != "FAKE" ] && d=(--dependency="afterok:$2")
  [ "$DRY" = 1 ] && { echo "  CE $1 $3 clip=$7 $5" >&2; return; }
  sbatch --parsable -J "ce_$1" ${d[@]+"${d[@]}"} \
    --export=ALL,"EXP=$1,CLIP_DIR=$6,CE_RAW_DIR=$6,CE_CLIP=$7,CE_TAG=$3,CE_SEED=$4,CE_REFBIAS=0.0,CE_ANCHOR=absolute,CE_FITVARIANT=False,CE_REFROOT=True,CE_REFROOT_FLOOR=True,CE_REFVEL_OBS=False,CE_CONTACT_TIMECONST=0.004,CE_NRENVS=64,$5" \
    "$CE3" > /dev/null
}
# cell <name> <base> <expv> <trainhold> <cefun> <seeds "1 2"> <zs 0|1> <cedir> <ceclip> [extra export]
cell() {
  local name=$1 b=$2 expv=$3 rh=$4 cef=$5 seeds=$6 zs=$7 cdir=$8 clip=$9 extra=${10:-}
  local PREV="-" s J
  for s in $seeds; do
    J=$(sub "${name}_s$s" "$PREV" "$b,$expv${extra:+,$extra},$(ea "$rh" "$s")")
    echo "${name}_s$s -> $J" >&2
    for h in 1 20; do
      local n=2; [ "$h" = "$rh" ] && n=4
      for ((cs=0; cs<n; cs++)); do ce "${name}_s$s" "$J" "h${h}_s$cs" "$cs" "$($cef $h)" "$cdir" "$clip"; done
    done
    [ "$zs" = 1 ] && for cs in 0 1; do ce "${name}_s$s" "$J" "zsh1_$cs" "$cs" "$($cef 1)" "$V2" walk1_subject1.npz; done
    PREV="$J"
  done
}

echo "##### SEEDS #####"
cell n6cot_tok   "$B" "$COT"   1 CECOT "3 4" 1 "$V2" dance2_subject4.npz
cell n6split_tok "$B" "$SPLIT" 1 CETOK "3"   1 "$V2" dance2_subject4.npz
cell n6aux_tok   "$B" "$SPLIT,AUX_COEFF=0.5,AUX_HORIZON=5" 1 CETOK "3" 1 "$V2" dance2_subject4.npz
echo "##### H20-3WAY #####"
cell n6h20_ref    "$B" "$REF" 20 CEREF "1 2" 0 "$V2" dance2_subject4.npz
cell n6h20_tok    "$B" "$TOK" 20 CETOK "1 2" 0 "$V2" dance2_subject4.npz
cell n6coth20_tok "$B" "$COT" 20 CECOT "1 2" 1 "$V2" dance2_subject4.npz
echo "##### RECON / REPLACES #####"
cell n6cot01_tok  "$B" "$COT,COTRAIN_RECON=0.1" 1 CECOT "1 2" 0 "$V2" dance2_subject4.npz
cell n6cotrep_tok "$B" "$COTREP" 1 CECOTR "1 2" 1 "$V2" dance2_subject4.npz
echo "##### MORPH #####"
cell n6cotm0_tok "$B,$M0" "$COT" 1 CECOT "1 2" 0 "$V2" dance2_subject4.npz
cell n6cotm7_tok "$B,$M7" "$COT" 1 CECOT "1 2" 0 "$V2" dance2_subject4.npz
echo "##### LEGS #####"
cell n6cotlegw5_tok "$B" "$COT,LEGW=5.0" 1 CECOT "1 2" 0 "$V2" dance2_subject4.npz
echo "##### MULTI-MOTION #####"
cell n6cotsup_tok "$BSUP" "$COT" 1 CECOT "1 2" 0 "$SUPER" $SUPERC

[ "$DRY" = 1 ] || touch "$ROOT/.w6c_submitted"
echo "##### WAVE 6c SUBMITTED #####"
squeue -u akalenik -h | wc -l
