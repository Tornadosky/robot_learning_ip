#!/usr/bin/env bash
# NIGHT5 WAVE 2 -- v2-tokenizer arms + the honest ZERO-SHOT probe.
# Requires clips/tokentest_v2 on Viper (H1/G1 dance4 + walk1, orig npz + z_q
# from tokenizer_3t_v2: 250 epochs, foot channels, walk1 HELD OUT of the fit).
#
#   n5v2_tok   T4 arm, v2 tokens, bundle recipe + swing 0.5 -- vs n5sw05_tok
#              (same recipe, mc tokens): isolates the tokenizer version/A5.
#   n5v2rep    token-only (LATENT_REPLACES), v2 tokens, swing 0.5.
#   Zero-shot CEs: every arm also scored on walk1_subject1 -- a motion neither
#   the policy nor the tokenizer ever saw. T4 walk1 CE runs at REFHOLD=20 so
#   the token, not the fresh reference, must carry the motion.
#
# 4 trains + 32 CEs.  DRY=1 to print.  Guard: .night5b_submitted.
set -u
ROOT=/ptmp/akalenik/urma
cd "$ROOT"
DRY="${DRY:-0}"
[ -f "$ROOT/.night5b_submitted" ] && [ "$DRY" != "1" ] && { echo "already submitted"; exit 0; }

V2=$ROOT/clips/tokentest_v2
LAF=$ROOT/clips/LAFAN1
XLA="--xla_gpu_enable_command_buffer="
CE3=$ROOT/crosseval_token3.sbatch
[ "$DRY" = "1" ] || { [ -f "$V2/UnitreeH1/walk1_subject1_zq.npz" ] || { echo "ABORT: no v2 sidecars"; exit 1; }; }

B="STAGE=mmtrain,NR_ENVS=768,TOTAL_STEPS=19660800,SAVE_EVERY=1966080"
B="$B,CLIP_FILE=dance2_subject4.npz,CLIP_DIR=$V2"
B="$B,REFVEL_OBS=False,REFROOT=True,REFROOT_FLOOR=True,REFBIAS=0.0"
B="$B,TRACK_COEFF=30.0,TRACK_TEMP=0.05,FITVARIANT=False,ANCHOR=absolute"
B="$B,GAITMODE=floor,GAITCOEFF=0.25,QVEL_TEMP=10"
B="$B,FOOTZVEL=1.0,FOOTH_TEMP=0.05,FOOTH=0.3333"
B="$B,FOOTSLIP=6.6667,GROUNDPEN=1000,POSTCONTACT=True"
B="$B,CONTACT_TIMECONST=0.004,HEADING_RATIO=0.20,HEADING_TEMP=2.0"
B="$B,XLA_EXTRA=$XLA"

T4="LATENT_OBS=True,LATENT_DIM=32,LATENT_REPLACES=False,LATENT_SCOPE=per_joint,LATENT_DIVISOR=10.0,JLAT_ENC_DIM=4"
TREP="LATENT_OBS=True,LATENT_DIM=32,LATENT_REPLACES=True,LATENT_SCOPE=per_joint,LATENT_DIVISOR=10.0,JLAT_ENC_DIM=4"
HOBS="--environment.command.tracking_clip_observe_root_heading=True"
SWF="--environment.reward.deepmimic_swing_match_weight_ratio=0.5"

sub() {  local name="$1" dep="$2" exp="$3"
  local d=(); [ -n "$dep" ] && [ "$dep" != "FAKE" ] && d=(--dependency="afterany:$dep")
  if [ "$DRY" = "1" ]; then echo "TRAIN $name dep=${dep:-none}" >&2; echo "  $exp" >&2; echo "FAKE"; return; fi
  sbatch --parsable -J "$name" ${d[@]+"${d[@]}"} --export=ALL,"EXP_NAME=$name,$exp" "$ROOT/viper_train.sbatch"
}
ce() {  # ce <arm> <jid> <sd> <clip> <extra>
  local arm="$1" jid="$2" sd="$3" cf="$4" extra="$5"
  local d=(); [ -n "$jid" ] && [ "$jid" != "FAKE" ] && d=(--dependency="afterok:$jid")
  [ "$DRY" = "1" ] && { echo "  CE $arm s$sd clip=$cf $extra" >&2; return; }
  local tag="s$sd"; [ "$cf" = "walk1_subject1.npz" ] && tag="zs$sd"
  sbatch --parsable -J "ce_$arm" ${d[@]+"${d[@]}"} \
    --export=ALL,"EXP=$arm,CLIP_DIR=$V2,CE_RAW_DIR=$V2,CE_CLIP=$cf,CE_TAG=$tag,CE_SEED=$sd,CE_REFBIAS=0.0,CE_ANCHOR=absolute,CE_FITVARIANT=False,CE_REFROOT=True,CE_REFROOT_FLOOR=True,CE_REFVEL_OBS=False,CE_CONTACT_TIMECONST=0.004,$extra" \
    "$CE3" > /dev/null
}
ea() {  local v="--environment.command.tracking_clip_reference_hold=1:$SWF"
  [ "$2" != "1" ] && v="$v:--environment.seed=$2"
  echo "EXTRA_ARGS=$v:$HOBS"
}
CET4="CE_LATENT=1,CE_SCOPE=per_joint,CE_DIVISOR=10.0,CE_REPLACES=False,CE_JLAT_ENC_DIM=4,CE_HEADING_OBS=True"
CETR="CE_LATENT=1,CE_SCOPE=per_joint,CE_DIVISOR=10.0,CE_REPLACES=True,CE_JLAT_ENC_DIM=4,CE_HEADING_OBS=True"

chainv2() {  # chainv2 <cell> <expv> <ce_extra> <zs_refhold>
  local cell="$1" expv="$2" cex="$3" zsh="$4" PREV="" s J
  for s in 1 2; do
    local name="${cell}_s${s}"
    J=$(sub "$name" "$PREV" "$B,$expv,$(ea 1 "$s")")
    echo "$name -> $J"
    for cs in 0 1 2 3; do ce "$name" "$J" "$cs" dance2_subject4.npz "$cex,CE_REFHOLD=1"; done
    for cs in 0 1 2 3; do ce "$name" "$J" "$cs" walk1_subject1.npz "$cex,CE_REFHOLD=$zsh"; done
    PREV="$J"
  done
}
echo "########## WAVE 2: v2 tokenizer + zero-shot ##########"
chainv2 n5v2_tok  "$T4"   "$CET4" 20
chainv2 n5v2rep   "$TREP" "$CETR" 1
[ "$DRY" = "1" ] || touch "$ROOT/.night5b_submitted"
echo "########## WAVE 2 SUBMITTED ##########"
squeue -u akalenik -h | wc -l
