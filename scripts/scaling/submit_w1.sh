#!/usr/bin/env bash
# W1 -- THE ASSEMBLY TEST, plus W3 -- mc-emission seed replication.
# Submitted 2026-08-30 morning, after V1/V1MC drained (72/72 CEs, 0 failures).
#
# W1: FINAL_CONFIG_2026-08-30.md was assembled from parts that each validated
# alone (tk4 token routing on the B4 base; rx_p3f0 contact dose; heading
# 0.20/2.0/observed) but the assembled config has never trained as ONE arm.
# {ref, token@mc} x {hold 1, 5} x 2 seeds on the MODERNIZED base answers:
# does the token benefit survive/compose with the practical recipe?
# Registered prediction: token deltas comparable to V1's (-7% at h1, -20% at
# h5 on H1); a token x recipe interaction shows up as a shrunken delta.
#
# W3: the mc-emission adoption rests on 1 train seed (v1mc_* s1). Three s2
# runs at the EXACT V1MC config (B4 base, bit-identical, only the seed moves)
# make it 2x4 like every other cell.
#
#   DRY=1  print, submit nothing
set -u
ROOT=/ptmp/akalenik/urma
cd "$ROOT"
DRY="${DRY:-0}"

TOKMC=$ROOT/clips/tokentest_mc   # ORIGINAL npz + mc z_q (md5-verified 00:50)
LAF=$ROOT/clips/LAFAN1
XLA="--xla_gpu_enable_command_buffer="

# ---- W1 base: wave-7 B4 + rx_p3f0 dose (from rx_p3f0_s2_11174732.out) +
# heading at CH-L's operating point. Heading OBSERVED via EXTRA_ARGS (HOBS).
BM="STAGE=mmtrain,NR_ENVS=768,TOTAL_STEPS=19660800,SAVE_EVERY=1966080"
BM="$BM,CLIP_FILE=dance2_subject4.npz,CLIP_DIR=$TOKMC"
BM="$BM,REFVEL_OBS=False,REFROOT=True,REFROOT_FLOOR=True,REFBIAS=0.0"
BM="$BM,TRACK_COEFF=30.0,TRACK_TEMP=0.05,FITVARIANT=False,ANCHOR=absolute"
BM="$BM,GAITMODE=floor,GAITCOEFF=0.25,QVEL_TEMP=10"
BM="$BM,FOOTZVEL=1.0,FOOTH_TEMP=0.01,FOOTH=0.0"
BM="$BM,FOOTSLIP=6.6667,GROUNDPEN=333.33,POSTCONTACT=True"
BM="$BM,HEADING_RATIO=0.20,HEADING_TEMP=2.0"
BM="$BM,XLA_EXTRA=$XLA"

# ---- W3 base: BIT-IDENTICAL to submit_v1mc.sh's B (which is submit_v1.sh's
# B except CLIP_DIR) -- do NOT modernize, that would break the s1/s2 pairing.
B0="STAGE=mmtrain,NR_ENVS=768,TOTAL_STEPS=19660800,SAVE_EVERY=1966080"
B0="$B0,CLIP_FILE=dance2_subject4.npz,CLIP_DIR=$TOKMC"
B0="$B0,REFVEL_OBS=False,REFROOT=True,REFROOT_FLOOR=True,REFBIAS=0.0"
B0="$B0,TRACK_COEFF=30.0,TRACK_TEMP=0.05,FITVARIANT=False,ANCHOR=absolute"
B0="$B0,GAITMODE=floor,GAITCOEFF=0.25,QVEL_TEMP=10"
B0="$B0,FOOTZVEL=1.0,FOOTH_TEMP=0.01,FOOTH=0.3333"
B0="$B0,FOOTSLIP=20.0,GROUNDPEN=1000,POSTCONTACT=True"
B0="$B0,XLA_EXTRA=$XLA"

REF="LATENT_OBS=False,JLAT_ENC_DIM=0"
T4="LATENT_OBS=True,LATENT_DIM=32,LATENT_REPLACES=False,LATENT_SCOPE=per_joint,LATENT_DIVISOR=10.0,JLAT_ENC_DIM=4"
HOBS="--environment.command.tracking_clip_observe_root_heading=True"

sub() {  # sub <name> <dep-or-empty> <exports> -> job id
  local name="$1" dep="$2" exp="$3"
  local d=(); [ -n "$dep" ] && [ "$dep" != "FAKE" ] && d=(--dependency="afterany:$dep")
  if [ "$DRY" = "1" ]; then echo "TRAIN $name dep=${dep:-none}" >&2; echo "  $exp" >&2; echo "FAKE"; return; fi
  sbatch --parsable -J "$name" ${d[@]+"${d[@]}"} --export=ALL,"EXP_NAME=$name,$exp" "$ROOT/viper_train.sbatch"
}
ce() {   # ce <arm> <train-jobid> <ce-seed> <extra exports>
  local arm="$1" jid="$2" sd="$3" extra="$4"
  local d=(); [ -n "$jid" ] && [ "$jid" != "FAKE" ] && d=(--dependency="afterok:$jid")
  [ "$DRY" = "1" ] && { echo "  CE $arm s$sd $extra" >&2; return; }
  sbatch --parsable -J "ce_$arm" ${d[@]+"${d[@]}"} \
    --export=ALL,"EXP=$arm,CLIP_DIR=$TOKMC,CE_RAW_DIR=$LAF,CE_TAG=s$sd,CE_SEED=$sd,CE_REFBIAS=0.0,CE_ANCHOR=absolute,CE_FITVARIANT=False,CE_REFROOT=True,CE_REFROOT_FLOOR=True,CE_REFVEL_OBS=False,$extra" \
    "$ROOT/crosseval_token.sbatch" > /dev/null
}
ce4() { for s in 0 1 2 3; do ce "$1" "$2" "$s" "$3"; done; }

ce_ref_w1() { echo "CE_LATENT=0,CE_JLAT_ENC_DIM=0,CE_REFHOLD=$1,CE_HEADING_OBS=True"; }
ce_t4_w1()  { echo "CE_LATENT=1,CE_SCOPE=per_joint,CE_DIVISOR=10.0,CE_REPLACES=False,CE_JLAT_ENC_DIM=4,CE_REFHOLD=$1,CE_HEADING_OBS=True"; }
ce_t4_w3()  { echo "CE_LATENT=1,CE_SCOPE=per_joint,CE_DIVISOR=10.0,CE_REPLACES=False,CE_JLAT_ENC_DIM=4,CE_REFHOLD=$1"; }

ea() {  # ea <hold> <seed> [extra] -> EXTRA_ARGS=...
  local v="--environment.command.tracking_clip_reference_hold=$1"
  [ "$2" = "2" ] && v="$v:--environment.seed=2"
  [ -n "${3:-}" ] && v="$v:$3"
  echo "EXTRA_ARGS=$v"
}

echo "########## W1 ASSEMBLY TEST (modernized base) ##########"
# 4 chains of 2 (the cell's two seeds pair with afterany).
w1_chain() {  # w1_chain <cellname> <REF-or-T4> <hold>
  local cell="$1" rt="$2" hold="$3" EXPV CEV PREV="" s
  for s in 1 2; do
    if [ "$rt" = "REF" ]; then EXPV="$REF"; CEV=$(ce_ref_w1 "$hold"); else EXPV="$T4"; CEV=$(ce_t4_w1 "$hold"); fi
    local name="w1h${hold}_${cell}_s${s}"
    local J=$(sub "$name" "$PREV" "$BM,$EXPV,$(ea "$hold" "$s" "$HOBS")")
    echo "$name -> $J"
    ce4 "$name" "$J" "$CEV"
    PREV="$J"
  done
}
w1_chain ref REF 1
w1_chain tok T4  1
w1_chain ref REF 5
w1_chain tok T4  5

echo "########## W3 MC SEED-2 REPLICATION (B4 base, bit-identical to v1mc s1) ##########"
PREV=""
for hold in 1 5; do
  name="v1mc_h${hold}_tok_s2"
  J=$(sub "$name" "$PREV" "$B0,$T4,$(ea "$hold" 2)")
  echo "$name -> $J"
  ce4 "$name" "$J" "$(ce_t4_w3 "$hold")"
  PREV="$J"
done
name="v1mc_h10_tok_s2"
J=$(sub "$name" "" "$B0,$T4,$(ea 10 2)")
echo "$name -> $J"
ce4 "$name" "$J" "$(ce_t4_w3 10)"

echo "########## SUBMITTED ##########"
