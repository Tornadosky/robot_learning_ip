#!/usr/bin/env bash
# NIGHT5 WAVE 1 (2026-08-31 evening) -- the swing-match campaign.
#
# Debug session findings this submits against:
#   * legs are PHASE-LOCKED but amplitude-suppressed (knee amp 0.13-0.77,
#     arm corr 0.99) -> the reward trade suppresses swing, info is there;
#   * the token carries legs faithfully (mc codec leg RMSE 0.037 rad);
#   * FOOTH and foot_air_time have no gradient for a planted foot; the new
#     SWING MATCH term (per-foot linear hinge, reference-airborne gated,
#     patch_swing_match.py) does -- smoke-tested locally, gate active 44% of
#     foot-time on dance2_subject4.
#
#   SW    swing ratio {0.15, 0.5, 1.5} x {ref, tok} x 2 seeds, hold=1 on the
#         night4 bundle recipe + render dumps on the 0.5/1.5 cells.
#   T1FIX the H1+T1 CE ROCm bisect: {no-XLA-flag, nr_envs 32, both} on the
#         existing n4t1_tok_s1 checkpoint.
#
# 12 trains + 48 CEs + 4 dumps + 3 bisect = 67 jobs.  DRY=1 to print.
set -u
ROOT=/ptmp/akalenik/urma
cd "$ROOT"
DRY="${DRY:-0}"

TOKMC=$ROOT/clips/tokentest_mc
LAF=$ROOT/clips/LAFAN1
XLA="--xla_gpu_enable_command_buffer="
CE2=$ROOT/crosseval_token2.sbatch

base() {  # night4 bundle recipe, verbatim
  local b="STAGE=mmtrain,NR_ENVS=768,TOTAL_STEPS=19660800,SAVE_EVERY=1966080"
  b="$b,CLIP_FILE=dance2_subject4.npz,CLIP_DIR=$TOKMC"
  b="$b,REFVEL_OBS=False,REFROOT=True,REFROOT_FLOOR=True,REFBIAS=0.0"
  b="$b,TRACK_COEFF=30.0,TRACK_TEMP=0.05,FITVARIANT=False,ANCHOR=absolute"
  b="$b,GAITMODE=floor,GAITCOEFF=0.25,QVEL_TEMP=10"
  b="$b,FOOTZVEL=1.0,FOOTH_TEMP=0.05,FOOTH=0.3333"
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
  local arm="$1" jid="$2" sd="$3" extra="$4"
  local d=(); [ -n "$jid" ] && [ "$jid" != "FAKE" ] && d=(--dependency="afterok:$jid")
  [ "$DRY" = "1" ] && { echo "  CE $arm s$sd $extra" >&2; return; }
  sbatch --parsable -J "ce_$arm" ${d[@]+"${d[@]}"} \
    --export=ALL,"EXP=$arm,CLIP_DIR=$TOKMC,CE_RAW_DIR=$LAF,CE_CLIP=dance2_subject4.npz,CE_TAG=s$sd,CE_SEED=$sd,CE_REFBIAS=0.0,CE_ANCHOR=absolute,CE_FITVARIANT=False,CE_REFROOT=True,CE_REFROOT_FLOOR=True,CE_REFVEL_OBS=False,CE_CONTACT_TIMECONST=0.004,$extra" \
    "$CE2" > /dev/null
}
ce4() { for s in 0 1 2 3; do ce "$1" "$2" "$s" "$3"; done; }
cedump() {
  local arm="$1" jid="$2" extra="$3"
  local d=(); [ -n "$jid" ] && [ "$jid" != "FAKE" ] && d=(--dependency="afterok:$jid")
  [ "$DRY" = "1" ] && { echo "  CEDUMP $arm" >&2; return; }
  sbatch --parsable -J "ced_$arm" ${d[@]+"${d[@]}"} \
    --export=ALL,"EXP=$arm,CLIP_DIR=$TOKMC,CE_RAW_DIR=$LAF,CE_CLIP=dance2_subject4.npz,CE_TAG=dump,CE_SEED=0,CE_DUMP=1,CE_REFBIAS=0.0,CE_ANCHOR=absolute,CE_FITVARIANT=False,CE_REFROOT=True,CE_REFROOT_FLOOR=True,CE_REFVEL_OBS=False,CE_CONTACT_TIMECONST=0.004,$extra" \
    "$CE2" > /dev/null
}

ce_ref()  { echo "CE_LATENT=0,CE_JLAT_ENC_DIM=0,CE_REFHOLD=$1,CE_HEADING_OBS=True"; }
ce_t4()   { echo "CE_LATENT=1,CE_SCOPE=per_joint,CE_DIVISOR=10.0,CE_REPLACES=False,CE_JLAT_ENC_DIM=4,CE_REFHOLD=$1,CE_HEADING_OBS=True"; }

ea() {  local v="--environment.command.tracking_clip_reference_hold=$1"
  [ "$2" != "1" ] && v="$v:--environment.seed=$2"
  [ -n "${3:-}" ] && v="$v:$3"
  echo "EXTRA_ARGS=$v:$HOBS"
}

seedrun() {  # <cell> <base> <expv> <ce_extra> <refhold> <seed> <dep> [eargs] -> jid
  local name="${1}_s${6}"
  local J; J=$(sub "$name" "$7" "$2,$3,$(ea "$5" "$6" "${8:-}")")
  echo "$name -> $J" >&2
  ce4 "$name" "$J" "$4"
  echo "$J"
}
chain2() {  # <cell> <base> <expv> <ce_extra> <refhold> [eargs] -> s1 jid
  local PREV="" s J J1=""
  for s in 1 2; do
    J=$(seedrun "$1" "$2" "$3" "$4" "$5" "$s" "$PREV" "${6:-}")
    [ "$s" = "1" ] && J1="$J"
    PREV="$J"
  done
  echo "$J1"
}

B=$(base)
SW() { echo "--environment.reward.deepmimic_swing_match_weight_ratio=$1"; }

echo "########## SW -- swing-match dose grid (hold=1, bundle recipe) ##########"
chain2 n5sw015_ref "$B" "$REF" "$(ce_ref 1)" 1 "$(SW 0.15)" > /dev/null
chain2 n5sw015_tok "$B" "$T4"  "$(ce_t4 1)"  1 "$(SW 0.15)" > /dev/null
J=$(chain2 n5sw05_ref "$B" "$REF" "$(ce_ref 1)" 1 "$(SW 0.5)")
cedump n5sw05_ref_s1 "$J" "$(ce_ref 1)"
J=$(chain2 n5sw05_tok "$B" "$T4" "$(ce_t4 1)" 1 "$(SW 0.5)")
cedump n5sw05_tok_s1 "$J" "$(ce_t4 1)"
J=$(chain2 n5sw15_ref "$B" "$REF" "$(ce_ref 1)" 1 "$(SW 1.5)")
cedump n5sw15_ref_s1 "$J" "$(ce_ref 1)"
J=$(chain2 n5sw15_tok "$B" "$T4" "$(ce_t4 1)" 1 "$(SW 1.5)")
cedump n5sw15_tok_s1 "$J" "$(ce_t4 1)"

echo "########## T1FIX -- H1+T1 CE ROCm bisect on n4t1_tok_s1 ##########"
t1fix() {  # t1fix <tag> <xla 0|1> <nrenvs>
  [ "$DRY" = "1" ] && { echo "  T1FIX $1 xla=$2 envs=$3" >&2; return; }
  sbatch --parsable -J "ce_t1fix_$1" \
    --export=ALL,"EXP=n4t1_tok_s1,CLIP_DIR=$ROOT/clips/clips_3t_token,CE_RAW_DIR=$ROOT/clips/clips_3t_token,CE_CLIP=dance2_subject4.npz,CE_TAG=fix$1,CE_SEED=0,CE_REFBIAS=0.0,CE_ANCHOR=absolute,CE_FITVARIANT=False,CE_REFROOT=True,CE_REFROOT_FLOOR=True,CE_REFVEL_OBS=False,CE_CONTACT_TIMECONST=0.004,CE_ROBOTS=unitree_h1:booster_t1,CE_XLA_OFF=$2,CE_NRENVS=$3,CE_LATENT=1,CE_SCOPE=per_joint,CE_DIVISOR=10.0,CE_REPLACES=False,CE_JLAT_ENC_DIM=4,CE_REFHOLD=1,CE_HEADING_OBS=True" \
    "$ROOT/crosseval_token3.sbatch" > /dev/null
  echo "  t1fix $1 submitted" >&2
}
t1fix a 1 64
t1fix b 0 32
t1fix c 1 32

echo "########## NIGHT5 WAVE 1 SUBMITTED ##########"
squeue -u akalenik -h | wc -l
