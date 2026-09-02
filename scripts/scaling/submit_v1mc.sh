#!/usr/bin/env bash
# V1MC -- does a BETTER tokenizer improve the policy? Post-V1 follow-up,
# submitted 2026-08-30 ~00:50 after V1 drained (60/60 CEs, 0 failures).
#
# F0 forensics (see OVERNIGHT_EXEC_2026-08-29.md) proved the production
# clips/tokentest _zq came from the BASE `tokenizer` fit (now-RMSE ~0.18 rad),
# while kevin_tokenizer_mc reconstructs at 0.035-0.056 rad with flat lookahead
# rows. clips/tokentest_mc = the SAME original LAFAN1 npz (md5-verified
# 29702a85/316cecfc) + the mc emission's z_q (shape-identical 9025xJx32).
#
# ARMS: {hold 1, 5, 10} x token@tk4-routing, seed 1, 4 CE seeds each.
# A/B partner cells: v1h1_tok/tk4_perjsep, v1h5_tok_s1, v1h10_tok_s1 -- the
# ONLY difference is which fit emitted the sidecar.
#
# REGISTERED EXPECTATION: if token quality matters, v1mc arms beat their V1
# partners; if the token is only a coarse phase/window signal, they tie.
#
#   DRY=1  print, submit nothing
set -u
ROOT=/ptmp/akalenik/urma
cd "$ROOT"
DRY="${DRY:-0}"

TOKMC=$ROOT/clips/tokentest_mc
LAF=$ROOT/clips/LAFAN1
XLA="--xla_gpu_enable_command_buffer="

# BIT-IDENTICAL to submit_v1.sh's base except CLIP_DIR.
B="STAGE=mmtrain,NR_ENVS=768,TOTAL_STEPS=19660800,SAVE_EVERY=1966080"
B="$B,CLIP_FILE=dance2_subject4.npz,CLIP_DIR=$TOKMC"
B="$B,REFVEL_OBS=False,REFROOT=True,REFROOT_FLOOR=True,REFBIAS=0.0"
B="$B,TRACK_COEFF=30.0,TRACK_TEMP=0.05,FITVARIANT=False,ANCHOR=absolute"
B="$B,GAITMODE=floor,GAITCOEFF=0.25,QVEL_TEMP=10"
B="$B,FOOTZVEL=1.0,FOOTH_TEMP=0.01,FOOTH=0.3333"
B="$B,FOOTSLIP=20.0,GROUNDPEN=1000,POSTCONTACT=True"
B="$B,XLA_EXTRA=$XLA"

T4="LATENT_OBS=True,LATENT_DIM=32,LATENT_REPLACES=False,LATENT_SCOPE=per_joint,LATENT_DIVISOR=10.0,JLAT_ENC_DIM=4"

sub() {  # sub <name> <exports> -> job id
  local name="$1" exp="$2"
  if [ "$DRY" = "1" ]; then echo "TRAIN $name" >&2; echo "  $exp" >&2; echo "FAKE"; return; fi
  sbatch --parsable -J "$name" --export=ALL,"EXP_NAME=$name,$exp" "$ROOT/viper_train.sbatch"
}
ce() {   # ce <arm> <train-jobid> <ce-seed> <extra>
  local arm="$1" jid="$2" sd="$3" extra="$4"
  local d=(); [ -n "$jid" ] && [ "$jid" != "FAKE" ] && d=(--dependency="afterok:$jid")
  [ "$DRY" = "1" ] && { echo "  CE $arm s$sd $extra" >&2; return; }
  sbatch --parsable -J "ce_$arm" ${d[@]+"${d[@]}"} \
    --export=ALL,"EXP=$arm,CLIP_DIR=$TOKMC,CE_RAW_DIR=$LAF,CE_TAG=s$sd,CE_SEED=$sd,CE_REFBIAS=0.0,CE_ANCHOR=absolute,CE_FITVARIANT=False,CE_REFROOT=True,CE_REFROOT_FLOOR=True,CE_REFVEL_OBS=False,$extra" \
    "$ROOT/crosseval_token.sbatch" > /dev/null
}
ce4() { for s in 0 1 2 3; do ce "$1" "$2" "$s" "$3"; done; }
ce_t4() { echo "CE_LATENT=1,CE_SCOPE=per_joint,CE_DIVISOR=10.0,CE_REPLACES=False,CE_JLAT_ENC_DIM=4,CE_REFHOLD=$1"; }
rh()  { echo "EXTRA_ARGS=--environment.command.tracking_clip_reference_hold=$1"; }

echo "########## V1MC TOKENIZER-QUALITY A/B ##########"
for spec in "v1mc_h1_tok_s1|1" "v1mc_h5_tok_s1|5" "v1mc_h10_tok_s1|10"; do
  name="${spec%|*}"; hold="${spec#*|}"
  J=$(sub "$name" "$B,$T4,$(rh "$hold")")
  echo "$name -> $J"
  ce4 "$name" "$J" "$(ce_t4 "$hold")"
done
echo "########## SUBMITTED ##########"
