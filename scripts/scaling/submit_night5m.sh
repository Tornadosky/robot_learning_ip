#!/usr/bin/env bash
# NIGHT5 M-EVAL -- does the token's advantage survive BODY MORPHOLOGY
# perturbation? Zero training: every Viper arm already trained under the
# default morphology schedule (0.2 -> ramp toward 0.7, ~0.45 reached at
# 19.66M), so the policies have seen randomized bodies; what was never
# measured is the REF-vs-TOKEN delta when the EVAL body is perturbed.
# The old "token gain vanishes under randomized bodies" claim predates the
# tk4 routing fix -- this retests it on the current recipe for free.
#
# Checkpoints: w2h20_{ref,tok}_s{1,2} (staleness showcase, hold=20) and
# n3gb33h1_{ref,tok}_s{1,2} (bundle recipe, hold=1).
# Eval morphology_coeff {0.3, 0.6} x CE seeds {0,1} = 32 CEs.
set -u
ROOT=/ptmp/akalenik/urma
cd "$ROOT"
DRY="${DRY:-0}"
TOKMC=$ROOT/clips/tokentest_mc
LAF=$ROOT/clips/LAFAN1
CE3=$ROOT/crosseval_token3.sbatch

REFX="CE_LATENT=0,CE_JLAT_ENC_DIM=0,CE_HEADING_OBS=True"
T4X="CE_LATENT=1,CE_SCOPE=per_joint,CE_DIVISOR=10.0,CE_REPLACES=False,CE_JLAT_ENC_DIM=4,CE_HEADING_OBS=True"

mce() {  # mce <arm> <extra> <mc> <seed> <tc>
  local arm="$1" extra="$2" mc="$3" sd="$4" tc="$5"
  local mctag="${mc/0./m}"
  [ "$DRY" = "1" ] && { echo "  MCE $arm mc=$mc s$sd" >&2; return; }
  sbatch --parsable -J "ce_${arm}_mc" \
    --export=ALL,"EXP=$arm,CLIP_DIR=$TOKMC,CE_RAW_DIR=$LAF,CE_CLIP=dance2_subject4.npz,CE_TAG=${mctag}_s$sd,CE_SEED=$sd,CE_MORPH=$mc,CE_REFBIAS=0.0,CE_ANCHOR=absolute,CE_FITVARIANT=False,CE_REFROOT=True,CE_REFROOT_FLOOR=True,CE_REFVEL_OBS=False,CE_CONTACT_TIMECONST=$tc,$extra" \
    "$CE3" > /dev/null
  echo "  $arm mc=$mc s$sd" >&2
}

for s in 1 2; do
  for mc in 0.3 0.6; do
    for cs in 0 1; do
      mce "w2h20_ref_s$s"   "$REFX,CE_REFHOLD=20" "$mc" "$cs" 0.0
      mce "w2h20_tok_s$s"   "$T4X,CE_REFHOLD=20"  "$mc" "$cs" 0.0
      mce "n3gb33h1_ref_s$s" "$REFX,CE_REFHOLD=1" "$mc" "$cs" 0.004
      mce "n3gb33h1_tok_s$s" "$T4X,CE_REFHOLD=1"  "$mc" "$cs" 0.004
    done
  done
done
echo "M-EVAL SUBMITTED"
squeue -u akalenik -h | wc -l
