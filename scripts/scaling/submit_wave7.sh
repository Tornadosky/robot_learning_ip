#!/usr/bin/env bash
# WAVE 7 -- the arms the paper needs, on the fixed code.
#
# The post-contact double count is now FIXED in Viper's loco_mjx (deployed
# 2026-08-29). `tracking_post_contact_penalties=True` therefore means something
# different than it did in waves 4-6, so EVERY wave-7 comparison runs against a
# wave-7 control. Nothing here is compared to an older arm.
#
# CH-P  the model fix: UnitreeH1's foot is a 0.295 x 0.028 m blade, ~3x narrower
#       than the real ~0.09 m foot, giving H1 1.60 cm2/kg of support against G1's
#       3.08. `foot_half_width_override=0.045` widens the y half-extent only, so
#       the underside height -- and standing height, and every reference-height
#       statistic -- is unchanged (verified: verify_foot_width.py).
# CH-Q  heading at wave 6 CH-L's operating point (80.7 -> 8.7 deg H1 for +8.1 %
#       joint RMSE). dance2_subject4 accumulates 12 058 deg of yaw and spends
#       74.8 % of frames >90 deg from its start, and joint RMSE is blind to all
#       of it, so this is the half of the motion nothing has been measuring.
# CH-R  do they compose? Both defects are about the feet and the root, so an
#       interaction is plausible in either direction.
#
# REGISTERED PREDICTION for CH-P, before the arms land: widening H1's foot raises
# H1's ankle FVE toward G1's and shrinks the H1/G1 penetration ratio. It should
# NOT fix the ankle outright -- BoosterT1 has 4.4x H1's support per kg and its
# ankles also sit at FVE ~0, so support area cannot be the whole story. A null on
# H1 kills the blade as a mechanism.
#
#   ONLY=P|Q|R|G   submit one chain (default: all)
#   DRY=1          print, submit nothing
set -u
ROOT=/ptmp/akalenik/urma
cd "$ROOT"
DRY="${DRY:-0}"
ONLY="${ONLY:-all}"
want() { [ "$ONLY" = "all" ] || [ "$ONLY" = "$1" ]; }

LAF=$ROOT/clips/LAFAN1
FIX=$ROOT/clips/LAFAN1_fixed
XLA="--xla_gpu_enable_command_buffer="

B4="STAGE=mmtrain,NR_ENVS=768,TOTAL_STEPS=19660800,SAVE_EVERY=1966080"
B4="$B4,CLIP_FILE=dance2_subject4.npz,CLIP_DIR=$LAF"
B4="$B4,REFVEL_OBS=False,REFROOT=True,REFROOT_FLOOR=True,REFBIAS=0.0"
B4="$B4,TRACK_COEFF=30.0,TRACK_TEMP=0.05,FITVARIANT=False,ANCHOR=absolute"
B4="$B4,GAITMODE=floor,GAITCOEFF=0.25,QVEL_TEMP=10"
B4="$B4,FOOTZVEL=1.0,FOOTH_TEMP=0.01,FOOTH=0.3333"
B4="$B4,FOOTSLIP=20.0,GROUNDPEN=1000,POSTCONTACT=True"
B4="$B4,XLA_EXTRA=$XLA"

HOBS="--environment.command.tracking_clip_observe_root_heading=True"
FOOT="--environment.terrain.foot_half_width_override=0.045"

sub() {
  local name="$1" dep="$2" exp="$3"
  local d=(); [ -n "$dep" ] && d=(--dependency="$dep")
  # DRY output goes to STDERR: this function's STDOUT is captured as the job id,
  # so echoing the plan there would splice the whole plan into $PREV and then
  # into the next arm's --dependency string.
  if [ "$DRY" = "1" ]; then echo "TRAIN $name dep=$dep" >&2; echo "  $exp" >&2; echo "FAKE"; return; fi
  sbatch --parsable -J "$name" ${d[@]+"${d[@]}"} \
    --export=ALL,"EXP_NAME=$name,$exp" "$ROOT/viper_train.sbatch"
}
ce() {
  local tag="$1" arm="$2" jid="$3" sd="$4" extra="$5"
  local d=(); [ -n "$jid" ] && [ "$jid" != "FAKE" ] && d=(--dependency="afterok:$jid")
  [ "$DRY" = "1" ] && { echo "  CE $tag $arm s$sd $extra" >&2; return; }
  sbatch --parsable -J "$tag" \
    ${d[@]+"${d[@]}"} \
    --export=ALL,"EXP=$arm,CE_TAG=s$sd,CE_SEED=$sd,CE_REFBIAS=0.0,CE_ANCHOR=absolute,CE_FITVARIANT=False,CE_REFROOT=True,CE_REFROOT_FLOOR=True,CE_REFVEL_OBS=False,$extra" \
    "$ROOT/crosseval2.sbatch" > /dev/null
}
# CE_HEADING_OBS must MATCH the arm: the flag adds 2 channels to policy AND
# critic, so a mismatch is an observation-width error, not a bad number.
ce4() { local a="$1" j="$2" ho="$3" k
  for k in 0 1 2 3; do ce "c_${a}_$k" "$a" "$j" "$k" \
    "CLIP_DIR=$LAF,CE_CLIP=dance2_subject4.npz,CE_RAW_DIR=$LAF,CE_ROBOTS=unitree_h1:unitree_g1,CE_HEADING_OBS=$ho"; done; }

submit_chain() {   # submit_chain <chain-name> <specs on stdin: name|extra|seed|headingobs>
  local PREV=""
  while IFS='|' read -r N EA S HO; do
    [ -z "$N" ] && continue
    local DEP=""; [ -n "$PREV" ] && DEP="afterany:$PREV"
    local EXTRA=""; [ -n "$EA" ] && EXTRA=",EXTRA_ARGS=$EA"
    local HEAD=""; [ "$HO" = "True" ] && HEAD=",HEADING_RATIO=0.20,HEADING_TEMP=2.0"
    local J=$(sub "$N" "$DEP" "$B4,SEED=$S$HEAD$EXTRA")
    echo "$N -> $J"; ce4 "$N" "$J" "$HO"; PREV="$J"
  done
}

if want P; then
  echo "########## CH-P -- H1 foot width (the model fix) ##########"
  submit_chain <<SPECS
w7_ctl_s1||1|False
w7_foot_s1|$FOOT|1|False
w7_ctl_s2||2|False
w7_foot_s2|$FOOT|2|False
SPECS
fi

if want Q; then
  echo "########## CH-Q -- heading on the fixed code ##########"
  submit_chain <<SPECS
w7_head_s1|$HOBS|1|True
w7_head_s2|$HOBS|2|True
SPECS
fi

if want R; then
  echo "########## CH-R -- do the foot and heading fixes COMPOSE? ##########"
  submit_chain <<SPECS
w7_both_s1|$HOBS:$FOOT|1|True
w7_both_s2|$HOBS:$FOOT|2|True
SPECS
fi

# ---------------------------------------------------------------- GEN-B'
# HELD-OUT BODY, done correctly this time. The first GEN-B asked for atlas,
# talos and unitree_h1v2 and produced NOTHING -- h1v2 has no loco_mjx model at
# all, and Atlas/Talos clips do not map onto their loco_mjx models
# (derive_clip_signs residual 0.043 / 0.830 against ~1.5e-4 for the families that
# work). BoosterT1 is the one body that IS correctly mapped (23/23 joints,
# residual 1.5e-4) and that `best79` never trained on, so it is the only honest
# zero-shot test available -- scored against the REGENERATED clip, because the
# shipped one is unreachable.
if want G; then
  echo "########## GEN-B' -- BoosterT1 held out, on the fixed reference ##########"
  for k in 0 1 2 3; do
    for arm in best79_s1 best79_s2; do
      ce "b2_t1_${arm}_$k" "$arm" "" "$k" \
        "CLIP_DIR=$FIX,CE_CLIP=dance2_subject4.npz,CE_RAW_DIR=$FIX,CE_ROBOTS=booster_t1,CE_HEADING_OBS=False,CE_OUTDIR=$ROOT/crosseval_gen"
    done
    ce "b2_t1_zero_$k" best79_s1 "" "$k" \
      "CLIP_DIR=$FIX,CE_CLIP=dance2_subject4.npz,CE_RAW_DIR=$FIX,CE_ROBOTS=booster_t1,CE_ZERO=1,CE_HEADING_OBS=False,CE_OUTDIR=$ROOT/crosseval_gen"
  done
fi

echo "WAVE7 SUBMITTED (ONLY=$ONLY)"
