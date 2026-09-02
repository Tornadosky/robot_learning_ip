#!/usr/bin/env bash
# V1 -- THE FSQ REFERENCE-RATE CURVE, at the CORRECTED token routing.
#
# WHY. CH-N showed the design-B token flips from a +10.5% penalty (fresh
# reference, hold=1) to a -10/-15% gain (stale reference, hold=5). One point
# against one point; the CURVE is the claim. But §4.2 of the 2026-08-29 plan
# found every historical token number was measured with the token entering the
# policy 559x louder than its neighbours, so CH-N measured rate x scale
# CONFOUNDED. The tk0-tk4 campaign (landed 2026-08-29 ~21:30) settled the scale
# axis at hold=1, n=4 CE seeds, raw_rmse_rad vs tk0_ctl (H1 0.2042 / G1 0.1608):
#
#   tk1 perj/div1.0    +4.2% H1 / +4.3% G1   reproduces the historical penalty
#   tk2 perj/div10     -3.6% / -3.0%          normalisation alone recovers it
#   tk3 global          -0.3% / -1.7%          ~control, as registered
#   tk4 perj/div10/sep  -3.7% / -9.9%          BEST -- the token PAYS at hold=1
#
# The scale/bottleneck account is CONFIRMED, so the rate curve runs at tk4's
# routing (per_joint, divisor 10, separate Dense(4) projection). hold=1 points
# already exist on this exact base: tk0_ctl (ref), tk4_perjsep (token),
# tk1_perj1 (token at the historical divisor).
#
# ARMS. hold in {2,5,10} x {reference only, reference+token(tk4 routing)},
# 2 runs each (second run at --environment.seed=2; NOTE: wave5/6's SEED= export
# was never consumed by viper_train.sbatch, so their _s2 arms were resubmissions
# under the default seed -- this campaign seeds the second run for real).
# Plus ONE bridge arm: hold=5 at the HISTORICAL routing (divisor 1.0, shared
# Dense), so the CH-N sign flip is reproduced on this clip and this code, and
# the new curve can be placed against every historical number.
# hold=1 gets its second run via v1h1_ref_s2 / v1h1_tok_s2 (same recipes as
# tk0_ctl / tk4_perjsep).
#
# REGISTERED PREDICTION (the 250ms-window mechanism): the token's delta should
# cross from ~0/positive-gain at hold=1-2 to a clear gain by hold ~3-5 (the
# window it averages), growing at hold=10. The bridge arm should reproduce
# CH-N's sign at divisor 1.0. If the corrected-routing token pays at EVERY
# hold, design B is a general result, not a low-rate niche (F2's question,
# answered for free).
#
# SCOPE. 15 training arms x 19.7M steps + 4 crosseval seeds each, chained in
# pairs with afterany (a failed first arm must not strand its partner) and
# crossevals afterok on their own arm -- everything lands in slurm at submit
# time, so a dropped WSL/ssh tunnel cannot interrupt the campaign.
#
#   DRY=1  print, submit nothing
set -u
ROOT=/ptmp/akalenik/urma
cd "$ROOT"
DRY="${DRY:-0}"

TOK=$ROOT/clips/tokentest      # ORIGINAL clip + _zq sidecar -- the arm varies
LAF=$ROOT/clips/LAFAN1         # the token and nothing else (submit_token.sh).
XLA="--xla_gpu_enable_command_buffer="

# Wave-7's B4 base, BIT-IDENTICAL to submit_token.sh, so tk0/tk1/tk4 are this
# campaign's hold=1 points. Do NOT modernise the recipe here (rx_p3f0 etc.) --
# that would disconnect the curve from its own left edge.
B="STAGE=mmtrain,NR_ENVS=768,TOTAL_STEPS=19660800,SAVE_EVERY=1966080"
B="$B,CLIP_FILE=dance2_subject4.npz,CLIP_DIR=$TOK"
B="$B,REFVEL_OBS=False,REFROOT=True,REFROOT_FLOOR=True,REFBIAS=0.0"
B="$B,TRACK_COEFF=30.0,TRACK_TEMP=0.05,FITVARIANT=False,ANCHOR=absolute"
B="$B,GAITMODE=floor,GAITCOEFF=0.25,QVEL_TEMP=10"
B="$B,FOOTZVEL=1.0,FOOTH_TEMP=0.01,FOOTH=0.3333"
B="$B,FOOTSLIP=20.0,GROUNDPEN=1000,POSTCONTACT=True"
B="$B,XLA_EXTRA=$XLA"

# The three routings, named for the tk arm that validated each.
REF="LATENT_OBS=False,JLAT_ENC_DIM=0"
T4="LATENT_OBS=True,LATENT_DIM=32,LATENT_REPLACES=False,LATENT_SCOPE=per_joint,LATENT_DIVISOR=10.0,JLAT_ENC_DIM=4"
T1="LATENT_OBS=True,LATENT_DIM=32,LATENT_REPLACES=False,LATENT_SCOPE=per_joint,LATENT_DIVISOR=1.0,JLAT_ENC_DIM=0"

sub() {  # sub <name> <dep-jobid-or-empty> <exports>  -> prints job id
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
    --export=ALL,"EXP=$arm,CLIP_DIR=$TOK,CE_RAW_DIR=$LAF,CE_TAG=s$sd,CE_SEED=$sd,CE_REFBIAS=0.0,CE_ANCHOR=absolute,CE_FITVARIANT=False,CE_REFROOT=True,CE_REFROOT_FLOOR=True,CE_REFVEL_OBS=False,$extra" \
    "$ROOT/crosseval_token.sbatch" > /dev/null
}
ce4() { for s in 0 1 2 3; do ce "$1" "$2" "$s" "$3"; done; }

# Crosseval must mirror the TRAINING condition (hold AND routing), or the
# config->effect guard's train-vs-eval diff flags the arm.
ce_ref() { echo "CE_LATENT=0,CE_JLAT_ENC_DIM=0,CE_REFHOLD=$1"; }
ce_t4()  { echo "CE_LATENT=1,CE_SCOPE=per_joint,CE_DIVISOR=10.0,CE_REPLACES=False,CE_JLAT_ENC_DIM=4,CE_REFHOLD=$1"; }
ce_t1()  { echo "CE_LATENT=1,CE_SCOPE=per_joint,CE_DIVISOR=1.0,CE_REPLACES=False,CE_JLAT_ENC_DIM=0,CE_REFHOLD=$1"; }

rh()  { echo "EXTRA_ARGS=--environment.command.tracking_clip_reference_hold=$1"; }
rh2() { echo "EXTRA_ARGS=--environment.command.tracking_clip_reference_hold=$1:--environment.seed=2"; }

echo "########## V1 RATE CURVE ##########"

# arm spec: name|routing(REF/T4/T1)|hold|seed2(0/1)|chain
# Chains pair a cell's two runs; the two hold=1 backfills and the bridge share
# chains with light partners so no chain exceeds 2 trains.
run_chain() {  # run_chain <spec lines...> (newline-separated on stdin)
  local PREV="" line name rt hold s2 EXPV CEV RHV
  while IFS='|' read -r name rt hold s2; do
    [ -z "$name" ] && continue
    case "$rt" in
      REF) EXPV="$REF"; CEV=$(ce_ref "$hold");;
      T4)  EXPV="$T4";  CEV=$(ce_t4  "$hold");;
      T1)  EXPV="$T1";  CEV=$(ce_t1  "$hold");;
    esac
    if [ "$s2" = "1" ]; then RHV=$(rh2 "$hold"); else RHV=$(rh "$hold"); fi
    J=$(sub "$name" "$PREV" "$B,$EXPV,$RHV")
    echo "$name -> $J"
    ce4 "$name" "$J" "$CEV"
    PREV="$J"
  done
}

run_chain <<'C1'
v1h2_ref_s1|REF|2|0
v1h2_ref_s2|REF|2|1
C1
run_chain <<'C2'
v1h2_tok_s1|T4|2|0
v1h2_tok_s2|T4|2|1
C2
run_chain <<'C3'
v1h5_ref_s1|REF|5|0
v1h5_ref_s2|REF|5|1
C3
run_chain <<'C4'
v1h5_tok_s1|T4|5|0
v1h5_tok_s2|T4|5|1
C4
run_chain <<'C5'
v1h10_ref_s1|REF|10|0
v1h10_ref_s2|REF|10|1
C5
run_chain <<'C6'
v1h10_tok_s1|T4|10|0
v1h10_tok_s2|T4|10|1
C6
run_chain <<'C7'
v1h5_tokd1_s1|T1|5|0
v1h1_ref_s2|REF|1|1
C7
run_chain <<'C8'
v1h1_tok_s2|T4|1|1
C8

echo "########## SUBMITTED ##########"
