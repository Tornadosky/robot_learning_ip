#!/usr/bin/env bash
# TOKEN ROUTING -- the quick test of WHERE the FSQ code should enter the policy.
#
# WHY THIS CAMPAIGN EXISTS. Every design-B arm to date paid 8-26% and the cost
# was independent of the tokenizer, of M, and of the token rate. Two facts found
# by reading the code on 2026-08-29 give a mechanism that no arm has controlled
# for:
#
#  1. urma's joint encoder is ONE `nn.Dense(4)` over the joint's channels
#     (algorithms/urma/mjx/policy.py). Per-joint token conditioning takes that
#     input from 5 channels to 37 and pushes all of them through the same 4
#     units that must also carry position, velocity, previous action and the
#     reference.
#  2. environment.py normalises every observation group in a `set_norm` table --
#     position /6.5, velocity /180, previous action /10 -- except the latent
#     block, which had NO entry because it had no index list. Measured on
#     clips_kevin/UnitreeH1/dance2_subject4_zq.npz the codes have std 0.508
#     against post-norm neighbours at 0.012-0.100, so the 32-channel block
#     enters with ~559x the input energy of all five real channels combined.
#
# The existing evidence already leans this way: under a "the token is redundant
# with the reference" account, token-ONLY should beat reference+token. Measured,
# it is worse (+26.4/+12.7% against +17.0/+8.0%). A bottleneck account predicts
# exactly that ordering; a redundancy account cannot produce it.
#
# ARMS. One control and four routings, everything else identical.
#   TK-0 ctl     no token at all -- the wave-7-era control, on this code.
#   TK-1 perj1   per-joint, divisor 1.0, shared Dense(4). BIT-IDENTICAL to every
#                design-B arm ever run. It is here so the campaign contains its
#                own reproduction of the effect it claims to explain.
#   TK-2 perj10  per-joint, divisor 10.0. Isolates NORMALISATION alone.
#   TK-3 glob    one pooled motion embedding in the general observation, which
#                reaches the 512-wide core with no bottleneck. This is the
#                "policy conditioned on a motion embedding" formulation.
#   TK-4 perjsep per-joint, divisor 10.0, and the token gets its OWN projection
#                beside the base Dense(4). Isolates the BOTTLENECK alone.
#
# REGISTERED PREDICTIONS, before the arms land:
#   * TK-1 reproduces roughly the historical penalty against TK-0.
#   * TK-2 and TK-4 each recover a large part of it; if BOTH are nulls, the
#     bottleneck/scale account is dead and the cost is genuinely the token's
#     content, which would close design B for good.
#   * TK-3 is the cheapest routing and should be at or near TK-0. If TK-3 is
#     ALSO negative, then no routing saves a per-frame motion code here.
# A null on TK-2/TK-3/TK-4 is a publishable negative; it is not a failure.
#
# SCOPE. Two topologies (H1+G1), one seed, 19.7M steps: this is a DIRECTION
# test, not a verdict. Rollout seed alone moves a crosseval 4.95% on this stack,
# which is the size of the effect, so the verdict needs the 4-seed crossevals
# queued below and a longer arm afterwards. Three topologies stay LOCAL: the
# fused urma2.mjx path faults on ROCm at every env count with 3 robots.
#
#   ONLY=0|1|2|3|4   submit one arm (default: all)
#   DRY=1            print, submit nothing
set -u
ROOT=/ptmp/akalenik/urma
cd "$ROOT"
DRY="${DRY:-0}"
ONLY="${ONLY:-all}"
want() { [ "$ONLY" = "all" ] || [ "$ONLY" = "$1" ]; }

TOK=$ROOT/clips/tokentest      # ORIGINAL clip + _zq.npz sidecar, NOT the
LAF=$ROOT/clips/LAFAN1         # reconstruction -- else the arm measures design
XLA="--xla_gpu_enable_command_buffer="   # A and design B at the same time.

# Wave-7's B4 base, unchanged, so this campaign sits on a known operating point.
B="STAGE=mmtrain,NR_ENVS=768,TOTAL_STEPS=19660800,SAVE_EVERY=1966080"
B="$B,CLIP_FILE=dance2_subject4.npz,CLIP_DIR=$TOK"
B="$B,REFVEL_OBS=False,REFROOT=True,REFROOT_FLOOR=True,REFBIAS=0.0"
B="$B,TRACK_COEFF=30.0,TRACK_TEMP=0.05,FITVARIANT=False,ANCHOR=absolute"
B="$B,GAITMODE=floor,GAITCOEFF=0.25,QVEL_TEMP=10"
B="$B,FOOTZVEL=1.0,FOOTH_TEMP=0.01,FOOTH=0.3333"
B="$B,FOOTSLIP=20.0,GROUNDPEN=1000,POSTCONTACT=True"
B="$B,XLA_EXTRA=$XLA"

sub() {
  local name="$1" exp="$2"
  # DRY output goes to STDERR: this function's STDOUT is the job id.
  if [ "$DRY" = "1" ]; then echo "TRAIN $name" >&2; echo "  $exp" >&2; echo "FAKE"; return; fi
  sbatch --parsable -J "$name" --export=ALL,"EXP_NAME=$name,$exp" "$ROOT/viper_train.sbatch"
}
ce() {
  local arm="$1" jid="$2" sd="$3" extra="$4"
  local d=(); [ -n "$jid" ] && [ "$jid" != "FAKE" ] && d=(--dependency="afterok:$jid")
  [ "$DRY" = "1" ] && { echo "  CE $arm s$sd $extra" >&2; return; }
  sbatch --parsable -J "ce_$arm" ${d[@]+"${d[@]}"} \
    --export=ALL,"EXP=$arm,CLIP_DIR=$TOK,CE_RAW_DIR=$LAF,CE_TAG=s$sd,CE_SEED=$sd,CE_REFBIAS=0.0,CE_ANCHOR=absolute,CE_FITVARIANT=False,CE_REFROOT=True,CE_REFROOT_FLOOR=True,CE_REFVEL_OBS=False,$extra" \
    "$ROOT/crosseval_token.sbatch" > /dev/null
}
# Four seeds per arm, because a single crosseval on this stack is noise the size
# of the effect being measured.
ce4() { for s in 0 1 2 3; do ce "$1" "$2" "$s" "$3"; done; }

echo "########## TOKEN ROUTING ##########"

if want 0; then
  J=$(sub tk0_ctl "$B,LATENT_OBS=False,JLAT_ENC_DIM=0")
  ce4 tk0_ctl "$J" "CE_LATENT=0"
fi
if want 1; then
  J=$(sub tk1_perj1 "$B,LATENT_OBS=True,LATENT_DIM=32,LATENT_REPLACES=False,LATENT_SCOPE=per_joint,LATENT_DIVISOR=1.0,JLAT_ENC_DIM=0")
  ce4 tk1_perj1 "$J" "CE_LATENT=1,CE_SCOPE=per_joint,CE_DIVISOR=1.0,CE_REPLACES=False,CE_JLAT_ENC_DIM=0"
fi
if want 2; then
  J=$(sub tk2_perj10 "$B,LATENT_OBS=True,LATENT_DIM=32,LATENT_REPLACES=False,LATENT_SCOPE=per_joint,LATENT_DIVISOR=10.0,JLAT_ENC_DIM=0")
  ce4 tk2_perj10 "$J" "CE_LATENT=1,CE_SCOPE=per_joint,CE_DIVISOR=10.0,CE_REPLACES=False,CE_JLAT_ENC_DIM=0"
fi
if want 3; then
  J=$(sub tk3_glob "$B,LATENT_OBS=True,LATENT_DIM=32,LATENT_REPLACES=False,LATENT_SCOPE=global,LATENT_DIVISOR=1.0,JLAT_ENC_DIM=0")
  ce4 tk3_glob "$J" "CE_LATENT=1,CE_SCOPE=global,CE_DIVISOR=1.0,CE_REPLACES=False,CE_JLAT_ENC_DIM=0"
fi
if want 4; then
  J=$(sub tk4_perjsep "$B,LATENT_OBS=True,LATENT_DIM=32,LATENT_REPLACES=False,LATENT_SCOPE=per_joint,LATENT_DIVISOR=10.0,JLAT_ENC_DIM=4")
  ce4 tk4_perjsep "$J" "CE_LATENT=1,CE_SCOPE=per_joint,CE_DIVISOR=10.0,CE_REPLACES=False,CE_JLAT_ENC_DIM=4"
fi

echo "########## SUBMITTED ##########"
