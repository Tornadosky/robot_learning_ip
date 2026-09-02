#!/usr/bin/env bash
# NIGHT3 (2026-08-30 -> 08-31) -- the overnight Viper campaign toward the paper
# (docs/notes/PAPER_ROADMAP_2026-08-30.md). Six campaigns, priority order:
#
#   G   F1+F2 gait dose grid at hold=5 on the practical recipe. THE fix list's
#       load-bearing items: FOOTH re-dose (ratio 0.33/0.5 at temp 0.05 vs the
#       "glues feet" 1.0/0.01) x contact bundle (CONTACT_TIMECONST=0.004 +
#       GROUNDPEN back to 1000). Baseline cell (FOOTH=0, old contact) is the
#       existing W1 h5 pair -- not re-run. 5 cells x {ref,tok} x 2 seeds.
#   A4G the "does URMA+FSQ solve a single DANCE" arm: 5-dance super-clip at
#       hold=1 WITH the mid gait dose (fh 0.3333/0.05 + contact bundle) so the
#       renders answer the visual question, not just RMSE. + dump CEs.
#   A4  multi-clip generalization at the baseline dose (comparable to the
#       W1/W2 single-clip curves): super5dance {ref,tok} x hold {1,5}.
#   A2  token-only (LATENT_REPLACES) at 2x and 3x budget: is the +14% penalty
#       an information gap or a compute gap?
#   A1  LATENT_HOLD sweep (hold the TOKEN -- never done): token-only at token
#       hold {5,20}; token+ref (fresh ref) at token hold 5.
#   S   seed-3 replication of the headline W1/W2 cells (n=2 -> n=3).
#
# 49 trains + 196 CEs + 6 render-dump jobs = 251 (MaxSubmit 300, MaxJobs 8).
# CEs go through crosseval_token2.sbatch (adds CE_CONTACT_TIMECONST, CE_HOLD,
# CE_DUMP) -- requires patch_ce_timeconst.py applied to crosseval_motion.py.
#
#   DRY=1  print, submit nothing
set -u
ROOT=/ptmp/akalenik/urma
cd "$ROOT"
DRY="${DRY:-0}"

TOKMC=$ROOT/clips/tokentest_mc
SUPER=$ROOT/clips/clips_super
LAF=$ROOT/clips/LAFAN1
XLA="--xla_gpu_enable_command_buffer="
CE2=$ROOT/crosseval_token2.sbatch

# The practical recipe, bit-identical to submit_w1/w2's BM except the five
# parametrized knobs: clip dir/file, total steps, and the night3 dose axes
# (FOOTH, FOOTH_TEMP, GROUNDPEN, CONTACT_TIMECONST).
base() {  # base <clipdir> <clipfile> <total_steps> <footh> <ftemp> <gpen> <tc>
  local b="STAGE=mmtrain,NR_ENVS=768,TOTAL_STEPS=$3,SAVE_EVERY=1966080"
  b="$b,CLIP_FILE=$2,CLIP_DIR=$1"
  b="$b,REFVEL_OBS=False,REFROOT=True,REFROOT_FLOOR=True,REFBIAS=0.0"
  b="$b,TRACK_COEFF=30.0,TRACK_TEMP=0.05,FITVARIANT=False,ANCHOR=absolute"
  b="$b,GAITMODE=floor,GAITCOEFF=0.25,QVEL_TEMP=10"
  b="$b,FOOTZVEL=1.0,FOOTH_TEMP=$5,FOOTH=$4"
  b="$b,FOOTSLIP=6.6667,GROUNDPEN=$6,POSTCONTACT=True"
  b="$b,CONTACT_TIMECONST=$7"
  b="$b,HEADING_RATIO=0.20,HEADING_TEMP=2.0"
  b="$b,XLA_EXTRA=$XLA"
  echo "$b"
}

REF="LATENT_OBS=False,JLAT_ENC_DIM=0"
T4="LATENT_OBS=True,LATENT_DIM=32,LATENT_REPLACES=False,LATENT_SCOPE=per_joint,LATENT_DIVISOR=10.0,JLAT_ENC_DIM=4"
TREP="LATENT_OBS=True,LATENT_DIM=32,LATENT_REPLACES=True,LATENT_SCOPE=per_joint,LATENT_DIVISOR=10.0,JLAT_ENC_DIM=4"
HOBS="--environment.command.tracking_clip_observe_root_heading=True"

sub() {  # sub <name> <dep-or-empty> <exports> -> job id
  local name="$1" dep="$2" exp="$3"
  local d=(); [ -n "$dep" ] && [ "$dep" != "FAKE" ] && d=(--dependency="afterany:$dep")
  if [ "$DRY" = "1" ]; then echo "TRAIN $name dep=${dep:-none}" >&2; echo "  $exp" >&2; echo "FAKE"; return; fi
  sbatch --parsable -J "$name" ${d[@]+"${d[@]}"} --export=ALL,"EXP_NAME=$name,$exp" "$ROOT/viper_train.sbatch"
}
# ce <arm> <train-jid> <ce-seed> <clipdir> <rawdir> <clipfile> <extra exports>
ce() {
  local arm="$1" jid="$2" sd="$3" cdir="$4" rdir="$5" cf="$6" extra="$7"
  local d=(); [ -n "$jid" ] && [ "$jid" != "FAKE" ] && d=(--dependency="afterok:$jid")
  [ "$DRY" = "1" ] && { echo "  CE $arm s$sd clip=$cf $extra" >&2; return; }
  sbatch --parsable -J "ce_$arm" ${d[@]+"${d[@]}"} \
    --export=ALL,"EXP=$arm,CLIP_DIR=$cdir,CE_RAW_DIR=$rdir,CE_CLIP=$cf,CE_TAG=s$sd,CE_SEED=$sd,CE_REFBIAS=0.0,CE_ANCHOR=absolute,CE_FITVARIANT=False,CE_REFROOT=True,CE_REFROOT_FLOOR=True,CE_REFVEL_OBS=False,$extra" \
    "$CE2" > /dev/null
}
ce4() { for s in 0 1 2 3; do ce "$1" "$2" "$s" "$3" "$4" "$5" "$6"; done; }
# dump CE: one extra eval that also writes render npz-s (CE_SEED=0, TAG=dump)
cedump() {
  local arm="$1" jid="$2" cdir="$3" rdir="$4" cf="$5" extra="$6"
  local d=(); [ -n "$jid" ] && [ "$jid" != "FAKE" ] && d=(--dependency="afterok:$jid")
  [ "$DRY" = "1" ] && { echo "  CEDUMP $arm $extra" >&2; return; }
  sbatch --parsable -J "ced_$arm" ${d[@]+"${d[@]}"} \
    --export=ALL,"EXP=$arm,CLIP_DIR=$cdir,CE_RAW_DIR=$rdir,CE_CLIP=$cf,CE_TAG=dump,CE_SEED=0,CE_DUMP=1,CE_REFBIAS=0.0,CE_ANCHOR=absolute,CE_FITVARIANT=False,CE_REFROOT=True,CE_REFROOT_FLOOR=True,CE_REFVEL_OBS=False,$extra" \
    "$CE2" > /dev/null
}

ce_ref()  { echo "CE_LATENT=0,CE_JLAT_ENC_DIM=0,CE_REFHOLD=$1,CE_HEADING_OBS=True"; }
ce_t4()   { echo "CE_LATENT=1,CE_SCOPE=per_joint,CE_DIVISOR=10.0,CE_REPLACES=False,CE_JLAT_ENC_DIM=4,CE_REFHOLD=$1,CE_HEADING_OBS=True"; }
ce_trep() { echo "CE_LATENT=1,CE_SCOPE=per_joint,CE_DIVISOR=10.0,CE_REPLACES=True,CE_JLAT_ENC_DIM=4,CE_REFHOLD=$1,CE_HEADING_OBS=True"; }

ea() {  # ea <refhold> <seed>  (seed 1 = trainer default, no flag)
  local v="--environment.command.tracking_clip_reference_hold=$1"
  [ "$2" != "1" ] && v="$v:--environment.seed=$2"
  echo "EXTRA_ARGS=$v:$HOBS"
}

# chain2 <cell> <base> <expv> <ce_extra> <refhold> <cdir> <rdir> <clip> [extra-exports] -> echoes s1 jid
chain2() {
  local cell="$1" b="$2" expv="$3" cex="$4" hold="$5" cdir="$6" rdir="$7" cf="$8" xx="${9:-}" PREV="" s J J1=""
  for s in 1 2; do
    local name="${cell}_s${s}"
    local exports="$b,$expv,$(ea "$hold" "$s")"
    [ -n "$xx" ] && exports="$exports,$xx"
    J=$(sub "$name" "$PREV" "$exports")
    echo "$name -> $J" >&2
    ce4 "$name" "$J" "$cdir" "$rdir" "$cf" "$cex"
    [ "$s" = "1" ] && J1="$J"
    PREV="$J"
  done
  echo "$J1"
}

DANCE=dance2_subject4.npz
SUPERC=super5dance.npz

echo "########## G -- F1+F2 gait dose grid (hold=5, dance2_subject4) ##########"
# cell name : FOOTH FTEMP GPEN TC
gcell() {  # gcell <cell> <footh> <ftemp> <gpen> <tc> -> s1 jid
  local cell="$1" fh="$2" ft="$3" gp="$4" tc="$5"
  local b; b=$(base "$TOKMC" "$DANCE" 19660800 "$fh" "$ft" "$gp" "$tc")
  local ceref cetok
  ceref="$(ce_ref 5),CE_CONTACT_TIMECONST=$tc"
  cetok="$(ce_t4 5),CE_CONTACT_TIMECONST=$tc"
  R1=$(chain2 "${cell}_ref" "$b" "$REF" "$ceref" 5 "$TOKMC" "$LAF" "$DANCE")
  T1=$(chain2 "${cell}_tok" "$b" "$T4"  "$cetok" 5 "$TOKMC" "$LAF" "$DANCE")
}
gcell n3gct  0.0    0.01 1000   0.004   # contact bundle only
gcell n3gf33 0.3333 0.05 333.33 0.0     # FOOTH re-dose only, mid
gcell n3gf50 0.5    0.05 333.33 0.0     # FOOTH re-dose only, high
gcell n3gb33 0.3333 0.05 1000   0.004   # both, mid   <- video candidate
GB33_REF_J1="$R1"; GB33_TOK_J1="$T1"
gcell n3gb50 0.5    0.05 1000   0.004   # both, high
cedump n3gb33_ref_s1 "$GB33_REF_J1" "$TOKMC" "$LAF" "$DANCE" "$(ce_ref 5),CE_CONTACT_TIMECONST=0.004"
cedump n3gb33_tok_s1 "$GB33_TOK_J1" "$TOKMC" "$LAF" "$DANCE" "$(ce_t4 5),CE_CONTACT_TIMECONST=0.004"

echo "########## A4G -- super-clip + gait dose, hold=1 (the DANCE question) ##########"
BA4G=$(base "$SUPER" "$SUPERC" 19660800 0.3333 0.05 1000 0.004)
J1=$(chain2 n3a4g_ref "$BA4G" "$REF" "$(ce_ref 1),CE_CONTACT_TIMECONST=0.004" 1 "$SUPER" "$SUPER" "$SUPERC")
cedump n3a4g_ref_s1 "$J1" "$SUPER" "$SUPER" "$SUPERC" "$(ce_ref 1),CE_CONTACT_TIMECONST=0.004"
J1=$(chain2 n3a4g_tok "$BA4G" "$T4" "$(ce_t4 1),CE_CONTACT_TIMECONST=0.004" 1 "$SUPER" "$SUPER" "$SUPERC")
cedump n3a4g_tok_s1 "$J1" "$SUPER" "$SUPER" "$SUPERC" "$(ce_t4 1),CE_CONTACT_TIMECONST=0.004"

echo "########## A4 -- super-clip at baseline dose, {ref,tok} x hold {1,5} ##########"
BA4=$(base "$SUPER" "$SUPERC" 19660800 0.0 0.01 333.33 0.0)
J1=$(chain2 n3a4h1_ref "$BA4" "$REF" "$(ce_ref 1)" 1 "$SUPER" "$SUPER" "$SUPERC")
cedump n3a4h1_ref_s1 "$J1" "$SUPER" "$SUPER" "$SUPERC" "$(ce_ref 1)"
J1=$(chain2 n3a4h1_tok "$BA4" "$T4" "$(ce_t4 1)" 1 "$SUPER" "$SUPER" "$SUPERC")
cedump n3a4h1_tok_s1 "$J1" "$SUPER" "$SUPER" "$SUPERC" "$(ce_t4 1)"
chain2 n3a4h5_ref "$BA4" "$REF" "$(ce_ref 5)" 5 "$SUPER" "$SUPER" "$SUPERC" > /dev/null
chain2 n3a4h5_tok "$BA4" "$T4" "$(ce_t4 5)" 5 "$SUPER" "$SUPER" "$SUPERC" > /dev/null

echo "########## A2 -- token-only at 2x / 3x budget (hold=1) ##########"
B2X=$(base "$TOKMC" "$DANCE" 39321600 0.0 0.01 333.33 0.0)
B3X=$(base "$TOKMC" "$DANCE" 58982400 0.0 0.01 333.33 0.0)
chain2 n3a2rep2x "$B2X" "$TREP" "$(ce_trep 1)" 1 "$TOKMC" "$LAF" "$DANCE" > /dev/null
chain2 n3a2rep3x "$B3X" "$TREP" "$(ce_trep 1)" 1 "$TOKMC" "$LAF" "$DANCE" > /dev/null

echo "########## A1 -- LATENT_HOLD sweep (hold the TOKEN) ##########"
BLH=$(base "$TOKMC" "$DANCE" 19660800 0.0 0.01 333.33 0.0)
chain2 n3a1rep_lh5  "$BLH" "$TREP" "$(ce_trep 1),CE_HOLD=5"  1 "$TOKMC" "$LAF" "$DANCE" "LATENT_HOLD=5"  > /dev/null
chain2 n3a1rep_lh20 "$BLH" "$TREP" "$(ce_trep 1),CE_HOLD=20" 1 "$TOKMC" "$LAF" "$DANCE" "LATENT_HOLD=20" > /dev/null
chain2 n3a1tok_lh5  "$BLH" "$T4"   "$(ce_t4 1),CE_HOLD=5"    1 "$TOKMC" "$LAF" "$DANCE" "LATENT_HOLD=5"  > /dev/null

echo "########## S -- seed-3 replication of the headline W1/W2 cells ##########"
BS=$(base "$TOKMC" "$DANCE" 19660800 0.0 0.01 333.33 0.0)
s3() {  # s3 <name> <expv> <cefn> <hold>
  local name="$1" expv="$2" cefn="$3" hold="$4"
  local J; J=$(sub "$name" "" "$BS,$expv,$(ea "$hold" 3)")
  echo "$name -> $J" >&2
  ce4 "$name" "$J" "$TOKMC" "$LAF" "$DANCE" "$($cefn "$hold")"
}
s3 w1h1_ref_s3  "$REF"  ce_ref  1
s3 w1h1_tok_s3  "$T4"   ce_t4   1
s3 w1h5_ref_s3  "$REF"  ce_ref  5
s3 w1h5_tok_s3  "$T4"   ce_t4   5
s3 w2h20_ref_s3 "$REF"  ce_ref  20
s3 w2h20_tok_s3 "$T4"   ce_t4   20
s3 w2rep_h1_s3  "$TREP" ce_trep 1

echo "########## SUBMITTED ##########"
squeue -u akalenik -h | wc -l
