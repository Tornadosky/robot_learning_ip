#!/usr/bin/env bash
# GEN-B' -- BoosterT1 as a HELD-OUT BODY, on the regenerated reference.
#
# `best79` trained on H1+G1 and dance2_subject4 only. BoosterT1 is the one body
# it never saw that is also CORRECTLY MAPPED onto its loco_mjx model
# (derive_clip_signs residual 1.5e-4, 23/23 joints) -- Atlas and Talos are not
# (0.043 / 0.830) and unitree_h1v2 has no loco_mjx model at all, which is why
# the first GEN-B produced nothing. Scored against LAFAN1_fixed, because the
# shipped T1 clip is unreachable.
#
# TWO BUGS IN THE FIRST ATTEMPT, both fixed here:
#
#  1. ARTIFACT COLLISION. crosseval2.sbatch writes ${EXP}__${CE_TAG}.json. The
#     first version varied only the SLURM job NAME and passed CE_TAG=s<seed>, so
#     all twelve cells -- policy AND zero-action floor -- wrote the same four
#     filenames. The floor would have overwritten the policy. This is the exact
#     shape of the collision that destroyed the 2026-08-24 FSQ control. CE_TAG
#     now carries the cell identity.
#  2. TWELVE AT ONCE. All twelve were submitted dependency-free and ran
#     concurrently beside wave-7 training; after 1h12m they had produced three
#     lines of log and zero artifacts, against a ~7 min norm. Suspected GPU
#     contention. So: PROBE=1 submits ONE and nothing else, to be timed before
#     committing the rest, and the full set is chained rather than parallel.
#
#   PROBE=1  submit a single policy cell and stop
#   DRY=1    print, submit nothing
set -u
ROOT=/ptmp/akalenik/urma
FIX=$ROOT/clips/LAFAN1_fixed
OUTDIR=$ROOT/crosseval_gen
DRY="${DRY:-0}"
PROBE="${PROBE:-0}"

# ce <cell-tag> <arm> <seed> <zero:0|1> <dep-jobid|"">
ce() {
  local tag="$1" arm="$2" sd="$3" zero="$4" dep="$5"
  local extra=""
  [ "$zero" = "1" ] && extra=",CE_ZERO=1"
  local d=(); [ -n "$dep" ] && d=(--dependency="afterany:$dep")
  if [ "$DRY" = "1" ]; then
    echo "CE $arm tag=$tag seed=$sd zero=$zero dep=$dep -> ${arm}__${tag}.json" >&2
    echo "FAKE"; return
  fi
  sbatch --parsable -J "$tag" ${d[@]+"${d[@]}"} \
    --export=ALL,"EXP=$arm,CE_TAG=$tag,CE_SEED=$sd,CE_REFBIAS=0.0,CE_ANCHOR=absolute,CE_FITVARIANT=False,CE_REFROOT=True,CE_REFROOT_FLOOR=True,CE_REFVEL_OBS=False,CE_HEADING_OBS=False,CE_ROBOTS=booster_t1,CLIP_DIR=$FIX,CE_RAW_DIR=$FIX,CE_CLIP=dance2_subject4.npz,CE_OUTDIR=$OUTDIR$extra" \
    "$ROOT/crosseval2.sbatch"
}

if [ "$PROBE" = "1" ]; then
  j=$(ce "b2probe_best79_s1_0" best79_s1 0 0 "")
  echo "PROBE submitted: $j  -> $OUTDIR/best79_s1__b2probe_best79_s1_0.json"
  echo "time it against the ~7 min norm before submitting the rest"
  exit 0
fi

# Chained, not parallel: one cell at a time, so twelve of them cannot contend
# with each other or with wave-7 training. afterany, so one bad cell does not
# stall the rest.
PREV=""
for k in 0 1 2 3; do
  for arm in best79_s1 best79_s2; do
    PREV=$(ce "b2_${arm}_$k" "$arm" "$k" 0 "$PREV")
    echo "policy $arm seed$k -> $PREV"
  done
  PREV=$(ce "b2_zero_$k" best79_s1 "$k" 1 "$PREV")
  echo "floor  seed$k -> $PREV"
done
echo "GEN-B' SUBMITTED (12 cells, chained)"
