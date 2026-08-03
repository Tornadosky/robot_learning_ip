#!/bin/bash
# Submit the matched cross-topology architecture comparison to Viper.
#
#   ssh viper11 "cd /ptmp/akalenik/frontier/repo && STEPS=100000000 bash scripts/scaling/submit_viper_crosstopo_batch.sh"
#
# Six jobs, never more: the account is shared and saturating all eight APU slots
# blocks colleagues.  Jobs 5-6 are the single-topology H1 controls that separate
# "URMA is worse in general" from "URMA is worse only where there is no topology
# variation to exploit"; without them the comparison is uninterpretable.
set -u

STEPS=${STEPS:-100000000}
ENVS=${ENVS:-4096}
# ROCm: URMAv2 with 4-epoch nested updates segfaults *after* a successful
# compile (reproduced 2026-08-02, job 10811782, "compile complete in 1535.8s"
# then SIGSEGV).  The masked MLP survives 4 epochs on the same node, so the
# fault is specific to the URMA update.  Both arms therefore run at 1 epoch, so
# the architecture comparison stays matched rather than confounded by optimiser
# epochs.  CUDA runs 4 epochs fine; the local queue keeps that configuration.
EPOCHS=${EPOCHS:-1}
SB=scripts/scaling/viper_cross_topology.sbatch

running=$(squeue -u "$USER" --noheader | wc -l)
if [[ "$running" -gt 2 ]]; then
  echo "Refusing to submit: $running of our jobs already queued/running." >&2
  exit 1
fi

submit() {
  local tag=$1 backbone=$2 robots=$3 onehot=$4 seed=$5
  sbatch --export=ALL,BACKBONE="$backbone",ROBOTS="$robots",ONE_HOT="$onehot",\
SEED="$seed",TOTAL_ENVS="$ENVS",TOTAL_TIMESTEPS="$STEPS",UPDATE_EPOCHS="$EPOCHS",TAG="$tag" \
    "$SB"
}

submit urmav2_x3_s1   urmav2     "h1 g1 atlas" 0 1
submit urmav2_x3_s2   urmav2     "h1 g1 atlas" 0 2
submit maskedmlp_x3_s1 masked_mlp "h1 g1 atlas" 1 1
submit maskedmlp_x3_s2 masked_mlp "h1 g1 atlas" 1 2
submit urmav2_h1_s1   urmav2     "h1"          0 1
submit maskedmlp_h1_s1 masked_mlp "h1"         1 1

echo "--- queue ---"
squeue -u "$USER"
