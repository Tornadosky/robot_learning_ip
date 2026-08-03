set -u
cd /mnt/c/Users/smirn/Desktop/robot_learning_ip
export PYTHONPATH=$PWD/scripts
export XLA_PYTHON_CLIENT_PREALLOCATE=false
PY=$HOME/jaxgpu/bin/python
AWARE=MorphologyAwareRootPoseTrajTerminalStateHandler
R=experiments/scaling_1000
for S in 1 2; do
  $PY scripts/scaling/evaluate_catalog_policy.py \
    --checkpoint $R/wide_walk_fixed/s$S/checkpoint_final/PPOJax_saved.pkl \
    --catalogs $R/catalogs_wide/iid_256.json $R/catalogs_wide/boundary_256.json \
    --replicas 4 --horizon 1000 --terminal-handler $AWARE \
    --output $R/evaluations/wide_walk_s$S.json
done
for S in 1 2; do
  $PY scripts/scaling/evaluate_online_amp_policy.py \
    --checkpoint $R/online_amp/amp_1000x4_s$S/checkpoint_final/OnlineMorphAMP_saved.pkl \
    --replicas 8 --horizon 1000 --heldout-clips dance2_subject5 \
    --output $R/evaluations/amp_1000x4_s$S.json
done
echo EVAL_PIPELINE_COMPLETE
