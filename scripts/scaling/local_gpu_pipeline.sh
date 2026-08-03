set -u
cd /mnt/c/Users/smirn/Desktop/robot_learning_ip
export PYTHONPATH=$PWD/scripts
# Leave most of the card free for the user's other work; peak live need is ~2 GiB.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
PY=$HOME/jaxgpu/bin/python
AWARE=MorphologyAwareRootPoseTrajTerminalStateHandler
R=experiments/scaling_1000

run () { echo "=== $1 START $(date +%H:%M:%S) ==="; shift; "$@" 2>&1 | grep -E '^\[online|^\[online-amp'; echo "=== DONE $(date +%H:%M:%S) ==="; }

for S in 1 2; do
  run "wide_walk_s$S" $PY scripts/scaling/online_h1_train.py \
    --catalog $R/catalogs_wide/train_1000.json --catalog-mode fixed_balanced \
    --terminal-handler $AWARE \
    --num-envs 8000 --num-steps 64 --total-timesteps 300000000 \
    --num-minibatches 32 --update-epochs 4 --hidden 256 128 --seed $S \
    --run-tag wide_walk_s$S --output-dir $R/wide_walk_fixed/s$S
done

for S in 1 2; do
  run "amp_1000x4_s$S" $PY scripts/scaling/online_amp_train.py \
    --catalog $R/catalogs/train_1000.json --catalog-mode catalog_resample \
    --num-envs 8000 --num-steps 64 --total-timesteps 100000000 \
    --num-minibatches 32 --update-epochs 4 --n-disc-epochs 2 --disc-lr 1e-5 \
    --hidden 256 128 --seed $S \
    --run-tag amp_1000x4_s$S --output-dir $R/online_amp/amp_1000x4_s$S
done

run "ablation_fullmimic_1body" $PY scripts/scaling/online_amp_train.py \
  --catalog $R/catalogs/train_1.json --catalog-mode fixed_balanced \
  --full-mimic-reward \
  --num-envs 1024 --num-steps 64 --total-timesteps 100000000 \
  --num-minibatches 32 --update-epochs 4 --n-disc-epochs 10 --disc-lr 5e-5 \
  --hidden 256 128 --seed 1 \
  --run-tag ablation_fullmimic_1body --output-dir $R/online_amp/ablation_fullmimic_1body

echo "PIPELINE_COMPLETE $(date +%H:%M:%S)"
