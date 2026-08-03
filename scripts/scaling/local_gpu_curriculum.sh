set -u
cd /mnt/c/Users/smirn/Desktop/robot_learning_ip
export PYTHONPATH=$PWD/scripts
export XLA_PYTHON_CLIENT_PREALLOCATE=false
PY=$HOME/jaxgpu/bin/python
AWARE=MorphologyAwareRootPoseTrajTerminalStateHandler
R=experiments/scaling_1000
LOG=/tmp/curriculum.log
: > $LOG

# Naive wide-bounds training collapsed (length ~61 after 300M). The roadmap's own
# advice is to widen progressively, so each stage warm-starts from the previous.
# NOTE: filter must show failures too - an earlier version grepped only for
# '^[online]' and silently hid a TypeError, making a crash look like success.
stage () {
  local name=$1 catalog=$2 init=$3
  echo "=== STAGE $name START $(date +%H:%M:%S) ==="
  $PY scripts/scaling/online_h1_train.py \
    --catalog "$catalog" --catalog-mode fixed_balanced \
    --terminal-handler $AWARE --init-checkpoint "$init" \
    --num-envs 8000 --num-steps 64 --total-timesteps 150000000 \
    --num-minibatches 32 --update-epochs 4 --hidden 256 128 --seed 1 \
    --run-tag curriculum_$name --output-dir $R/curriculum/$name >>$LOG 2>&1
  local rc=$?
  grep -E '^\[online\]|Error|Traceback|error:' $LOG | tail -6
  if [ $rc -ne 0 ]; then echo "STAGE $name FAILED rc=$rc"; tail -20 $LOG; return 1; fi
  echo "=== STAGE $name DONE $(date +%H:%M:%S) ==="
}

stage mid  $R/catalogs_mid/train_1000.json \
  $R/walk_continuous_control/s1_10806778/checkpoint_final/PPOJax_saved.pkl || exit 1
stage wide $R/catalogs_wide/train_1000.json \
  $R/curriculum/mid/checkpoint_final/PPOJax_saved.pkl || exit 1
echo "CURRICULUM_COMPLETE $(date +%H:%M:%S)"
