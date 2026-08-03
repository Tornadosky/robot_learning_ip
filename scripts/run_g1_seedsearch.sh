#!/usr/bin/env bash
# Reliably land a CONVERGED G1 nominal crosser, GPU-efficiently.
#
# The balance transition is stochastic (~1 in 3 single-seed 3x-gain runs cross) and
# the GPU is throughput-bound (~5.5M env-steps/min), so parallel vmap seeds cost ~N x
# wall-clock just like sequential -- no free lunch. Instead we run single seeds (the
# proven 2048-env recipe) sequentially and EARLY-KILL non-crossers: a crosser is
# obvious by the 90M checkpoint (len ~635) vs a stuck seed (len ~100). A failed seed
# therefore costs ~17 min, not a full 55. Expected ~80-90 min to a converged crosser.
set -u
source ~/miniconda3/bin/activate locodm
cd /mnt/c/Users/smirn/Desktop/robot_learning_ip/scripts
export XLA_FLAGS='--xla_gpu_triton_gemm_any=True'
LOGDIR=/mnt/c/Users/smirn/Desktop/robot_learning_ip/logs/g1_experiments
mkdir -p "$LOGDIR"

DECIDE="3/10"      # the 90M checkpoint line ("[train] checkpoint 3/10 ...")
THRESH=350         # episode length at 90M: crosser ~635, stuck ~100
MAXSEED=6
WIN=""

for seed in $(seq 1 $MAXSEED); do
  SUF="seed${seed}"
  LOG="$LOGDIR/seedsearch_${SUF}.log"
  echo "[driver] === seed $seed start $(date) ==="
  python train_deepmimic_morphology.py \
    --robot g1 --preset nominal --clip dance2_subject4 --duration 30 \
    --num-envs 2048 --total-timesteps 300e6 --num-checkpoints 10 \
    --raw-reference --control pd --lr 3e-4 --init-std 0.2 --pd-gain-scale 3.0 \
    --seed "$seed" --out-suffix "$SUF" > "$LOG" 2>&1 &
  PID=$!

  while kill -0 "$PID" 2>/dev/null; do
    line=$(grep "checkpoint ${DECIDE}" "$LOG" | tail -1)
    if [ -n "$line" ]; then
      len=$(echo "$line" | sed 's/.*length=//' | tr -d ' ')
      cross=$(awk -v l="$len" -v t="$THRESH" 'BEGIN{print (l+0>t)?1:0}')
      if [ "$cross" = "1" ]; then
        echo "[driver] seed $seed CROSSED at 90M (len=$len) -> training to convergence"
        wait "$PID"; WIN="$SUF"
      else
        echo "[driver] seed $seed stuck at 90M (len=$len) -> killing, trying next seed"
        kill "$PID" 2>/dev/null; wait "$PID" 2>/dev/null
      fi
      break
    fi
    sleep 15
  done

  if [ -n "$WIN" ]; then break; fi
done

echo "[driver] WINNER=$WIN  $(date)"
echo ALL_DONE
