#!/usr/bin/env bash
# rsync the wave-7 data from BOX-A (this WSL) to BOX-B (RTX 5080). Resumable; run from WSL:
#   bash scripts/scaling/wave7/sync_boxb.sh
set -u
REPO=${REPO:-/mnt/c/Users/smirn/Desktop/robot_learning_ip}
B=${BOXB:-melo@192.168.178.41}
D=${BOXB_DIR:-/home/melo/Projects/ip_project/robot_learning_ip}
cd "$REPO"
ssh $B "mkdir -p $D/experiments/fsq_khaendler $D/external_data/amass_converted $D/handoff"
rsync -a --partial --info=progress2 experiments/fsq_khaendler/tokenizer_m20 experiments/fsq_khaendler/tokenizer_3t_v2 experiments/fsq_khaendler/clips_m20 experiments/fsq_khaendler/clips_3t_v2 experiments/fsq_khaendler/clips_5r $B:$D/experiments/fsq_khaendler/
rsync -a --partial --info=progress2 external_data/amass_converted/LAFAN1_allfix external_data/amass_converted/LAFAN1_all $B:$D/external_data/amass_converted/
rsync -a handoff/ $B:$D/handoff/
echo "=== SYNC DONE $(date -Is)"
