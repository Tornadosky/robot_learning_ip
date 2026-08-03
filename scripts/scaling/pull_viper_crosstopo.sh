#!/bin/bash
# Pull finished Viper cross-topology runs (manifest + checkpoint + slurm log)
# into experiments/cross_embodiment/ so the local GPU can evaluate them.
#
# Only runs whose manifest.json exists are pulled: the trainer writes the
# manifest last, so its presence means training completed rather than crashed.
set -u

cd /mnt/c/Users/smirn/Desktop/robot_learning_ip
DEST=experiments/cross_embodiment
REMOTE=viper11:/ptmp/akalenik/frontier
mkdir -p "$DEST" "$DEST/viper_logs"

finished=$(ssh viper11 'for d in /ptmp/akalenik/frontier/crosstopo_*/; do [ -f "$d/manifest.json" ] && basename "$d"; done' 2>/dev/null)
if [[ -z "$finished" ]]; then
  echo "[pull] no finished Viper runs yet"
  exit 0
fi

for run in $finished; do
  if [[ -f "$DEST/$run/manifest.json" ]]; then
    echo "[pull] SKIP $run (already local)"
    continue
  fi
  echo "[pull] $run"
  rsync -az "$REMOTE/$run" "$DEST/" || echo "[pull] FAILED $run"
done

# Slurm logs are the record of what actually ran, including the segfaults.
rsync -az "$REMOTE/"crosstopo_*.out "$DEST/viper_logs/" 2>/dev/null || true
echo "[pull] done $(date -Is)"
ls -d "$DEST"/crosstopo_* 2>/dev/null
