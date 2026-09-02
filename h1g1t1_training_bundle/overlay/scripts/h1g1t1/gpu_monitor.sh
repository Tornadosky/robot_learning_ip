#!/usr/bin/env bash
set -u
OUT=${1:?usage: gpu_monitor.sh OUTPUT.csv [INTERVAL_SECONDS]}
INTERVAL=${2:-30}
mkdir -p "$(dirname "$OUT")"
echo 'timestamp,index,name,utilization_gpu_percent,memory_used_mib,memory_total_mib,temperature_c,power_w' > "$OUT"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "$(date -Is),n/a,nvidia-smi-not-found,n/a,n/a,n/a,n/a,n/a" >> "$OUT"
  exit 0
fi
while true; do
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
    --format=csv,noheader,nounits 2>/dev/null | while IFS= read -r line; do
      printf '%s,%s\n' "$(date -Is)" "$line" >> "$OUT"
    done
  sleep "$INTERVAL" || exit 0
done
