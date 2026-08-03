#!/usr/bin/env bash
# Sample GPU memory.used every 3s for N seconds, print the peak. Arg1 = seconds (default 200).
secs="${1:-200}"
peak=0
end=$(( $(date +%s) + secs ))
while [ "$(date +%s)" -lt "$end" ]; do
  cur=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  [ -n "$cur" ] && [ "$cur" -gt "$peak" ] && peak=$cur
  sleep 3
done
echo "PEAK_MEM_MiB=$peak"
