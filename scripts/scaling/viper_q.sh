#!/bin/bash
# Run a command on Viper through the WSL ssh mux. Usage: viper_q.sh <remote command...>
set -euo pipefail
HOST="${VIPER_HOST:-viper11}"
exec wsl.exe -d Ubuntu -- bash -lc "ssh $HOST \"$*\""
