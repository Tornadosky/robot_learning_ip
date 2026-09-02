#!/usr/bin/env bash
# ONE COMMAND after the PC restart -- relaunches everything local for night5.
# Run from Git Bash or WSL:   bash scripts/scaling/night5_restart.sh
#
# What it starts (all detached, all resumable):
#   1. tokenizer_3t_v2 on the GPU (WSL jaxgpu; ~30-60 min instead of CPU 7.5h)
#      -> emits clips_3t_v2 sidecars when done
#   2. local_night5.sh orchestrator (waits for those sidecars, then
#      ln5_ref -> ln5_tok -> 4-seed eval + render dumps)
#   3. tokenizer_mb on the CPU (5-family multibody codec, parallel, no GPU)
#
# Then: reopen Claude (claude --continue) and say "restarted" so the ops cron
# gets re-armed -- the session-level cron died with the old session.
set -eu
REPO_WIN="C:/Users/smirn/Desktop/robot_learning_ip"
REPO_WSL="/mnt/c/Users/smirn/Desktop/robot_learning_ip"

wsl -d Ubuntu -- bash -c "cd $REPO_WSL && setsid nohup bash scripts/scaling/train_tokenizer_3t_v2_gpu.sh > experiments/fsq_khaendler/_tok_logs/tok3t_v2_gpu.log 2>&1 < /dev/null & echo tokenizer_v2_gpu launched"
wsl -d Ubuntu -- bash -c "cd $REPO_WSL && setsid nohup bash scripts/scaling/local_night5.sh > experiments/local_3t/ln5_run.log 2>&1 < /dev/null & echo ln5_orchestrator launched"
# CPU multibody tokenizer from Git Bash side (Windows venv). If this script is
# itself running under WSL, launch it through cmd so the Windows python runs.
if grep -qi microsoft /proc/version 2>/dev/null; then
  (cd "$REPO_WSL" && setsid nohup cmd.exe /c "bash scripts/scaling/train_tokenizer_multibody.sh > experiments/fsq_khaendler/_tok_logs/tok_mb.log 2>&1" < /dev/null > /dev/null 2>&1 &)
else
  (cd "$REPO_WIN" && nohup bash scripts/scaling/train_tokenizer_multibody.sh > experiments/fsq_khaendler/_tok_logs/tok_mb.log 2>&1 < /dev/null &)
fi
echo "tokenizer_mb (CPU) launched"
echo ""
echo "ALL LAUNCHED. Now run:  claude --continue   and say: restarted"
