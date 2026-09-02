#!/usr/bin/env bash
# ONE command after a PC restart (run from a Windows terminal):
#   wsl -d Ubuntu -e bash /mnt/c/Users/smirn/Desktop/robot_learning_ip/scripts/scaling/wave7/after_restart.sh
# Relaunches, detached, everything the restart killed, each step resumable:
#   1. Atlas+Apollo feasibility re-issue (Windows .venv, skips finished clips)
#   2. the 5-robot clip assembly waiter (build_clips_5r.sh, waits for 1 + tokenizer)
#   3. the local overnight orchestrator (local_night7.sh: smoke -> 5-robot arms; skips done)
# Viper is unaffected by the restart.
set -u
REPO=/mnt/c/Users/smirn/Desktop/robot_learning_ip
W7=$REPO/scripts/scaling/wave7
L=$REPO/experiments/fsq_khaendler/_tok_logs
cd "$REPO"
CL="dance1_subject1 dance1_subject2 dance1_subject3 dance2_subject1 dance2_subject2 dance2_subject3 dance2_subject4 dance2_subject5 fight1_subject2 fight1_subject3 fight1_subject5 fightAndSports1_subject1 fightAndSports1_subject4 jumps1_subject1 jumps1_subject2 jumps1_subject5 run1_subject2 run1_subject5 run2_subject1 run2_subject4 sprint1_subject2 sprint1_subject4 walk1_subject1 walk1_subject2 walk1_subject5 walk2_subject1 walk2_subject3 walk2_subject4 walk3_subject1 walk3_subject2 walk3_subject3 walk3_subject4 walk3_subject5 walk4_subject1"
echo "== state: tokenizer $(grep -c 'PIPELINE DONE' $L/tokenizer_m20.log 2>/dev/null)/1, T1 re-issue $(grep -c '^OK' $L/reissue_t1_all.log 2>/dev/null)/34, Atlas+Apollo $(grep -c '^OK' $L/reissue_aa_all.log 2>/dev/null)/68, 5r READY: $([ -f experiments/fsq_khaendler/clips_5r/READY ] && echo yes || echo no)"
# 1. re-issue (T1 first if incomplete, then Atlas+Apollo); the script skips existing outputs
ARGS=$(echo $CL | sed "s/ /','/g")
if [ "$(grep -c '^OK\|^!!' $L/reissue_t1_all.log 2>/dev/null)" -lt 34 ]; then
  powershell.exe -NoProfile -Command "Start-Process -FilePath 'C:\Users\smirn\Desktop\robot_learning_ip\.venv\Scripts\python.exe' -ArgumentList 'experiments/fsq_khaendler/reissue_clips.py','--src-dir','external_data/amass_converted/LAFAN1_all','--out-root','external_data/amass_converted/LAFAN1_allfix','--targets','BoosterT1','Atlas','Apollo','--clips','$ARGS' -WorkingDirectory 'C:\Users\smirn\Desktop\robot_learning_ip' -RedirectStandardOutput 'C:\Users\smirn\Desktop\robot_learning_ip\experiments\fsq_khaendler\_tok_logs\reissue_aa_all.log' -RedirectStandardError 'C:\Users\smirn\Desktop\robot_learning_ip\experiments\fsq_khaendler\_tok_logs\reissue_aa_all.err' -WindowStyle Hidden"
  echo "launched T1+Atlas+Apollo re-issue (resume)"
elif [ "$(grep -c '^OK\|^!!' $L/reissue_aa_all.log 2>/dev/null)" -lt 68 ]; then
  powershell.exe -NoProfile -Command "Start-Process -FilePath 'C:\Users\smirn\Desktop\robot_learning_ip\.venv\Scripts\python.exe' -ArgumentList 'experiments/fsq_khaendler/reissue_clips.py','--src-dir','external_data/amass_converted/LAFAN1_all','--out-root','external_data/amass_converted/LAFAN1_allfix','--targets','Atlas','Apollo','--clips','$ARGS' -WorkingDirectory 'C:\Users\smirn\Desktop\robot_learning_ip' -RedirectStandardOutput 'C:\Users\smirn\Desktop\robot_learning_ip\experiments\fsq_khaendler\_tok_logs\reissue_aa_all.log' -RedirectStandardError 'C:\Users\smirn\Desktop\robot_learning_ip\experiments\fsq_khaendler\_tok_logs\reissue_aa_all.err' -WindowStyle Hidden"
  echo "launched Atlas+Apollo re-issue (resume)"
fi
# 2. clip assembly waiter
[ -f experiments/fsq_khaendler/clips_5r/READY ] || powershell.exe -NoProfile -Command "Start-Process wsl.exe -ArgumentList '-d','Ubuntu','-e','bash','/mnt/c/Users/smirn/Desktop/robot_learning_ip/scripts/scaling/wave7/launch_build_5r.sh' -WindowStyle Hidden"
# 3. local overnight orchestrator (waits for READY, skips done arms)
powershell.exe -NoProfile -Command "Start-Process wsl.exe -ArgumentList '-d','Ubuntu','-e','bash','/mnt/c/Users/smirn/Desktop/robot_learning_ip/scripts/scaling/wave7/launch_local_night7.sh' -WindowStyle Hidden"
sleep 5
echo "== running:"; pgrep -fa "build_clips_5r|local_night7|reissue" | grep -v pgrep | cut -c1-100
echo "logs: $L/{reissue_aa_all,build_clips_5r,local_night7}.log ; arms: experiments/local_w7/"
