#!/usr/bin/env bash
# Upload the regenerated BoosterT1 clips and the wave-7 launcher, then dry-run.
# A script file rather than an inline `wsl bash -lc` string: variables and
# `$(...)` in an inline string get expanded by the LOCAL shell, which has bitten
# this three times today.
set -eu
REPO=/mnt/c/Users/smirn/Desktop/robot_learning_ip
RCLIP=/ptmp/akalenik/urma/clips

ssh viper11 "mkdir -p $RCLIP/LAFAN1_fixed"
cd "$REPO/external_data/amass_converted/LAFAN1_fixed"
tar -czf /tmp/fixedclips.tgz BoosterT1
scp -q /tmp/fixedclips.tgz "viper11:$RCLIP/"
ssh viper11 "cd $RCLIP/LAFAN1_fixed && tar -xzf ../fixedclips.tgz && echo -n 'clips uploaded: ' && ls BoosterT1 | wc -l && md5sum BoosterT1/dance2_subject4.npz"
echo -n "local md5:                        "
md5sum "$REPO/external_data/amass_converted/LAFAN1_fixed/BoosterT1/dance2_subject4.npz"

tr -d '\r' < "$REPO/scripts/scaling/submit_wave7.sh" > /tmp/w7.sh
scp -q /tmp/w7.sh viper11:/ptmp/akalenik/urma/submit_wave7.sh
ssh viper11 "cd /ptmp/akalenik/urma && file submit_wave7.sh && bash -n submit_wave7.sh && echo SYNTAX_OK && DRY=1 bash submit_wave7.sh 2>&1 | grep -E '^(TRAIN|  CE|##########|WAVE7)' | head -40"
