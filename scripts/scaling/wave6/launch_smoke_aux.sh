#!/usr/bin/env bash
# aux-head smoke: ON (coeff 0.5, horizon 5) then OFF control; 64 envs, 10 updates each.
W=/mnt/c/Users/smirn/Desktop/robot_learning_ip/scripts/scaling/wave6/local_train.sh
L=/mnt/c/Users/smirn/Desktop/robot_learning_ip/experiments/local_w6
C=/mnt/c/Users/smirn/Desktop/robot_learning_ip/experiments/fsq_khaendler/clips_3t_v2
NAME=w6smoke_aux CLIPDIR=$C LATENT=1 NR_ENVS=64 MINIBATCH=2048 TOTAL=40960 SAVE_EVERY=40960 AUX_COEFF=0.5 AUX_HORIZON=5 PROJECT=local_w6 bash $W > $L/smoke_aux.log 2>&1
echo "SMOKE_AUX_RC=$?" >> $L/smoke_aux.log
NAME=w6smoke_off CLIPDIR=$C LATENT=1 NR_ENVS=64 MINIBATCH=2048 TOTAL=40960 SAVE_EVERY=40960 AUX_COEFF=0.0 PROJECT=local_w6 bash $W > $L/smoke_off.log 2>&1
echo "SMOKE_OFF_RC=$?" >> $L/smoke_off.log
