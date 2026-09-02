#!/usr/bin/env bash
cd /mnt/c/Users/smirn/Desktop/robot_learning_ip
R=/mnt/c/Users/smirn/Desktop/robot_learning_ip
env NAME=w7smoke_cot3 CLIPDIR=$R/experiments/fsq_khaendler/clips_3t_v2 ROBOTS=unitree_h1:unitree_g1:booster_t1 NR_ENVS=96 MINIBATCH=1536 TOTAL=61440 SAVE_EVERY=61440 LATENT=1 LATENT_DIM=44 LATENT_DIVISOR=1.0 SIDECAR=_win COTRAIN_ROWS=11 COTRAIN_CH=4 COTRAIN_INIT=$R/experiments/fsq_khaendler/tokenizer_3t_v2/params.msgpack PROJECT=local_w7 bash scripts/scaling/wave6/local_train.sh > experiments/local_w7/smoke_cot3.log 2>&1
echo "SMOKE3_RC=$?" >> experiments/local_w7/smoke_cot3.log
