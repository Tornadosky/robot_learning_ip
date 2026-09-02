#!/usr/bin/env bash
# Wave-6 smokes, sequential: (1) aux head on the REAL split routing, (2) SONIC
# co-training with the v2 encoder init, (3) co-training from scratch.
# 64 envs x 64 steps x 10 updates each.
W=/mnt/c/Users/smirn/Desktop/robot_learning_ip/scripts/scaling/wave6/local_train.sh
L=/mnt/c/Users/smirn/Desktop/robot_learning_ip/experiments/local_w6
C=/mnt/c/Users/smirn/Desktop/robot_learning_ip/experiments/fsq_khaendler/clips_3t_v2
TOK=/mnt/c/Users/smirn/Desktop/robot_learning_ip/experiments/fsq_khaendler/tokenizer_3t_v2/params.msgpack
COMMON="CLIPDIR=$C NR_ENVS=64 MINIBATCH=2048 TOTAL=40960 SAVE_EVERY=40960 PROJECT=local_w6"
env $COMMON NAME=w6smoke_aux2 LATENT=1 JLAT_CH=-1 AUX_COEFF=0.5 AUX_HORIZON=5 bash $W > $L/smoke_aux2.log 2>&1
echo "SMOKE_AUX2_RC=$?" >> $L/smoke_aux2.log
env $COMMON NAME=w6smoke_cot LATENT=1 LATENT_DIM=44 LATENT_DIVISOR=1.0 SIDECAR=_win COTRAIN_ROWS=11 COTRAIN_CH=4 COTRAIN_INIT=$TOK bash $W > $L/smoke_cot.log 2>&1
echo "SMOKE_COT_RC=$?" >> $L/smoke_cot.log
env $COMMON NAME=w6smoke_cotaux LATENT=1 LATENT_DIM=44 LATENT_DIVISOR=1.0 SIDECAR=_win COTRAIN_ROWS=11 COTRAIN_CH=4 AUX_COEFF=0.5 AUX_HORIZON=5 bash $W > $L/smoke_cotaux.log 2>&1
echo "SMOKE_COTAUX_RC=$?" >> $L/smoke_cotaux.log
