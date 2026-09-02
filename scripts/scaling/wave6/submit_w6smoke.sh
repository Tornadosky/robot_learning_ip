#!/usr/bin/env bash
# Wave-6 runtime smokes on Viper's apudev partition (15 min cap): does the
# patched tree still run the legacy path, and do aux / co-training run on ROCm?
# 768 envs (ROCm floor), 2 iterations (98304 steps).
set -u
ROOT=/ptmp/akalenik/urma
cd "$ROOT"
V2=$ROOT/clips/tokentest_v2
XLA="--xla_gpu_enable_command_buffer="
B="STAGE=mmtrain,NR_ENVS=768,TOTAL_STEPS=98304,SAVE_EVERY=98304"
B="$B,CLIP_FILE=dance2_subject4.npz,CLIP_DIR=$V2"
B="$B,REFVEL_OBS=False,REFROOT=True,REFROOT_FLOOR=True,REFBIAS=0.0"
B="$B,TRACK_COEFF=30.0,TRACK_TEMP=0.05,FITVARIANT=False,ANCHOR=absolute"
B="$B,GAITMODE=floor,GAITCOEFF=0.25,QVEL_TEMP=10,FOOTZVEL=1.0,FOOTH_TEMP=0.05,FOOTH=0.3333"
B="$B,FOOTSLIP=6.6667,GROUNDPEN=1000,POSTCONTACT=True,CONTACT_TIMECONST=0.004,HEADING_RATIO=0.20,HEADING_TEMP=2.0"
B="$B,XLA_EXTRA=$XLA"
EA="EXTRA_ARGS=--environment.command.tracking_clip_reference_hold=1:--environment.reward.deepmimic_swing_match_weight_ratio=0.5:--environment.command.tracking_clip_observe_root_heading=True"
TOK="LATENT_OBS=True,LATENT_DIM=32,LATENT_REPLACES=False,LATENT_SCOPE=per_joint,LATENT_DIVISOR=10.0,JLAT_ENC_DIM=4"
COT="LATENT_OBS=True,LATENT_DIM=44,LATENT_REPLACES=False,LATENT_SCOPE=per_joint,LATENT_DIVISOR=1.0,SIDECAR=_win,JLAT_ENC_DIM=4,COTRAIN_ROWS=11,COTRAIN_CH=4,COTRAIN_INIT=$ROOT/tokenizer_3t_v2/params.msgpack"
sub() { sbatch --parsable -p apudev -t 15:00 -J "$1" --export=ALL,"EXP_NAME=$1,$2" "$ROOT/viper_train.sbatch"; }
echo "w6smk_legacy -> $(sub w6smk_legacy "$B,$TOK,$EA")"
echo "w6smk_aux    -> $(sub w6smk_aux "$B,$TOK,JLAT_CH=-1,AUX_COEFF=0.5,AUX_HORIZON=5,$EA")"
echo "w6smk_cot    -> $(sub w6smk_cot "$B,$COT,$EA")"
echo "w6smk_legw   -> $(sub w6smk_legw "$B,LATENT_OBS=False,JLAT_ENC_DIM=0,LEGW=3.0,$EA")"
squeue -u akalenik -h -p apudev -o "%j %T"
