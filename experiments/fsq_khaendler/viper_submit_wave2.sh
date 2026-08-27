#!/bin/bash
# Overnight-fix WAVE 2 (submitted ~02:35; jobs queue behind wave 1's last arms).
# Picks per OVERNIGHT_BASELINE_FIX_GOAL.md + the night's own findings:
#   W1 morph-0.7 walk    (A1 config, the aggressive-random-bodies claim)
#   W2 morph-0.7 dance   (B1 config = best dance arm at 95M; doubles as B4)
#   W3 fx_super_z        (A6 + scz z-token obs; launched unconditionally --
#                         slots are not scarce; ignore it if A4/A6 fail gates)
#   W4 fx_dance_nohead   (B2r single-delta diagnosis: HEADING off. B2 stalled
#                         at 2.9 [REFBIAS=0], B2r crawls at ~10 [REFBIAS=1];
#                         heading-free B3 bootstraps fine on walk)
set -u
cd /ptmp/akalenik/urma

COMMON="ALL,STAGE=mmtrain,NR_ENVS=768,ROBOTS_LIST=unitree_h1:unitree_g1,ANCHOR=absolute,FITVARIANT=False,REFBIAS=0.0,REFROOT=True,VELCMD=True,GAITMODE=fixed,GAITCOEFF=0.0,NOMINAL_TARGET=reference,CURTRACK=0.0,DEVRATIO=0.0,CURMAX=0.6,MORPH_MODE=schedule,MORPH_COEFF=0.5,MORPH_START=0.2,MORPH_RAMP=40000000,TOTAL_STEPS=98304000,SAVE_EVERY=4915200,CLIP_DIR=/ptmp/akalenik/urma/clips/LAFAN1,XLA_EXTRA=--xla_gpu_enable_command_buffer="

sbatch -J w2_walk07  --export="$COMMON,CLIP_FILE=walk1_subject1.npz,CONTACT_TIMECONST=0.01,MORPH_COEFF=0.7,EXP_NAME=w2_walk07" viper_train.sbatch
sbatch -J w2_dance07 --export="$COMMON,CLIP_FILE=dance2_subject4.npz,QVEL_TEMP=10,MORPH_COEFF=0.7,EXP_NAME=w2_dance07" viper_train.sbatch
sbatch -J w2_superz  --export="$COMMON,CLIP_DIR=/ptmp/akalenik/urma/clips/clips_super,CLIP_FILE=super5dance.npz,EXTRA_ARGS=--environment.command.tracking_clip_latent_obs=True:--environment.command.tracking_clip_latent_dim=32:--environment.command.tracking_clip_latent_replaces_reference=True,EXP_NAME=w2_superz" viper_train.sbatch
sbatch -J w2_dance_nohead --export="$COMMON,CLIP_FILE=dance2_subject4.npz,QVEL_TEMP=10,TRACK_DEVIATION=0.5,REFBIAS=1.0,EXP_NAME=w2_dance_nohead" viper_train.sbatch

squeue -u akalenik -o "%.9i %.ames %.2t %.8M %R" 2>/dev/null || squeue -u akalenik
