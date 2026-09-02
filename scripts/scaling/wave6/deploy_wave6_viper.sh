#!/usr/bin/env bash
# Deploy the wave-6 code (aux head, co-training, leg kernel, routing fix) to
# Viper by ANCHORED PATCH (never whole-file copies of the trainer), plus the
# window sidecars. Run from WSL: bash deploy_wave6_viper.sh
set -u
LOCAL=/mnt/c/Users/smirn/Desktop/robot_learning_ip
W6=$LOCAL/scripts/scaling/wave6
ROOT=/ptmp/akalenik/urma
ssh viper11 "mkdir -p $ROOT/scripts/scaling/wave6 $ROOT/clips/tokentest_v2/UnitreeH1 $ROOT/clips/tokentest_v2/UnitreeG1"
# 1. patch scripts + new module
scp $W6/patch_aux_head.py $W6/patch_cotrain.py $W6/patch_legweight.py $W6/patch_jlat_default.py viper11:$ROOT/scripts/scaling/wave6/
scp $LOCAL/loco_mjx/loco_mjx/algorithms/urma2/mjx/fsq_cotrain.py viper11:$ROOT/loco_mjx/loco_mjx/algorithms/urma2/mjx/fsq_cotrain.py
# 2. window sidecars (H1/G1 dance4 + walk1)
for r in UnitreeH1 UnitreeG1; do
  scp $LOCAL/experiments/fsq_khaendler/clips_3t_v2/$r/dance2_subject4_win.npz $LOCAL/experiments/fsq_khaendler/clips_3t_v2/$r/walk1_subject1_win.npz viper11:$ROOT/clips/tokentest_v2/$r/
done
# 3. tokenizer params for the encoder init
ssh viper11 "mkdir -p $ROOT/tokenizer_3t_v2"
scp $LOCAL/experiments/fsq_khaendler/tokenizer_3t_v2/params.msgpack $LOCAL/experiments/fsq_khaendler/tokenizer_3t_v2/config.json viper11:$ROOT/tokenizer_3t_v2/
# 3b. the trainer sbatch: local == Viper's + the wave-6 knobs (verified by diff
#     against viper_mirror/root_scripts 2026-09-02), so upload it whole; the
#     .bak_wave6 backup below is taken before the copy lands.
ssh viper11 "cd $ROOT && [ -f viper_train.sbatch.bak_wave6 ] || cp viper_train.sbatch viper_train.sbatch.bak_wave6"
scp $LOCAL/experiments/urma2_h1g1/viper_train.sbatch viper11:$ROOT/viper_train.sbatch
# 4. apply on Viper (all-or-nothing per file), then compile-check with the run env
ssh viper11 bash -s <<'EOF'
set -u
ROOT=/ptmp/akalenik/urma
cd $ROOT
for f in scripts/scaling/wave6/*.py viper_train.sbatch; do sed -i 's/\r$//' "$f"; done
sed -i 's/\r$//' loco_mjx/loco_mjx/algorithms/urma2/mjx/fsq_cotrain.py
# backups of every file the patches touch
for f in loco_mjx/loco_mjx/algorithms/urma2/mjx/policy.py loco_mjx/loco_mjx/algorithms/urma2/mjx/critic.py \
         loco_mjx/loco_mjx/algorithms/urma2/mjx/urma2.py loco_mjx/loco_mjx/algorithms/urma2/mjx/default_config.py \
         loco_mjx/loco_mjx/environments/locomotion/urma2/mjx/default_config.py \
         loco_mjx/loco_mjx/environments/locomotion/urma2/mjx/reward_functions/tracking.py \
         loco_mjx/loco_mjx/environments/locomotion/urma2/mjx/command_functions/tracking_clip.py \
         viper_train.sbatch crosseval_token3.sbatch crosseval_motion.py; do
  [ -f "$f.bak_wave6" ] || cp "$f" "$f.bak_wave6"
done
module load python-waterboa/2025.06 >/dev/null 2>&1
eval "$(conda shell.bash hook)"; conda activate $ROOT/env
python scripts/scaling/wave6/patch_jlat_default.py || exit 1
python scripts/scaling/wave6/patch_legweight.py || exit 1
python scripts/scaling/wave6/patch_aux_head.py || exit 1
python scripts/scaling/wave6/patch_cotrain.py || exit 1
python -m py_compile loco_mjx/loco_mjx/algorithms/urma2/mjx/urma2.py loco_mjx/loco_mjx/algorithms/urma2/mjx/policy.py \
  loco_mjx/loco_mjx/algorithms/urma2/mjx/critic.py loco_mjx/loco_mjx/algorithms/urma2/mjx/fsq_cotrain.py \
  loco_mjx/loco_mjx/environments/locomotion/urma2/mjx/reward_functions/tracking.py \
  loco_mjx/loco_mjx/environments/locomotion/urma2/mjx/command_functions/tracking_clip.py crosseval_motion.py && echo VIPER_COMPILE_OK
bash -n viper_train.sbatch && bash -n crosseval_token3.sbatch && echo VIPER_SH_OK
echo "--- diff summary vs .bak_wave6:"
for f in loco_mjx/loco_mjx/algorithms/urma2/mjx/urma2.py loco_mjx/loco_mjx/algorithms/urma2/mjx/policy.py loco_mjx/loco_mjx/algorithms/urma2/mjx/critic.py viper_train.sbatch crosseval_token3.sbatch crosseval_motion.py; do
  echo "$f: +$(diff $f.bak_wave6 $f | grep -c '^>') -$(diff $f.bak_wave6 $f | grep -c '^<')"
done
grep -c "sow(" loco_mjx/loco_mjx/algorithms/urma2/mjx/policy.py
ls -la clips/tokentest_v2/UnitreeH1/ tokenizer_3t_v2/
EOF
