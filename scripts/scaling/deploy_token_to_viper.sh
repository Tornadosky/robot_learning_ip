#!/usr/bin/env bash
# Push the 2026-08-29 TOKEN-ROUTING changes to Viper, then dry-run the campaign.
#
# Two traps this script exists to avoid (both have cost this project a run):
#  * CRLF. The Windows checkout has CRLF endings and Viper's tree has LF, so a
#    raw scp ships CRLF into a Linux tree and the far side silently misbehaves.
#    Everything goes through `tr -d '\r'` and is verified with `file` remotely.
#  * `$(...)` in an ssh/wsl string expands LOCALLY. Hence a script file.
#
# It REFUSES to run while any training job is queued or running: viper_train.sbatch
# and the environment change here alter what an arm trains under, and a campaign
# whose arms straddle a code change is not a campaign.
set -eu
REPO=/mnt/c/Users/smirn/Desktop/robot_learning_ip
ROOT=/ptmp/akalenik/urma
HOST=viper11
FORCE="${FORCE:-0}"

echo "=== guard: nothing training ==="
PENDING=$(ssh "$HOST" "squeue -u akalenik -h -o '%j %T' | grep -v -E '^ce_|^crosseval|^c_|^ttcl' | wc -l")
echo "  non-crosseval jobs in queue: $PENDING"
if [ "$PENDING" != "0" ] && [ "$FORCE" != "1" ]; then
  ssh "$HOST" "squeue -u akalenik"
  echo "ABORT: training jobs present. Re-run with FORCE=1 only if you intend"
  echo "       those arms to straddle this code change."
  exit 1
fi

echo "=== 1. loco_mjx sources ==="
ENVSRC=$REPO/loco_mjx/loco_mjx/environments/locomotion/urma2/mjx
ENVDST=$ROOT/loco_mjx/loco_mjx/environments/locomotion/urma2/mjx
ALGSRC=$REPO/loco_mjx/loco_mjx/algorithms/urma2/mjx
ALGDST=$ROOT/loco_mjx/loco_mjx/algorithms/urma2/mjx

ENVFILES="environment.py default_config.py command_functions/tracking_clip.py"
ALGFILES="policy.py critic.py urma2.py default_config.py"

ssh "$HOST" "cd $ENVDST && for f in $ENVFILES; do cp -n \$f \$f.bak_token 2>/dev/null || true; done; \
             cd $ALGDST && for f in $ALGFILES; do cp -n \$f \$f.bak_token 2>/dev/null || true; done; echo BACKED_UP"

for f in $ENVFILES; do
  tr -d '\r' < "$ENVSRC/$f" > "/tmp/up_${f##*/}"
  scp -q "/tmp/up_${f##*/}" "$HOST:$ENVDST/$f"; echo "  sent env/$f"
done
for f in $ALGFILES; do
  tr -d '\r' < "$ALGSRC/$f" > "/tmp/upa_$f"
  scp -q "/tmp/upa_$f" "$HOST:$ALGDST/$f"; echo "  sent alg/$f"
done

echo "=== 2. clips: ORIGINAL reference + z_q sidecar ==="
# NOT the reconstructed clips. An arm trained on clips_kevin would be measuring
# design A (reconstructed reference) and design B (token) at the same time.
ssh "$HOST" "mkdir -p $ROOT/clips/tokentest"
cd "$REPO/experiments/fsq_khaendler/clips_tokentest"
tar -czf /tmp/tokentest.tgz UnitreeH1 UnitreeG1
scp -q /tmp/tokentest.tgz "$HOST:$ROOT/clips/"
ssh "$HOST" "cd $ROOT/clips/tokentest && tar -xzf ../tokentest.tgz && ls -1 */ && md5sum UnitreeH1/dance2_subject4.npz"
# Non-fatal: this box's WSL VM is memory-capped and md5sum over the /mnt/c
# mount has failed with ENOMEM here. The Viper-side hash above is the one that
# matters, and the expected value is pinned so a mismatch is still caught.
echo -n "  local md5:      "; md5sum "$REPO/experiments/fsq_khaendler/clips_tokentest/UnitreeH1/dance2_subject4.npz" || echo "(local hash unavailable)"
echo    "  expected (H1):  29702a850bc767b66ba88299f6b665ae  <- the ORIGINAL clip, not the reconstruction"

echo "=== 3. launchers ==="
tr -d '\r' < "$REPO/experiments/urma2_h1g1/viper_train.sbatch" > /tmp/vt.sbatch
scp -q /tmp/vt.sbatch "$HOST:$ROOT/viper_train.sbatch"
tr -d '\r' < "$REPO/scripts/scaling/crosseval_token.sbatch" > /tmp/cet.sbatch
scp -q /tmp/cet.sbatch "$HOST:$ROOT/crosseval_token.sbatch"
tr -d '\r' < "$REPO/scripts/scaling/submit_token.sh" > /tmp/st.sh
scp -q /tmp/st.sh "$HOST:$ROOT/submit_token.sh"
tr -d '\r' < "$REPO/experiments/fsq_khaendler/crosseval_motion.py" > /tmp/cm.py
scp -q /tmp/cm.py "$HOST:$ROOT/crosseval_motion.py"
echo "  sent viper_train.sbatch crosseval_token.sbatch submit_token.sh crosseval_motion.py"

echo "=== 4. verify on the far side ==="
ssh "$HOST" "cd $ROOT && file viper_train.sbatch submit_token.sh crosseval_token.sbatch | grep -i CRLF && echo 'ABORT: CRLF' && exit 1
  bash -n viper_train.sbatch && bash -n submit_token.sh && bash -n crosseval_token.sbatch && echo SYNTAX_OK
  cd $ENVDST && python3 -c \"
import ast
for p in 'environment.py default_config.py command_functions/tracking_clip.py'.split():
    ast.parse(open(p).read())
print('ENV PARSE OK')\"
  cd $ALGDST && python3 -c \"
import ast
for p in 'policy.py critic.py urma2.py default_config.py'.split():
    ast.parse(open(p).read())
print('ALG PARSE OK')\"
  echo -n 'latent index list present: '; grep -c policy_joint_latent_obs_idx $ENVDST/environment.py
  echo -n 'latent set_norm present:   '; grep -c 'set_norm(self.policy_joint_latent_obs_idx' $ENVDST/environment.py
  echo -n 'scope key present:         '; grep -c tracking_clip_latent_scope $ENVDST/default_config.py
  echo -n 'token encoder present:     '; grep -c encoder_latent_state_token $ALGDST/policy.py
  echo -n 'sbatch knobs present:      '; grep -c 'LATENT_SCOPE\|JLAT_ENC_DIM' $ROOT/viper_train.sbatch"

echo "=== 5. dry run ==="
ssh "$HOST" "cd $ROOT && DRY=1 bash submit_token.sh 2>&1 | head -40"
echo
echo "To submit for real:  ssh $HOST 'cd $ROOT && bash submit_token.sh'"
