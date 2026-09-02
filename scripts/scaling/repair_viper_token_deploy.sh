#!/usr/bin/env bash
# REPAIR the token deploy on Viper.
#
# The first deploy copied whole files across, which was wrong: the two trees
# have diverged in BOTH directions. Viper's tracking_clip.py carries
# `tracking_clip_reference_lead` (the wave-2 observed-reference lookahead) that
# the local checkout does not have, and the local algorithms/urma2 carries
# `update_guard.py` that Viper does not have -- so the uploaded urma2.py failed
# at import with ModuleNotFoundError. Copying either way deletes work.
#
# This restores every touched file from its .bak_token copy -- Viper's own
# pre-deploy version -- and then applies ONLY the token-routing edits to it, by
# anchored replacement. Each anchor must match exactly once or the patcher
# aborts before writing anything.
set -eu
REPO=/mnt/c/Users/smirn/Desktop/robot_learning_ip
ROOT=/ptmp/akalenik/urma
HOST=viper11
ENVDST=$ROOT/loco_mjx/loco_mjx/environments/locomotion/urma2/mjx
ALGDST=$ROOT/loco_mjx/loco_mjx/algorithms/urma2/mjx

echo "=== 0. guard: nothing training ==="
PENDING=$(ssh "$HOST" "squeue -u akalenik -h -o '%j' | grep -v -E '^ce_|^crosseval|^c_|^ttcl|^layout_test' | wc -l")
echo "  non-crosseval jobs in queue: $PENDING"
if [ "$PENDING" != "0" ] && [ "${FORCE:-0}" != "1" ]; then
  ssh "$HOST" "squeue -u akalenik"; echo "ABORT: training jobs present."; exit 1
fi

echo "=== 1. restore Viper's own files from .bak_token ==="
ssh "$HOST" "
set -e
for f in $ENVDST/environment.py $ENVDST/default_config.py $ENVDST/command_functions/tracking_clip.py \
         $ALGDST/policy.py $ALGDST/critic.py $ALGDST/urma2.py $ALGDST/default_config.py; do
  if [ -f \"\$f.bak_token\" ]; then cp \"\$f.bak_token\" \"\$f\"; echo \"  restored \$(basename \$f)\"; else echo \"  NO BACKUP for \$(basename \$f)\"; fi
done
echo '  reference_lead back in tracking_clip.py:' \$(grep -c reference_lead $ENVDST/command_functions/tracking_clip.py)
"

echo "=== 2. upload the anchored patcher ==="
tr -d '\r' < "$REPO/scripts/scaling/apply_token_patch.py" > /tmp/atp.py
scp -q /tmp/atp.py "$HOST:$ROOT/apply_token_patch.py"
ssh "$HOST" "file $ROOT/apply_token_patch.py"

echo "=== 3. apply ONLY the token edits to Viper's own tree ==="
ssh "$HOST" "cd $ROOT && python3 apply_token_patch.py $ROOT/loco_mjx/loco_mjx"

echo "=== 4. verify ==="
ssh "$HOST" "
cd $ROOT
python3 -c \"
import ast
paths = [
 '$ENVDST/environment.py', '$ENVDST/default_config.py',
 '$ENVDST/command_functions/tracking_clip.py',
 '$ALGDST/policy.py', '$ALGDST/critic.py', '$ALGDST/urma2.py', '$ALGDST/default_config.py',
]
for p in paths:
    ast.parse(open(p).read())
print('PARSE OK for', len(paths), 'files')
\"
echo -n '  reference_lead preserved:     '; grep -c reference_lead $ENVDST/command_functions/tracking_clip.py
echo -n '  latent index list:            '; grep -c policy_joint_latent_obs_idx $ENVDST/environment.py
echo -n '  latent set_norm:              '; grep -c 'set_norm(self.policy_joint_latent_obs_idx' $ENVDST/environment.py
echo -n '  motion latent obs idx:        '; grep -c policy_motion_latent_obs_idx $ENVDST/environment.py
echo -n '  scope key:                    '; grep -c tracking_clip_latent_scope $ENVDST/default_config.py
echo -n '  divisor key:                  '; grep -c tracking_clip_latent_obs_divisor $ENVDST/default_config.py
echo -n '  pooled global latents:        '; grep -c clip_latents_global $ENVDST/command_functions/tracking_clip.py
echo -n '  token encoder (policy):       '; grep -c encoder_latent_state_token $ALGDST/policy.py
echo -n '  token encoder (critic):       '; grep -c encoder_joint_latent_state_token $ALGDST/critic.py
echo -n '  channel resolution (trainer): '; grep -c _env_latent_channels $ALGDST/urma2.py
echo -n '  update_guard import present:  '; grep -c update_guard $ALGDST/urma2.py
"
echo
echo "REPAIR DONE"
