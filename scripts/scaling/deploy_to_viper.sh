#!/usr/bin/env bash
# Push the 2026-08-29 loco_mjx fixes to Viper's own copy.
#
# Two traps this script exists to avoid:
#  * CRLF. The Windows checkout has CRLF line endings and Viper's copy has LF, so
#    a raw diff reports every line changed (3562 lines on a 1785-line file) and a
#    raw scp ships CRLF into a Linux tree. `tr -d '\r'` on the way over, and the
#    result is verified with `file` on the far side. A CRLF upload has already
#    killed one submission on this project.
#  * `$(...)` inside an ssh/wsl string is expanded LOCALLY, not remotely. Hence a
#    script file rather than an inline command.
#
# Safe to run only when no TRAINING arm is pending: a reward change mid-chain
# means arms in one campaign train under different code.
set -eu
REPO=/mnt/c/Users/smirn/Desktop/robot_learning_ip
SRC=$REPO/loco_mjx/loco_mjx/environments/locomotion/urma2/mjx
DST=/ptmp/akalenik/urma/loco_mjx/loco_mjx/environments/locomotion/urma2/mjx
FILES="environment.py default_config.py reward_functions/default.py"

ssh viper11 "cd $DST && for f in $FILES; do cp -n \$f \$f.bak_0829 2>/dev/null || true; done; echo BACKED_UP"

for f in $FILES; do
  base=${f##*/}
  tr -d '\r' < "$SRC/$f" > "/tmp/up_$base"
  scp -q "/tmp/up_$base" "viper11:$DST/$f"
  echo "sent $f"
done

ssh viper11 "cd $DST && file $FILES && python3 -c \"
import ast
for p in '$FILES'.split():
    ast.parse(open(p).read())
print('PARSE OK')
\" && grep -c foot_half_width_override environment.py default_config.py && grep -c move_contact_quality_past_clip reward_functions/default.py"
