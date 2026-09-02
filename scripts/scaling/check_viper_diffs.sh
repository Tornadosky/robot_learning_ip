#!/usr/bin/env bash
# What did the token deploy ACTUALLY change on Viper? The deploy overwrote whole
# files with the local checkout's copies, which is only safe if the far side was
# byte-identical apart from the intended edits. .bak_token holds Viper's own
# pre-deploy version, so this prints the real delta per file.
set -u
ROOT=/ptmp/akalenik/urma
ENVDST=$ROOT/loco_mjx/loco_mjx/environments/locomotion/urma2/mjx
ALGDST=$ROOT/loco_mjx/loco_mjx/algorithms/urma2/mjx
ssh viper11 "
for f in $ENVDST/environment.py $ENVDST/default_config.py $ENVDST/command_functions/tracking_clip.py \
         $ALGDST/policy.py $ALGDST/critic.py $ALGDST/urma2.py $ALGDST/default_config.py; do
  if [ -f \"\$f.bak_token\" ]; then
    added=\$(diff \"\$f.bak_token\" \"\$f\" | grep -c '^>' || true)
    removed=\$(diff \"\$f.bak_token\" \"\$f\" | grep -c '^<' || true)
    echo \"\$(basename \$f): +\$added -\$removed\"
    if [ \"\$removed\" != \"0\" ]; then
      echo '    REMOVED LINES (Viper had these, the upload does not):'
      diff \"\$f.bak_token\" \"\$f\" | grep '^<' | head -12 | sed 's/^/      /'
    fi
  else
    echo \"\$(basename \$f): NO BACKUP\"
  fi
done
echo '--- update_guard.py present on Viper? ---'
ls -la $ALGDST/update_guard.py 2>/dev/null || echo '  MISSING'
"
