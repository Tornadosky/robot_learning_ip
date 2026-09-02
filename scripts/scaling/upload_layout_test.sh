#!/usr/bin/env bash
# Variables live in a FILE, never inside a `wsl -- bash -lc '...'` string: the
# LOCAL shell expands them there and they arrive empty. Cost one attempt today,
# and the repo's own notes record it costing runs before.
set -eu
REPO=/mnt/c/Users/smirn/Desktop/robot_learning_ip
ROOT=/ptmp/akalenik/urma

tr -d '\r' < "$REPO/tests/test_latent_obs_layout.py" > /tmp/tl.py
python3 - <<'PY'
from pathlib import Path
p = Path("/tmp/tl.py"); s = p.read_text()
old = 'CLIP_DIR = REPO / "experiments" / "fsq_khaendler" / "clips_tokentest"'
assert s.count(old) == 1, "clip-dir line not found"
p.write_text(s.replace(old, 'CLIP_DIR = Path("/ptmp/akalenik/urma/clips/tokentest")'))
print("clip dir repointed at Viper's tokentest bundle")
PY
tr -d '\r' < "$REPO/scripts/scaling/layout_test.sbatch" > /tmp/lt.sbatch

ssh viper11 "mkdir -p $ROOT/tests"
scp -q /tmp/tl.py "viper11:$ROOT/tests/test_latent_obs_layout.py"
scp -q /tmp/lt.sbatch "viper11:$ROOT/layout_test.sbatch"
ssh viper11 "cd $ROOT && file tests/test_latent_obs_layout.py layout_test.sbatch && grep -n 'CLIP_DIR =' tests/test_latent_obs_layout.py && sbatch --parsable layout_test.sbatch"
