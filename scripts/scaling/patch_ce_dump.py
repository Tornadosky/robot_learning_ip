#!/usr/bin/env python3
"""Let crosseval2.sbatch dump render trajectories.

`crosseval_motion.py` already supports --dump_render/--record_envs; the sbatch
never exposed them, so every rendered video on this project came from the LOCAL
eval script and none from a Viper arm. Wave-7 checkpoints live on Viper, so
without this there is no way to see what the heading fix actually looks like.

Off unless CE_DUMP is set, so nothing that runs today changes.
"""
import sys

P = "/ptmp/akalenik/urma/crosseval2.sbatch"
s = open(P).read()
if "CE_DUMP" in s:
    print("already patched")
    sys.exit(0)

anchor = '[ -n "${CE_REPLACES:-}" ] && CE_FLAGS+=(--latent_replaces "$CE_REPLACES")\n'
assert anchor in s, "anchor not found"
add = anchor + '''# CE_DUMP: write per-frame policy AND reference trajectories for rendering.
# CE_RECORD_ENVS defaults to 2 -- a couple of episodes is enough for a video and
# keeps the artifact small.
if [ -n "${CE_DUMP:-}" ]; then
  CE_FLAGS+=(--dump_render "$CE_DUMP" --record_envs "${CE_RECORD_ENVS:-2}")
fi
'''
open(P, "w").write(s.replace(anchor, add, 1))
print("patched")
