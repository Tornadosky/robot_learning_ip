#!/usr/bin/env python3
"""Anchored patch: teach crosseval_motion.py a --contact_timeconst flag.

Night3 (2026-08-30) doses CONTACT_TIMECONST=0.004 in training arms; an eval in
the default 0.0-timeconst world is a train-vs-eval physics diff and the feet
metrics (airborne/slip/penetration) are exactly what the campaign measures.
Default 0.0 keeps every pre-night3 arm bit-identical.

Runs ON VIPER against /ptmp/akalenik/urma/crosseval_motion.py.
Anchored (aborts if an anchor is missing or already patched), backs up first.
"""
import shutil
import sys

PATH = "/ptmp/akalenik/urma/crosseval_motion.py"

ARG_ANCHOR = '    p.add_argument("--dump_render", default=None,'
ARG_INSERT = (
    '    p.add_argument("--contact_timeconst", type=float, default=0.0,\n'
    '                   help="terrain.contact_solref_timeconst; must match the '
    'training arm (0.0 = model default, every pre-night3 arm)")\n'
)

CFG_ANCHOR = '    config.environment.terrain.type = "plane"\n'
CFG_INSERT = (
    '    config.environment.terrain.contact_solref_timeconst = '
    'args.contact_timeconst\n'
)

src = open(PATH).read()
if "contact_timeconst" in src:
    print("ALREADY PATCHED, nothing to do")
    sys.exit(0)
for anchor in (ARG_ANCHOR, CFG_ANCHOR):
    if src.count(anchor) != 1:
        print(f"ABORT: anchor not unique ({src.count(anchor)}x): {anchor!r}")
        sys.exit(1)

shutil.copy2(PATH, PATH + ".bak_night3")
src = src.replace(ARG_ANCHOR, ARG_INSERT + ARG_ANCHOR)
src = src.replace(CFG_ANCHOR, CFG_ANCHOR + CFG_INSERT)
open(PATH, "w").write(src)

compile(src, PATH, "exec")
print("PATCHED OK (backup: crosseval_motion.py.bak_night3)")
