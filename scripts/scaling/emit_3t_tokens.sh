#!/usr/bin/env bash
# Emit the FSQ token sidecars for the 3-topology campaign, then assemble the
# clip directory the RL arms actually consume.
#
# Run after scripts/scaling/khaendler_fsq_clip.py train finishes writing
# experiments/fsq_khaendler/tokenizer_3t.
#
# TWO OUTPUT DIRECTORIES, and the difference matters:
#   clips_3t_fsq/      reconstructed clips + _zq.npz -- this is DESIGN A, an
#                      FSQ-reconstructed REFERENCE. Useful on its own, but an
#                      arm trained here measures reconstruction quality.
#   clips_3t_token/    ORIGINAL clips + the same _zq.npz -- this is DESIGN B,
#                      the token added beside an untouched reference. This is
#                      the one a token-conditioning arm must use; otherwise the
#                      arm varies two things at once and neither is measured.
set -eu
cd "$(dirname "$0")/../.."
REPO=$PWD
PY="$REPO/.venv/Scripts/python.exe"
[ -x "$PY" ] || PY=python

# RELATIVE paths, deliberately. This runs the WINDOWS interpreter from Git
# Bash, whose $PWD is POSIX (/c/Users/...) and which the interpreter cannot
# open. Everything below is relative to the repo root we just cd'd into, which
# both shells agree on.
TOK=experiments/fsq_khaendler/tokenizer_3t
SRC=external_data/amass_converted/LAFAN1_3t
FSQ=experiments/fsq_khaendler/clips_3t_fsq
TOKEN=experiments/fsq_khaendler/clips_3t_token
ROBOTS="UnitreeH1 UnitreeG1 BoosterT1"
CLIPS="dance2_subject1.npz dance2_subject2.npz dance2_subject3.npz dance2_subject4.npz dance2_subject5.npz walk1_subject1.npz walk_cycle_s6764_n46.npz walk_cycle_s720_n62.npz walk_cycle_s7925_n54.npz walk_cycle_s8883_n57.npz"

[ -f "$TOK/params.msgpack" ] || { echo "ABORT: no tokenizer at $TOK"; exit 1; }
echo "=== tokenizer ==="
"$PY" -c "
import json, sys
c = json.load(open('$TOK/config.json'))
print('  robots      :', ' '.join(c['robots']))
print('  clips       :', len(c.get('clips', [])))
print('  held out    :', ' '.join(c.get('heldout_clips') or ['(tail split only)']))
print('  final eval  :', c['final_eval_loss'])
"

echo "=== 1. reconstruct + emit z_q ==="
"$PY" scripts/scaling/khaendler_fsq_clip.py reconstruct \
  --robots $ROBOTS \
  --clip $CLIPS \
  --clip-dir "$SRC" \
  --tokenizer "$TOK" \
  --out "$FSQ"

echo "=== 2. assemble the DESIGN B dir: original reference + token ==="
rm -rf "$TOKEN"
for r in $ROBOTS; do
  mkdir -p "$TOKEN/$r"
  for c in $CLIPS; do
    [ -f "$SRC/$r/$c" ] || continue
    cp -f "$SRC/$r/$c" "$TOKEN/$r/$c"
    stem="${c%.npz}"
    cp -f "$FSQ/$r/${stem}_zq.npz" "$TOKEN/$r/${stem}_zq.npz"
  done
  echo "  $r: $(ls "$TOKEN/$r"/*_zq.npz 2>/dev/null | wc -l) token sidecars beside $(ls "$TOKEN/$r"/*.npz 2>/dev/null | grep -vc _zq) clips"
done

echo "=== 3. verify the reference is the ORIGINAL, not the reconstruction ==="
"$PY" - <<'PY'
import hashlib, sys
from pathlib import Path
repo = Path.cwd()
src = repo / "external_data/amass_converted/LAFAN1_3t"
fsq = repo / "experiments/fsq_khaendler/clips_3t_fsq"
token = repo / "experiments/fsq_khaendler/clips_3t_token"
h = lambda p: hashlib.md5(p.read_bytes()).hexdigest()[:12]
bad = 0
for r in ("UnitreeH1", "UnitreeG1", "BoosterT1"):
    for c in sorted((token / r).glob("*.npz")):
        if c.name.endswith("_zq.npz"):
            continue
        o, f = src / r / c.name, fsq / r / c.name
        if h(c) != h(o):
            print(f"  FAIL {r}/{c.name}: staged clip is NOT the original"); bad += 1
        elif f.exists() and h(o) == h(f):
            print(f"  WARN {r}/{c.name}: reconstruction is byte-identical to the original?"); bad += 1
print("  OK: every staged reference is the original clip" if not bad else f"  {bad} PROBLEMS")
sys.exit(1 if bad else 0)
PY

echo
echo "DESIGN A (reconstructed reference): $FSQ"
echo "DESIGN B (original reference + token): $TOKEN"
