#!/usr/bin/env bash
set -Eeuo pipefail

BUNDLE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-}
PYTHON=${PYTHON:-python}
PROFILE=${PROFILE:-probe}
FORCE_INSTALL=${FORCE_INSTALL:-0}
INSTALL_ONLY=${INSTALL_ONLY:-0}

if [[ -z "$REPO" ]]; then
  echo "REPO is required. Example: REPO=/path/to/robot_learning_ip PYTHON=~/jaxgpu/bin/python PROFILE=probe bash install_and_run.sh" >&2
  exit 2
fi
REPO=$(cd "$REPO" && pwd)
PYTHON=$(realpath -m "${PYTHON/#\~/$HOME}")
[[ -x "$PYTHON" ]] || { echo "Python executable not found: $PYTHON" >&2; exit 2; }
[[ -f "$REPO/loco_mjx/experiments/experiment.py" ]] || { echo "Not the expected repository: $REPO" >&2; exit 2; }

BACKUP_ROOT="$REPO/.h1g1t1_bundle_backups/$(date +%Y%m%d_%H%M%S)"
export BUNDLE_DIR REPO BACKUP_ROOT FORCE_INSTALL

"$PYTHON" - <<'PY'
from __future__ import annotations
import hashlib, json, os, shutil, stat, sys
from pathlib import Path

bundle = Path(os.environ["BUNDLE_DIR"])
repo = Path(os.environ["REPO"])
backup = Path(os.environ["BACKUP_ROOT"])
force = os.environ.get("FORCE_INSTALL") == "1"
manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

# Refuse to overwrite a materially different checkout unless explicitly forced.
for item in manifest.get("source_requirements", []):
    dst = repo / item["path"]
    if not dst.is_file():
        print(f"missing required source file: {dst}", file=sys.stderr)
        raise SystemExit(3)
    current = sha(dst)
    accepted = {item["source_sha256"], item["installed_sha256"]}
    if current not in accepted and not force:
        print(f"source hash mismatch: {item['path']}", file=sys.stderr)
        print(f"  current:  {current}", file=sys.stderr)
        print(f"  expected: {item['source_sha256']} (original) or {item['installed_sha256']} (already installed)", file=sys.stderr)
        print("Review the local changes. Use FORCE_INSTALL=1 only after deciding they may be replaced.", file=sys.stderr)
        raise SystemExit(4)

for item in manifest["overlay_files"]:
    src = bundle / "overlay" / item["path"]
    dst = repo / item["path"]
    if not src.is_file() or sha(src) != item["sha256"]:
        raise SystemExit(f"bundle integrity failure for {src}")
    if dst.exists() and sha(dst) != item["sha256"]:
        prior = backup / item["path"]
        prior.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dst, prior)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    os.chmod(dst, int(item.get("mode", "0644"), 8))
    if sha(dst) != item["sha256"]:
        raise SystemExit(f"post-copy hash mismatch for {dst}")

print(f"installed {len(manifest['overlay_files'])} files into {repo}")
print(f"backup directory: {backup}")
PY

# Syntax/contract verification is deliberately repeated after installation.
"$PYTHON" -m unittest discover -s "$BUNDLE_DIR/tests" -p 'test_*.py'
find "$REPO/scripts/h1g1t1" -maxdepth 1 -name '*.py' -print0 | xargs -0 "$PYTHON" -m py_compile
find "$REPO/scripts/h1g1t1" -maxdepth 1 -name '*.sh' -print0 | xargs -0 -n1 bash -n

if [[ "$INSTALL_ONLY" == 1 ]]; then
  echo "Installation-only verification complete."
  exit 0
fi

export REPO PYTHON PROFILE
exec bash "$REPO/scripts/h1g1t1/run_experiment.sh" all
