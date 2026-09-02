#!/usr/bin/env python3
"""Build the small return artifact needed for post-run diagnosis."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_tree_filtered(src: Path, dst: Path, include_model: bool) -> None:
    skip_names = {"diagnostics_return.zip", ".partial_diagnostics_return.zip"}
    model_suffixes = {".model", ".npz"}
    for path in src.rglob("*"):
        if not path.is_file() or path.name in skip_names:
            continue
        rel = path.relative_to(src)
        rel_text = rel.as_posix()
        is_model = path.suffix in model_suffixes and ("models/" in rel_text or path.suffix == ".model")
        if is_model and not include_model:
            continue
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--include-model", action="store_true")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    models = []
    if args.model_root.exists():
        for path in sorted(args.model_root.rglob("*.model")):
            if path.is_file():
                models.append({
                    "path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)
                })
    manifest = {
        "run_dir": str(args.run_dir.resolve()),
        "model_root": str(args.model_root.resolve()),
        "include_model": args.include_model,
        "models": models,
        "environment": {
            key: os.environ.get(key) for key in (
                "CUDA_VISIBLE_DEVICES", "XLA_FLAGS", "XLA_PYTHON_CLIENT_PREALLOCATE", "MUJOCO_GL"
            )
        },
    }

    with tempfile.TemporaryDirectory(prefix="h1g1t1_collect_") as tmp:
        root = Path(tmp) / args.run_dir.name
        root.mkdir(parents=True)
        copy_tree_filtered(args.run_dir, root, args.include_model)
        (root / "model_hashes.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        if args.include_model and args.model_root.exists():
            copy_tree_filtered(args.model_root, root / "model_artifacts", True)
        with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    zf.write(path, path.relative_to(Path(tmp)).as_posix())
    print(json.dumps({"artifact": str(args.out), "bytes": args.out.stat().st_size, "sha256": sha256(args.out)}, indent=2))


if __name__ == "__main__":
    main()
