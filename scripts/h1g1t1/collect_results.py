#!/usr/bin/env python3
"""Build a compact H1+G1+T1 diagnostic archive."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import zipfile
from pathlib import Path
from typing import Any, Iterable, Sequence

MODEL_SUFFIX = ".model"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "file_manifest.json":
            yield path


def build_checkpoint_inventory(result_dir: str | Path) -> dict[str, Any]:
    result_dir = Path(result_dir)
    models: list[dict[str, Any]] = []
    config_dir = result_dir / "checkpoint_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    for model in sorted(result_dir.rglob(f"*{MODEL_SUFFIX}")):
        item: dict[str, Any] = {
            "path": str(model.relative_to(result_dir)),
            "bytes": model.stat().st_size,
            "sha256": _sha256(model),
        }
        try:
            with zipfile.ZipFile(model) as zf:
                member = next((name for name in zf.namelist()
                               if name.endswith("config_algorithm.json")), None)
                if member:
                    data = zf.read(member)
                    target = config_dir / f"{model.name}.config_algorithm.json"
                    target.write_bytes(data)
                    item["config_algorithm"] = str(target.relative_to(result_dir))
                progress = next((name for name in zf.namelist()
                                 if name.endswith("training_progress.json")), None)
                if progress:
                    item["training_progress"] = json.loads(zf.read(progress))
        except (zipfile.BadZipFile, OSError, StopIteration, json.JSONDecodeError) as exc:
            item["inspection_error"] = f"{type(exc).__name__}: {exc}"
        models.append(item)
    inventory = {"models": models, "count": len(models)}
    (result_dir / "checkpoint_inventory.json").write_text(
        json.dumps(inventory, indent=2), encoding="utf-8")
    return inventory


def create_archive(result_dir: str | Path, output_zip: str | Path,
                   include_model: bool = False) -> Path:
    result_dir = Path(result_dir).resolve()
    output_zip = Path(output_zip).resolve()
    if not result_dir.is_dir():
        raise FileNotFoundError(result_dir)
    build_checkpoint_inventory(result_dir)

    entries: list[dict[str, Any]] = []
    files: list[Path] = []
    for path in _iter_files(result_dir):
        if path.suffix == MODEL_SUFFIX:
            if not include_model or path.name != "latest.model":
                continue
        files.append(path)
        entries.append({"path": str(path.relative_to(result_dir)),
                        "bytes": path.stat().st_size, "sha256": _sha256(path)})
    manifest_path = result_dir / "file_manifest.json"
    manifest_path.write_text(json.dumps({"include_model": include_model,
                                         "files": entries}, indent=2), encoding="utf-8")
    files.append(manifest_path)

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    if output_zip.exists():
        output_zip.unlink()
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6) as zf:
        for path in files:
            zf.write(path, arcname=f"{result_dir.name}/{path.relative_to(result_dir)}")
    return output_zip


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--include-model", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    out = create_archive(args.result_dir, args.out, args.include_model)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
