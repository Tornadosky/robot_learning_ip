"""Generate the frozen catalogs for the 1,000-robot scaling experiments.

One command produces every catalog the experiments need and writes them under
``experiments/scaling_1000/catalogs/``.  Regenerating with the same arguments
reproduces identical files and hashes; the held-out sets must be created once,
before training, and never regenerated after results are seen.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco

WORKSPACE = Path(__file__).resolve().parents[2]
SCRIPTS = WORKSPACE / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from scaling.embodiment_catalog import build_catalog  # noqa: E402
from scaling.online_h1 import MorphologyBounds  # noqa: E402

NOMINAL_XML = WORKSPACE / "generated_variants" / "h1_morphology_nominal" / "h1.xml"

# Seeds are fixed here so that the catalogs are a property of the repository,
# not of whoever runs the script.
SEEDS = {
    "train_1000": 1_000_001,
    "iid_256": 2_000_001,
    "boundary_256": 3_000_001,
    "ood_128": 4_000_001,
    "validation_128": 5_000_001,
}


def topology_signature(xml_path: Path) -> str:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    parts = [
        f"nq={model.nq}",
        f"nv={model.nv}",
        f"nu={model.nu}",
        f"njnt={model.njnt}",
        f"nbody={model.nbody}",
        f"ngeom={model.ngeom}",
        f"nmesh={model.nmesh}",
        f"nsite={model.nsite}",
    ]
    return ",".join(parts)


def main() -> None:
    args = parse_args()
    bounds = MorphologyBounds(tuple(args.morph_low), tuple(args.morph_high))
    bounds.validate()
    signature = topology_signature(args.xml)
    common = {
        "bounds_low": bounds.low,
        "bounds_high": bounds.high,
        "family": "h1",
        "topology_signature": signature,
    }

    specs = [
        (
            "train_1000",
            dict(
                num_bodies=args.num_train_bodies,
                seed=SEEDS["train_1000"],
                sampling_method="latin_hypercube",
                split="train",
                body_id_offset=0,
                notes=(
                    "Primary 1,000-body H1 training catalog. Latin-hypercube "
                    "stratification over the four online morphology coordinates."
                ),
            ),
        ),
        (
            "validation_128",
            dict(
                num_bodies=128,
                seed=SEEDS["validation_128"],
                sampling_method="latin_hypercube",
                split="validation",
                body_id_offset=1_000_000,
                notes="Seed-disjoint in-bounds bodies for checkpoint selection.",
            ),
        ),
        (
            "iid_256",
            dict(
                num_bodies=256,
                seed=SEEDS["iid_256"],
                sampling_method="latin_hypercube",
                split="test",
                body_id_offset=2_000_000,
                notes="Frozen IID interpolation test set inside the training bounds.",
            ),
        ),
        (
            "boundary_256",
            dict(
                num_bodies=256,
                seed=SEEDS["boundary_256"],
                sampling_method="boundary",
                split="test",
                body_id_offset=3_000_000,
                notes=(
                    "Frozen boundary test set: 1-4 coordinates pinned into a thin "
                    "band at a face, edge or corner of the training box."
                ),
            ),
        ),
        (
            "ood_128",
            dict(
                num_bodies=128,
                seed=SEEDS["ood_128"],
                sampling_method="extrapolation",
                split="ood",
                body_id_offset=4_000_000,
                notes=(
                    "Frozen mild extrapolation set: exactly one coordinate outside "
                    "its training bound, all scales positive."
                ),
            ),
        ),
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    index = {
        "topology_signature": signature,
        "nominal_xml": str(args.xml),
        "morphology_low": list(bounds.low),
        "morphology_high": list(bounds.high),
        "catalogs": {},
    }
    for name, spec in specs:
        catalog = build_catalog(**common, **spec)
        path = args.output_dir / f"{name}.json"
        if path.is_file() and not args.overwrite:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing.get("content_hash") != catalog.content_hash:
                raise RuntimeError(
                    f"{path} exists with a different hash. Frozen catalogs must "
                    "not be silently regenerated; pass --overwrite deliberately."
                )
        catalog.save(path)
        resolved = path.resolve()
        index["catalogs"][name] = {
            "path": str(
                resolved.relative_to(WORKSPACE)
                if resolved.is_relative_to(WORKSPACE)
                else resolved
            ),
            "split": catalog.split,
            "sampling_method": catalog.sampling_method,
            "seed": catalog.seed,
            "num_bodies": catalog.num_bodies,
            "content_hash": catalog.content_hash,
            "validity": catalog.validity_summary(),
        }
        print(
            f"[catalog] {name:<16} n={catalog.num_bodies:<5} "
            f"unique={catalog.num_unique:<5} hash={catalog.content_hash[:12]}",
            flush=True,
        )

    index_path = args.output_dir / "index.json"
    index_path.write_text(
        json.dumps(index, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"[catalog] wrote {index_path}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-train-bodies", type=int, default=1000)
    parser.add_argument("--xml", type=Path, default=NOMINAL_XML)
    parser.add_argument(
        "--morph-low", type=float, nargs=4, default=list(MorphologyBounds().low)
    )
    parser.add_argument(
        "--morph-high", type=float, nargs=4, default=list(MorphologyBounds().high)
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORKSPACE / "experiments" / "scaling_1000" / "catalogs",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
