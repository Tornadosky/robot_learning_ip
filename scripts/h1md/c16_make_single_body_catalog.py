"""C16 -- one-body catalogs for a breadth control with honest descriptor scaling.

C14 collapsed the morphology bounds to pin a single body, which had a side
effect: `_normalized_morphology` divides by the sampling half-range, so a
degenerate range turns the descriptor into four dimensions of noise. That biases
the single-body control downward and makes a "breadth helps" reading unsafe.

`online_h1_train.py` adopts a catalog's bounds unless `--keep-morph-bounds` is
given. So a **one-body catalog plus `--keep-morph-bounds`** pins the body while
leaving the descriptor normalised by the real bounds — the control C14 should
have been.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE / "scripts"))
sys.path.insert(0, str(WORKSPACE / "scripts" / "scaling"))

from embodiment_catalog import EmbodimentCatalog  # noqa: E402
from online_h1 import MORPHOLOGY_NAMES, MorphologyBounds  # noqa: E402

BODIES = {
    "nominal": (1.0, 1.0, 1.0, 1.0),
    "offnominal": (0.90, 1.10, 1.10, 1.30),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path,
                    default=WORKSPACE / "experiments" / "h1_morphology_deepmimic_20260808" / "body_catalog")
    args = ap.parse_args()

    bounds = MorphologyBounds()
    for name, desc in BODIES.items():
        catalog = EmbodimentCatalog(
            schema_version=1,
            generator_revision="c16_single_body",
            family="h1",
            topology_signature="h1_19dof",
            seed=0,
            sampling_method="explicit",
            split="train",
            bounds_low=tuple(bounds.low),
            bounds_high=tuple(bounds.high),
            descriptor_names=tuple(MORPHOLOGY_NAMES),
            body_ids=np.array([0], dtype=np.int64),
            descriptors=np.array([desc], dtype=np.float64),
            notes=("single-body breadth control; use with --keep-morph-bounds so the "
                   "descriptor stays normalised by the real morphology bounds"),
        )
        path = catalog.save(args.out_dir / f"c16_single_{name}.json")
        print(f"{name}: {desc} -> {path} (hash {catalog.content_hash[:12]}, "
              f"n={len(catalog)}, unique={catalog.num_unique})")


if __name__ == "__main__":
    main()
