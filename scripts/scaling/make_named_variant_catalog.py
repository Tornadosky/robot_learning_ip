"""Build the named-static-XML evaluation catalog required by roadmap step 2C.

The online path drives morphology through four dynamic model arrays, while
`generated_variants/h1_morphology_*` are separately compiled XML robots.
`tests/test_online_h1.py::test_dynamic_arrays_match_existing_static_generator`
already asserts that the dynamic arrays reproduce the static XML's `body_pos`,
`body_ipos`, `body_mass`, `body_inertia` and `site_pos`. Evaluating the named
descriptors through the online environment is therefore evaluating those exact
robots, without compiling one MJX branch per variant.

Variants that differ only in parameters the four-coordinate descriptor cannot
express are excluded by name rather than silently deduplicated.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

WORKSPACE = Path(__file__).resolve().parents[2]
SCRIPTS = WORKSPACE / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from h1_morphology_variants import PRESETS  # noqa: E402
from scaling.embodiment_catalog import build_catalog_from_descriptors  # noqa: E402
from scaling.online_h1 import MorphologyBounds  # noqa: E402

# `big_feet` varies foot geometry only; its four online coordinates are identical
# to `nominal`, so the online descriptor cannot represent it. Excluded, and the
# limitation is recorded rather than hidden.
UNREPRESENTABLE = {
    "big_feet": "differs from nominal only in foot geometry, which is not one of "
    "the four online morphology coordinates",
}


def main() -> None:
    args = parse_args()
    bounds = MorphologyBounds(tuple(args.morph_low), tuple(args.morph_high))
    bounds.validate()
    low = np.asarray(bounds.low)
    high = np.asarray(bounds.high)

    names, descriptors, rows = [], [], []
    for name, preset in PRESETS.items():
        if name in UNREPRESENTABLE:
            continue
        xml = WORKSPACE / "generated_variants" / f"h1_morphology_{name}" / "h1.xml"
        descriptor = [
            float(preset.leg_length_scale),
            float(preset.arm_length_scale),
            float(preset.shoulder_width_scale),
            float(preset.torso_mass_scale),
        ]
        names.append(name)
        descriptors.append(descriptor)
        outside = (np.asarray(descriptor) < low) | (np.asarray(descriptor) > high)
        rows.append(
            {
                "name": name,
                "xml": str(xml.relative_to(WORKSPACE)) if xml.is_file() else None,
                "xml_present": xml.is_file(),
                "descriptor": descriptor,
                "num_coordinates_outside_training_bounds": int(outside.sum()),
                "inside_training_bounds": bool(not outside.any()),
            }
        )

    catalog = build_catalog_from_descriptors(
        descriptors,
        bounds_low=bounds.low,
        bounds_high=bounds.high,
        split="test",
        topology_signature=topology_signature(args.xml),
        body_id_offset=5_000_000,
        notes=(
            "Named generated_variants H1 morphologies evaluated through the online "
            "descriptor path; dynamic arrays are asserted equal to the static XML "
            "in tests/test_online_h1.py."
        ),
    )
    path = catalog.save(args.output_dir / "named_variants.json")
    index = {
        "catalog": str(path.relative_to(WORKSPACE)),
        "content_hash": catalog.content_hash,
        "num_variants": len(names),
        "excluded": UNREPRESENTABLE,
        "morphology_low": list(bounds.low),
        "morphology_high": list(bounds.high),
        "variants": [
            {**row, "body_id": int(catalog.body_ids[i])} for i, row in enumerate(rows)
        ],
    }
    index_path = args.output_dir / "named_variants_index.json"
    index_path.write_text(
        json.dumps(index, indent=2, allow_nan=False), encoding="utf-8"
    )
    inside = sum(r["inside_training_bounds"] for r in rows)
    print(
        f"[named] {len(names)} variants ({inside} inside training bounds, "
        f"{len(names) - inside} extrapolating) hash={catalog.content_hash[:12]}",
        flush=True,
    )
    for row in rows:
        flag = "" if row["inside_training_bounds"] else "  <-- outside bounds"
        missing = "" if row["xml_present"] else "  [no XML on disk]"
        print(f"    {row['name']:<24}{row['descriptor']}{flag}{missing}", flush=True)
    print(f"[named] wrote {index_path}", flush=True)


def topology_signature(xml_path: Path) -> str:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    return ",".join(
        [
            f"nq={model.nq}",
            f"nv={model.nv}",
            f"nu={model.nu}",
            f"njnt={model.njnt}",
            f"nbody={model.nbody}",
            f"ngeom={model.ngeom}",
            f"nmesh={model.nmesh}",
            f"nsite={model.nsite}",
        ]
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--xml",
        type=Path,
        default=WORKSPACE / "generated_variants" / "h1_morphology_nominal" / "h1.xml",
    )
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
    return parser.parse_args()


if __name__ == "__main__":
    main()
