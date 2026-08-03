import json
from dataclasses import replace

import numpy as np
import pytest

from scaling.embodiment_catalog import (
    EmbodimentCatalog,
    build_catalog,
    catalog_resample_assignment,
    exposure_summary,
    fixed_balanced_assignment,
)


BOUNDS_LOW = (0.85, 0.85, 0.85, 0.70)
BOUNDS_HIGH = (1.20, 1.20, 1.20, 1.50)


def make(num_bodies=1000, seed=1_000_001, method="latin_hypercube", split="train"):
    return build_catalog(
        num_bodies=num_bodies,
        seed=seed,
        bounds_low=BOUNDS_LOW,
        bounds_high=BOUNDS_HIGH,
        sampling_method=method,
        split=split,
        topology_signature="test-signature",
    )


def test_identical_seed_and_revision_reproduce_the_same_catalog_and_hash():
    first = make()
    second = make()
    np.testing.assert_array_equal(first.descriptors, second.descriptors)
    np.testing.assert_array_equal(first.body_ids, second.body_ids)
    assert first.content_hash == second.content_hash

    different_seed = make(seed=1_000_002)
    assert different_seed.content_hash != first.content_hash


def test_training_catalog_has_exactly_1000_unique_descriptors_and_ids():
    catalog = make()
    assert catalog.num_bodies == 1000
    assert catalog.num_unique == 1000
    assert len(np.unique(catalog.body_ids)) == 1000
    validity = catalog.validity_summary()
    assert validity["all_descriptors_unique"]
    assert validity["all_positive"]
    assert validity["fraction_inside_training_bounds"] == 1.0
    assert validity["min_pairwise_l2"] > 0.0


def test_fixed_balanced_assignment_gives_every_body_the_requested_replicas():
    assignment = fixed_balanced_assignment(num_envs=8000, num_bodies=1000)
    summary = exposure_summary(assignment, 1000)
    assert summary["min_exposure"] == 8
    assert summary["max_exposure"] == 8
    assert summary["num_bodies_with_zero_exposure"] == 0
    assert summary["balanced"]

    capacity = exposure_summary(fixed_balanced_assignment(200_000, 1000), 1000)
    assert capacity["min_exposure"] == capacity["max_exposure"] == 200


def test_catalog_resampling_stays_balanced_over_a_registered_window():
    num_bodies, num_envs, window = 1000, 8000, 1000
    slots = np.arange(num_envs)
    seen = np.concatenate(
        [
            catalog_resample_assignment(slots, np.full(num_envs, g), num_bodies)
            for g in range(window)
        ]
    )
    summary = exposure_summary(seen, num_bodies)
    assert summary["min_exposure"] == summary["max_exposure"]
    assert summary["num_bodies_with_zero_exposure"] == 0
    # Every slot walks the entire catalog exactly once per window.
    per_slot = catalog_resample_assignment(
        np.zeros(window, dtype=np.int64), np.arange(window), num_bodies
    )
    assert len(np.unique(per_slot)) == num_bodies


def test_saved_catalog_reloads_without_numerical_drift(tmp_path):
    catalog = make()
    path = catalog.save(tmp_path / "train.json")
    reloaded = EmbodimentCatalog.load(path)
    np.testing.assert_array_equal(catalog.descriptors, reloaded.descriptors)
    np.testing.assert_array_equal(catalog.body_ids, reloaded.body_ids)
    assert reloaded.content_hash == catalog.content_hash
    assert reloaded.bounds_low == catalog.bounds_low


def test_tampered_catalog_is_rejected_on_load(tmp_path):
    catalog = make(num_bodies=16)
    path = catalog.save(tmp_path / "train.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["descriptors"][0][0] += 0.01
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="content hash mismatch"):
        EmbodimentCatalog.load(path)


def test_invalid_bounds_and_descriptors_are_rejected():
    with pytest.raises(ValueError):
        build_catalog(
            num_bodies=8,
            seed=1,
            bounds_low=(1.0, 1.0, 1.0, 1.0),
            bounds_high=(0.9, 1.1, 1.1, 1.1),
        )
    with pytest.raises(ValueError):
        build_catalog(
            num_bodies=8,
            seed=1,
            bounds_low=(0.0, 0.85, 0.85, 0.7),
            bounds_high=(1.2, 1.2, 1.2, 1.5),
        )
    with pytest.raises(ValueError):
        build_catalog(
            num_bodies=8,
            seed=1,
            bounds_low=BOUNDS_LOW,
            bounds_high=BOUNDS_HIGH,
            sampling_method="not_a_method",
        )
    catalog = make(num_bodies=8)
    with pytest.raises(ValueError, match="unique"):
        replace(catalog, body_ids=np.zeros(8, dtype=np.int64))
    with pytest.raises(ValueError, match="positive"):
        replace(catalog, descriptors=np.zeros_like(catalog.descriptors))
    with pytest.raises(ValueError, match="finite"):
        bad = catalog.descriptors.copy()
        bad[0, 0] = np.nan
        replace(catalog, descriptors=bad)


def test_boundary_catalog_hugs_the_bounds_and_ood_catalog_leaves_them():
    boundary = make(num_bodies=256, seed=3_000_001, method="boundary", split="test")
    normalized = boundary.normalized()
    # Every boundary body pins at least one coordinate near a face.
    assert np.all(np.max(np.abs(normalized), axis=1) > 0.9)
    assert boundary.validity_summary()["fraction_inside_training_bounds"] == 1.0

    ood = build_catalog(
        num_bodies=128,
        seed=4_000_001,
        bounds_low=BOUNDS_LOW,
        bounds_high=BOUNDS_HIGH,
        sampling_method="extrapolation",
        split="ood",
        topology_signature="test-signature",
    )
    low = np.asarray(BOUNDS_LOW)
    high = np.asarray(BOUNDS_HIGH)
    outside = (ood.descriptors < low) | (ood.descriptors > high)
    # Exactly one coordinate outside the training box per body, all positive.
    np.testing.assert_array_equal(outside.sum(axis=1), np.ones(len(ood)))
    assert np.all(ood.descriptors > 0.0)


def test_single_body_catalog_serialises():
    """A one-body catalog has no pairwise distance; it must still round-trip.

    ``inf`` is not JSON-serialisable under allow_nan=False, so the validity
    summary reports None rather than a sentinel.
    """
    catalog = make(num_bodies=1)
    assert catalog.validity_summary()["min_pairwise_l2"] is None
    payload = json.dumps(catalog.to_dict(), allow_nan=False)
    reloaded = EmbodimentCatalog.from_dict(json.loads(payload))
    np.testing.assert_array_equal(reloaded.descriptors, catalog.descriptors)
    assert reloaded.content_hash == catalog.content_hash
