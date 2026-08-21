"""Unit gates for the canonical FSQ motion-latent machinery."""

from __future__ import annotations

import numpy as np
import pytest

from scaling.fsq_motion import (
    FRAME_FEATURE_DIM,
    FSQ,
    NormalizationStats,
    feature_group_slices,
    future_window,
    oracle_codes,
)


def test_fsq_roundtrip_and_codebook_size():
    fsq = FSQ(levels=[8, 5, 5, 5])
    assert fsq.codebook_size == 1000
    z = np.random.default_rng(0).normal(size=(64, 4)).astype(np.float32)
    quantized = np.asarray(fsq.quantize(z))
    indices = np.asarray(fsq.codes_to_indexes(quantized))
    recovered = np.asarray(fsq.indexes_to_codes(indices))
    np.testing.assert_allclose(quantized, recovered, atol=1e-6)
    assert quantized.min() >= -1.0 and quantized.max() <= 1.0


def test_future_window_clamps_at_clip_end_without_cross_clip_leak():
    features = np.arange(10, dtype=np.float32)[:, None]  # clip A: 0-5, B: 6-9
    split_points = np.asarray([0, 6, 10])
    windows = future_window(features, split_points, window=3, stride=2)
    # timestamp 4 (clip A): frames 4, min(6->5), min(8->5) — never frame 6+
    np.testing.assert_array_equal(windows[4], [4.0, 5.0, 5.0])
    # last timestamp of clip A repeats itself
    np.testing.assert_array_equal(windows[5], [5.0, 5.0, 5.0])
    # clip B starts fresh and clamps at ITS end (frame 9), never clip A's
    np.testing.assert_array_equal(windows[6], [6.0, 8.0, 9.0])


def test_feature_group_slices_cover_frame_dim_exactly():
    slices = feature_group_slices()
    stops = sorted(s.stop for s in slices.values())
    assert stops[-1] == FRAME_FEATURE_DIM
    covered = np.zeros(FRAME_FEATURE_DIM, dtype=int)
    for s in slices.values():
        covered[s] += 1
    assert (covered == 1).all()


def test_oracle_codes_are_deterministic_and_on_the_fsq_grid():
    rng = np.random.default_rng(3)
    windows = rng.normal(size=(50, 20)).astype(np.float32)
    stats = NormalizationStats.fit(windows)
    levels = [8, 5, 5, 5]
    first = oracle_codes(windows, stats, levels, seed=9)
    again = oracle_codes(windows, stats, levels, seed=9)
    other_seed = oracle_codes(windows, stats, levels, seed=10)
    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, other_seed)
    # every row is a valid codebook entry
    fsq = FSQ(levels)
    indices = np.asarray(fsq.codes_to_indexes(first))
    np.testing.assert_allclose(
        first, np.asarray(fsq.indexes_to_codes(indices)), atol=1e-6
    )
    # temporally meaningful: not one constant code
    assert np.unique(first, axis=0).shape[0] > 1


@pytest.mark.parametrize("bad_split", [[0, 5], [0, 11]])
def test_future_window_rejects_mismatched_split_points(bad_split):
    features = np.zeros((10, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        future_window(features, np.asarray(bad_split), window=2, stride=1)
