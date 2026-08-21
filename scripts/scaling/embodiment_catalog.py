"""Deterministic, versioned catalogs of online H1 embodiments.

The online H1 path samples a fresh morphology at every reset.  That is the right
default for maximum embodiment exposure, but it cannot answer "which thousand
robots did this policy actually train on?".  A catalog fixes a reproducible set
of bodies: stable integer IDs, unique normalized descriptors, a recorded seed and
generator revision, a split label, morphology bounds, and a content hash.

The catalog stores compact descriptors only.  The physical model arrays are still
produced by ``OnlineMorphMjxUnitreeH1._apply_morphology`` inside one MJX graph, so
a thousand bodies cost one XLA branch, not a thousand.  ``audit_catalog_bodies.py``
verifies that the intended arrays really differ per catalog entry.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np

CATALOG_SCHEMA_VERSION = 1
CATALOG_GENERATOR_REVISION = "h1-online-morph-4coord-v1"

MORPHOLOGY_NAMES = (
    "leg_length_scale",
    "arm_length_scale",
    "shoulder_width_scale",
    "torso_mass_scale",
)

SAMPLING_METHODS = (
    "latin_hypercube",
    "uniform_iid",
    "boundary",
    "extrapolation",
    "explicit",
)
SPLITS = ("train", "validation", "test", "ood")

# Descriptors are rounded to this many decimals before uniqueness and hashing so
# that a catalog written on one machine matches one written on another.
DESCRIPTOR_DECIMALS = 9


@dataclass(frozen=True)
class EmbodimentCatalog:
    """A frozen, reproducible set of same-topology embodiments."""

    schema_version: int
    generator_revision: str
    family: str
    topology_signature: str
    seed: int
    sampling_method: str
    split: str
    bounds_low: tuple[float, ...]
    bounds_high: tuple[float, ...]
    descriptor_names: tuple[str, ...]
    body_ids: np.ndarray
    descriptors: np.ndarray
    notes: str = ""

    # ---------------------------------------------------------------- helpers

    def __post_init__(self) -> None:
        if self.sampling_method not in SAMPLING_METHODS:
            raise ValueError(f"Unknown sampling method {self.sampling_method!r}.")
        if self.split not in SPLITS:
            raise ValueError(f"Unknown split {self.split!r}.")
        descriptors = np.asarray(self.descriptors, dtype=np.float64)
        body_ids = np.asarray(self.body_ids, dtype=np.int64)
        if descriptors.ndim != 2 or descriptors.shape[1] != len(self.descriptor_names):
            raise ValueError(
                f"Descriptors must be (N, {len(self.descriptor_names)}); "
                f"got {descriptors.shape}."
            )
        if body_ids.shape != (descriptors.shape[0],):
            raise ValueError("body_ids must have one entry per descriptor row.")
        if len(np.unique(body_ids)) != len(body_ids):
            raise ValueError("Catalog body IDs must be unique.")
        if not np.all(np.isfinite(descriptors)):
            raise ValueError("Catalog descriptors must be finite.")
        # Positivity applies to the multiplicative SCALE dims only. The
        # additive dims (torso_com_x_offset, joint_range_shift) are legitimately
        # negative or zero, so the old blanket check rejected valid bodies --
        # latent since the 4->11 dim expansion, because nothing exercised the
        # catalog path with an additive dim present. Same rule as
        # online_h1.MORPHOLOGY_SCALE_MASK, derived from the names held here so
        # this module keeps no import back into the env.
        scale_mask = np.array(
            [str(n).endswith("_scale") for n in self.descriptor_names], dtype=bool
        )
        if scale_mask.shape[0] != descriptors.shape[1]:
            raise ValueError(
                "descriptor_names must have one entry per descriptor column; "
                f"got {scale_mask.shape[0]} names for {descriptors.shape[1]} columns."
            )
        if np.any(descriptors[:, scale_mask] <= 0.0):
            raise ValueError("Every morphology scale must be strictly positive.")
        object.__setattr__(self, "descriptors", descriptors)
        object.__setattr__(self, "body_ids", body_ids)

    def __len__(self) -> int:
        return int(self.descriptors.shape[0])

    @property
    def num_bodies(self) -> int:
        return len(self)

    @property
    def rounded_descriptors(self) -> np.ndarray:
        return np.round(self.descriptors, DESCRIPTOR_DECIMALS)

    @property
    def num_unique(self) -> int:
        return int(len(np.unique(self.rounded_descriptors, axis=0)))

    def normalized(self, low=None, high=None) -> np.ndarray:
        """Map descriptors into the policy's [-1, 1] descriptor coordinates.

        ``low``/``high`` default to this catalog's own bounds.  Held-out and OOD
        catalogs must instead be normalized with the *training* bounds, which is
        why the arguments exist.
        """
        low = np.asarray(self.bounds_low if low is None else low, dtype=np.float64)
        high = np.asarray(self.bounds_high if high is None else high, dtype=np.float64)
        midpoint = 0.5 * (low + high)
        half_range = 0.5 * (high - low)
        return (self.descriptors - midpoint) / half_range

    # ------------------------------------------------------------- validity

    def validity_summary(self) -> dict:
        """Numpy-level physical plausibility of the compact descriptors.

        This deliberately does *not* claim MJX-array correctness; see
        ``audit_catalog_bodies.py`` for the model-array checks required by the
        roadmap's "make different physically auditable" item.
        """
        descriptors = self.descriptors
        low = np.asarray(self.bounds_low, dtype=np.float64)
        high = np.asarray(self.bounds_high, dtype=np.float64)
        inside = np.all((descriptors >= low) & (descriptors <= high), axis=1)
        return {
            "num_bodies": len(self),
            "num_unique_descriptors": self.num_unique,
            "all_descriptors_unique": self.num_unique == len(self),
            "all_positive": bool(np.all(descriptors > 0.0)),
            "all_finite": bool(np.all(np.isfinite(descriptors))),
            "num_inside_training_bounds": int(np.sum(inside)),
            "fraction_inside_training_bounds": float(np.mean(inside)),
            "descriptor_min": descriptors.min(axis=0).tolist(),
            "descriptor_max": descriptors.max(axis=0).tolist(),
            "descriptor_mean": descriptors.mean(axis=0).tolist(),
            "min_pairwise_l2": _min_pairwise_distance(self.normalized()),
        }

    # ------------------------------------------------------------ hash / io

    def _hash_payload(self) -> bytes:
        header = json.dumps(
            {
                "schema_version": self.schema_version,
                "generator_revision": self.generator_revision,
                "family": self.family,
                "topology_signature": self.topology_signature,
                "seed": int(self.seed),
                "sampling_method": self.sampling_method,
                "split": self.split,
                "bounds_low": [float(v) for v in self.bounds_low],
                "bounds_high": [float(v) for v in self.bounds_high],
                "descriptor_names": list(self.descriptor_names),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        body = self.body_ids.astype(np.int64).tobytes()
        descriptors = np.ascontiguousarray(
            self.rounded_descriptors, dtype=np.float64
        ).tobytes()
        return header + b"|" + body + b"|" + descriptors

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self._hash_payload()).hexdigest()

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "generator_revision": self.generator_revision,
            "family": self.family,
            "topology_signature": self.topology_signature,
            "seed": int(self.seed),
            "sampling_method": self.sampling_method,
            "split": self.split,
            "bounds_low": [float(v) for v in self.bounds_low],
            "bounds_high": [float(v) for v in self.bounds_high],
            "descriptor_names": list(self.descriptor_names),
            "notes": self.notes,
            "content_hash": self.content_hash,
            "validity": self.validity_summary(),
            "body_ids": self.body_ids.astype(np.int64).tolist(),
            # float64 -> repr round-trips exactly through JSON, so a reload is
            # bit-identical and the content hash is stable.
            "descriptors": self.descriptors.tolist(),
        }

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, allow_nan=False), encoding="utf-8"
        )
        return path

    @classmethod
    def from_dict(cls, payload: dict) -> "EmbodimentCatalog":
        catalog = cls(
            schema_version=int(payload["schema_version"]),
            generator_revision=str(payload["generator_revision"]),
            family=str(payload["family"]),
            topology_signature=str(payload["topology_signature"]),
            seed=int(payload["seed"]),
            sampling_method=str(payload["sampling_method"]),
            split=str(payload["split"]),
            bounds_low=tuple(float(v) for v in payload["bounds_low"]),
            bounds_high=tuple(float(v) for v in payload["bounds_high"]),
            descriptor_names=tuple(str(v) for v in payload["descriptor_names"]),
            body_ids=np.asarray(payload["body_ids"], dtype=np.int64),
            descriptors=np.asarray(payload["descriptors"], dtype=np.float64),
            notes=str(payload.get("notes", "")),
        )
        recorded = payload.get("content_hash")
        if recorded is not None and recorded != catalog.content_hash:
            raise ValueError(
                "Catalog content hash mismatch on load: "
                f"recorded {recorded}, recomputed {catalog.content_hash}."
            )
        return catalog

    @classmethod
    def load(cls, path: Path) -> "EmbodimentCatalog":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def with_split(self, split: str) -> "EmbodimentCatalog":
        return replace(self, split=split)

    def subset(self, indices: Sequence[int]) -> "EmbodimentCatalog":
        indices = np.asarray(indices, dtype=np.int64)
        return replace(
            self,
            body_ids=self.body_ids[indices],
            descriptors=self.descriptors[indices],
        )


def _min_pairwise_distance(points: np.ndarray, max_points: int = 2048) -> float | None:
    """Smallest normalized L2 gap, used as a cheap "are these really distinct" check.

    Returns ``None`` for a single-body catalog: there is no pair to measure, and
    ``inf`` is not JSON-serialisable under ``allow_nan=False``.
    """
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return None
    if len(points) > max_points:
        points = points[:max_points]
    diff = points[:, None, :] - points[None, :, :]
    distances = np.sqrt(np.sum(np.square(diff), axis=-1))
    np.fill_diagonal(distances, np.inf)
    return float(np.min(distances))


# --------------------------------------------------------------------- sampling


def _latin_hypercube(rng: np.random.Generator, num: int, dim: int) -> np.ndarray:
    """Stratified unit-cube samples: one draw per stratum on every axis."""
    strata = (np.arange(num, dtype=np.float64)[:, None] + rng.random((num, dim))) / num
    for axis in range(dim):
        strata[:, axis] = strata[rng.permutation(num), axis]
    return strata


def _uniform(rng: np.random.Generator, num: int, dim: int) -> np.ndarray:
    return rng.random((num, dim))


def _boundary_unit_cube(
    rng: np.random.Generator, num: int, dim: int, *, edge_width: float
) -> np.ndarray:
    """Points concentrated on faces, edges and corners of the unit cube.

    ``num`` is split as evenly as the dimension allows across "how many
    coordinates are pinned to a face": 1 (face), 2 (edge), ... up to ``dim``
    (corner).  Pinned coordinates land inside a thin band at a bound; the rest
    stay in the interior.
    """
    points = np.empty((num, dim), dtype=np.float64)
    pinned_counts = 1 + np.arange(num) % dim
    for row in range(num):
        n_pinned = int(pinned_counts[row])
        axes = rng.choice(dim, size=n_pinned, replace=False)
        sample = rng.uniform(edge_width, 1.0 - edge_width, size=dim)
        at_high = rng.random(n_pinned) < 0.5
        band = rng.random(n_pinned) * edge_width
        sample[axes] = np.where(at_high, 1.0 - band, band)
        points[row] = sample
    return points


def _extrapolation(
    rng: np.random.Generator,
    num: int,
    low: np.ndarray,
    high: np.ndarray,
    *,
    margin: float,
    floor: np.ndarray,
) -> np.ndarray:
    """Bodies outside exactly one training bound at a time.

    ``margin`` is a fraction of each coordinate's training range.  ``floor``
    keeps the low-side extrapolation physically sensible (positive scales that
    still produce a standing robot).
    """
    dim = low.shape[0]
    span = high - low
    points = np.empty((num, dim), dtype=np.float64)
    for row in range(num):
        # interior on every axis, then push exactly one axis outside its bound
        sample = low + rng.uniform(0.15, 0.85, size=dim) * span
        axis = int(row % dim)
        overshoot = span[axis] * rng.uniform(0.05, margin)
        if row // dim % 2 == 0:
            sample[axis] = high[axis] + overshoot
        else:
            sample[axis] = max(low[axis] - overshoot, floor[axis])
        points[row] = sample
    return points


def build_catalog(
    *,
    num_bodies: int,
    seed: int,
    bounds_low: Sequence[float],
    bounds_high: Sequence[float],
    sampling_method: str = "latin_hypercube",
    split: str = "train",
    family: str = "h1",
    topology_signature: str = "unspecified",
    body_id_offset: int = 0,
    boundary_edge_width: float = 0.04,
    extrapolation_margin: float = 0.5,
    extrapolation_floor: Sequence[float] | None = None,
    notes: str = "",
) -> EmbodimentCatalog:
    """Generate one deterministic catalog.

    The same ``seed``, ``num_bodies``, ``sampling_method`` and bounds always
    produce the same descriptors and therefore the same content hash.
    """
    if num_bodies <= 0:
        raise ValueError("num_bodies must be positive.")
    low = np.asarray(bounds_low, dtype=np.float64)
    high = np.asarray(bounds_high, dtype=np.float64)
    if low.shape != high.shape or low.ndim != 1:
        raise ValueError("Bounds must be one-dimensional and the same length.")
    if np.any(low <= 0.0):
        raise ValueError("Morphology lower bounds must be positive.")
    if np.any(high <= low):
        raise ValueError("Every upper bound must exceed its lower bound.")

    rng = np.random.default_rng(seed)
    dim = low.shape[0]
    if sampling_method == "latin_hypercube":
        descriptors = low + _latin_hypercube(rng, num_bodies, dim) * (high - low)
    elif sampling_method == "uniform_iid":
        descriptors = low + _uniform(rng, num_bodies, dim) * (high - low)
    elif sampling_method == "boundary":
        unit = _boundary_unit_cube(
            rng, num_bodies, dim, edge_width=float(boundary_edge_width)
        )
        descriptors = low + unit * (high - low)
    elif sampling_method == "extrapolation":
        floor = (
            np.asarray(extrapolation_floor, dtype=np.float64)
            if extrapolation_floor is not None
            else low * 0.6
        )
        descriptors = _extrapolation(
            rng,
            num_bodies,
            low,
            high,
            margin=float(extrapolation_margin),
            floor=floor,
        )
    else:
        raise ValueError(f"Unknown sampling method {sampling_method!r}.")

    descriptors = np.round(descriptors, DESCRIPTOR_DECIMALS)
    unique = np.unique(descriptors, axis=0)
    if len(unique) != num_bodies:
        raise ValueError(
            f"Catalog generation produced {len(unique)} unique descriptors for "
            f"{num_bodies} requested bodies."
        )
    body_ids = np.arange(num_bodies, dtype=np.int64) + int(body_id_offset)
    return EmbodimentCatalog(
        schema_version=CATALOG_SCHEMA_VERSION,
        generator_revision=CATALOG_GENERATOR_REVISION,
        family=family,
        topology_signature=topology_signature,
        seed=int(seed),
        sampling_method=sampling_method,
        split=split,
        bounds_low=tuple(float(v) for v in low),
        bounds_high=tuple(float(v) for v in high),
        descriptor_names=tuple(MORPHOLOGY_NAMES[:dim]),
        body_ids=body_ids,
        descriptors=descriptors,
        notes=notes,
    )


def build_catalog_from_descriptors(
    descriptors: Sequence[Sequence[float]],
    *,
    bounds_low: Sequence[float],
    bounds_high: Sequence[float],
    split: str,
    family: str = "h1",
    topology_signature: str = "unspecified",
    body_id_offset: int = 0,
    notes: str = "",
) -> EmbodimentCatalog:
    """Wrap an explicit descriptor list, e.g. the named static XML variants.

    The bounds are the *training* bounds so the policy's descriptor
    normalisation is unchanged; explicit descriptors may legitimately fall
    outside them, which is what makes the extreme named variants a real
    extrapolation test.
    """
    descriptors = np.round(
        np.asarray(descriptors, dtype=np.float64), DESCRIPTOR_DECIMALS
    )
    if descriptors.ndim != 2:
        raise ValueError("Explicit descriptors must be two-dimensional.")
    unique = np.unique(descriptors, axis=0)
    if len(unique) != len(descriptors):
        raise ValueError(
            "Explicit descriptors contain duplicates. Variants that differ only "
            "in parameters outside the online descriptor (e.g. foot size) cannot "
            "be represented and must be excluded explicitly."
        )
    return EmbodimentCatalog(
        schema_version=CATALOG_SCHEMA_VERSION,
        generator_revision=CATALOG_GENERATOR_REVISION,
        family=family,
        topology_signature=topology_signature,
        seed=-1,
        sampling_method="explicit",
        split=split,
        bounds_low=tuple(float(v) for v in bounds_low),
        bounds_high=tuple(float(v) for v in bounds_high),
        descriptor_names=tuple(MORPHOLOGY_NAMES[: descriptors.shape[1]]),
        body_ids=np.arange(len(descriptors), dtype=np.int64) + int(body_id_offset),
        descriptors=descriptors,
        notes=notes,
    )


# ------------------------------------------------------------------ assignment


def fixed_balanced_assignment(num_envs: int, num_bodies: int) -> np.ndarray:
    """Body index for every environment slot under the fixed-balanced schedule.

    ``slot % num_bodies`` gives exactly ``num_envs // num_bodies`` replicas per
    body when the counts divide, and differs by at most one otherwise.
    """
    if num_envs <= 0 or num_bodies <= 0:
        raise ValueError("num_envs and num_bodies must be positive.")
    return np.arange(num_envs, dtype=np.int64) % num_bodies


def catalog_resample_assignment(
    slots: np.ndarray, generations: np.ndarray, num_bodies: int, stride: int = 1
) -> np.ndarray:
    """Body index after ``generations`` episode resets on each environment slot.

    The schedule is ``(slot + generation * stride) % num_bodies``.  Every slot
    walks the whole catalog, and with equal generations the exposure counts are
    identical to the fixed-balanced counts.
    """
    slots = np.asarray(slots, dtype=np.int64)
    generations = np.asarray(generations, dtype=np.int64)
    return (slots + generations * int(stride)) % int(num_bodies)


def exposure_counts(body_indices: np.ndarray, num_bodies: int) -> np.ndarray:
    return np.bincount(
        np.asarray(body_indices, dtype=np.int64).ravel(), minlength=num_bodies
    )


def exposure_summary(body_indices: np.ndarray, num_bodies: int) -> dict:
    counts = exposure_counts(body_indices, num_bodies)
    return {
        "num_bodies": int(num_bodies),
        "num_assignments": int(np.size(body_indices)),
        "min_exposure": int(counts.min()),
        "max_exposure": int(counts.max()),
        "mean_exposure": float(counts.mean()),
        "std_exposure": float(counts.std()),
        "num_bodies_with_zero_exposure": int(np.sum(counts == 0)),
        "balanced": bool(counts.max() - counts.min() <= 1),
    }
