"""Canonical future-motion features + FSQ codes for motion-latent commands.

The motion command z given to the shared URMA policy is keyed by canonical
(motion_id, timestamp) — never by one topology's private trajectory index.
This module produces every z variant behind one interface
(:class:`loco_mujoco.algorithms.TrajectoryLatentBuffer`):

- ``fake``     — `TrajectoryLatentBuffer.fake_from_trajectory_data` (upstream);
- ``oracle``   — fixed seeded projection of normalized canonical features,
  quantized onto the FSQ grid (no learning, temporally meaningful);
- ``learned``  — codes produced by the offline FSQ autoencoder
  (`train_fsq_motion.py`) and cached per canonical timestamp.

The ``FSQ`` class below is copied VERBATIM from the official Google Research
notebook shipped in this repo at ``fsq/fsq.ipynb`` (Mentzer et al.,
"Finite Scalar Quantization: VQ-VAE Made Simple", arXiv:2309.15505;
Apache 2.0).  Only cosmetic import wiring differs.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# Official FSQ quantizer (verbatim copy, see module docstring for provenance)
# ---------------------------------------------------------------------------

Codeword = jax.Array
Indices = jax.Array


def round_ste(z):
    """Round with straight through gradients."""
    zhat = jnp.round(z)
    return z + jax.lax.stop_gradient(zhat - z)


class FSQ:
    """Quantizer."""

    def __init__(self, levels: list[int], eps: float = 1e-3):
        self._levels = levels
        self._eps = eps
        self._levels_np = np.asarray(levels)
        self._basis = np.concatenate(
            ([1], np.cumprod(self._levels_np[:-1]))).astype(np.uint32)

        self._implicit_codebook = self.indexes_to_codes(
            np.arange(self.codebook_size))

    @property
    def num_dimensions(self) -> int:
        """Number of dimensions expected from inputs."""
        return len(self._levels)

    @property
    def codebook_size(self) -> int:
        """Size of the codebook."""
        return int(np.prod(self._levels_np))

    @property
    def codebook(self):
        """Returns the implicit codebook. Shape (prod(levels), num_dimensions)."""
        return self._implicit_codebook

    def bound(self, z: jax.Array) -> jax.Array:
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self._levels_np - 1) * (1 - self._eps) / 2
        offset = jnp.where(self._levels_np % 2 == 1, 0.0, 0.5)
        shift = jnp.tan(offset / half_l)
        return jnp.tanh(z + shift) * half_l - offset

    def quantize(self, z: jax.Array) -> Codeword:
        """Quanitzes z, returns quantized zhat, same shape as z."""
        quantized = round_ste(self.bound(z))

        # Renormalize to [-1, 1].
        half_width = self._levels_np // 2
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized):
        # Scale and shift to range [0, ..., L-1]
        half_width = self._levels_np // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat):
        half_width = self._levels_np // 2
        return (zhat - half_width) / half_width

    def codes_to_indexes(self, zhat: Codeword) -> Indices:
        """Converts a `code` to an index in the codebook."""
        assert zhat.shape[-1] == self.num_dimensions
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(axis=-1).astype(jnp.uint32)

    def indexes_to_codes(self, indices: Indices) -> Codeword:
        """Inverse of `indexes_to_codes`."""
        indices = indices[..., jnp.newaxis]
        codes_non_centered = np.mod(
            np.floor_divide(indices, self._basis), self._levels_np
        )
        return self._scale_and_shift_inverse(codes_non_centered)


# ---------------------------------------------------------------------------
# Canonical, embodiment-independent per-frame motion features
# ---------------------------------------------------------------------------

#: End-effector sites of the canonical (source) trajectory.  The mimic sites
#: exist on every LocoMuJoCo humanoid, but canonical features are always
#: computed from ONE source topology's trajectory so H1/G1/Atlas receive the
#: same code for the same motion.
CANONICAL_EE_SITES = (
    "left_foot_mimic",
    "right_foot_mimic",
    "left_hand_mimic",
    "right_hand_mimic",
)

#: name -> width of each canonical feature group, in emission order.
FEATURE_GROUPS = (
    ("root_height", 1),
    ("gravity_in_root", 3),
    ("root_linvel_in_root", 3),
    ("root_angvel", 3),
    ("ee_rel_in_root", 3 * len(CANONICAL_EE_SITES)),
)
FRAME_FEATURE_DIM = sum(width for _, width in FEATURE_GROUPS)


def feature_group_slices() -> dict[str, slice]:
    slices = {}
    start = 0
    for name, width in FEATURE_GROUPS:
        slices[name] = slice(start, start + width)
        start += width
    return slices


def _quat_to_rotmat(quat: np.ndarray) -> np.ndarray:
    """(N, 4) wxyz quaternions -> (N, 3, 3) rotation matrices."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    rot = np.empty((quat.shape[0], 3, 3), dtype=np.float64)
    rot[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rot[:, 0, 1] = 2 * (x * y - z * w)
    rot[:, 0, 2] = 2 * (x * z + y * w)
    rot[:, 1, 0] = 2 * (x * y + z * w)
    rot[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rot[:, 1, 2] = 2 * (y * z - x * w)
    rot[:, 2, 0] = 2 * (x * z - y * w)
    rot[:, 2, 1] = 2 * (y * z + x * w)
    rot[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rot


def canonical_frame_features(trajectory) -> np.ndarray:
    """(n_samples, FRAME_FEATURE_DIM) canonical features for one Trajectory.

    Root-relative task-space quantities only: no joint angles, no topology
    layout.  Heading (yaw) is NOT removed — the same motion played facing a
    different direction is still keyed by its clip identity, and all families
    share the source clip anyway.
    """
    data = trajectory.data
    info = trajectory.info
    qpos = np.asarray(data.qpos, dtype=np.float64)
    qvel = np.asarray(data.qvel, dtype=np.float64)
    site_names = list(info.site_names)
    site_xpos = np.asarray(data.site_xpos, dtype=np.float64)
    if site_xpos.ndim != 3:
        raise ValueError(
            "Trajectory has no site_xpos; canonical features need the "
            "completed (extended) trajectory."
        )
    missing = [s for s in CANONICAL_EE_SITES if s not in site_names]
    if missing:
        raise ValueError(f"Trajectory lacks canonical EE sites: {missing}")
    ee_indices = [site_names.index(s) for s in CANONICAL_EE_SITES]

    root_pos = qpos[:, 0:3]
    root_quat = qpos[:, 3:7]  # wxyz
    rot = _quat_to_rotmat(root_quat)  # root->world
    world_to_root = np.transpose(rot, (0, 2, 1))

    gravity_world = np.tile(np.asarray([0.0, 0.0, -1.0]), (qpos.shape[0], 1))
    gravity_in_root = np.einsum("nij,nj->ni", world_to_root, gravity_world)

    # MuJoCo free joint: linear velocity in world frame, angular in body frame.
    linvel_in_root = np.einsum("nij,nj->ni", world_to_root, qvel[:, 0:3])
    angvel = qvel[:, 3:6]

    ee_world = site_xpos[:, ee_indices, :]
    ee_rel = ee_world - root_pos[:, None, :]
    ee_in_root = np.einsum("nij,nkj->nki", world_to_root, ee_rel)

    features = np.concatenate(
        [
            root_pos[:, 2:3],
            gravity_in_root,
            linvel_in_root,
            angvel,
            ee_in_root.reshape(qpos.shape[0], -1),
        ],
        axis=-1,
    ).astype(np.float32)
    if features.shape[1] != FRAME_FEATURE_DIM:
        raise AssertionError(
            f"feature width {features.shape[1]} != {FRAME_FEATURE_DIM}"
        )
    return features


def future_window(
    features: np.ndarray,
    split_points: np.ndarray,
    window: int,
    stride: int,
) -> np.ndarray:
    """(N, window*D) future-motion windows, clamped at each clip's end.

    For timestamp t the window covers frames t, t+stride, ..., within the
    SAME clip; indices past the clip end repeat the final frame (documented
    clamp — no cross-clip leakage, no phase reset artifact).
    """
    split_points = np.asarray(split_points, dtype=np.int64)
    n = features.shape[0]
    if int(split_points[-1]) != n:
        raise ValueError("split_points do not match features")
    out = np.empty((n, window * features.shape[1]), dtype=np.float32)
    offsets = np.arange(window, dtype=np.int64) * stride
    for c in range(split_points.size - 1):
        start, stop = int(split_points[c]), int(split_points[c + 1])
        t = np.arange(start, stop, dtype=np.int64)
        idx = np.minimum(t[:, None] + offsets[None, :], stop - 1)
        out[start:stop] = features[idx].reshape(stop - start, -1)
    return out


# ---------------------------------------------------------------------------
# Providers — every z variant is a TrajectoryLatentBuffer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NormalizationStats:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "NormalizationStats":
        return cls(
            mean=x.mean(axis=0).astype(np.float32),
            std=(x.std(axis=0) + 1e-6).astype(np.float32),
        )

    def apply(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std


def oracle_codes(
    windows: np.ndarray,
    stats: NormalizationStats,
    levels: list[int],
    seed: int = 0,
) -> np.ndarray:
    """Deterministic FSQ-grid codes from a fixed projection (no learning)."""
    fsq = FSQ(levels)
    rng = np.random.default_rng(int(seed))
    projection = rng.normal(
        size=(windows.shape[1], fsq.num_dimensions)
    ).astype(np.float32) / np.sqrt(windows.shape[1])
    z = stats.apply(windows) @ projection
    return np.asarray(fsq.quantize(jnp.asarray(z)), dtype=np.float32)


def buffer_from_values(trajectory_data, values: np.ndarray):
    # Imported lazily so the FSQ trainer runs on machines whose loco-mujoco
    # checkout predates the latent module (e.g. the Viper repo copy).
    from loco_mujoco.algorithms import TrajectoryLatentBuffer

    return TrajectoryLatentBuffer.from_trajectory_data(trajectory_data, values)


def buffer_from_codes_npz(path):
    """Load a canonical token cache written by train_fsq_motion.py."""
    from loco_mujoco.algorithms import TrajectoryLatentBuffer

    with np.load(path) as payload:
        return TrajectoryLatentBuffer(
            values=payload["codes"].astype(np.float32),
            split_points=payload["split_points"].astype(np.int32),
        )
