"""Mandatory acceptance tests from TARGET_SPHERE_RENDERER_PATCH.md.

A target-sphere video is only evidence if the markers were produced by the same
computation the reward scored, on the same body, at the same trajectory phase.
These tests check exactly that; the video itself proves nothing.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "wgl" if sys.platform == "win32" else "osmesa")

WORKSPACE = Path(__file__).resolve().parents[1]
SCRIPTS = WORKSPACE / "scripts"
for _p in (str(SCRIPTS), str(SCRIPTS / "h1md")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jax.numpy as jnp  # noqa: E402

from loco_mujoco.core.utils.math import (  # noqa: E402
    calculate_relative_site_quatities,
)

from scaling.family_morphology import (  # noqa: E402
    FAMILY_MORPHOLOGY_HIGH,
    FAMILY_MORPHOLOGY_LOW,
)
from scaling.parallel_cross_humanoid_train import (  # noqa: E402
    _build_robot_env,
    _ensure_latent_defaults,
)
from scaling.render_cross_topology_policy import (  # noqa: E402
    BodyCorrectTargets,
    enforce_provider_agreement,
    render_track_with_targets,
)

REFERENCE_ROOT = WORKSPACE / "external_data" / "cross_humanoid"
ROBOTS = ("h1", "g1")
#: verify_fk_targets.py's tolerance for "FK on the nominal body reproduces the
#: trajectory's own stored site data".
NOMINAL_STORED_TOL_M = 5e-3
#: the patch document's tolerance for CPU-vs-production agreement.
PROVIDER_TOL_M = 1e-5

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not (REFERENCE_ROOT / "h1_source" / "dance2_subject4" / "h1").is_dir(),
        reason="cross-humanoid reference cache not present",
    ),
]


_ENV_CACHE: dict = {}


def _env(robot: str):
    if robot not in _ENV_CACHE:
        args = _ensure_latent_defaults(
            SimpleNamespace(
                source="h1",
                reference_mode="direct",
                reference_root=REFERENCE_ROOT,
                clip=None,
                start_frame=None,
                frames=None,
                clip_windows=["dance2_subject4:19482:800"],
                morphology="continuous",
                use_mjwarp=False,
                reward_type="MorphMimicReward",
                goal_type="MorphGoalTrajMimicRootErr",
            )
        )
        env, _ = _build_robot_env(args, robot)
        _ENV_CACHE[robot] = env
    return _ENV_CACHE[robot]


def _targets(robot: str) -> BodyCorrectTargets:
    env = _env(robot)
    return BodyCorrectTargets(env, robot, f"{robot}.npz")


@pytest.mark.parametrize("robot", ROBOTS)
def test_nominal_markers_match_stored_trajectory_sites(robot):
    """Acceptance 1: at nominal morphology the markers reproduce the
    trajectory's own site data -- the one body where the two must agree."""
    env = _env(robot)
    targets = _targets(robot)
    nominal = np.ones(FAMILY_MORPHOLOGY_LOW.shape[0])

    for step in (3, 111, 400, 731):
        _, _, rpos = targets.targets(nominal, 0, step)
        stored = env.th.traj.data.get(0, step, jnp)
        stored_rpos, _, _ = calculate_relative_site_quatities(
            stored, targets.site_ids, targets.body_ids, env.sys.body_rootid, jnp
        )
        assert np.abs(rpos - np.asarray(stored_rpos)).max() < NOMINAL_STORED_TOL_M


@pytest.mark.parametrize("robot", ROBOTS)
@pytest.mark.parametrize("corner", ["low", "high"])
def test_corner_markers_match_reward_targets(robot, corner):
    """Acceptance 2: on corner bodies the markers equal the quantities
    ``MorphMimicReward`` scores, within the patch's 1e-5 m."""
    targets = _targets(robot)
    morphology = (
        FAMILY_MORPHOLOGY_LOW if corner == "low" else FAMILY_MORPHOLOGY_HIGH
    ).astype(np.float64)

    for step in (3, 400):
        error = targets.provider_error(morphology, 0, step)
        assert error < PROVIDER_TOL_M, (
            f"{robot} {corner} phase {step}: marker targets differ from the "
            f"scored targets by {error:.2e} m"
        )


@pytest.mark.parametrize("robot", ROBOTS)
def test_phase_p_plus_one_fails_the_phase_p_check(robot):
    """Acceptance 3: the same frame at p+1 must not pass p's check.

    Without this the whole overlay could be one phase off and still look right.
    """
    targets = _targets(robot)
    nominal = np.ones(FAMILY_MORPHOLOGY_LOW.shape[0])
    for step in (111, 400):
        _, _, rpos_p = targets.targets(nominal, 0, step)
        _, _, rpos_next = targets.targets(nominal, 0, step + 1)
        assert np.abs(rpos_p - rpos_next).max() > 100 * PROVIDER_TOL_M


@pytest.mark.parametrize("robot", ROBOTS)
def test_morphology_change_moves_markers_and_bumps_generation(robot):
    """Acceptance 4: a morphology reset must move the markers AND show up in
    the sidecar generation -- a cached-by-phase target would do neither."""
    targets = _targets(robot)
    env = _env(robot)
    qpos = np.asarray(env.th.traj.data.get(0, 111, jnp).qpos)
    nominal = np.ones(FAMILY_MORPHOLOGY_LOW.shape[0])
    tall = FAMILY_MORPHOLOGY_HIGH.astype(np.float64)

    log = {
        "qpos": np.stack([qpos, qpos]),
        "traj_no": np.array([0, 0]),
        "subtraj_step_no": np.array([111, 111]),
        "morphology": np.stack([nominal, tall]),
        "morphology_generation": np.array([1, 2]),
        "absorbing": np.array([False, False]),
        "reset_happened": np.array([False, True]),
    }
    frames, rows = render_track_with_targets(
        targets, log, 200, 240, 3.4, -12.0, True, True
    )

    assert len(frames) == 2
    assert rows[0]["morphology_generation"] != rows[1]["morphology_generation"]
    assert rows[1]["reset_happened"] is True

    _, _, rpos_nominal = targets.targets(nominal, 0, 111)
    _, _, rpos_tall = targets.targets(tall, 0, 111)
    assert np.abs(rpos_nominal - rpos_tall).max() > 5e-3, (
        "markers did not move with the body (target cached by phase alone?)"
    )
    # every row carries the fields the sidecar schema requires
    for row in rows:
        assert set(row) >= {
            "frame", "family", "traj_no", "subtraj_step_no", "morphology",
            "morphology_generation", "absorbing", "target_provider_max_error_m",
        }


def test_families_use_their_own_retargeted_trajectories():
    """Acceptance 5: H1 and G1 markers come from different reference files.

    If both families silently read one trajectory the site targets would still
    be self-consistent -- and completely wrong for one of them.
    """
    h1, g1 = _env("h1"), _env("g1")
    q_h1 = np.asarray(h1.th.traj.data.get(0, 400, jnp).qpos)
    q_g1 = np.asarray(g1.th.traj.data.get(0, 400, jnp).qpos)
    assert q_h1.shape != q_g1.shape or not np.allclose(q_h1, q_g1)

    nominal = np.ones(FAMILY_MORPHOLOGY_LOW.shape[0])
    _, _, rpos_h1 = _targets("h1").targets(nominal, 0, 400)
    _, _, rpos_g1 = _targets("g1").targets(nominal, 0, 400)
    assert rpos_h1.shape == rpos_g1.shape  # same 5-site mimic set
    assert not np.allclose(rpos_h1, rpos_g1)


def test_renderer_fails_closed_on_provider_disagreement():
    """Acceptance 6: exceeding the threshold exits non-zero."""
    enforce_provider_agreement(1e-7, PROVIDER_TOL_M)  # agreement: no raise
    with pytest.raises(SystemExit) as excinfo:
        enforce_provider_agreement(1e-3, PROVIDER_TOL_M)
    assert excinfo.value.code == 3
