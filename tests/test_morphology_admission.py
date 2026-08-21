"""P1 tests: bounded body admission, rejection accounting, fail-closed.

The sampler used to consume every draw and count nothing, so "zero bodies were
skipped" was not a claim the run could make -- only one it could fail to
contradict. These tests check that the counters exist, that they are produced by
the same pass that does the rejecting, that the resample budget is bounded, and
that exhausting it condemns the episode instead of quietly substituting a
nominal body.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

WORKSPACE = Path(__file__).resolve().parents[1]
SCRIPTS = WORKSPACE / "scripts"
for _p in (str(SCRIPTS), str(SCRIPTS / "h1md")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jax  # noqa: E402

from scaling.family_morphology import FAMILY_MORPHOLOGY_NAMES  # noqa: E402
from scaling.morphology_admission import (  # noqa: E402
    NUM_REASONS,
    REJECTION_REASONS,
    AdmissionConfig,
    admission_census,
    summarize_admission,
    zero_stats,
)
from scaling.parallel_cross_humanoid_train import (  # noqa: E402
    _build_robot_env,
    _ensure_latent_defaults,
)

REFERENCE_ROOT = WORKSPACE / "external_data" / "cross_humanoid"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not (REFERENCE_ROOT / "h1_source" / "dance2_subject4" / "h1").is_dir(),
        reason="cross-humanoid reference cache not present",
    ),
]


def _env_args(**overrides):
    base = dict(
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
    base.update(overrides)
    return _ensure_latent_defaults(SimpleNamespace(**base))


_ENV_CACHE: dict = {}


def _env(robot: str, **overrides):
    key = (robot, tuple(sorted((k, repr(v)) for k, v in overrides.items())))
    if key not in _ENV_CACHE:
        env, _ = _build_robot_env(_env_args(**overrides), robot)
        _ENV_CACHE[key] = env
    return _ENV_CACHE[key]


def test_reason_table_is_stable():
    """Reason index IS the histogram column; reordering silently corrupts
    every manifest written before the change."""
    assert REJECTION_REASONS == (
        "nonfinite_model",
        "invalid_mass_or_inertia",
        "invalid_joint_range",
        "reference_limit_violation",
        "initial_penetration",
        "initial_absorbing",
        "fk_nonfinite",
        "reference_screen_failed",
    )
    assert NUM_REASONS == len(REJECTION_REASONS)


@pytest.mark.parametrize("robot", ["h1", "g1"])
def test_conservative_box_admits_every_draw_and_counts_it(robot):
    """Zero rejections is fine -- but it has to be a number, not a silence."""
    env = _env(robot)
    states = jax.jit(jax.vmap(env.mjx_reset))(
        jax.random.split(jax.random.PRNGKey(7), 16)
    )
    stats = states.additional_carry.admission

    assert int(np.sum(np.asarray(stats.draws_total))) == 16
    assert int(np.sum(np.asarray(stats.accepted_total))) == 16
    assert int(np.sum(np.asarray(stats.rejected_total))) == 0
    assert int(np.sum(np.asarray(stats.resamples_total))) == 0
    assert int(np.sum(np.asarray(stats.resample_exhausted_total))) == 0
    assert not bool(np.any(np.asarray(states.additional_carry.admission_failed)))


@pytest.mark.parametrize("robot", ["h1", "g1"])
def test_impossible_screen_exhausts_the_budget_and_fails_closed(robot):
    """A check nothing can pass must exhaust a BOUNDED budget and condemn the
    episode -- never fall back to a nominal body.

    A negative reference tolerance rejects every draw, which is precisely the
    behaviour a genuinely over-tight future randomization would produce.
    """
    budget = 3
    env = _env(
        robot,
        admission_config=AdmissionConfig(
            max_resamples=budget, reference_limit_tol=-1.0
        ),
    )
    state = jax.jit(env.mjx_reset)(jax.random.PRNGKey(1))
    carry = state.additional_carry
    stats = carry.admission

    assert int(stats.draws_total) == budget + 1, "resampling is not bounded"
    assert int(stats.accepted_total) == 0
    assert int(stats.rejected_total) >= budget + 1
    assert int(stats.resamples_total) == budget
    assert int(stats.resample_exhausted_total) == 1
    assert bool(carry.admission_failed), "exhausted budget did not fail closed"

    reasons = np.asarray(stats.rejections_by_reason)
    assert reasons[REJECTION_REASONS.index("reference_limit_violation")] > 0

    # fail closed means: the episode ends, it does not run on a rejected body
    absorbing, _ = env._mjx_is_absorbing(
        state.observation, state.info, state.data, carry
    )
    assert bool(absorbing)

    # and the accepted-body statistics must not have absorbed the rejected draw
    assert not np.isfinite(np.asarray(stats.morphology_min)).any()

    # Counter invariants. A body rejected by BOTH the reference screen and a
    # post-reset check must not be subtracted from an accepted_total it was
    # never added to -- negative counters are worse than none, because they
    # still look like numbers.
    assert int(stats.accepted_total) >= 0
    assert int(stats.rejected_total) <= int(stats.draws_total)
    assert int(stats.accepted_total) + int(stats.rejected_total) == int(
        stats.draws_total
    )


@pytest.mark.parametrize("robot", ["h1", "g1"])
def test_disabled_admission_reproduces_the_old_sampler(robot):
    """Turning the checks off must change the distribution, not the counters.

    This is the migration guarantee: accounting can be switched on without
    altering what the policy trains on.
    """
    env = _env(robot, admission_config=AdmissionConfig(enabled=False))
    state = jax.jit(env.mjx_reset)(jax.random.PRNGKey(2))
    stats = state.additional_carry.admission
    assert int(stats.draws_total) == 1
    assert int(stats.resamples_total) == 0
    assert not bool(state.additional_carry.admission_failed)


def test_summary_uses_the_key_names_the_auditor_greps_for():
    """audit_h1g1_pipeline.py's body_rejection_accounting check looks for
    'rejected_bodies' / 'reject_reasons'; a differently-named counter block is
    invisible to it and the run stays 'unsupported'."""
    stats = {"h1": zero_stats(len(FAMILY_MORPHOLOGY_NAMES))}
    summary = summarize_admission(stats, FAMILY_MORPHOLOGY_NAMES)

    assert "rejected_bodies" in summary["total"]
    assert "reject_reasons" in summary["total"]
    assert set(summary["total"]["reject_reasons"]) == set(REJECTION_REASONS)
    assert summary["fail_closed"] is True
    assert set(summary["per_family"]["h1"]["morphology"]) == set(
        FAMILY_MORPHOLOGY_NAMES
    )
    # unpopulated extremes must serialise as null, not as inf
    assert summary["per_family"]["h1"]["morphology"]["leg_length_scale"]["min"] is None


def test_census_reports_quantiles_and_is_json_safe():
    import json

    env = _env("h1")
    payload = admission_census([env], ["h1"], seed=5, resets=2, envs_per_family=4)

    assert payload["supported"] is True
    assert payload["resets_per_family"] == 8
    assert payload["total"]["draws_total"] == 8
    assert payload["total"]["rejected_bodies"] == 0
    quantiles = payload["per_family"]["h1"]["morphology_quantiles"]
    assert set(quantiles) == set(FAMILY_MORPHOLOGY_NAMES)
    for coordinate in FAMILY_MORPHOLOGY_NAMES:
        block = quantiles[coordinate]
        assert block["q00"] <= block["q50"] <= block["q100"]
    # allow_nan=False is what the trainer writes manifests with
    json.dumps(payload, allow_nan=False)
