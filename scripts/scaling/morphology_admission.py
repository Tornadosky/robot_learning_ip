"""Bounded admission control for sampled bodies, with rejection accounting.

Why this exists
---------------
The conservative four-dimensional H1/G1 sampler uses **every** draw. Nothing
checks that the body it produced is finite, has positive mass and inertia, has
well-ordered joint ranges, can reach the reference at the phase it is about to
be reset into, or is standing inside its own terminal window. And because there
are no counters, a run cannot even report that zero bodies were skipped -- it
can only fail to mention it, which reads the same in a manifest and is not the
same claim.

Widening randomization (joint axes, joint ranges, foot size, more extreme
geometry) makes an invalid draw a matter of when, not if. This module is the
gate that has to exist first.

Design
------
``admit_morphology`` is a ``jax.lax.while_loop`` over draws, bounded by
``max_resamples``. Each iteration samples a body, applies it, runs the checks,
and stops on the first admissible one. The loop carries a per-reason rejection
histogram, so the counters below come out of the same pass that does the
rejecting -- there is no second, host-side estimate that could disagree.

**Fail closed.** If the budget is exhausted the loop does *not* fall back to a
nominal body: it keeps the last (rejected) draw and raises the
``resample_exhausted`` flag, which :class:`AdmissionStats` carries into the
carry. ``FamilyMorphMixin`` makes such an episode absorbing at reset, and
``summarize_admission`` reports it; a silent nominal substitution would turn a
sampler bug into a quietly narrower training distribution.

**Zero is a measurement.** For today's conservative bounds every draw is
expected to pass. That expectation is exactly what these counters exist to test,
so the numbers are logged whether or not they are all zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

#: Rejection reasons, in histogram order. The index of a reason IS its column in
#: ``AdmissionStats.rejections_by_reason``; never reorder without bumping the
#: manifests that read it back by name.
REJECTION_REASONS: tuple[str, ...] = (
    "nonfinite_model",
    "invalid_mass_or_inertia",
    "invalid_joint_range",
    "reference_limit_violation",
    "initial_penetration",
    "initial_absorbing",
    "fk_nonfinite",
    "reference_screen_failed",
)
REASON_INDEX = {name: i for i, name in enumerate(REJECTION_REASONS)}
NUM_REASONS = len(REJECTION_REASONS)

#: Fixed budget. Bounded on purpose: an unbounded resample loop turns an
#: impossible constraint into a hang instead of a reported failure.
DEFAULT_MAX_RESAMPLES = 8

#: Minimum acceptable body mass (kg) and diagonal inertia (kg m^2). Below these
#: MuJoCo's solver is not merely inaccurate, it is ill-conditioned.
MIN_BODY_MASS = 1e-6
MIN_BODY_INERTIA = 1e-9

#: How far the joint-range clamp may move the reference before the body counts
#: as unable to perform the motion (radians).
DEFAULT_REFERENCE_LIMIT_TOL = 1e-3

#: Contact penetration depth (m) tolerated at reset.
DEFAULT_MAX_INITIAL_PENETRATION = 5e-3


@struct.dataclass
class AdmissionStats:
    """Per-environment admission counters, carried in the env carry.

    Every field is a scalar or a fixed-width vector so the whole thing vmaps and
    sums across environments without a host round-trip. Aggregate with
    :func:`summarize_admission`.

    Attributes:
        draws_total: bodies drawn, including rejected ones.
        accepted_total: episodes that started on an admissible body.
        rejected_total: draws that failed at least one check.
        resamples_total: extra draws caused by a rejection (``draws - resets``).
        resample_exhausted_total: episodes that ran out of budget and started on
            a rejected body. Must be zero for a production claim.
        rejections_by_reason: histogram over :data:`REJECTION_REASONS`. A single
            draw can fail several checks and is counted under each.
        morphology_min / morphology_max: per-coordinate extremes of the ACCEPTED
            bodies, i.e. the distribution actually trained on.
        morphology_sum / morphology_sumsq: accumulators for mean/std; quantiles
            come from the sampled log the trainer writes, not from these.
    """

    draws_total: jax.Array
    accepted_total: jax.Array
    rejected_total: jax.Array
    resamples_total: jax.Array
    resample_exhausted_total: jax.Array
    rejections_by_reason: jax.Array
    morphology_min: jax.Array
    morphology_max: jax.Array
    morphology_sum: jax.Array
    morphology_sumsq: jax.Array


def zero_stats(morphology_dim: int) -> AdmissionStats:
    """A fresh counter block. ``min``/``max`` start at the identity for their
    reduction so the first accepted body sets them exactly."""
    return AdmissionStats(
        draws_total=jnp.zeros((), dtype=jnp.int32),
        accepted_total=jnp.zeros((), dtype=jnp.int32),
        rejected_total=jnp.zeros((), dtype=jnp.int32),
        resamples_total=jnp.zeros((), dtype=jnp.int32),
        resample_exhausted_total=jnp.zeros((), dtype=jnp.int32),
        rejections_by_reason=jnp.zeros((NUM_REASONS,), dtype=jnp.int32),
        morphology_min=jnp.full((morphology_dim,), jnp.inf, dtype=jnp.float32),
        morphology_max=jnp.full((morphology_dim,), -jnp.inf, dtype=jnp.float32),
        morphology_sum=jnp.zeros((morphology_dim,), dtype=jnp.float32),
        morphology_sumsq=jnp.zeros((morphology_dim,), dtype=jnp.float32),
    )


@dataclass(frozen=True)
class AdmissionConfig:
    """Which checks run, and how strict they are.

    ``enabled=False`` reproduces the pre-admission sampler exactly (one draw,
    always used) while still emitting counters, so the accounting can be turned
    on without changing the training distribution.
    """

    enabled: bool = True
    max_resamples: int = DEFAULT_MAX_RESAMPLES
    reference_limit_tol: float = DEFAULT_REFERENCE_LIMIT_TOL
    max_initial_penetration: float = DEFAULT_MAX_INITIAL_PENETRATION
    check_reference_limits: bool = True
    check_penetration: bool = True
    #: Phases screened for reference reachability, as fractions of the clip.
    reference_screen_fractions: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75)


# --------------------------------------------------------------------------- #
# the checks
# --------------------------------------------------------------------------- #
def _finite(*arrays) -> jax.Array:
    ok = jnp.asarray(True)
    for array in arrays:
        ok = jnp.logical_and(ok, jnp.all(jnp.isfinite(array)))
    return ok


def model_rejection_flags(body_model, config: AdmissionConfig) -> jax.Array:
    """Model-level checks. Returns a boolean vector over REJECTION_REASONS."""
    flags = [jnp.asarray(False)] * NUM_REASONS

    finite = _finite(
        body_model.body_pos,
        body_model.body_ipos,
        body_model.body_mass,
        body_model.body_inertia,
        body_model.dof_damping,
        body_model.actuator_gainprm,
        body_model.actuator_forcerange,
        body_model.jnt_range,
    )
    flags[REASON_INDEX["nonfinite_model"]] = jnp.logical_not(finite)

    # body 0 is the world body: massless and inertialess by construction.
    mass_ok = jnp.all(body_model.body_mass[1:] > MIN_BODY_MASS)
    inertia_ok = jnp.all(body_model.body_inertia[1:] > MIN_BODY_INERTIA)
    flags[REASON_INDEX["invalid_mass_or_inertia"]] = jnp.logical_not(
        jnp.logical_and(mass_ok, inertia_ok)
    )

    limited = jnp.asarray(np.asarray(body_model.jnt_limited) != 0)
    range_ok = jnp.all(
        jnp.where(limited, body_model.jnt_range[:, 1] > body_model.jnt_range[:, 0], True)
    )
    flags[REASON_INDEX["invalid_joint_range"]] = jnp.logical_not(range_ok)

    return jnp.stack(flags)


def reference_rejection_flags(env, body_model, screen_steps, config: AdmissionConfig):
    """Reference-reachability checks on the sampled body.

    ``reference_limit_violation`` fires when this body's joint ranges clip the
    reference by more than ``reference_limit_tol`` at the reset phase --
    i.e. the motion is not merely hard for the body, it is unreachable.
    ``reference_screen_failed`` is the same test applied across the screening
    phases, so a body that can hit the start pose but not the middle of the clip
    is still rejected.

    Deliberately allocation-free. An earlier version ran ``mjx.make_data`` plus a
    kinematics pass here to test ``fk_nonfinite``; inside the resample
    ``while_loop``, under vmap over 768 environments, that allocated a full MJX
    ``Data`` per loop body and aborted the MI300A run with
    ``HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION``. The FK check now rides on the
    forward pass the reset already performs -- see
    :func:`state_rejection_flags` -- which is both cheaper and a truer test,
    because it checks the FK that actually happens.
    """
    from scaling.body_correct_reference import clamp_reference_qpos

    flags = [jnp.asarray(False)] * NUM_REASONS
    if not config.check_reference_limits:
        return jnp.stack(flags)

    traj_no = screen_steps[0]
    steps = screen_steps[1]

    def clip_gap(step):
        raw = env.th.traj.data.get(traj_no, step, jnp).qpos
        clamped = clamp_reference_qpos(raw, body_model.jnt_range, env.sys, jnp)
        return jnp.max(jnp.abs(clamped - raw))

    gaps = jax.vmap(clip_gap)(steps)
    flags[REASON_INDEX["reference_limit_violation"]] = (
        gaps[0] > config.reference_limit_tol
    )
    flags[REASON_INDEX["reference_screen_failed"]] = jnp.any(
        gaps > config.reference_limit_tol
    )
    return jnp.stack(flags)


def state_rejection_flags(env, body_model, data, carry, config: AdmissionConfig):
    """Post-reset checks that need the simulator state.

    ``initial_penetration``: MuJoCo reports negative ``contact.dist`` for
    interpenetration, and the soft-contact defect makes deep initial penetration
    the failure mode a randomized leg length actually produces.
    ``initial_absorbing``: the body starts outside its own terminal window, so
    the episode would be a single wasted step.
    ``fk_nonfinite``: the forward pass the reset just ran produced a non-finite
    site or body frame -- checked here, on the real data, rather than on a
    throwaway copy inside the resample loop.
    """
    flags = [jnp.asarray(False)] * NUM_REASONS

    flags[REASON_INDEX["fk_nonfinite"]] = jnp.logical_not(
        _finite(data.site_xpos, data.xpos, data.xmat)
    )

    if config.check_penetration:
        # mujoco >= 3.9 moved Data.contact behind _impl and deprecated the
        # direct attribute; support both so this file is not version-locked.
        impl = getattr(data, "_impl", None)
        contact = getattr(impl, "contact", None) if impl is not None else None
        if contact is None:
            contact = getattr(data, "contact", None)
        dist = getattr(contact, "dist", None)
        if dist is not None and np.prod(jnp.shape(dist)) > 0:
            penetration = jnp.max(jnp.maximum(-dist, 0.0))
            flags[REASON_INDEX["initial_penetration"]] = (
                penetration > config.max_initial_penetration
            )

    handler = getattr(env, "_terminal_state_handler", None)
    if handler is not None and getattr(handler, "initialized", False):
        absorbing, _ = handler._is_absorbing_compat(
            env, None, {}, data, carry, jnp
        )
        flags[REASON_INDEX["initial_absorbing"]] = jnp.asarray(absorbing)

    return jnp.stack(flags)


# --------------------------------------------------------------------------- #
# the bounded sampler
# --------------------------------------------------------------------------- #
def admit_morphology(
    env,
    key: jax.Array,
    sample_fn,
    screen_steps,
    config: AdmissionConfig,
):
    """Draw an admissible body, bounded by ``config.max_resamples``.

    Only the model- and reference-level checks run inside the loop; the
    state-level ones (penetration, initial absorbing) need a stepped ``data``
    and are folded in afterwards by :func:`record_state_rejections`, which can
    still mark the episode.

    Args:
        env: the family-morph environment.
        key: PRNG key.
        sample_fn: ``key -> morphology`` (the env's own sampler).
        screen_steps: ``(traj_no, steps)`` to screen the reference at.
        config: see :class:`AdmissionConfig`.

    Returns:
        ``(morphology, flags, draws, exhausted)`` -- ``flags`` is the rejection
        histogram of the RETURNED body (all-False when admitted), ``draws`` the
        number of bodies drawn, and ``exhausted`` whether the budget ran out
        with no admissible body found.
    """
    if not config.enabled:
        key, draw_key = jax.random.split(key)
        morphology = sample_fn(draw_key)
        return (
            morphology,
            jnp.zeros((NUM_REASONS,), dtype=bool),
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(False),
        )

    max_draws = int(config.max_resamples) + 1

    def draw(carry):
        loop_key, _, _, draws = carry
        loop_key, draw_key = jax.random.split(loop_key)
        morphology = sample_fn(draw_key)
        body_model = env._apply_morphology(env.sys, morphology)
        flags = jnp.logical_or(
            model_rejection_flags(body_model, config),
            reference_rejection_flags(env, body_model, screen_steps, config),
        )
        return loop_key, morphology, flags, draws + 1

    def keep_going(carry):
        _, _, flags, draws = carry
        return jnp.logical_and(jnp.any(flags), draws < max_draws)

    init = draw(
        (
            key,
            jnp.zeros((), dtype=jnp.float32),
            jnp.zeros((NUM_REASONS,), dtype=bool),
            jnp.asarray(0, dtype=jnp.int32),
        )
    )
    _, morphology, flags, draws = jax.lax.while_loop(keep_going, draw, init)
    exhausted = jnp.any(flags)  # loop exits either admitted or out of budget
    return morphology, flags, draws, exhausted


def update_stats(
    stats: AdmissionStats,
    morphology: jax.Array,
    flags: jax.Array,
    draws: jax.Array,
    exhausted: jax.Array,
) -> AdmissionStats:
    """Fold one reset's outcome into the running counters."""
    rejected = jnp.any(flags)
    accepted = jnp.logical_not(rejected)
    morphology = morphology.astype(jnp.float32)
    # Only ACCEPTED bodies enter the morphology extremes: the point of the
    # quantiles is to describe the distribution actually trained on.
    keep = accepted
    return stats.replace(
        draws_total=stats.draws_total + draws,
        accepted_total=stats.accepted_total + accepted.astype(jnp.int32),
        rejected_total=stats.rejected_total + (draws - accepted.astype(jnp.int32)),
        resamples_total=stats.resamples_total + jnp.maximum(draws - 1, 0),
        resample_exhausted_total=(
            stats.resample_exhausted_total + exhausted.astype(jnp.int32)
        ),
        rejections_by_reason=stats.rejections_by_reason + flags.astype(jnp.int32),
        morphology_min=jnp.where(
            keep, jnp.minimum(stats.morphology_min, morphology), stats.morphology_min
        ),
        morphology_max=jnp.where(
            keep, jnp.maximum(stats.morphology_max, morphology), stats.morphology_max
        ),
        morphology_sum=stats.morphology_sum + jnp.where(keep, morphology, 0.0),
        morphology_sumsq=stats.morphology_sumsq
        + jnp.where(keep, jnp.square(morphology), 0.0),
    )


def record_state_rejections(
    stats: AdmissionStats, flags: jax.Array, was_accepted: jax.Array
) -> tuple[AdmissionStats, jax.Array]:
    """Fold post-reset checks in, and report whether the episode is condemned.

    These checks cannot drive a resample (they need a reset that already
    happened), so the body is counted as rejected and the caller makes the
    episode absorbing -- fail closed, not silently accepted.

    ``was_accepted`` is what :func:`update_stats` concluded from the model- and
    reference-level checks. Without it a body that failed BOTH stages would be
    subtracted from ``accepted_total`` it was never added to, and the counters
    would go negative -- which is worse than no counters, because it looks like
    a number.

    ``rejections_by_reason`` is incremented regardless: the histogram records
    what was observed about the body, and a single draw failing several checks
    already contributes to several reasons.
    """
    rejected = jnp.any(flags)
    newly_rejected = jnp.logical_and(rejected, was_accepted).astype(jnp.int32)
    stats = stats.replace(
        rejected_total=stats.rejected_total + newly_rejected,
        accepted_total=stats.accepted_total - newly_rejected,
        rejections_by_reason=stats.rejections_by_reason + flags.astype(jnp.int32),
    )
    return stats, rejected


# --------------------------------------------------------------------------- #
# host-side reporting
# --------------------------------------------------------------------------- #
def summarize_admission(
    stats_by_family: dict, morphology_names: Sequence[str]
) -> dict:
    """JSON-ready aggregate, one block per family plus a total.

    Key names are chosen so ``audit_h1g1_pipeline.py`` finds them:
    ``rejected_bodies`` and ``reject_reasons`` are what its
    ``body_rejection_accounting`` check greps for. A run that reports zeros here
    is making a measurement; a run with no such block is making no claim at all.
    """
    families = {}
    totals = {
        "draws_total": 0,
        "accepted_total": 0,
        "rejected_bodies": 0,
        "resamples_total": 0,
        "resample_exhausted_total": 0,
        "reject_reasons": {name: 0 for name in REJECTION_REASONS},
    }

    for family, stats in stats_by_family.items():
        draws = int(np.sum(np.asarray(stats.draws_total)))
        accepted = int(np.sum(np.asarray(stats.accepted_total)))
        rejected = int(np.sum(np.asarray(stats.rejected_total)))
        resamples = int(np.sum(np.asarray(stats.resamples_total)))
        exhausted = int(np.sum(np.asarray(stats.resample_exhausted_total)))
        reasons = np.asarray(stats.rejections_by_reason)
        if reasons.ndim > 1:
            reasons = reasons.sum(axis=tuple(range(reasons.ndim - 1)))
        reason_counts = {
            name: int(reasons[i]) for i, name in enumerate(REJECTION_REASONS)
        }

        lo = np.asarray(stats.morphology_min, dtype=np.float64)
        hi = np.asarray(stats.morphology_max, dtype=np.float64)
        total = np.asarray(stats.morphology_sum, dtype=np.float64)
        sumsq = np.asarray(stats.morphology_sumsq, dtype=np.float64)
        if lo.ndim > 1:
            lo = lo.min(axis=tuple(range(lo.ndim - 1)))
            hi = hi.max(axis=tuple(range(hi.ndim - 1)))
            total = total.sum(axis=tuple(range(total.ndim - 1)))
            sumsq = sumsq.sum(axis=tuple(range(sumsq.ndim - 1)))
        n = max(accepted, 1)
        mean = total / n
        var = np.maximum(sumsq / n - np.square(mean), 0.0)

        families[family] = {
            "draws_total": draws,
            "accepted_total": accepted,
            "rejected_bodies": rejected,
            "resamples_total": resamples,
            "resample_exhausted_total": exhausted,
            "reject_reasons": reason_counts,
            "morphology": {
                name: {
                    "min": _finite_or_none(lo[i]),
                    "max": _finite_or_none(hi[i]),
                    "mean": _finite_or_none(mean[i]),
                    "std": _finite_or_none(float(np.sqrt(var[i]))),
                }
                for i, name in enumerate(morphology_names)
            },
        }
        totals["draws_total"] += draws
        totals["accepted_total"] += accepted
        totals["rejected_bodies"] += rejected
        totals["resamples_total"] += resamples
        totals["resample_exhausted_total"] += exhausted
        for name, count in reason_counts.items():
            totals["reject_reasons"][name] += count

    return {
        "per_family": families,
        "total": totals,
        "fail_closed": totals["resample_exhausted_total"] == 0,
        "note": (
            "Zero rejected bodies is a measurement produced by these counters, "
            "not an assumption; resample_exhausted_total > 0 means at least one "
            "episode started on a rejected body and the run is not admissible."
        ),
    }


def _finite_or_none(value):
    value = float(value)
    return value if np.isfinite(value) else None


def admission_census(
    raw_envs, names: Sequence[str], seed: int, resets: int, envs_per_family: int
) -> dict:
    """Measure the admission behaviour of each family's sampler.

    The in-graph counters protect the training run (an inadmissible body ends
    its episode), but they live inside PPO's carry and never reach a manifest.
    This is the reportable measurement: ``resets x envs_per_family`` independent
    resets per family, drawn by the same sampler, screened by the same checks,
    with the accepted bodies kept so the distribution can be quantiled.

    It is a census of the sampler, not of the training run -- stated that way in
    the returned payload so a reader cannot mistake it for one.
    """
    import jax

    per_family_stats = {}
    accepted_bodies: dict[str, np.ndarray] = {}

    for name, env in zip(names, raw_envs):
        if not hasattr(env, "_apply_morphology"):
            continue
        total = int(resets) * int(envs_per_family)
        if total <= 0:
            continue
        keys = jax.random.split(jax.random.PRNGKey(int(seed)), total)
        states = jax.jit(jax.vmap(env.mjx_reset))(keys)
        carry = states.additional_carry
        per_family_stats[name] = carry.admission
        morphologies = np.asarray(carry.morphology)
        admitted = ~np.asarray(carry.admission_failed, dtype=bool)
        accepted_bodies[name] = morphologies[admitted]

    if not per_family_stats:
        return {
            "supported": False,
            "reason": "no family env exposes online morphology",
        }

    from scaling.family_morphology import FAMILY_MORPHOLOGY_NAMES

    summary = summarize_admission(per_family_stats, FAMILY_MORPHOLOGY_NAMES)
    for name, bodies in accepted_bodies.items():
        quantiles = {}
        if bodies.size:
            values = np.quantile(bodies, [0.0, 0.05, 0.5, 0.95, 1.0], axis=0)
            for i, coordinate in enumerate(FAMILY_MORPHOLOGY_NAMES):
                quantiles[coordinate] = {
                    "q00": float(values[0, i]),
                    "q05": float(values[1, i]),
                    "q50": float(values[2, i]),
                    "q95": float(values[3, i]),
                    "q100": float(values[4, i]),
                }
        summary["per_family"][name]["morphology_quantiles"] = quantiles
        summary["per_family"][name]["accepted_bodies_sampled"] = int(bodies.shape[0])

    summary["supported"] = True
    summary["resets_per_family"] = int(resets) * int(envs_per_family)
    summary["scope"] = (
        "census of the reset-time sampler under the training configuration; "
        "not a count over the training rollout itself"
    )
    return summary
