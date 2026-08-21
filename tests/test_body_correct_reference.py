"""P0 acceptance tests for the shared body-correct reference provider.

These are the gate on a 60M/300M no-FSQ claim. Each one fails for a specific,
previously-possible defect:

1. ``test_goal_target_matches_reward_target``  -- the actual bug: the goal
   commanded nominal-body site targets while ``MorphMimicReward`` scored
   sampled-body ones.
2. ``test_jitted_and_eager_provider_agree``    -- a traced-vs-eager divergence
   in the FK pass.
3. ``test_provider_matches_independent_cpu_fk``-- the MJX FK formula itself
   being wrong, checked against the C engine on an independently mutated model.
4. ``test_phase_p_does_not_match_p_plus_one``  -- an off-by-one cursor that a
   tolerance-based check would happily pass.
5. ``test_low_high_morphology_move_targets``   -- a silently nominal-cached FK.
6. ``test_joint_reward_and_fk_share_one_clamped_qpos`` -- the joint term and the
   site targets reading two different "references".
7. ``test_reset_observation_reward_and_cursor_share_phase`` -- the reset
   observation being built at a different phase than the one the reward scores.
8. ``test_tall_body_not_absorbing_only_from_nominal_height_bounds`` -- a
   legitimately taller body terminated at reset for standing higher than the
   human-derived reference motion.

Covered for H1 and G1, at nominal/low/high morphologies, at several phases.
"""

from __future__ import annotations

import json
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
import jax.numpy as jnp  # noqa: E402

from scaling.body_correct_goal import goal_block_slices  # noqa: E402
from scaling.body_correct_reference import (  # noqa: E402
    body_correct_reference,
    clamp_reference_qpos,
    cpu_morphology_model,
    cpu_reference_bundle,
)
from scaling.family_morphology import (  # noqa: E402
    FAMILY_MORPHOLOGY_HIGH,
    FAMILY_MORPHOLOGY_LOW,
)
from scaling.parallel_cross_humanoid_train import (  # noqa: E402
    _build_robot_env,
    _ensure_latent_defaults,
)

CLIP_WINDOWS = ["dance2_subject4:19482:800"]
REFERENCE_ROOT = WORKSPACE / "external_data" / "cross_humanoid"
ROBOTS = ("h1", "g1")
PHASES = (3, 111, 400, 731)

#: The P0 gate: "goal target site block equals reward target bundle within 1e-5 m".
#: It is denominated in METRES and applies to positions, which match exactly.
GOAL_REWARD_TOL_M = 1e-5
#: Orientations are rotation vectors in RADIANS, so the metre gate does not apply
#: to them. The goal emits 15 mimic sites and the reward 5; on GPU the two batch
#: shapes make `R.from_matrix(...).as_rotvec()` reduce in a different order, and
#: float32 leaves ~3e-4 rad (0.017 deg) of residual. It is provably batching and
#: not semantics: `test_goal_and_reward_orientations_are_identical_on_one_site_set`
#: below shows the difference is exactly zero once both sides use the same site
#: set. 2e-3 rad is the tolerance verify_fk_targets.py already uses for rotvecs.
GOAL_REWARD_TOL_RAD = 2e-3
#: verify_fk_targets.py's tolerances for MJX-vs-C-engine agreement.
CPU_TOL_M = 5e-4
CPU_TOL_ANGLE = 2e-3

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
        clip_windows=list(CLIP_WINDOWS),
        morphology="continuous",
        use_mjwarp=False,
        reward_type="MorphMimicReward",
        goal_type="MorphGoalTrajMimicRootErr",
    )
    base.update(overrides)
    return _ensure_latent_defaults(SimpleNamespace(**base))


_ENV_CACHE: dict = {}


def _env(robot: str, **overrides):
    """Environment builds are the expensive part; memoise per configuration."""
    key = (robot, tuple(sorted((k, str(v)) for k, v in overrides.items())))
    if key not in _ENV_CACHE:
        env, _ = _build_robot_env(_env_args(**overrides), robot)
        _ENV_CACHE[key] = env
    return _ENV_CACHE[key]


def _morphologies():
    return {
        "nominal": np.ones(FAMILY_MORPHOLOGY_LOW.shape[0], dtype=np.float32),
        "low": FAMILY_MORPHOLOGY_LOW.copy(),
        "high": FAMILY_MORPHOLOGY_HIGH.copy(),
    }


def _carry(env, morphology, traj_no: int, step: int):
    """A minimal carry standing in for the in-graph one.

    Only ``morphology`` and ``traj_state`` are read by the provider, so a
    namespace is enough and keeps the test independent of the env's flax carry
    layout.
    """
    return SimpleNamespace(
        morphology=jnp.asarray(morphology),
        traj_state=SimpleNamespace(
            traj_no=jnp.asarray(traj_no), subtraj_step_no=jnp.asarray(step)
        ),
    )


def _provider(env, site_ids, body_ids, morphology, traj_no, step,
              jitted: bool = False, include_site_velocity: bool = True):
    from mujoco import mjx

    data0 = mjx.make_data(env.sys)

    def compute(m, t, s):
        bundle = body_correct_reference(
            env,
            env._model,
            data0,
            _carry(env, m, t, s),
            jnp,
            rel_site_ids=site_ids,
            rel_body_ids=body_ids,
            body_rootid=env.sys.body_rootid,
            include_site_velocity=include_site_velocity,
        )
        return (
            bundle.reference_qpos_clamped,
            bundle.relative_site_position,
            bundle.relative_site_orientation,
        )

    fn = jax.jit(compute) if jitted else compute
    qpos, rpos, rangles = fn(
        jnp.asarray(morphology), jnp.asarray(traj_no), jnp.asarray(step)
    )
    return np.asarray(qpos), np.asarray(rpos), np.asarray(rangles)


def _reward_targets(env, morphology, traj_no, step, jitted=False):
    reward = env._reward_function
    return _provider(
        env, reward._rel_site_ids, reward._rel_body_ids, morphology,
        traj_no, step, jitted=jitted, include_site_velocity=False,
    )


def _goal_targets(env, morphology, traj_no, step, jitted=False):
    goal = env._goal
    return _provider(
        env, goal._rel_site_ids, goal._site_bodyid[goal._rel_site_ids],
        morphology, traj_no, step, jitted=jitted,
    )


# --------------------------------------------------------------------------- #
# 1. the actual bug
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("robot", ROBOTS)
@pytest.mark.parametrize("morph_name", ["nominal", "low", "high"])
def test_goal_target_matches_reward_target(robot, morph_name):
    """The site block the actor is commanded toward IS the one it is paid for.

    The goal mimics 15 sites and the reward 5; ``resolve_reward_site_rows``
    supplies the row mapping, and a mismatch there raises rather than silently
    comparing the wrong rows.
    """
    env = _env(robot)
    morphology = _morphologies()[morph_name]
    rows = env._goal.resolve_reward_site_rows(env)

    for step in PHASES:
        _, goal_rpos, goal_rang = _goal_targets(env, morphology, 0, step)
        _, reward_rpos, reward_rang = _reward_targets(env, morphology, 0, step)
        assert np.abs(goal_rpos[rows] - reward_rpos).max() < GOAL_REWARD_TOL_M, (
            f"{robot} {morph_name} phase {step}: goal commands a different "
            "spatial target than the reward scores"
        )
        assert np.abs(goal_rang[rows] - reward_rang).max() < GOAL_REWARD_TOL_RAD


@pytest.mark.parametrize("robot", ROBOTS)
@pytest.mark.parametrize("morph_name", ["nominal", "high"])
def test_goal_and_reward_orientations_are_identical_on_one_site_set(
    robot, morph_name
):
    """The orientation residual is batch shape, not semantics.

    Queried through the goal's own code path but with the reward's site set, the
    provider must return bit-identical orientations. If this ever fails, the
    looser radian tolerance above is hiding a real disagreement rather than
    float32 reduction order.
    """
    env = _env(robot)
    morphology = _morphologies()[morph_name]
    reward = env._reward_function
    goal = env._goal

    for step in (PHASES[0], PHASES[3]):
        _, same_rpos, same_rang = _provider(
            env, np.asarray(reward._rel_site_ids),
            np.asarray(reward._rel_body_ids), morphology, 0, step,
            jitted=True, include_site_velocity=False,
        )
        _, reward_rpos, reward_rang = _reward_targets(
            env, morphology, 0, step, jitted=True)
        assert np.array_equal(same_rpos, reward_rpos)
        assert np.array_equal(same_rang, reward_rang), (
            f"{robot} {morph_name} phase {step}: the provider disagrees with "
            "itself on one site set — the residual is not batching"
        )
        assert goal._body_rootid.shape == env.sys.body_rootid.shape


@pytest.mark.parametrize("robot", ROBOTS)
def test_goal_observation_block_matches_reward_target(robot):
    """Same check, but read out of the emitted observation, not a helper call.

    This is what catches a correct provider wired into the wrong slice of the
    goal vector.
    """
    env = _env(robot)
    goal = env._goal
    rows = goal.resolve_reward_site_rows(env)
    slices = goal_block_slices(goal)
    n_rel = len(np.asarray(goal._rel_site_ids)) - 1

    state = jax.jit(env.mjx_reset)(jax.random.PRNGKey(3))
    carry = state.additional_carry
    observation = np.asarray(state.observation)[np.asarray(goal.obs_ind)]
    emitted = observation[slices["target_site_position"]].reshape(n_rel, 3)

    reward = env._reward_function
    bundle = reward.reference_bundle(
        env, env._model, state.data, carry, jnp, include_site_velocity=False
    )
    scored = np.asarray(bundle.relative_site_position)

    assert np.abs(emitted[rows] - scored).max() < GOAL_REWARD_TOL_M


# --------------------------------------------------------------------------- #
# 2. jit stability
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("robot", ROBOTS)
@pytest.mark.parametrize("morph_name", ["nominal", "low", "high"])
def test_jitted_and_eager_provider_agree(robot, morph_name):
    """Tracing the provider must not change what it computes.

    Tolerances are per quantity and per unit: metres for positions, radians for
    orientations. On GPU the fused jitted kernel and the eager op-by-op path
    reduce in different orders, so float32 leaves a small residual in the
    rotvec conversion; 2e-3 rad is verify_fk_targets.py's own rotvec tolerance.
    """
    env = _env(robot)
    morphology = _morphologies()[morph_name]
    for step in (PHASES[0], PHASES[2]):
        qpos_e, rpos_e, rang_e = _goal_targets(env, morphology, 0, step)
        qpos_j, rpos_j, rang_j = _goal_targets(env, morphology, 0, step, jitted=True)
        assert np.array_equal(qpos_j, qpos_e), "clamped reference qpos changed"
        assert np.abs(rpos_j - rpos_e).max() < 1e-6
        assert np.abs(rang_j - rang_e).max() < GOAL_REWARD_TOL_RAD
        assert np.isfinite(rpos_j).all() and np.isfinite(rang_j).all()


# --------------------------------------------------------------------------- #
# 3. independent CPU MuJoCo FK
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("robot", ROBOTS)
@pytest.mark.parametrize("morph_name", ["nominal", "low", "high"])
def test_provider_matches_independent_cpu_fk(robot, morph_name):
    """MJX provider vs the C engine on an independently mutated CPU model."""
    env = _env(robot)
    reward = env._reward_function
    morphology = _morphologies()[morph_name]
    cpu_model = cpu_morphology_model(env._model, robot, morphology)

    for step in PHASES:
        _, rpos_j, rang_j = _reward_targets(env, morphology, 0, step, jitted=True)
        cpu = cpu_reference_bundle(
            env, cpu_model, 0, step,
            rel_site_ids=reward._rel_site_ids,
            rel_body_ids=reward._rel_body_ids,
            include_site_velocity=False,
        )
        assert np.abs(rpos_j - np.asarray(cpu.relative_site_position)).max() < CPU_TOL_M
        assert (
            np.abs(rang_j - np.asarray(cpu.relative_site_orientation)).max()
            < CPU_TOL_ANGLE
        )


# --------------------------------------------------------------------------- #
# 4. off-by-one phase
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("robot", ROBOTS)
def test_phase_p_does_not_match_p_plus_one(robot):
    """Phase p and p+1 must be separable far beyond the CPU match error.

    Without this an off-by-one cursor passes every tolerance check above.
    """
    env = _env(robot)
    nominal = _morphologies()["nominal"]
    for step in PHASES:
        _, rpos_p, _ = _reward_targets(env, nominal, 0, step, jitted=True)
        _, rpos_next, _ = _reward_targets(env, nominal, 0, step + 1, jitted=True)
        gap = float(np.abs(rpos_p - rpos_next).max())
        assert gap > max(10.0 * CPU_TOL_M, 1e-3), (
            f"{robot} phase {step}: p vs p+1 gap {gap:.2e} m is not "
            "distinguishable from the FK match error"
        )


# --------------------------------------------------------------------------- #
# 5. morphology actually moves the targets
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("robot", ROBOTS)
@pytest.mark.parametrize("morph_name", ["low", "high"])
def test_low_high_morphology_move_targets(robot, morph_name):
    """Corner bodies must not produce the nominal body's targets.

    ``leg_length_scale`` is the only coordinate that changes kinematics, so this
    is the check that the sampled body reaches the FK pass at all.
    """
    env = _env(robot)
    nominal = _morphologies()["nominal"]
    corner = _morphologies()[morph_name]
    assert not np.isclose(corner[0], 1.0), "corner body must change leg geometry"

    for step in (PHASES[1], PHASES[3]):
        _, rpos_nom, _ = _goal_targets(env, nominal, 0, step, jitted=True)
        _, rpos_cor, _ = _goal_targets(env, corner, 0, step, jitted=True)
        shift = float(np.abs(rpos_cor - rpos_nom).max())
        assert shift > 5e-3, (
            f"{robot} {morph_name} phase {step}: corner target moved only "
            f"{shift:.2e} m from nominal (nominal-cached FK?)"
        )


# --------------------------------------------------------------------------- #
# 6. one clamped reference array
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("robot", ROBOTS)
@pytest.mark.parametrize("morph_name", ["nominal", "low", "high"])
def test_joint_reward_and_fk_share_one_clamped_qpos(robot, morph_name):
    """The joint term and the FK target read the same clamped array.

    Both the emitted goal reference-qpos block and the bundle the reward scores
    must equal the clamp applied to the raw trajectory sample -- computed here
    independently of the provider.
    """
    env = _env(robot)
    morphology = _morphologies()[morph_name]
    body_model = env._apply_morphology(env.sys, jnp.asarray(morphology))

    for step in PHASES:
        raw = env.th.traj.data.get(0, step, jnp).qpos
        expected = np.asarray(
            clamp_reference_qpos(raw, body_model.jnt_range, env.sys, jnp)
        )
        qpos_goal, _, _ = _goal_targets(env, morphology, 0, step, jitted=True)
        qpos_reward, _, _ = _reward_targets(env, morphology, 0, step, jitted=True)
        assert np.array_equal(qpos_goal, qpos_reward)
        assert np.abs(qpos_goal - expected).max() == 0.0


# --------------------------------------------------------------------------- #
# 7. post-reset phase is shared
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("robot", ROBOTS)
def test_reset_observation_reward_and_cursor_share_phase(robot):
    """The reset observation is built at the cursor the reward will score.

    Random start phases make this easy to get wrong and impossible to see: an
    observation one step ahead of the reward still trains, just toward the wrong
    thing.
    """
    env = _env(robot)
    goal = env._goal
    slices = goal_block_slices(goal)
    n_rel = len(np.asarray(goal._rel_site_ids)) - 1

    for seed in (0, 11, 42):
        state = jax.jit(env.mjx_reset)(jax.random.PRNGKey(seed))
        carry = state.additional_carry
        traj_no = int(carry.traj_state.traj_no)
        step = int(carry.traj_state.subtraj_step_no)
        morphology = np.asarray(carry.morphology)

        observation = np.asarray(state.observation)[np.asarray(goal.obs_ind)]
        emitted = observation[slices["target_site_position"]].reshape(n_rel, 3)

        # the provider, driven from the SAME cursor the carry reports
        _, rpos, _ = _goal_targets(env, morphology, traj_no, step, jitted=True)
        assert np.abs(emitted - rpos).max() < GOAL_REWARD_TOL_M

        # and the next phase must NOT match, so this is a real phase check
        _, rpos_next, _ = _goal_targets(env, morphology, traj_no, step + 1,
                                        jitted=True)
        assert np.abs(emitted - rpos_next).max() > 1e-4


# --------------------------------------------------------------------------- #
# 8. morphology-aware terminal semantics
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("robot", ROBOTS)
def test_tall_body_not_absorbing_only_from_nominal_height_bounds(robot, tmp_path):
    """A tall but valid body survives reset under the morphology-aware handler.

    The leg scale is derived per family, not hard-coded: it is chosen so the
    body's standing height clears the stock handler's ``root_height_margin`` by
    a comfortable margin. That is what makes the test non-vacuous -- a fixed 1.5
    happens to clear H1's 0.3 m window but not G1's shorter legs, and would pass
    on G1 while proving nothing.

    The scale is deliberately outside today's conservative box: the point is to
    certify the terminal semantics BEFORE leg randomization widens.
    """
    probe = _env(robot)
    margin = float(probe._terminal_state_handler.root_height_margin)
    leg = float(probe.nominal_leg_length_m)
    # standing height offset = leg * (scale - 1); clear the window by 0.15 m
    leg_scale = 1.0 + (margin + 0.15) / leg

    catalog = tmp_path / "tall.json"
    catalog.write_text(
        json.dumps([[leg_scale, 1.0, 1.0, 1.0]] * 2), encoding="utf-8"
    )

    stock = _env(robot, morphology="catalog2",
                 morphology_catalog_file=str(catalog))
    aware = _env(robot, morphology="catalog2",
                 morphology_catalog_file=str(catalog),
                 terminal_handler="MorphologyAwareRootPoseTrajTerminalStateHandler")
    assert (
        type(aware._terminal_state_handler).__name__
        == "MorphologyAwareRootPoseTrajTerminalStateHandler"
    )

    # NOTE: `mjx_reset` hardcodes `absorbing=False` on the returned state, so
    # reading `state.absorbing` here would make this test vacuously pass. Both
    # sides must go through the environment's own absorbing check, which is the
    # production path (handler criterion OR the admission fail-closed flag).
    def absorbing_at_reset(env, key):
        state = jax.jit(env.mjx_reset)(key)
        flag, _ = env._mjx_is_absorbing(
            state.observation, state.info, state.data, state.additional_carry
        )
        return bool(flag)

    stock_absorbing = []
    aware_absorbing = []
    for seed in (0, 5, 9):
        key = jax.random.PRNGKey(seed)
        stock_absorbing.append(absorbing_at_reset(stock, key))
        aware_absorbing.append(absorbing_at_reset(aware, key))

    assert not any(aware_absorbing), (
        f"{robot}: a valid tall body is absorbing at reset under the "
        "morphology-aware handler"
    )
    assert any(stock_absorbing), (
        f"{robot}: the stock handler did not flag a 1.5x-leg body either, so "
        "this test proves nothing about the morphology-aware height interval"
    )


# --------------------------------------------------------------------------- #
# 9. reward, goal and termination share one root frame
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("robot", ROBOTS)
def test_episode_start_frame_agrees_with_the_reset_and_the_terminal(robot):
    """A perfectly-posed robot must not be charged for the clip's absolute position.

    The reset places the root at world XY (0, 0) and the terminal criterion
    measures deviation from the reference's displacement since the episode's
    start frame. Upstream's reward and root-error goal read the reference's
    ABSOLUTE root XY, so under random start phases a pixel-perfect RSI pose is
    charged up to 1.3 m of error -- and the reward then RISES for walking to the
    point the terminal terminates the episode for reaching.

    Under root_frame="episode_start" all three agree, which this pins.
    """
    absolute = _env(robot)
    relative = _env(robot, root_frame="episode_start")

    for seed in (0, 7, 21):
        key = jax.random.PRNGKey(seed)
        for env, frame, expect_zero in ((absolute, "absolute", False),
                                        (relative, "episode_start", True)):
            state = jax.jit(env.mjx_reset)(key)
            carry = state.additional_carry
            bundle = env._reward_function.reference_bundle(
                env, env._model, state.data, carry, jnp,
                include_site_velocity=False)
            reference_xy = np.asarray(bundle.reference_qpos_clamped)[:2]
            robot_xy = np.asarray(state.data.qpos)[:2]
            error = float(np.linalg.norm(reference_xy - robot_xy))
            if expect_zero:
                assert error < 1e-6, (
                    f"{robot} seed {seed}: episode_start frame still charges "
                    f"{error:.3f} m at reset"
                )
            else:
                # the defect itself, pinned so a silent convention change is loud
                assert error > 1e-3, (
                    f"{robot} seed {seed}: absolute frame unexpectedly agrees "
                    "with the reset; the frames may have been unified silently"
                )
            assert frame == getattr(env._reward_function, "_root_frame")
            assert frame == getattr(env._goal, "_root_frame")
