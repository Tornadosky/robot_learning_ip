"""Regression tests for the cross-family morphology-conditioned FK targets.

These would fail for (a) an implementation that silently caches nominal-body
FK, (b) an off-by-one phase/command, (c) a reward that never consumes the
sampled morphology at runtime, (d) JIT instability of the FK pass.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

WORKSPACE = Path(__file__).resolve().parents[1]
SCRIPTS = WORKSPACE / "scripts"
for p in (str(SCRIPTS), str(SCRIPTS / "h1md")):
    if p not in sys.path:
        sys.path.insert(0, p)

from scaling.parallel_cross_humanoid_train import (  # noqa: E402
    _build_robot_env,
    _ensure_latent_defaults,
    _reward_env_params,
)

CLIP_WINDOWS = ["dance2_subject4:19482:800", "dance2_subject1:2000:800"]
REFERENCE_ROOT = WORKSPACE / "external_data" / "cross_humanoid"

pytestmark = pytest.mark.skipif(
    not (REFERENCE_ROOT / "h1_source" / "dance2_subject4" / "h1").is_dir(),
    reason="cross-humanoid reference cache not present",
)


def _env_args(reward_type: str, morphology: str = "continuous"):
    return _ensure_latent_defaults(
        SimpleNamespace(
            source="h1",
            reference_mode="direct",
            reference_root=REFERENCE_ROOT,
            clip=None,
            start_frame=None,
            frames=None,
            clip_windows=CLIP_WINDOWS,
            morphology=morphology,
            use_mjwarp=False,
            reward_type=reward_type,
        )
    )


@pytest.fixture(scope="module")
def h1_fk_env():
    env, _ = _build_robot_env(_env_args("MorphMimicReward"), "h1")
    return env


def test_reward_env_params_default_is_stock():
    assert _reward_env_params(SimpleNamespace()) == {}
    assert _reward_env_params(SimpleNamespace(reward_type="MimicReward")) == {}
    with pytest.raises(ValueError):
        _reward_env_params(SimpleNamespace(reward_type="NopeReward"))


def test_fk_reward_selected(h1_fk_env):
    assert type(h1_fk_env._reward_function).__name__ == "MorphMimicReward"


def test_stock_reward_still_default():
    env, _ = _build_robot_env(_env_args("MimicReward"), "h1")
    assert type(env._reward_function).__name__ == "MimicReward"


def _targets(env, morphology, qpos, jitted=False):
    from mujoco import mjx

    reward = env._reward_function
    data0 = mjx.make_data(env.sys)
    carry = SimpleNamespace(morphology=jnp.asarray(morphology))

    def compute(m, q):
        body_model = env._apply_morphology(env.sys, m)
        rpos, rangles, _ = reward._traj_site_quantities(
            env, env._model, data0, q, carry, jnp, body_model=body_model
        )
        return rpos, rangles

    fn = jax.jit(compute) if jitted else compute
    rpos, rangles = fn(jnp.asarray(morphology), jnp.asarray(qpos))
    return np.asarray(rpos), np.asarray(rangles)


def test_fk_targets_move_with_morphology_and_jit_stable(h1_fk_env):
    """Nominal-cached FK would make corner targets equal nominal targets."""
    qpos = np.asarray(h1_fk_env.th.traj.data.get(0, 111, jnp).qpos)
    nominal = np.ones(4, dtype=np.float32)
    tall = np.array([1.12, 1.0, 1.0, 1.0], dtype=np.float32)

    rpos_nom, _ = _targets(h1_fk_env, nominal, qpos)
    rpos_tall, _ = _targets(h1_fk_env, tall, qpos)
    rpos_tall_jit, _ = _targets(h1_fk_env, tall, qpos, jitted=True)

    assert np.abs(rpos_tall - rpos_tall_jit).max() < 1e-6  # JIT stability
    shift = np.abs(rpos_tall - rpos_nom).max()
    assert shift > 0.01, f"corner morphology only moved targets {shift:.2e} m"
    assert np.isfinite(rpos_tall).all()


def test_fk_targets_match_independent_cpu_fk(h1_fk_env):
    from scaling.verify_fk_targets import _cpu_reference_quantities

    reward = h1_fk_env._reward_function
    qpos = np.asarray(h1_fk_env.th.traj.data.get(1, 400, jnp).qpos)
    tall = np.array([1.12, 1.3, 1.5, 1.1], dtype=np.float32)
    rpos_j, rang_j = _targets(h1_fk_env, tall, qpos, jitted=True)
    rpos_c, rang_c = _cpu_reference_quantities(h1_fk_env, reward, "h1", tall, qpos)
    assert np.abs(rpos_j - rpos_c).max() < 5e-4
    assert np.abs(rang_j - rang_c).max() < 2e-3


def test_off_by_one_phase_detectable(h1_fk_env):
    """FK targets at phase p and p+1 must be separable far beyond match error."""
    nominal = np.ones(4, dtype=np.float32)
    q0 = np.asarray(h1_fk_env.th.traj.data.get(0, 111, jnp).qpos)
    q1 = np.asarray(h1_fk_env.th.traj.data.get(0, 112, jnp).qpos)
    rpos0, _ = _targets(h1_fk_env, nominal, q0)
    rpos1, _ = _targets(h1_fk_env, nominal, q1)
    assert np.abs(rpos0 - rpos1).max() > 1e-3


def test_runtime_reward_consumes_fk_targets():
    """A full jitted env step must score corner bodies differently under
    stock vs FK rewards — fails if the reward never reads the sampled body."""
    from loco_mujoco.core.wrappers import LogWrapper

    rewards = {}
    for reward_type in ("MimicReward", "MorphMimicReward"):
        raw, _ = _build_robot_env(_env_args(reward_type, "catalog2"), "h1")
        env = LogWrapper(raw)
        key = jax.random.PRNGKey(5)
        obs, state = jax.jit(env.reset)(key)
        action = jnp.zeros(raw.info.action_space.shape, dtype=jnp.float32)
        stepped = jax.jit(env.step)(state, action)
        rewards[reward_type] = float(stepped[1])
        assert np.isfinite(rewards[reward_type])
    assert rewards["MimicReward"] != pytest.approx(
        rewards["MorphMimicReward"], abs=1e-6
    ), "stock and FK rewards identical on a corner body"


def test_ensure_latent_defaults_adds_reward_type():
    ns = SimpleNamespace()
    _ensure_latent_defaults(ns)
    assert ns.reward_type == "MimicReward"
