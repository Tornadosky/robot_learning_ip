"""Gate tests for the padded cross-topology URMA path.

The cheap tests build MuJoCo models straight from each robot's XML.  The two
expensive ones build the real three-robot ``ParallelMorphVecEnv`` and are marked
``slow`` so the fast suite stays fast; the overnight gate runs everything.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

WORKSPACE = Path(__file__).resolve().parents[1]
SCRIPTS = WORKSPACE / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from scaling.cross_humanoid_retarget import HUMANOIDS  # noqa: E402
from scaling.joint_descriptions import (  # noqa: E402
    GENERIC_JOINT_DESCRIPTION_DIM,
    JOINT_FEATURE_DIM,
    generic_joint_descriptions,
)

EXPECTED_JOINTS = {"h1": 19, "g1": 23, "atlas": 27, "toddlerbot": 30}
TRAIN_ROBOTS = ("h1", "g1", "atlas")


def _model(robot: str):
    return mujoco.MjModel.from_xml_path(str(HUMANOIDS[robot].xml_path))


@pytest.mark.parametrize("robot", list(EXPECTED_JOINTS))
def test_joint_descriptions_are_finite_and_correctly_shaped(robot):
    model = _model(robot)
    action_dim = int(model.nu)
    assert action_dim == EXPECTED_JOINTS[robot]
    descriptions = generic_joint_descriptions(model, action_dim)
    assert descriptions.shape == (action_dim, GENERIC_JOINT_DESCRIPTION_DIM)
    assert np.isfinite(descriptions).all()
    # The block is structural only: no family morphology coordinates are in it.
    assert descriptions.dtype == np.float32


def test_different_topologies_produce_different_description_blocks():
    blocks = {
        robot: generic_joint_descriptions(_model(robot), EXPECTED_JOINTS[robot])
        for robot in TRAIN_ROBOTS
    }
    for left, right in (("h1", "g1"), ("h1", "atlas"), ("g1", "atlas")):
        assert blocks[left].shape != blocks[right].shape
        shared = min(blocks[left].shape[0], blocks[right].shape[0])
        assert not np.allclose(blocks[left][:shared], blocks[right][:shared])


@pytest.fixture(scope="module")
def cross_env():
    from scaling.parallel_cross_humanoid_train import build_cross_humanoid_env

    args = SimpleNamespace(
        robots=list(TRAIN_ROBOTS),
        source="h1",
        clip="dance2_subject4",
        start_frame=19482,
        frames=800,
        reference_mode="direct",
        reference_root=WORKSPACE / "external_data" / "cross_humanoid",
        use_mjwarp=False,
        envs_per_robot=2,
        total_envs=6,
        robot_one_hot=False,
        append_joint_features=True,
        reserve_robots=[],
    )
    env, metadata = build_cross_humanoid_env(args)
    return env, metadata


@pytest.mark.slow
def test_padded_joint_mask_has_exactly_n_joints_per_robot(cross_env):
    env, _ = cross_env
    keys = jax.random.split(jax.random.PRNGKey(0), env.num_envs)
    observation = np.asarray(jax.jit(env.reset)(keys)[0])
    block = observation[:, env.joint_feature_start :].reshape(
        env.num_envs, env.num_joint_slots, JOINT_FEATURE_DIM
    )
    mask = block[..., -1]
    assert set(np.unique(mask).tolist()) <= {0.0, 1.0}
    for group in env.groups:
        counts = mask[group.start : group.stop].sum(axis=-1)
        np.testing.assert_array_equal(counts, EXPECTED_JOINTS[group.name])
    # Padded joints carry no description or state signal either.
    for group, n_joints in zip(
        env.groups, [EXPECTED_JOINTS[g.name] for g in env.groups], strict=True
    ):
        padded = block[group.start : group.stop, n_joints:, :-1]
        np.testing.assert_allclose(padded, 0.0, atol=0.0)


@pytest.mark.slow
def test_padded_action_means_are_zero_after_one_forward_pass(cross_env):
    from scaling.cross_topology_urma import CrossTopologyURMAPPO
    from scaling.parallel_cross_humanoid_train import build_config

    env, _ = cross_env
    args = SimpleNamespace(
        num_steps=4,
        num_minibatches=2,
        update_epochs=1,
        hidden=[32, 32],
        backbone="urmav2",
        urma_latent_slots=8,
        urma_joint_value_dim=2,
        lr=1e-4,
        init_std=0.2,
        learnable_std=True,
        no_normalize_reward=False,
    )
    config = build_config(args, env.num_envs, args.num_steps * env.num_envs)
    agent_conf = CrossTopologyURMAPPO.init_agent_conf(env, config)
    keys = jax.random.split(jax.random.PRNGKey(0), env.num_envs)
    observation = jax.jit(env.reset)(keys)[0]
    variables = agent_conf.network.init(jax.random.PRNGKey(1), observation)
    (policy, value), _ = agent_conf.network.apply(
        variables, observation, mutable=["run_stats"]
    )
    mean = np.asarray(policy.mean())
    stddev = np.asarray(policy.stddev())
    assert mean.shape == (env.num_envs, env.num_joint_slots)
    assert np.isfinite(mean).all() and np.isfinite(np.asarray(value)).all()
    for group in env.groups:
        n_joints = EXPECTED_JOINTS[group.name]
        padded_mean = mean[group.start : group.stop, n_joints:]
        padded_std = stddev[group.start : group.stop, n_joints:]
        np.testing.assert_allclose(padded_mean, 0.0, atol=1e-7)
        np.testing.assert_allclose(padded_std, 1e-3, rtol=1e-5)


@pytest.mark.slow
def test_one_jitted_ppo_update_over_three_robots_is_finite(cross_env):
    from scaling.cross_topology_urma import CrossTopologyURMAPPO
    from scaling.parallel_cross_humanoid_train import build_config

    env, metadata = cross_env
    assert metadata["group_sizes"] == [2, 2, 2]
    assert metadata["joint_counts"] == {r: EXPECTED_JOINTS[r] for r in TRAIN_ROBOTS}

    args = SimpleNamespace(
        num_steps=4,
        num_minibatches=2,
        update_epochs=1,
        hidden=[32, 32],
        backbone="urmav2",
        urma_latent_slots=8,
        urma_joint_value_dim=2,
        lr=1e-4,
        init_std=0.2,
        learnable_std=True,
        no_normalize_reward=False,
    )
    steps = args.num_steps * env.num_envs
    config = build_config(args, env.num_envs, steps)
    agent_conf = CrossTopologyURMAPPO.init_agent_conf(env, config)
    train_fn = jax.jit(CrossTopologyURMAPPO.build_train_fn(env, agent_conf, mh=None))
    output = train_fn(jax.random.PRNGKey(0))
    jax.block_until_ready(output["agent_state"])
    leaves = jax.tree_util.tree_leaves(output["agent_state"].train_state.params)
    assert leaves
    for leaf in leaves:
        assert np.isfinite(np.asarray(leaf)).all()
    returns = np.asarray(output["training_metrics"].mean_episode_return)
    assert returns.shape[0] >= 1


@pytest.mark.slow
def test_reserved_slots_keep_a_heldout_topology_representable():
    """A no-one-hot policy trained with reserved slots must accept ToddlerBot."""
    from scaling.parallel_cross_humanoid_train import build_cross_humanoid_env

    common = dict(
        source="h1",
        clip="dance2_subject4",
        start_frame=19482,
        frames=800,
        reference_mode="direct",
        reference_root=WORKSPACE / "external_data" / "cross_humanoid",
        use_mjwarp=False,
        envs_per_robot=1,
        robot_one_hot=False,
        append_joint_features=True,
    )
    trained, _ = build_cross_humanoid_env(
        SimpleNamespace(
            robots=list(TRAIN_ROBOTS),
            total_envs=3,
            reserve_robots=["toddlerbot"],
            **common,
        )
    )
    heldout, _ = build_cross_humanoid_env(
        SimpleNamespace(
            robots=["toddlerbot"],
            total_envs=1,
            reserve_robots=list(TRAIN_ROBOTS),
            **common,
        )
    )
    assert trained.output_observation_dim == heldout.output_observation_dim
    assert trained.max_action_dim == heldout.max_action_dim == 30
    assert trained.joint_feature_start == heldout.joint_feature_start
    keys = jax.random.split(jax.random.PRNGKey(0), heldout.num_envs)
    observation = jax.jit(heldout.reset)(keys)[0]
    assert observation.shape[-1] == trained.output_observation_dim
    block = np.asarray(observation)[:, heldout.joint_feature_start :].reshape(
        heldout.num_envs, heldout.num_joint_slots, JOINT_FEATURE_DIM
    )
    np.testing.assert_array_equal(block[..., -1].sum(axis=-1), 30)


@pytest.mark.slow
def test_general_indices_stop_before_the_action_mask(cross_env):
    env, _ = cross_env
    layout = env.urma_input_layout
    assert max(layout.general_indices) < env.action_mask_observation_start
    assert layout.joint_feature_start == env.joint_feature_start
    assert layout.num_joints == env.num_joint_slots
    assert layout.joint_feature_stop == env.output_observation_dim
    assert jnp.asarray(layout.general_indices).shape[0] == env.max_observation_dim
