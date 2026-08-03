import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaling.urma_networks import URMAActorCritic


def _network(variant):
    return URMAActorCritic(
        action_dim=3,
        base_observation_dim=8,
        joint_feature_start=12,
        num_joints=3,
        joint_description_dim=4,
        joint_state_dim=3,
        joint_feature_dim=8,
        general_indices=(0, 1, 8, 9, 10, 11),
        variant=variant,
        core_hidden=(16, 8),
        latent_slots=8,
        joint_value_dim=2,
        learnable_std=True,
    )


def _observations(batch_size=4):
    observations = np.zeros((batch_size, 36), dtype=np.float32)
    rng = np.random.default_rng(7)
    observations[:] = rng.normal(size=observations.shape)
    features = observations[:, 12:].reshape(batch_size, 3, 8)
    features[..., -1] = 1.0
    return jnp.asarray(observations)


@pytest.mark.parametrize("variant", ["urma", "urmav2"])
def test_urma_actor_critic_has_joint_sized_outputs(variant):
    network = _network(variant)
    observations = _observations()
    variables = network.init(jax.random.PRNGKey(0), observations)
    (policy, value), updates = network.apply(
        variables, observations, mutable=["run_stats"]
    )

    assert policy.mean().shape == (4, 3)
    assert policy.stddev().shape == (4, 3)
    assert value.shape == (4,)
    assert "run_stats" in updates
    assert np.isfinite(np.asarray(policy.mean())).all()
    assert np.isfinite(np.asarray(value)).all()


@pytest.mark.parametrize("variant", ["urma", "urmav2"])
def test_invalid_padded_joint_has_zero_action_mean(variant):
    network = _network(variant)
    observations = np.array(_observations(), copy=True)
    observations[:, 12:].reshape(4, 3, 8)[:, -1, -1] = 0.0
    observations = jnp.asarray(observations)
    variables = network.init(jax.random.PRNGKey(1), observations)
    (policy, _), _ = network.apply(variables, observations, mutable=["run_stats"])
    np.testing.assert_allclose(np.asarray(policy.mean())[:, -1], 0.0, atol=1e-7)
    np.testing.assert_allclose(np.asarray(policy.stddev())[:, -1], 1e-3, rtol=1e-5)


@pytest.mark.parametrize("variant", ["urma", "urmav2"])
def test_padding_mask_stays_binary_after_running_statistics_update(variant):
    network = _network(variant)
    all_valid = _observations()
    variables = network.init(jax.random.PRNGKey(2), all_valid)
    (_, _), first_updates = network.apply(variables, all_valid, mutable=["run_stats"])

    one_invalid = np.array(all_valid, copy=True)
    one_invalid[:, 12:].reshape(4, 3, 8)[:, -1, -1] = 0.0
    updated_variables = {
        "params": variables["params"],
        "run_stats": first_updates["run_stats"],
    }
    (policy, _), _ = network.apply(
        updated_variables,
        jnp.asarray(one_invalid),
        mutable=["run_stats"],
    )

    np.testing.assert_allclose(np.asarray(policy.mean())[:, -1], 0.0, atol=1e-7)
