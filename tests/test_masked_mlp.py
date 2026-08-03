import jax
import jax.numpy as jnp
import numpy as np

from scaling.masked_mlp import MaskedActorCritic


def test_masked_mlp_suppresses_padded_action_mean_and_variance():
    network = MaskedActorCritic(
        action_dim=3,
        action_mask_start=5,
        hidden_layer_dims=(16, 8),
        learnable_std=True,
    )
    observation = jnp.asarray(
        [
            [0.2, -0.1, 0.3, 1.0, 0.0, 1.0, 0.0, 0.0],
            [-0.2, 0.4, 0.1, 0.0, 1.0, 1.0, 1.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    variables = network.init(jax.random.PRNGKey(0), observation)
    (policy, value), _ = network.apply(variables, observation, mutable=["run_stats"])

    assert policy.mean().shape == (2, 3)
    assert value.shape == (2,)
    np.testing.assert_allclose(np.asarray(policy.mean())[0, 1:], 0.0, atol=1e-7)
    np.testing.assert_allclose(np.asarray(policy.stddev())[0, 1:], 1e-3, rtol=1e-5)
    assert np.isfinite(np.asarray(policy.mean())).all()
