import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaling.online_amp import DescriptorBlindNet, keep_indices_excluding


OBS_DIM = 438
DESCRIPTOR_DIM = 4
DESCRIPTOR_START = OBS_DIM - DESCRIPTOR_DIM


def _net(keep):
    return DescriptorBlindNet(
        hidden_layer_dims=[32, 16], output_dim=1, keep_indices=keep, activation="tanh"
    )


def _logits(net, params, x):
    logits, _ = net.apply(params, x, mutable=["run_stats"])
    return np.asarray(logits)


def _batches(seed=0):
    """Two batches identical except in the morphology descriptor columns."""
    base = jax.random.normal(jax.random.PRNGKey(seed), (64, OBS_DIM))
    perturbed = base.at[:, DESCRIPTOR_START:].set(
        jax.random.normal(jax.random.PRNGKey(seed + 1), (64, DESCRIPTOR_DIM))
    )
    return base, perturbed


def test_keep_indices_drop_exactly_the_descriptor_slice():
    keep = keep_indices_excluding(OBS_DIM, DESCRIPTOR_START, OBS_DIM)
    assert len(keep) == DESCRIPTOR_START
    assert keep == tuple(range(DESCRIPTOR_START))
    with pytest.raises(ValueError):
        keep_indices_excluding(OBS_DIM, 10, 5)
    with pytest.raises(ValueError):
        keep_indices_excluding(OBS_DIM, 0, OBS_DIM + 1)


def test_blind_discriminator_cannot_see_the_descriptor_channel():
    """A constant or mismatched descriptor must not separate expert from policy.

    Expert transitions come from one canonical body; policy transitions carry
    per-environment descriptors.  If the discriminator reads that channel the
    AMP reward degenerates into body identification.
    """
    keep = keep_indices_excluding(OBS_DIM, DESCRIPTOR_START, OBS_DIM)
    net = _net(keep)
    base, perturbed = _batches()
    params = net.init(jax.random.PRNGKey(3), base)
    np.testing.assert_array_equal(
        _logits(net, params, base), _logits(net, params, perturbed)
    )

    # A zero-padded "expert" batch is likewise indistinguishable from the same
    # batch carrying real descriptors.
    padded = base.at[:, DESCRIPTOR_START:].set(0.0)
    np.testing.assert_array_equal(
        _logits(net, params, base), _logits(net, params, padded)
    )


def test_unblinded_discriminator_is_the_negative_control():
    """Without blinding the descriptor channel does change the logits."""
    net = _net(tuple(range(OBS_DIM)))
    base, perturbed = _batches()
    params = net.init(jax.random.PRNGKey(3), base)
    difference = np.max(
        np.abs(_logits(net, params, base) - _logits(net, params, perturbed))
    )
    assert difference > 0.0


def test_blind_network_normalisation_covers_only_kept_columns():
    keep = keep_indices_excluding(OBS_DIM, DESCRIPTOR_START, OBS_DIM)
    net = _net(keep)
    base, _ = _batches()
    params = net.init(jax.random.PRNGKey(3), base)
    stats = jax.tree_util.tree_leaves(params["run_stats"])
    assert stats, "the discriminator should keep running input statistics"
    for leaf in stats:
        array = np.asarray(leaf)
        if array.ndim >= 1 and array.shape[-1] > 1:
            assert array.shape[-1] == len(keep)


def test_expert_padding_matches_policy_width():
    from scaling.online_amp_train import pad_expert_to_policy_width
    from loco_mujoco.trajectory import TrajectoryTransitions

    expert = TrajectoryTransitions(
        observations=jnp.ones((7, DESCRIPTOR_START)),
        next_observations=jnp.ones((7, DESCRIPTOR_START)),
        absorbings=jnp.zeros((7,)),
        dones=jnp.zeros((7,)),
    )
    padded, pad = pad_expert_to_policy_width(expert, OBS_DIM)
    assert pad == DESCRIPTOR_DIM
    assert padded.observations.shape == (7, OBS_DIM)
    assert padded.next_observations.shape == (7, OBS_DIM)
    np.testing.assert_array_equal(
        np.asarray(padded.observations)[:, DESCRIPTOR_START:], 0.0
    )
    unchanged, pad = pad_expert_to_policy_width(padded, OBS_DIM)
    assert pad == 0 and unchanged is padded
