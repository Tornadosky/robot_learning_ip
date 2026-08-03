import numpy as np

from loco_mujoco.algorithms import AMPJax, GAILJax


def test_gail_and_amp_use_their_intended_discriminator_targets():
    gail_policy, gail_expert = GAILJax._get_discriminator_targets(3, 2)
    amp_policy, amp_expert = AMPJax._get_discriminator_targets(3, 2)

    np.testing.assert_array_equal(gail_policy, np.zeros(3))
    np.testing.assert_array_equal(gail_expert, np.ones(2))
    np.testing.assert_array_equal(amp_policy, -np.ones(3))
    np.testing.assert_array_equal(amp_expert, np.ones(2))
