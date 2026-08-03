"""URMA PPO over a padded, heterogeneous-topology environment batch.

``MaskedParallelPPO`` gives the fixed-width MLP a ``ParallelMorphVecEnv``; this
is the same bridge for URMA/URMAv2.  The only difference from ``URMAPPO`` is the
environment wrapper: the grouped environment needs the per-group reward
normalisation from :class:`scaling.parallel_env.ParallelMorphPPO` rather than
the stock single-model one.
"""

from __future__ import annotations

from scaling.parallel_env import ParallelMorphPPO
from scaling.urma_networks import URMAPPO


class CrossTopologyURMAPPO(URMAPPO):
    """URMA over several robot topologies inside one shared PPO update."""

    _wrap_env = staticmethod(ParallelMorphPPO._wrap_env)

    @classmethod
    def init_agent_conf(cls, env, config):
        if not getattr(env, "append_joint_features", False):
            raise TypeError(
                "CrossTopologyURMAPPO requires joint_block_specs in the parallel "
                "environment (append_joint_features)."
            )
        return super().init_agent_conf(env, config)
