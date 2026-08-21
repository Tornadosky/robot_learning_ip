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
    def init_agent_conf(cls, env, config, actor_latent_buffer=None):
        if not getattr(env, "append_joint_features", False):
            raise TypeError(
                "CrossTopologyURMAPPO requires joint_block_specs in the parallel "
                "environment (append_joint_features)."
            )
        latent_requested = actor_latent_buffer is not None or int(
            config.experiment.get("actor_latent_dim", 0)
        ) > 0
        if latent_requested and not getattr(
            env, "route_reset_observation_on_done", False
        ):
            raise ValueError(
                "A trajectory-keyed actor latent needs "
                "route_reset_observation_on_done=True on ParallelMorphVecEnv; "
                "otherwise done steps pair the reset cursor's z with the "
                "terminal observation."
            )
        return super().init_agent_conf(env, config, actor_latent_buffer)

    @staticmethod
    def _next_actor_observation(actor_latent_buffer, returned_obs, env_state, done):
        # ParallelMorphVecEnv already routes the reset observation on done
        # steps (a grouped heterogeneous state exposes no flat observation
        # view, so the stock env_state.observation read cannot work here).
        return returned_obs
