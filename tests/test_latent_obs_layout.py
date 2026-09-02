"""The FSQ latent block's observation layout, which was previously unverifiable.

`environment.py` maintains the per-joint channel ORDER in two places by hand --
the index loops in `get_observation_space` and the concatenation in
`get_observation` -- with only a comment tying them together. The latent block
was the one group with no index list at all, which is also why it is the one
group missing from the `set_norm` normalisation table: there was nothing to
address.

These tests assert, on the real environment:
  * the recorded latent indices actually select the clip's z_q values, so an
    ordering slip between the two hand-maintained sites is detectable;
  * `set_norm` reaches them, so the divisor knob is not a silent no-op;
  * "global" scope keeps the per-joint block at its token-free width and puts
    exactly `latent_dim` pooled channels into the general observation;
  * that pooled value is the mean over MAPPED joints, not over all actuators.

Runs under pytest, or as a plain script -- the cluster the arms actually train
on has no pytest installed.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

try:
    import pytest
except ModuleNotFoundError:            # Viper's conda env ships no pytest, and
    pytest = None                      # this file has to run there as a script.

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MUJOCO_GL", "disable")

REPO = Path(__file__).resolve().parents[1]
CLIP_DIR = REPO / "experiments" / "fsq_khaendler" / "clips_tokentest"
CLIP = "dance2_subject4.npz"
ZQ = "dance2_subject4_zq.npz"
LATENT_DIM = 32

_HAVE_ZQ = (CLIP_DIR / "UnitreeH1" / ZQ).exists()
if pytest is not None:
    pytestmark = pytest.mark.skipif(not _HAVE_ZQ, reason="needs the H1 z_q sidecar")


def _build(scope=None, divisor=None, latent=True, replaces=False):
    """Build the real single-robot training environment."""
    import importlib
    from ml_collections import config_dict
    from rl_x.algorithms.algorithm_manager import get_algorithm_config
    from rl_x.environments.environment_manager import get_environment_config
    from rl_x.runner.default_config import get_config as get_runner_config
    from rl_x.runner.runner_mode import RunnerMode

    importlib.import_module("loco_mjx.environments.locomotion.urma2.mjx")
    importlib.import_module("loco_mjx.algorithms.urma2.mjx")
    from loco_mjx.environments.locomotion.urma2.mjx.create_env import create_env

    config = config_dict.ConfigDict()
    config.runner = get_runner_config(RunnerMode.TRAIN)
    config.algorithm = get_algorithm_config("urma2.mjx")
    config.environment = get_environment_config("locomotion.urma2.mjx")
    config.environment.name = "locomotion.urma2.mjx"
    config.environment.train_robots = ("unitree_h1",)
    config.environment.eval_robots = ()
    config.environment.nr_envs = 1
    config.environment.nr_eval_envs = 0
    config.environment.render = False
    config.environment.seed = 0
    config.environment.terrain.type = "plane"
    config.environment.critic_exteroceptive_observation_type = "none"
    config.environment.command.type = "tracking_clip"
    config.environment.reward.type = "tracking"
    config.environment.command.tracking_clip_dir = str(CLIP_DIR)
    config.environment.command.tracking_clip_file = CLIP
    config.environment.domain_randomization.initial_state.type = "reference"
    config.environment.domain_randomization.observation_noise.type = "none"
    if latent:
        config.environment.command.tracking_clip_latent_obs = True
        config.environment.command.tracking_clip_latent_dim = LATENT_DIM
        config.environment.command.tracking_clip_latent_replaces_reference = replaces
        if scope is not None:
            config.environment.command.tracking_clip_latent_scope = scope
        if divisor is not None:
            config.environment.command.tracking_clip_latent_obs_divisor = divisor
    train_env, _ = create_env(config)
    return train_env


def _attr(env, name):
    """Per-robot attribute off the MultiEnvironment wrapper (one robot here)."""
    return np.asarray(env.call(name)[0])


def _observation(env):
    """One real observation vector from a reset environment."""
    import jax

    keys = jax.random.split(jax.random.PRNGKey(0), 1)
    states = env.init(keys)
    multi_state = env.create_multi_state(states)
    return np.asarray(multi_state["next_observation"]).reshape(-1)


def _codes():
    z = np.load(CLIP_DIR / "UnitreeH1" / ZQ, allow_pickle=True)
    return np.asarray(z["z_q"], dtype=np.float32)


def test_per_joint_indices_point_at_the_token():
    """The recorded indices must select the clip's actual z_q values.

    This is what makes a channel-order slip between the index loops and the
    concatenation detectable instead of silent.
    """
    env = _build(scope="per_joint", divisor=1.0)
    idx = _attr(env, "policy_joint_latent_obs_idx")
    nr_actuators = int(_attr(env, "nr_actuators"))
    assert idx.size == nr_actuators * LATENT_DIM

    obs = _observation(env)
    seen = obs[idx].reshape(nr_actuators, LATENT_DIM)
    mapped = seen[np.abs(seen).sum(axis=1) > 0]
    assert mapped.shape[0] > 0, "no actuator observed a token"

    # Every observed row must be a row the tokenizer actually emitted. An
    # ordering slip would put values there that are not codes at all.
    z_q = _codes()
    codes = np.unique(z_q.reshape(-1, LATENT_DIM), axis=0)
    for row in mapped:
        assert np.min(np.abs(codes - row).sum(axis=1)) < 1e-5


def test_set_norm_reaches_the_latent_block():
    """The divisor knob must actually divide -- the bug was that it could not."""
    env1 = _build(scope="per_joint", divisor=1.0)
    div1 = _attr(env1, "obs_norm_divisor")
    idx1 = _attr(env1, "policy_joint_latent_obs_idx")
    assert np.allclose(div1[idx1], 1.0)

    env10 = _build(scope="per_joint", divisor=10.0)
    div10 = _attr(env10, "obs_norm_divisor")
    idx10 = _attr(env10, "policy_joint_latent_obs_idx")
    assert np.allclose(div10[idx10], 10.0)

    # and it must not have touched any other channel's normalisation. The
    # CRITIC's latent block takes the same divisor, so it is excluded too --
    # leaving it in was this test's own bug, not the environment's.
    touched = np.concatenate([idx10, _attr(env10, "critic_joint_latent_obs_idx")])
    other = np.setdiff1d(np.arange(div10.size), touched)
    assert np.allclose(div10[other], div1[other])
    assert np.allclose(div10[_attr(env10, "critic_joint_latent_obs_idx")], 10.0)


def test_global_scope_keeps_the_joint_block_narrow():
    """Global routing must not widen the per-joint bottleneck's input."""
    none_env = _build(latent=False)
    per_joint = _build(scope="per_joint")
    glob = _build(scope="global")

    assert per_joint.joint_observation_size == none_env.joint_observation_size + LATENT_DIM
    assert glob.joint_observation_size == none_env.joint_observation_size
    assert _attr(glob, "policy_joint_latent_obs_idx").size == 0
    assert _attr(glob, "policy_motion_latent_obs_idx").size == LATENT_DIM
    assert _attr(per_joint, "policy_motion_latent_obs_idx").size == 0

    # the pooled embedding must land inside the general observation the policy
    # actually reads, or it is present but unreachable
    general = set(_attr(glob, "policy_general_obs_idx").tolist())
    assert set(_attr(glob, "policy_motion_latent_obs_idx").tolist()) <= general


def test_global_embedding_is_the_pooled_code():
    """The general-block values must equal the mean over MAPPED joints."""
    env = _build(scope="global", divisor=1.0)
    obs = _observation(env)
    pooled = obs[_attr(env, "policy_motion_latent_obs_idx")]
    frame_means = _codes().mean(axis=1)
    assert np.min(np.abs(frame_means - pooled).sum(axis=1)) < 1e-4


def test_both_scope_is_the_union():
    env = _build(scope="both")
    nr_actuators = int(_attr(env, "nr_actuators"))
    assert _attr(env, "policy_joint_latent_obs_idx").size == nr_actuators * LATENT_DIM
    assert _attr(env, "policy_motion_latent_obs_idx").size == LATENT_DIM


if __name__ == "__main__":
    # Runnable without pytest, because the machine that most needs to run this
    # -- the cluster the arms actually train on -- does not have it installed.
    import sys
    import traceback

    if not _HAVE_ZQ:
        print(f"SKIP: no z_q sidecar under {CLIP_DIR}")
        sys.exit(0)
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}", flush=True)
        except Exception:
            failed.append(name)
            print(f"FAIL {name}", flush=True)
            traceback.print_exc()
    print(f"LAYOUT SUMMARY: {len(tests) - len(failed)} passed, {len(failed)} failed"
          + (f" ({', '.join(failed)})" if failed else ""), flush=True)
    sys.exit(1 if failed else 0)
