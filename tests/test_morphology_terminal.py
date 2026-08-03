"""The terminal handler must judge a body against its own standing height.

The stock ``RootPoseTrajTerminalStateHandler`` derives its root-height window
from the reference trajectory alone, so a legitimately taller robot is declared
absorbing for standing higher than the reference. That silently caps the
morphology range independently of how well the policy controls the body.
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE / "scripts") not in sys.path:
    sys.path.insert(0, str(WORKSPACE / "scripts"))

from morphology_deepmimic import (  # noqa: E402
    crop_trajectory,
    get_robot,
    ground_trajectory_constant,
    make_mimic_env,
    resolve_window,
)
from loco_mujoco.task_factories import ImitationFactory, LAFAN1DatasetConf  # noqa: E402
from scaling.embodiment_catalog import build_catalog_from_descriptors  # noqa: E402
from scaling.morphology_terminal import (  # noqa: E402
    MorphologyAwareRootPoseTrajTerminalStateHandler,
)
from scaling.online_h1 import MorphologyBounds, register_online_h1_env  # noqa: E402

NOMINAL_XML = WORKSPACE / "generated_variants" / "h1_morphology_nominal" / "h1.xml"

# leg 1.5 is the `extreme_tall_light` variant: 0.8 * (1.5 - 1) = 0.40 m above the
# reference, beyond the handler's 0.3 m margin.
TALL = [1.5, 0.72, 0.72, 0.55]
NOMINAL = [1.0, 1.0, 1.0, 1.0]


def _make_env(handler: str, run_tag: str):
    robot = get_robot("h1")
    source = ImitationFactory.make(
        robot.cpu_env_name,
        lafan1_dataset_conf=LAFAN1DatasetConf(["walk1_subject1"]),
    )
    start, frames = resolve_window(source.th.traj, 4.0, None)
    trajectory, _ = ground_trajectory_constant(
        NOMINAL_XML, crop_trajectory(source.th.traj, start, frames)
    )
    catalog = build_catalog_from_descriptors(
        [NOMINAL, TALL],
        bounds_low=MorphologyBounds().low,
        bounds_high=MorphologyBounds().high,
        split="test",
    )
    name = f"MjxH1TerminalTest_{run_tag}"
    register_online_h1_env(name, NOMINAL_XML)
    env = make_mimic_env(
        name,
        trajectory,
        headless=True,
        nconmax=7000,
        catalog_descriptors=catalog.descriptors,
        catalog_mode="fixed_balanced",
        terminal_state_type=handler,
    )
    return env


@pytest.fixture(scope="module")
def stock_env():
    return _make_env("RootPoseTrajTerminalStateHandler", "stock")


@pytest.fixture(scope="module")
def aware_env():
    return _make_env(
        "MorphologyAwareRootPoseTrajTerminalStateHandler",
        "aware",
    )


def _absorbing_at_reset(env, slot):
    state = env.mjx_reset_with_slot(jax.random.PRNGKey(3), jnp.asarray(slot, jnp.int32))
    absorbing, _ = env._terminal_state_handler.mjx_is_absorbing(
        env, state.observation, state.info, state.data, state.additional_carry
    )
    return bool(absorbing)


def test_handler_is_registered_under_its_class_name():
    from loco_mujoco.core.terminal_state_handler import TerminalStateHandler

    assert (
        TerminalStateHandler.registered[
            "MorphologyAwareRootPoseTrajTerminalStateHandler"
        ]
        is MorphologyAwareRootPoseTrajTerminalStateHandler
    )


def test_root_height_offset_matches_the_reset_shift(aware_env):
    """Reset and termination must share one definition of the offset."""
    assert float(aware_env.root_height_offset(jnp.asarray(NOMINAL))) == 0.0
    np.testing.assert_allclose(
        float(aware_env.root_height_offset(jnp.asarray(TALL))), 0.4, atol=1e-6
    )


def test_stock_handler_kills_a_tall_body_at_reset(stock_env):
    """Documents the defect: the tall body is terminal before it acts."""
    assert not _absorbing_at_reset(stock_env, 0)  # nominal survives
    assert _absorbing_at_reset(stock_env, 1)  # 1.5x leg is already terminal


def test_morphology_aware_handler_lets_the_tall_body_start(aware_env):
    assert not _absorbing_at_reset(aware_env, 0)
    assert not _absorbing_at_reset(aware_env, 1)
