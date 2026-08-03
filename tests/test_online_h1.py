from pathlib import Path

import jax
import mujoco
import numpy as np
import pytest

import jax.numpy as jnp

from h1_morphology_variants import PRESETS
from scaling.embodiment_catalog import build_catalog, exposure_summary
from scaling.online_h1 import MorphologyBounds, register_online_h1_env


WORKSPACE = Path(__file__).resolve().parents[1]

CATALOG = build_catalog(
    num_bodies=64,
    seed=7_000_001,
    bounds_low=MorphologyBounds().low,
    bounds_high=MorphologyBounds().high,
    topology_signature="test-signature",
)


@pytest.fixture(scope="module")
def online_env():
    xml_path = WORKSPACE / "generated_variants" / "h1_morphology_nominal" / "h1.xml"
    env_cls = register_online_h1_env("MjxH1OnlineMorphTest", xml_path)
    return env_cls()


@pytest.fixture(scope="module")
def resampling_urma_env():
    xml_path = WORKSPACE / "generated_variants" / "h1_morphology_nominal" / "h1.xml"
    env_cls = register_online_h1_env("MjxH1OnlineMorphResamplingTest", xml_path)
    return env_cls(
        append_urma_joint_features=True,
        resample_morphology_on_episode_reset=True,
    )


@pytest.fixture(scope="module")
def fixed_balanced_env():
    xml_path = WORKSPACE / "generated_variants" / "h1_morphology_nominal" / "h1.xml"
    env_cls = register_online_h1_env("MjxH1CatalogFixedTest", xml_path)
    return env_cls(
        catalog_descriptors=CATALOG.descriptors,
        catalog_mode="fixed_balanced",
    )


@pytest.fixture(scope="module")
def catalog_resample_env():
    xml_path = WORKSPACE / "generated_variants" / "h1_morphology_nominal" / "h1.xml"
    env_cls = register_online_h1_env("MjxH1CatalogResampleTest", xml_path)
    return env_cls(
        catalog_descriptors=CATALOG.descriptors,
        catalog_mode="catalog_resample",
    )


def _descriptor(preset_name):
    preset = PRESETS[preset_name]
    return np.asarray(
        [
            preset.leg_length_scale,
            preset.arm_length_scale,
            preset.shoulder_width_scale,
            preset.torso_mass_scale,
        ],
        dtype=np.float32,
    )


@pytest.mark.parametrize(
    "preset_name",
    ["nominal", "tall_legs", "long_arms", "broad_shoulders", "heavy_torso"],
)
def test_dynamic_arrays_match_existing_static_generator(online_env, preset_name):
    static_xml = (
        WORKSPACE / "generated_variants" / f"h1_morphology_{preset_name}" / "h1.xml"
    )
    static_model = mujoco.MjModel.from_xml_path(str(static_xml))
    dynamic_model = online_env._apply_morphology(
        online_env.sys, _descriptor(preset_name)
    )

    np.testing.assert_allclose(dynamic_model.body_pos, static_model.body_pos, atol=2e-6)
    np.testing.assert_allclose(
        dynamic_model.body_ipos, static_model.body_ipos, atol=2e-6
    )
    # XML serialization rounds decimal text slightly before MuJoCo recompiles it.
    np.testing.assert_allclose(
        dynamic_model.body_mass, static_model.body_mass, atol=1e-4
    )
    np.testing.assert_allclose(
        dynamic_model.body_inertia, static_model.body_inertia, atol=1e-4
    )
    np.testing.assert_allclose(dynamic_model.site_pos, static_model.site_pos, atol=2e-6)


def test_invalid_morphology_bounds_are_rejected():
    with pytest.raises(ValueError):
        MorphologyBounds(low=(1.0, 1.0, 1.0, 1.0), high=(0.9, 1.1, 1.1, 1.1)).validate()


def test_urma_joint_feature_layout_is_explicit_and_finite(resampling_urma_env):
    state = resampling_urma_env.mjx_reset(jax.random.PRNGKey(11))
    layout = resampling_urma_env.urma_input_layout
    assert layout is not None
    assert layout.num_joints == resampling_urma_env.info.action_space.shape[0]
    assert layout.joint_description_dim == 26
    features = np.asarray(state.observation)[
        layout.joint_feature_start : layout.joint_feature_stop
    ].reshape(layout.num_joints, layout.joint_feature_dim)
    assert np.isfinite(features).all()
    np.testing.assert_array_equal(features[:, layout.joint_mask_offset], 1.0)


def test_episode_reset_resamples_body_and_returns_matching_observation(
    resampling_urma_env,
):
    state = resampling_urma_env.mjx_reset(jax.random.PRNGKey(23))
    old_morphology = np.asarray(state.additional_carry.morphology)
    old_generation = int(state.additional_carry.morphology_generation)

    reset_state = resampling_urma_env._mjx_reset_in_step(state)
    new_morphology = np.asarray(reset_state.additional_carry.morphology)
    assert not np.allclose(old_morphology, new_morphology)
    assert int(reset_state.additional_carry.morphology_generation) == old_generation + 1
    np.testing.assert_allclose(
        reset_state.additional_carry.final_observation,
        reset_state.observation,
    )
    layout = resampling_urma_env.urma_input_layout
    expected = np.asarray(
        resampling_urma_env._normalized_morphology(
            reset_state.additional_carry.morphology
        )
    )
    actual = np.asarray(reset_state.observation)[
        layout.morphology_start : layout.morphology_start + layout.morphology_dim
    ]
    np.testing.assert_allclose(actual, expected)


def test_slot_assignment_is_exactly_balanced_and_matches_the_catalog(
    fixed_balanced_env,
):
    num_envs = 256
    slots = jnp.arange(num_envs, dtype=jnp.int32)
    keys = jax.random.split(jax.random.PRNGKey(5), num_envs)
    state = jax.vmap(fixed_balanced_env.mjx_reset_with_slot, in_axes=(0, 0))(
        keys, slots
    )

    body_index = np.asarray(state.additional_carry.body_index)
    summary = exposure_summary(body_index, CATALOG.num_bodies)
    assert summary["min_exposure"] == summary["max_exposure"] == num_envs // 64
    np.testing.assert_allclose(
        np.asarray(state.additional_carry.morphology),
        CATALOG.descriptors[body_index].astype(np.float32),
        atol=1e-6,
    )
    # The descriptor channel of the observation describes the assigned body.
    expected = np.asarray(
        jax.vmap(fixed_balanced_env._normalized_morphology)(
            state.additional_carry.morphology
        )
    )
    np.testing.assert_allclose(
        np.asarray(state.observation)[:, -4:], expected, atol=1e-5
    )


def test_fixed_balanced_keeps_its_body_across_episode_resets(fixed_balanced_env):
    state = fixed_balanced_env.mjx_reset_with_slot(
        jax.random.PRNGKey(9), jnp.asarray(37, dtype=jnp.int32)
    )
    before = np.asarray(state.additional_carry.morphology)
    for _ in range(3):
        state = fixed_balanced_env._mjx_reset_in_step(state)
        np.testing.assert_allclose(
            np.asarray(state.additional_carry.morphology), before
        )
        assert int(state.additional_carry.body_index) == 37 % CATALOG.num_bodies


def test_catalog_resample_walks_the_catalog_and_reports_the_new_body(
    catalog_resample_env,
):
    state = catalog_resample_env.mjx_reset_with_slot(
        jax.random.PRNGKey(13), jnp.asarray(5, dtype=jnp.int32)
    )
    seen = [int(state.additional_carry.body_index)]
    assert seen[0] == 5
    for step in range(4):
        state = catalog_resample_env._mjx_reset_in_step(state)
        index = int(state.additional_carry.body_index)
        assert index == (5 + step + 1) % CATALOG.num_bodies
        np.testing.assert_allclose(
            np.asarray(state.additional_carry.morphology),
            CATALOG.descriptors[index].astype(np.float32),
            atol=1e-6,
        )
        # A reset that changes the body must expose the new body's observation.
        np.testing.assert_allclose(
            state.additional_carry.final_observation, state.observation
        )
        seen.append(index)
    assert len(set(seen)) == len(seen)


def test_slot_reset_matches_plain_reset_when_the_key_draws_that_slot(
    fixed_balanced_env,
):
    key = jax.random.PRNGKey(21)
    plain = fixed_balanced_env.mjx_reset(key)
    slot = plain.additional_carry.body_slot
    forced = fixed_balanced_env.mjx_reset_with_slot(key, slot)
    np.testing.assert_allclose(
        np.asarray(forced.observation), np.asarray(plain.observation)
    )
    np.testing.assert_allclose(
        np.asarray(forced.data.qpos), np.asarray(plain.data.qpos)
    )
    assert int(forced.additional_carry.body_index) == int(
        plain.additional_carry.body_index
    )


def test_catalog_bodies_produce_different_model_arrays(fixed_balanced_env):
    masses = []
    positions = []
    for index in (0, 17, 63):
        model = fixed_balanced_env._apply_morphology(
            fixed_balanced_env.sys,
            jnp.asarray(CATALOG.descriptors[index], dtype=jnp.float32),
        )
        masses.append(np.asarray(model.body_mass))
        positions.append(np.asarray(model.body_pos))
        assert np.all(np.asarray(model.body_mass)[1:] > 0.0)
        assert np.all(np.asarray(model.body_inertia)[1:] > 0.0)
    assert len(np.unique(np.round(masses, 9), axis=0)) == 3
    assert len(np.unique(np.round(positions, 9), axis=0)) == 3


def test_catalog_configuration_errors_are_rejected():
    xml_path = WORKSPACE / "generated_variants" / "h1_morphology_nominal" / "h1.xml"
    env_cls = register_online_h1_env("MjxH1CatalogErrorTest", xml_path)
    with pytest.raises(ValueError):
        env_cls(catalog_mode="fixed_balanced")
    with pytest.raises(ValueError):
        env_cls(catalog_mode="not_a_mode", catalog_descriptors=CATALOG.descriptors)
    with pytest.raises(ValueError):
        env_cls(
            catalog_mode="fixed_balanced",
            catalog_descriptors=np.zeros((4, 4), dtype=np.float32),
        )
    with pytest.raises(ValueError):
        env_cls(catalog_descriptors=CATALOG.descriptors)
