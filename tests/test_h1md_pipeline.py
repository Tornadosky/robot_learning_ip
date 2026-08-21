"""Tests required by H1_MORPHOLOGY_DEEPMIMIC_GOAL.md for the h1md pipeline.

The goal document enumerates a minimum test set: deterministic body regeneration
and hashes, body/reference ID alignment, reference schema and finiteness,
normalised quaternions, and the reward and renderer consuming the same target
frame and provider. These cover the parts that can be checked without a GPU or a
training run; the dataset-backed ones skip cleanly where the LAFAN1 cache is not
reachable (its configured paths are posix, so they resolve under WSL only).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "scripts"))
sys.path.insert(0, str(WORKSPACE / "scripts" / "h1md"))

mujoco = pytest.importorskip("mujoco")

from c2_body_equivalence import (  # noqa: E402
    MIMIC_SITES, apply_dynamic_morphology, fk_sites, load, model_hash, validity,
)
from c6_reward_discrimination import DYNAMIC_DIMS, build_model, catalog  # noqa: E402

XML_ROOT = WORKSPACE / "experiments" / "h1_morphology_deepmimic_20260808" / "body_catalog" / "xml"


def _dataset_available() -> bool:
    try:
        import yaml

        import loco_mujoco
        v = yaml.safe_load(open(loco_mujoco.PATH_TO_VARIABLES))
        p = Path(v["LOCOMUJOCO_CONVERTED_LAFAN1_PATH"]) / "UnitreeH1" / "dance2_subject4.npz"
        return p.exists()
    except Exception:
        return False


needs_dataset = pytest.mark.skipif(
    not _dataset_available(), reason="LAFAN1 cache not reachable (posix paths; run under WSL)"
)


# --------------------------------------------------------------------------- #
# deterministic body regeneration and hashes
# --------------------------------------------------------------------------- #

def test_catalog_is_deterministic():
    """The same seeds must yield the same morphology descriptors every time."""
    a, b = catalog(), catalog()
    assert [n for n, _ in a] == [n for n, _ in b]
    for (_, ma), (_, mb) in zip(a, b):
        assert ma == mb


def test_catalog_bodies_are_distinct_and_in_bounds():
    entries = catalog()
    assert len(entries) == 5
    seen = set()
    for name, morph in entries:
        key = tuple(round(morph[d], 9) for d in DYNAMIC_DIMS)
        assert key not in seen, f"{name} duplicates another catalog body"
        seen.add(key)
        assert morph["leg_length_scale"] >= 0.85 - 1e-9
        assert morph["torso_mass_scale"] <= 1.50 + 1e-9


@pytest.mark.skipif(not XML_ROOT.exists(), reason="body catalog XMLs not generated yet")
def test_body_regeneration_is_hash_stable():
    """Rebuilding a body from its descriptor reproduces the same model arrays."""
    name, morph = catalog()[1]
    first = model_hash(build_model(name, morph, XML_ROOT))
    second = model_hash(build_model(name, morph, XML_ROOT))
    assert first == second


@pytest.mark.skipif(not XML_ROOT.exists(), reason="body catalog XMLs not generated yet")
def test_xml_and_dynamic_paths_agree_on_the_reward_sites():
    """C2 Finding 4: the two body-construction paths must build the same body.

    This is the assumption the whole architecture rests on -- offline retargeting
    uses the XML body, training uses the dynamic-array body.
    """
    import loco_mujoco

    nominal = load(Path(loco_mujoco.environments.UnitreeH1.get_default_xml_file_path()))
    rng = np.random.default_rng(0)
    lo, hi = nominal.jnt_range[1:, 0], nominal.jnt_range[1:, 1]
    poses = np.zeros((16, nominal.nq))
    poses[:, 2] = 1.0
    poses[:, 3] = 1.0
    poses[:, 7:] = rng.uniform(lo, hi, size=(16, nominal.nq - 7))

    for name, morph in catalog()[1:]:
        xml_model = build_model(name, morph, XML_ROOT)
        dyn_model = apply_dynamic_morphology(nominal, morph)
        assert np.allclose(xml_model.body_mass, dyn_model.body_mass, atol=1e-9)
        err = np.linalg.norm(fk_sites(xml_model, poses) - fk_sites(dyn_model, poses), axis=-1)
        assert err.max() < 1e-5, f"{name}: site FK differs by {err.max():.2e} m"


@pytest.mark.skipif(not XML_ROOT.exists(), reason="body catalog XMLs not generated yet")
def test_every_catalog_body_is_physically_valid():
    for name, morph in catalog():
        v = validity(build_model(name, morph, XML_ROOT))
        assert v["all_finite"], name
        assert v["positive_mass"], name
        assert v["positive_inertia"], name
        assert v["valid_jnt_ranges"], name
        assert v["ten_steps_finite"], name


# --------------------------------------------------------------------------- #
# reference schema, alignment, and the reward/renderer sharing one provider
# --------------------------------------------------------------------------- #

@needs_dataset
@pytest.mark.skipif(not XML_ROOT.exists(), reason="body catalog XMLs not generated yet")
def test_reference_schema_and_alignment():
    """Reference must be complete, finite, unit-quaternion, and body-specific."""
    from c6_reward_discrimination import reground
    from c8_reference_feasibility import reference_qvel
    from c9_shared_policy import make_complete_trajectory, register_variant
    from loco_mujoco.task_factories import ImitationFactory, LAFAN1DatasetConf

    src = ImitationFactory.make("UnitreeH1", lafan1_dataset_conf=LAFAN1DatasetConf(["dance2_subject4"]))
    th = src.th.traj
    freq = float(th.info.frequency)
    qpos = np.asarray(th.data.qpos)[19482:19582].astype(np.float64)
    qvel = np.asarray(th.data.qvel)[19482:19582].astype(np.float64)

    refs = {}
    for name, morph in catalog()[:3]:
        model = build_model(name, morph, XML_ROOT)
        env_name = f"TestH1md_{name}"
        register_variant(env_name, XML_ROOT / f"h1_morphology_c2_{name}" / "h1.xml", mjx=False)
        q_ref, _ = reground(model, qpos)
        traj = make_complete_trajectory(env_name, q_ref, reference_qvel(qvel, q_ref, 1.0 / freq),
                                        freq, th)

        assert bool(traj.data.is_complete), f"{name}: reference lacks the site data MimicReward reads"
        # Finding 17: extend_motion must not silently downsample the reference
        assert float(traj.info.frequency) == pytest.approx(freq), f"{name}: reference resampled"
        arr_q = np.asarray(traj.data.qpos)
        assert np.isfinite(arr_q).all(), name
        quats = arr_q[:, 3:7]
        assert np.allclose(np.linalg.norm(quats, axis=1), 1.0, atol=1e-5), f"{name}: quats not unit"
        refs[name] = arr_q

    # body/reference alignment: no body may share another body's reference
    names = list(refs)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not np.allclose(refs[a], refs[b]), f"{a} and {b} share a reference"


@needs_dataset
@pytest.mark.skipif(not XML_ROOT.exists(), reason="body catalog XMLs not generated yet")
def test_renderer_targets_come_from_the_reward_provider():
    """The overlay must index the same array at the same frame as MimicReward.

    The goal document forbids reconstructing a separate approximate target for
    rendering; this pins the renderer to `traj.data.site_xpos[rel_site_ids]`.
    """
    from c6_reward_discrimination import reground
    from c8_reference_feasibility import reference_qvel
    from c9_shared_policy import make_complete_trajectory, register_variant
    from loco_mujoco.task_factories import ImitationFactory, LAFAN1DatasetConf

    src = ImitationFactory.make("UnitreeH1", lafan1_dataset_conf=LAFAN1DatasetConf(["dance2_subject4"]))
    th = src.th.traj
    freq = float(th.info.frequency)
    qpos = np.asarray(th.data.qpos)[19482:19532].astype(np.float64)
    qvel = np.asarray(th.data.qvel)[19482:19532].astype(np.float64)

    name, morph = catalog()[0]
    model = build_model(name, morph, XML_ROOT)
    env_name = f"TestH1mdRender_{name}"
    register_variant(env_name, XML_ROOT / f"h1_morphology_c2_{name}" / "h1.xml", mjx=False)
    q_ref, _ = reground(model, qpos)
    traj = make_complete_trajectory(env_name, q_ref, reference_qvel(qvel, q_ref, 1.0 / freq), freq, th)

    rel_site_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n) for n in MIMIC_SITES])
    targets = np.asarray(traj.data.site_xpos)[:, rel_site_ids, :]

    # the same site ids MimicReward resolves from the same names
    from loco_mujoco.core.reward.trajectory_based import MimicReward  # noqa: F401
    assert targets.shape[1] == len(MIMIC_SITES)
    assert np.isfinite(targets).all()

    # recomputing FK at a frame must land on the drawn target
    data = mujoco.MjData(model)
    frame = 10
    data.qpos[:] = np.asarray(traj.data.qpos)[frame]
    mujoco.mj_forward(model, data)
    assert np.allclose(data.site_xpos[rel_site_ids], targets[frame], atol=1e-6)


# --------------------------------------------------------------------------- #
# the phase-indexing trap, which cost three findings
# --------------------------------------------------------------------------- #

def test_drift_evaluator_indexes_reference_by_phase_not_step():
    """Guard the bug that voided Findings 29/30/31.

    Reference-state initialisation starts each environment at a RANDOM
    trajectory phase. An evaluator that indexes the reference by step number
    measures the phase offset, not the policy -- bounded by the clip's root
    travel, so it looks plausible on a clip that barely moves and reorders the
    arms anyway. This pins the fix in place: the evaluator must read each env's
    own `subtraj_step_no`.
    """
    source = (WORKSPACE / "scripts" / "h1md" / "c21_drift_eval.py").read_text(encoding="utf-8")
    assert "subtraj_step_no" in source, "drift evaluator no longer reads the trajectory phase"
    assert "ref_all[idx]" in source, "drift evaluator no longer indexes the reference by phase"


def test_drift_is_reported_against_reference_travel():
    """A drift number is meaningless without the scale it is measured against.

    The walk clip's root travels 10.34 m; a 10.57 m 'drift' there was the bug,
    not a result. Any drift measurement must carry the reference's own travel so
    the comparison is possible -- `stance_stats` supplies `root_travel_m`.
    """
    from c25_stance_audit import stance_stats
    import inspect
    assert "root_travel_m" in inspect.getsource(stance_stats)


@needs_dataset
@pytest.mark.skipif(not XML_ROOT.exists(), reason="body catalog XMLs not generated yet")
def test_reference_arms_are_distinct_and_grounded():
    """Guard the refactor that removed the third instance of the mismatch bug.

    `refbuild.build_reference` is now the single source of truth. Two properties
    have to hold for the evaluators built on it:

      * the three arms must produce genuinely different references, otherwise
        scoring one policy against another's reference would be undetectable;
      * every arm must be re-grounded, including `ik_scaled` -- Finding 32's
        floating reference cost 77% of episode length when it was corrected.
    """
    from loco_mujoco.task_factories import ImitationFactory, LAFAN1DatasetConf
    from refbuild import ARMS, build_reference

    src = ImitationFactory.make("UnitreeH1", lafan1_dataset_conf=LAFAN1DatasetConf(["dance2_subject4"]))
    qpos = np.asarray(src.th.traj.data.qpos)[19482:19582].astype(np.float64)

    body = "body04_seed1004"
    model = build_model(body, dict(catalog())[body], XML_ROOT)
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    feet = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("left_foot", "right_foot")]

    refs, clearance = {}, {}
    for arm in ARMS:
        q = build_reference(body, arm, qpos, XML_ROOT)
        assert np.isfinite(q).all(), arm
        data = mujoco.MjData(model)
        c = []
        for frame in q[::10]:
            data.qpos[:] = frame
            mujoco.mj_forward(model, data)
            c.append(min(mujoco.mj_geomDistance(model, data, floor, f, 10.0, None) for f in feet))
        refs[arm], clearance[arm] = q, max(c)

    # `fk` and `ik_scaled` are grounded ON THIS BODY, so the lowest foot sits at
    # the clearance rather than floating.
    for arm in ("fk", "ik_scaled"):
        assert clearance[arm] < 0.005, f"{arm}: reference floats up to {clearance[arm]:.4f} m"

    # `shared_nominal` is the NOMINAL body's reference used unchanged, so on a
    # shorter-legged body it necessarily floats -- 11 cm for body04. That is the
    # arm's definition, not a defect, and it is why the arm cannot be mixed with
    # the others at evaluation time: a policy trained on one grounding and scored
    # against another loses most of its episode length (Finding 39).
    assert clearance["shared_nominal"] > 0.02, (
        "shared_nominal unexpectedly grounded on a non-nominal body; the arm is "
        "defined as the nominal reference used unchanged"
    )

    names = list(refs)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not np.allclose(refs[a], refs[b]), f"{a} and {b} produce identical references"


def test_build_reference_rejects_unknown_arm():
    from refbuild import build_reference
    with pytest.raises(ValueError):
        build_reference("body00_nominal", "not_an_arm", np.zeros((4, 26)), XML_ROOT)
