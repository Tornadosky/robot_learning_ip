from pathlib import Path
from types import SimpleNamespace
import zipfile
from experiments.fsq_khaendler.crosseval_motion import (
    apply_evaluation_settings, build_arg_parser, make_eval_checkpoint, make_eval_condition,
)


def config():
    seen = SimpleNamespace(morphology_coeff_mode="curriculum", morphology_coeff_value=0.,
                           body_pool_size=0, exact_inertia_rescale=False,
                           torque_scaling_exponent=1.)
    return SimpleNamespace(environment=SimpleNamespace(
        command=SimpleNamespace(tracking_clip_observe_root_heading=False),
        domain_randomization=SimpleNamespace(sampling_type="none", sampling_probability=0.,
                                             seen_robot=seen)))


def args(*extra):
    return build_arg_parser().parse_args(["--model_path", "model/latest.model",
        "--clip_dir", "clips", "--raw_clip_dir", "raw", "--refbias", "0.0",
        "--out", "out.json", *extra])


def test_parser_accepts_heading_and_morphology_flags():
    a = args("--root_heading_obs", "True", "--morphology_coeff", ".3",
             "--torque_scaling_exponent", "4", "--exact_inertia_rescale", "True",
             "--body_pool_size", "128")
    assert (a.root_heading_obs, a.morphology_coeff, a.torque_scaling_exponent,
            a.exact_inertia_rescale, a.body_pool_size) == ("True", .3, 4., "True", 128)


def test_randomized_eval_is_reset_only_and_stamped():
    a = args("--root_heading_obs", "True", "--morphology_coeff", ".3",
             "--torque_scaling_exponent", "4", "--exact_inertia_rescale", "True",
             "--body_pool_size", "64")
    c = config()
    apply_evaluation_settings(c, a)
    condition = make_eval_condition(a, ("unitree_h1", "unitree_g1", "booster_t1"))
    assert c.environment.command.tracking_clip_observe_root_heading is True
    assert c.environment.domain_randomization.sampling_type == "step_probability_and_reset"
    assert c.environment.domain_randomization.sampling_probability == 0
    assert c.environment.domain_randomization.seen_robot.morphology_coeff_value == .3
    assert c.environment.domain_randomization.seen_robot.torque_scaling_exponent == 4
    assert c.environment.domain_randomization.seen_robot.exact_inertia_rescale is True
    assert condition["morphology_coeff"] == .3


def test_nominal_eval_disables_randomization():
    c, a = config(), args("--morphology_coeff", "0")
    apply_evaluation_settings(c, a)
    assert c.environment.domain_randomization.sampling_type == "none"


def test_eval_checkpoint_sanitizes_only_selected_archive(tmp_path: Path):
    source_dir = tmp_path / "models"
    source_dir.mkdir()
    latest = source_dir / "latest.model"
    with zipfile.ZipFile(latest, "w") as zf:
        zf.writestr("config_algorithm.json", "{}")
        zf.writestr("training_progress.json", "{}")
        zf.writestr("resume_manifest.json", "{}")
        zf.writestr("resume_state.npz", b"resume")
        zf.writestr("checkpoint/_METADATA", "metadata")
    with zipfile.ZipFile(source_dir / "model_100.model", "w") as zf:
        zf.writestr("should_not_be_copied.txt", "x")

    sanitized = Path(make_eval_checkpoint(str(latest), str(tmp_path / "work")))
    assert sanitized.name == "latest.model"
    assert not (sanitized.parent / "model_100.model").exists()
    with zipfile.ZipFile(sanitized) as zf:
        names = set(zf.namelist())
    assert "config_algorithm.json" in names
    assert "checkpoint/_METADATA" in names
    assert not {"training_progress.json", "resume_manifest.json", "resume_state.npz"} & names
