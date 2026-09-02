from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scripts.h1g1t1.diagnostics import derive_metrics, parse_console_log, summarize_final, validate_recipe


def test_parse_console_log_flushes_one_row_per_step(tmp_path: Path):
    log = tmp_path / "train.log"
    log.write_text("\n".join([
        "[x] INFO - ┌───────────────────────────────┬────────────────┐",
        "[x] INFO - │ env_info/joint_tracking_error/h1│ 0.25 │",
        "[x] INFO - │ env_info/root_heading_error/h1│ 1.0 │",
        "[x] INFO - │ steps/nr_env_steps │ 36864 │",
        "[x] INFO - │ steps/nr_updates │ 30 │",
        "[x] INFO - └───────────────────────────────┴────────────────┘",
        "noise",
        "[x] INFO - ┌───────────────────────────────┬────────────────┐",
        "[x] INFO - │ env_info/joint_tracking_error/h1│ 0.09 │",
        "[x] INFO - │ steps/nr_env_steps │ 73728 │",
        "[x] INFO - │ steps/nr_updates │ 60 │",
        "[x] INFO - └───────────────────────────────┴────────────────┘",
    ]))
    frame = parse_console_log(log)
    assert frame["steps/nr_env_steps"].tolist() == [36864, 73728]
    assert frame["steps/nr_updates"].tolist() == [30, 60]
    assert frame["env_info/joint_tracking_error/h1"].tolist() == [0.25, 0.09]
    assert np.isnan(frame.loc[1, "env_info/root_heading_error/h1"])


def test_parse_console_log_merges_duplicate_step_fragments(tmp_path: Path):
    log = tmp_path / "train.log"
    log.write_text("\n".join([
        "[x] INFO - ┌───────────────────────────────┬────────────────┐",
        "[x] INFO - │ env_info/joint_tracking_error/h1│ 0.04 │",
        "[x] INFO - │ steps/nr_env_steps │ 100 │",
        "[x] INFO - │ steps/nr_updates │ 10 │",
        # A copied/final console fragment can start with another metric and end
        # with the same step. The two rows must be coalesced, not overwritten.
        "[x] INFO - │ gradients/policy_grad_norm │ 3.0 │",
        "[x] INFO - │ steps/nr_env_steps │ 100 │",
        "[x] INFO - │ steps/nr_updates │ 10 │",
        "[x] INFO - └───────────────────────────────┴────────────────┘",
    ]))
    frame = parse_console_log(log)
    assert frame["steps/nr_env_steps"].tolist() == [100]
    assert frame.loc[0, "env_info/joint_tracking_error/h1"] == pytest.approx(.04)
    assert frame.loc[0, "gradients/policy_grad_norm"] == pytest.approx(3.0)
    assert frame.loc[0, "steps/nr_updates"] == pytest.approx(10)


def test_derive_metrics_converts_squared_errors_and_heading_units():
    raw = pd.DataFrame({"steps/nr_env_steps": [1],
                        "env_info/joint_tracking_error/h1": [.25],
                        "env_info/qvel_tracking_error/h1": [9.0],
                        "env_info/root_heading_error/h1": [np.pi / 2],
                        "gradients/policy_grad_norm": [4.0],
                        "gradients/critic_grad_norm": [5.0],
                        "policy/std_dev": [.6],
                        "v_value/explained_variance": [.7]})
    out = derive_metrics(raw)
    assert out.loc[0, "tracking/joint_rmse_rad/h1"] == pytest.approx(.5)
    assert out.loc[0, "tracking/joint_rmse_deg/h1"] == pytest.approx(np.degrees(.5))
    assert out.loc[0, "tracking/qvel_rmse/h1"] == pytest.approx(3)
    assert out.loc[0, "tracking/root_heading_error_deg/h1"] == pytest.approx(90)
    assert out.loc[0, "optimizer/policy_grad_norm"] == pytest.approx(4.0)
    assert out.loc[0, "optimizer/critic_grad_norm"] == pytest.approx(5.0)
    assert out.loc[0, "optimizer/policy_std"] == pytest.approx(.6)
    assert out.loc[0, "optimizer/value_explained_variance"] == pytest.approx(.7)


def test_validate_recipe_rejects_alignment_errors():
    valid = validate_recipe(192, 64, 6144, 9_830_400, 117_964_800, 3)
    assert valid["batch_size"] == 12_288
    assert valid["envs_per_robot"] == 64
    assert valid["updates_per_save"] == 800
    with pytest.raises(ValueError, match="nr_envs.*robot"):
        validate_recipe(190, 64, 6080, 9_830_400, 117_964_800, 3)
    with pytest.raises(ValueError, match="minibatch_size.*robot"):
        validate_recipe(192, 64, 4096, 9_830_400, 117_964_800, 3)
    with pytest.raises(ValueError, match="batch_size.*minibatch_size"):
        validate_recipe(192, 64, 5760, 9_830_400, 117_964_800, 3)
    with pytest.raises(ValueError, match="save_every.*batch_size"):
        validate_recipe(192, 64, 6144, 9_830_401, 117_964_800, 3)


def test_summary_prioritizes_motion_quality():
    frame = pd.DataFrame({"steps/nr_env_steps": [100, 200],
                          "tracking/joint_rmse_rad/h1": [.5, .3],
                          "tracking/root_heading_error_deg/h1": [15, 80],
                          "episode/return/h1": [100, 500]})
    summary = summarize_final(frame)
    assert summary["final_step"] == 200
    assert summary["robots"]["h1"]["joint_rmse_rad"] == pytest.approx(.3)
    assert summary["quality_flags"]["h1"]["heading_error_over_45_deg"] is True
