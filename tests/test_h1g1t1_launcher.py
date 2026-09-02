import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

from scripts.h1g1t1.collect_results import create_archive
from scripts.h1g1t1.preflight import inspect_clip_tree

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/h1g1t1/run_h1g1t1.sh"


def write_clip(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    qpos = np.zeros((4, 10), np.float32)
    qpos[:, 2] = 1.0
    qpos[:, 3] = 1.0
    np.savez_compressed(path, qpos=qpos, qvel=np.zeros((4, 9), np.float32),
                        joint_names=np.array(["root", "joint"], dtype=object),
                        frequency=np.array(30.))


def test_clip_preflight_requires_three_families(tmp_path: Path):
    for family in ("UnitreeH1", "UnitreeG1", "BoosterT1"):
        write_clip(tmp_path / family / "dance2_subject4.npz")
    report = inspect_clip_tree(tmp_path, "dance2_subject4.npz")
    assert report["ok"] is True
    assert set(report["robots"]) == {"unitree_h1", "unitree_g1", "booster_t1"}
    (tmp_path / "BoosterT1/dance2_subject4.npz").unlink()
    with pytest.raises(FileNotFoundError, match="BoosterT1"):
        inspect_clip_tree(tmp_path, "dance2_subject4.npz")


def test_dry_run_has_heading_randomization_checkpoints_and_tb(tmp_path: Path):
    env = os.environ.copy()
    env.update({"REPO": str(tmp_path / "repo"), "PY": sys.executable,
                "RUN_ID": "pytest_run", "CLIPS": str(tmp_path / "clips")})
    result = subprocess.run(["bash", str(LAUNCHER), "dry-run"], env=env,
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "--environment.nr_envs=576" in out
    assert "--algorithm.evaluation_and_save_frequency=11796480" in out
    assert "--environment.command.tracking_clip_observe_root_heading=True" in out
    assert "--environment.reward.root_heading_tracking_weight_ratio=0.75" in out
    assert "--environment.reward.root_heading_tracking_temperature=2.0" in out
    assert "CUDA_VISIBLE_DEVICES=0" in out
    assert "--environment.domain_randomization.seen_robot.morphology_coeff_mode=schedule" in out
    assert "--environment.domain_randomization.seen_robot.morphology_coeff_value=0.3" in out
    assert "--algorithm.save_intermediate_models=True" in out
    assert "--runner.track_tb=True" in out
    launcher_text = LAUNCHER.read_text(encoding="utf-8")
    assert "--dump_render" in launcher_text
    assert "last_step != TOTAL" in launcher_text


def test_dry_run_rejects_invalid_three_family_geometry(tmp_path: Path):
    env = os.environ.copy()
    env.update({"REPO": str(ROOT), "PY": sys.executable,
                "RUN_ID": "pytest_invalid", "NR_ENVS": "190"})
    result = subprocess.run(["bash", str(LAUNCHER), "dry-run"], env=env,
                            cwd=ROOT, text=True, capture_output=True)
    assert result.returncode != 0
    assert "must be divisible by robot count" in (result.stdout + result.stderr)


def test_dry_run_rejects_multiple_visible_devices():
    env = os.environ.copy()
    env.update({"REPO": str(ROOT), "PY": sys.executable,
                "RUN_ID": "pytest_multigpu", "CUDA_VISIBLE_DEVICES": "0,1"})
    result = subprocess.run(["bash", str(LAUNCHER), "dry-run"], env=env,
                            cwd=ROOT, text=True, capture_output=True)
    assert result.returncode != 0
    assert "exactly one accelerator" in (result.stdout + result.stderr)


def test_full_stage_packages_partial_diagnostics_on_preflight_failure(tmp_path: Path):
    repo = tmp_path / "repo"
    for rel in (
        "loco_mjx/experiments/experiment.py",
        "RL-X/rl_x/runner/runner.py",
        "experiments/fsq_khaendler/crosseval_motion.py",
    ):
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n")
    signs = repo / "loco_mjx/loco_mjx/environments/locomotion/urma2/mjx/clip_reference.py"
    signs.parent.mkdir(parents=True, exist_ok=True)
    signs.write_text("T1_CLIP_SIGNS = {}\n")
    shutil.copytree(ROOT / "scripts/h1g1t1", repo / "scripts/h1g1t1")

    run_id = "pytest_failure_package"
    env = os.environ.copy()
    env.update({"REPO": str(repo), "PY": sys.executable, "RUN_ID": run_id,
                "CLIPS": str(tmp_path / "missing_clips"),
                "CUDA_VISIBLE_DEVICES": "0"})
    result = subprocess.run(["bash", str(LAUNCHER), "full"], env=env,
                            cwd=ROOT, text=True, capture_output=True, timeout=60)
    assert result.returncode != 0
    package = repo / f"experiments/h1g1t1_debug/{run_id}_diagnostics.zip"
    assert package.is_file(), result.stdout + result.stderr
    with zipfile.ZipFile(package) as zf:
        names = set(zf.namelist())
    assert any(name.endswith("failure_exit_code.txt") for name in names)
    assert any(name.endswith("preflight.log") for name in names)
    assert any(name.endswith("recipe.json") for name in names)


def test_collector_excludes_models_by_default_and_includes_latest(tmp_path: Path):
    result = tmp_path / "result"
    models = result / "runner/models"
    models.mkdir(parents=True)
    (result / "train.log").write_text("steps/nr_env_steps 100\n")
    (result / "summary.json").write_text(json.dumps({"ok": True}))
    with zipfile.ZipFile(models / "latest.model", "w") as zf:
        zf.writestr("config_algorithm.json", "{}")
    with zipfile.ZipFile(models / "model_100.model", "w") as zf:
        zf.writestr("config_algorithm.json", "{}")
    compact = tmp_path / "compact.zip"
    create_archive(result, compact, include_model=False)
    with zipfile.ZipFile(compact) as zf:
        names = set(zf.namelist())
    assert not any(name.endswith(".model") for name in names)
    assert any(name.endswith("config_algorithm.json") for name in names)
    assert any(name.endswith("checkpoint_inventory.json") for name in names)
    full = tmp_path / "full.zip"
    create_archive(result, full, include_model=True)
    with zipfile.ZipFile(full) as zf:
        names = set(zf.namelist())
    assert any(name.endswith("latest.model") for name in names)
    assert not any(name.endswith("model_100.model") for name in names)


def test_clip_preflight_reports_suspicious_full_clip_segments(tmp_path: Path):
    for family in ("UnitreeH1", "UnitreeG1", "BoosterT1"):
        write_clip(tmp_path / family / "dance2_subject4.npz")
    path = tmp_path / "BoosterT1/dance2_subject4.npz"
    with np.load(path, allow_pickle=True) as data:
        payload = {key: data[key] for key in data.files}
    payload["qpos"] = payload["qpos"].copy()
    payload["qpos"][:, 2] = .05
    payload["qvel"] = payload["qvel"].copy()
    payload["qvel"][0, 0] = 100.0
    np.savez_compressed(path, **payload)

    report = inspect_clip_tree(tmp_path, "dance2_subject4.npz")
    assert report["robots"]["booster_t1"]["root_z_min_m"] == pytest.approx(.05)
    assert report["robots"]["booster_t1"]["qvel_abs_max"] == pytest.approx(100.0)
    assert any("booster_t1" in warning and "root z" in warning for warning in report["warnings"])
    assert any("booster_t1" in warning and "qvel" in warning for warning in report["warnings"])
