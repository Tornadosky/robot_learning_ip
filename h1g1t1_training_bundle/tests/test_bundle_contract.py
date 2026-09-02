from __future__ import annotations

import ast
import importlib.util
import re
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class BundleContractTests(unittest.TestCase):
    def test_required_entrypoints_exist(self):
        required = [
            ROOT / "install_and_run.sh",
            ROOT / "README.md",
            ROOT / "manifest.json",
            ROOT / "overlay/scripts/h1g1t1/run_experiment.sh",
            ROOT / "overlay/scripts/h1g1t1/preflight.py",
            ROOT / "overlay/scripts/h1g1t1/parse_training_log.py",
            ROOT / "overlay/scripts/h1g1t1/plot_training.py",
            ROOT / "overlay/scripts/h1g1t1/collect_diagnostics.py",
        ]
        self.assertEqual([], [str(p) for p in required if not p.is_file()])

    def test_profiles_have_valid_batch_geometry(self):
        mod = load_module(
            "h1g1t1_profiles",
            ROOT / "overlay/scripts/h1g1t1/profiles.py",
        )
        for name in ("smoke", "probe", "full"):
            p = mod.get_profile(name)
            self.assertEqual(p.total_envs % 3, 0)
            self.assertEqual((p.total_envs * p.rollout_steps) % p.minibatch_size, 0)
            self.assertEqual(p.minibatch_size % 3, 0)
            self.assertEqual(p.total_timesteps % (p.total_envs * p.rollout_steps), 0)
            self.assertEqual(p.total_timesteps % p.save_frequency, 0)

    def test_log_parser_extracts_non_return_metrics(self):
        mod = load_module(
            "h1g1t1_parser",
            ROOT / "overlay/scripts/h1g1t1/parse_training_log.py",
        )
        text = "\n".join(
            [
                "[08-27 14:45:27] [urma2.py:866] INFO - │ env_info/joint_tracking_error/h1│ 0.25 │",
                "[08-27 14:45:27] [urma2.py:866] INFO - │ env_info/root_heading_error/h1│ 1.57 │",
                "[08-27 14:45:27] [urma2.py:866] INFO - │ policy_ratio/approx_kl        │ 0.04 │",
                "[08-27 14:45:27] [urma2.py:866] INFO - │ steps/nr_env_steps            │ 36864 │",
            ]
        )
        rows = mod.parse_lines(text.splitlines())
        self.assertEqual(1, len(rows))
        self.assertAlmostEqual(0.25, rows[0]["env_info/joint_tracking_error/h1"])
        self.assertAlmostEqual(1.57, rows[0]["env_info/root_heading_error/h1"])
        self.assertAlmostEqual(0.04, rows[0]["policy_ratio/approx_kl"])
        self.assertEqual(36864.0, rows[0]["steps/nr_env_steps"])

    def test_manifest_paths_and_hashes_are_well_formed(self):
        manifest = json.loads((ROOT / "manifest.json").read_text())
        self.assertEqual(1, manifest["format_version"])
        self.assertGreaterEqual(len(manifest["overlay_files"]), 8)
        for item in manifest["overlay_files"]:
            self.assertFalse(pathlib.PurePosixPath(item["path"]).is_absolute())
            self.assertRegex(item["sha256"], r"^[0-9a-f]{64}$")



    def test_launcher_flags_exist_in_config_schema(self):
        launch = (ROOT / "overlay/scripts/h1g1t1/run_experiment.sh").read_text()
        flags = re.findall(r"--(environment|algorithm|runner)\.([A-Za-z0-9_.]+)=", launch)

        env_path = ROOT / "overlay/loco_mjx/loco_mjx/environments/locomotion/urma2/mjx/default_config.py"
        module = ast.parse(env_path.read_text())
        config_node = next(
            node.value
            for node in ast.walk(module)
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Dict)
            and any(isinstance(target, ast.Name) and target.id == "config" for target in node.targets)
        )
        env_paths = set()
        def walk_dict(node, prefix=""):
            if not isinstance(node, ast.Dict):
                return
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    path = f"{prefix}.{key.value}" if prefix else key.value
                    env_paths.add(path)
                    walk_dict(value, path)
        walk_dict(config_node)

        source_root = ROOT.parent / "src/h1g1t1_pipeline"
        # During bundle-only extraction the original tree is unavailable, so use
        # the known source schemas copied into the clean install test when present.
        algorithm_source = source_root / "loco_mjx/loco_mjx/algorithms/urma2/mjx/default_config.py"
        runner_source = source_root / "RL-X/rl_x/runner/default_config.py"
        if not algorithm_source.is_file():
            algorithm_source = pathlib.Path("/mnt/data/rebuild_h1g1t1/src/h1g1t1_pipeline/loco_mjx/loco_mjx/algorithms/urma2/mjx/default_config.py")
            runner_source = pathlib.Path("/mnt/data/rebuild_h1g1t1/src/h1g1t1_pipeline/RL-X/rl_x/runner/default_config.py")
        if algorithm_source.is_file() and runner_source.is_file():
            algorithm_keys = set(re.findall(r"config\.([A-Za-z0-9_]+)\s*=", algorithm_source.read_text()))
            runner_keys = set(re.findall(r"config\.([A-Za-z0-9_]+)\s*=", runner_source.read_text()))
            self.assertEqual([], [path for group, path in flags if group == "algorithm" and path not in algorithm_keys])
            self.assertEqual([], [path for group, path in flags if group == "runner" and path not in runner_keys])
        self.assertEqual([], [path for group, path in flags if group == "environment" and path not in env_paths])

    def test_heading_kernel_patch_is_present(self):
        default_cfg = ROOT / "overlay/loco_mjx/loco_mjx/environments/locomotion/urma2/mjx/default_config.py"
        tracking = ROOT / "overlay/loco_mjx/loco_mjx/environments/locomotion/urma2/mjx/reward_functions/tracking.py"
        self.assertTrue(default_cfg.is_file())
        self.assertTrue(tracking.is_file())
        self.assertIn('"root_heading_tracking_kernel": "exponential"', default_cfg.read_text())
        code = tracking.read_text()
        self.assertIn('root_heading_tracking_kernel', code)
        self.assertIn('0.5 * (1.0 + jnp.cos(heading_error))', code)

    def test_shell_scripts_parse(self):
        scripts = [ROOT / "install_and_run.sh"] + list(
            (ROOT / "overlay/scripts/h1g1t1").glob("*.sh")
        )
        for script in scripts:
            subprocess.run(["bash", "-n", str(script)], check=True)


if __name__ == "__main__":
    unittest.main()
