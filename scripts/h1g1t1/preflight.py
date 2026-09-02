#!/usr/bin/env python3
"""Validate H1+G1+T1 assets and arithmetic before GPU compilation."""
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    from scripts.h1g1t1.diagnostics import validate_recipe
except ModuleNotFoundError:  # direct execution from scripts/h1g1t1
    from diagnostics import validate_recipe

ROBOT_SUBDIRS = {
    "unitree_h1": "UnitreeH1",
    "unitree_g1": "UnitreeG1",
    "booster_t1": "BoosterT1",
}


def inspect_clip_tree(clip_dir: str | Path, clip_file: str) -> dict[str, Any]:
    """Validate all three retargets and report full-clip integrity signals.

    Hard structural defects abort before JAX compilation.  Large joint-speed
    spikes or implausibly low root heights are warnings rather than hard gates:
    they can be real motion, but they must be visible in the returned artifact.
    """
    clip_dir = Path(clip_dir)
    robots: dict[str, Any] = {}
    warnings: list[str] = []
    for robot, subdir in ROBOT_SUBDIRS.items():
        path = clip_dir / subdir / clip_file
        if not path.is_file():
            raise FileNotFoundError(f"Missing {subdir} clip: {path}")
        with np.load(path, allow_pickle=True) as data:
            missing = [key for key in ("qpos", "joint_names", "frequency") if key not in data]
            if missing:
                raise ValueError(f"{path} is missing keys: {missing}")
            qpos = np.asarray(data["qpos"], dtype=np.float64)
            qvel = np.asarray(data["qvel"], dtype=np.float64) if "qvel" in data else None
            names = [str(name) for name in np.asarray(data["joint_names"]).tolist()]
            frequency = float(np.asarray(data["frequency"]))
            if qpos.ndim != 2 or qpos.shape[0] < 2 or qpos.shape[1] < 8:
                raise ValueError(f"Invalid qpos shape in {path}: {qpos.shape}")
            if not names or not all(names):
                raise ValueError(f"Invalid joint_names in {path}")
            if not np.isfinite(qpos).all():
                raise ValueError(f"Non-finite qpos in {path}")
            if not np.isfinite(frequency) or frequency <= 0:
                raise ValueError(f"Invalid frequency in {path}: {frequency}")
            if qvel is not None:
                if qvel.ndim != 2 or qvel.shape[0] != qpos.shape[0]:
                    raise ValueError(f"Invalid qvel shape in {path}: {qvel.shape}")
                if not np.isfinite(qvel).all():
                    raise ValueError(f"Non-finite qvel in {path}")

            quat_norm = np.linalg.norm(qpos[:, 3:7], axis=1)
            if np.any(quat_norm < 0.5) or np.any(quat_norm > 1.5):
                raise ValueError(f"Grossly invalid root quaternion norm in {path}: "
                                 f"[{quat_norm.min():.6g}, {quat_norm.max():.6g}]")
            root_z_min = float(np.min(qpos[:, 2]))
            root_z_max = float(np.max(qpos[:, 2]))
            qvel_abs_max = float(np.max(np.abs(qvel))) if qvel is not None and qvel.size else None
            qvel_abs_p99 = float(np.percentile(np.abs(qvel), 99)) if qvel is not None and qvel.size else None
            quat_max_abs_error = float(np.max(np.abs(quat_norm - 1.0)))

            if root_z_min < 0.20:
                warnings.append(
                    f"{robot}: root z falls to {root_z_min:.4f} m; inspect the retarget/full clip")
            if qvel_abs_max is not None and qvel_abs_max > 50.0:
                warnings.append(
                    f"{robot}: |qvel| reaches {qvel_abs_max:.2f} rad/s; inspect spike frames")
            if quat_max_abs_error > 1e-3:
                warnings.append(
                    f"{robot}: root quaternion max norm error is {quat_max_abs_error:.3g}")

            robots[robot] = {
                "path": str(path), "frames": int(qpos.shape[0]),
                "duration_s": float((qpos.shape[0] - 1) / frequency),
                "qpos_dim": int(qpos.shape[1]), "joint_names": len(names),
                "frequency_hz": frequency,
                "root_z_min_m": root_z_min, "root_z_max_m": root_z_max,
                "root_quaternion_norm_min": float(quat_norm.min()),
                "root_quaternion_norm_max": float(quat_norm.max()),
                "qvel_present": qvel is not None,
                "qvel_abs_p99": qvel_abs_p99,
                "qvel_abs_max": qvel_abs_max,
            }
    return {"ok": True, "clip_dir": str(clip_dir), "clip_file": clip_file,
            "robots": robots, "warnings": warnings}


def inspect_t1_sign_fix(repo: str | Path) -> dict[str, Any]:
    path = Path(repo) / "loco_mjx/loco_mjx/environments/locomotion/urma2/mjx/clip_reference.py"
    if not path.is_file():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    if "T1_CLIP_SIGNS" not in text:
        raise RuntimeError(f"T1_CLIP_SIGNS is absent from {path}; T1 crosseval is convention-invalid")
    return {"ok": True, "path": str(path), "symbol": "T1_CLIP_SIGNS"}


def inspect_repo(repo: str | Path) -> dict[str, Any]:
    repo = Path(repo).resolve()
    required = [
        "loco_mjx/experiments/experiment.py",
        "experiments/fsq_khaendler/crosseval_motion.py",
        "RL-X/rl_x/runner/runner.py",
    ]
    missing = [str(repo / item) for item in required if not (repo / item).is_file()]
    if missing:
        raise FileNotFoundError("Repository is missing required files: " + ", ".join(missing))
    git: dict[str, Any] = {"available": False}
    if (repo / ".git").exists():
        try:
            git = {
                "available": True,
                "commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
                "dirty": bool(subprocess.check_output(
                    ["git", "status", "--porcelain"], cwd=repo, text=True).strip()),
            }
        except (OSError, subprocess.CalledProcessError) as exc:
            git = {"available": True, "error": str(exc)}
    return {"ok": True, "path": str(repo), "git": git}


def inspect_python_dependencies(modules: Sequence[str]) -> dict[str, Any]:
    found = {module: importlib.util.find_spec(module) is not None for module in modules}
    missing = [module for module, present in found.items() if not present]
    return {"ok": not missing, "modules": found, "missing": missing,
            "python": sys.executable, "version": platform.python_version()}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--clip-dir", required=True)
    parser.add_argument("--clip-file", default="dance2_subject4.npz")
    parser.add_argument("--nr-envs", type=int, default=576)
    parser.add_argument("--nr-steps", type=int, default=64)
    parser.add_argument("--minibatch-size", type=int, default=6144)
    parser.add_argument("--save-every", type=int, default=11_796_480)
    parser.add_argument("--total-timesteps", type=int, default=117_964_800)
    parser.add_argument("--out", required=True)
    parser.add_argument("--skip-dependencies", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report: dict[str, Any] = {
        "recipe": validate_recipe(args.nr_envs, args.nr_steps, args.minibatch_size,
                                  args.save_every, args.total_timesteps, 3),
        "repository": inspect_repo(args.repo),
        "clips": inspect_clip_tree(args.clip_dir, args.clip_file),
        "t1_sign_fix": inspect_t1_sign_fix(args.repo),
    }
    if not args.skip_dependencies:
        report["dependencies"] = inspect_python_dependencies(
            ("jax", "flax", "optax", "mujoco", "ml_collections", "tensorboard", "torch", "pandas", "matplotlib"))
        if not report["dependencies"]["ok"]:
            raise RuntimeError("Missing Python modules: " + ", ".join(report["dependencies"]["missing"]))
    report["ok"] = True
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
