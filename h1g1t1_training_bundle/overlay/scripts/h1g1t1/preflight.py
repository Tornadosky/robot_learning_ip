#!/usr/bin/env python3
"""Fail-fast checks for the H1+G1+T1 dance experiment."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
from pathlib import Path

ROBOTS = {
    "unitree_h1": "UnitreeH1",
    "unitree_g1": "UnitreeG1",
    "booster_t1": "BoosterT1",
}
REQUIRED_NPZ_KEYS = {"qpos", "joint_names"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def add(checks: list[dict[str, object]], name: str, ok: bool, detail: object) -> None:
    checks.append({"name": name, "ok": bool(ok), "detail": detail})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--python", dest="python_path", required=True)
    parser.add_argument("--clip-dir", type=Path, required=True)
    parser.add_argument("--clip", default="dance2_subject4.npz")
    parser.add_argument("--profile-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()

    checks: list[dict[str, object]] = []
    repo = args.repo.resolve()
    add(checks, "repository_exists", repo.is_dir(), str(repo))
    add(checks, "python_executable_exists", Path(os.path.expanduser(args.python_path)).is_file(), args.python_path)

    required_repo = [
        "loco_mjx/experiments/experiment.py",
        "loco_mjx/loco_mjx/environments/locomotion/urma2/mjx/create_env.py",
        "loco_mjx/loco_mjx/environments/locomotion/urma2/mjx/clip_reference.py",
        "loco_mjx/loco_mjx/environments/locomotion/urma2/mjx/reward_functions/tracking.py",
        "RL-X/rl_x/algorithms/algorithm_manager.py",
    ]
    for rel in required_repo:
        add(checks, f"repo_file:{rel}", (repo / rel).is_file(), rel)

    profile: dict[str, object] = {}
    try:
        profile = json.loads(args.profile_json.read_text(encoding="utf-8"))
        total_envs = int(profile["total_envs"])
        rollout = int(profile["rollout_steps"])
        minibatch = int(profile["minibatch_size"])
        geometry_ok = (
            total_envs % 3 == 0
            and (total_envs * rollout) % minibatch == 0
            and minibatch % 3 == 0
        )
        add(checks, "batch_geometry", geometry_ok, profile)
    except Exception as exc:  # noqa: BLE001
        add(checks, "batch_geometry", False, f"{type(exc).__name__}: {exc}")

    try:
        import numpy as np
        for robot, subdir in ROBOTS.items():
            clip_path = args.clip_dir / subdir / args.clip
            if not clip_path.is_file():
                add(checks, f"clip:{robot}", False, str(clip_path))
                continue
            with np.load(clip_path, allow_pickle=True) as data:
                keys = set(data.files)
                qpos_shape = tuple(data["qpos"].shape) if "qpos" in keys else None
                names = [str(x) for x in data["joint_names"]] if "joint_names" in keys else []
                valid = REQUIRED_NPZ_KEYS.issubset(keys) and qpos_shape is not None and len(qpos_shape) == 2 and qpos_shape[0] > 10 and len(names) > 2
                add(checks, f"clip:{robot}", valid, {
                    "path": str(clip_path), "sha256": sha256(clip_path),
                    "keys": sorted(keys), "qpos_shape": qpos_shape,
                    "joint_name_count": len(names),
                })
    except Exception as exc:  # noqa: BLE001
        add(checks, "clip_loader", False, f"{type(exc).__name__}: {exc}")

    capability_files = {
        "current_total_env_semantics": (repo / "loco_mjx/loco_mjx/environments/locomotion/urma2/mjx/create_env.py", "nr_envs_per_train_robot = nr_envs_per_device // len(train_robots)"),
        "t1_sign_map": (repo / "loco_mjx/loco_mjx/environments/locomotion/urma2/mjx/clip_reference.py", "T1_CLIP_SIGNS"),
        "heading_observation": (repo / "loco_mjx/loco_mjx/environments/locomotion/urma2/mjx/environment.py", "tracking_clip_observe_root_heading"),
        "morphology_schedule": (repo / "loco_mjx/loco_mjx/environments/locomotion/urma2/mjx/environment.py", 'morphology_coeff_mode == "schedule"'),
        "exact_inertia": (repo / "loco_mjx/loco_mjx/environments/locomotion/urma2/mjx/domain_randomization/seen_robot_functions/default.py", "exact_inertia_rescale"),
        "cosine_heading_kernel": (repo / "loco_mjx/loco_mjx/environments/locomotion/urma2/mjx/reward_functions/tracking.py", "root_heading_tracking_kernel"),
    }
    for name, (path, needle) in capability_files.items():
        try:
            text = path.read_text(encoding="utf-8")
            add(checks, name, needle in text, {"path": str(path), "needle": needle})
        except Exception as exc:  # noqa: BLE001
            add(checks, name, False, f"{type(exc).__name__}: {exc}")

    import_status = {}
    for module in ("jax", "mujoco", "ml_collections", "rl_x", "loco_mjx"):
        import_status[module] = importlib.util.find_spec(module) is not None
    add(checks, "python_imports", all(import_status.values()), import_status)

    devices: list[str] = []
    accelerator_ok = False
    try:
        import jax
        devices = [f"{d.platform}:{d.device_kind}:{d.id}" for d in jax.devices()]
        accelerator_ok = any(d.platform in ("gpu", "cuda", "rocm") for d in jax.devices())
        add(checks, "jax_devices", bool(devices), devices)
        add(checks, "accelerator", accelerator_ok or args.allow_cpu, {
            "accelerator_found": accelerator_ok, "allow_cpu": args.allow_cpu, "devices": devices,
        })
    except Exception as exc:  # noqa: BLE001
        add(checks, "jax_devices", False, f"{type(exc).__name__}: {exc}")
        add(checks, "accelerator", args.allow_cpu, {"error": str(exc), "allow_cpu": args.allow_cpu})

    result = {
        "status": "PASS" if all(c["ok"] for c in checks) else "FAIL",
        "repo": str(repo),
        "clip_dir": str(args.clip_dir.resolve()),
        "clip": args.clip,
        "profile": profile,
        "python": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "checks": checks,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
