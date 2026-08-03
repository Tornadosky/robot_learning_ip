"""Retarget a LAFAN1 dance clip onto one morphology variant -> cached reference NPZ.

This builds the *physically feasible reference motion* a DeepMimic policy will
track on a modified body. The stock joint trajectory is not usable directly: an
extreme variant (e.g. legs 1.55x) puts feet through the floor or in mid-air, so
we go stock-robot -> SMPL -> variant-robot via loco-mujoco's native retargeting,
then clamp to the floor and extend the motion (adds the qpos/qvel/site fields the
Mimic goal needs).

The heavy SMPL fit of the *source* motion is cached once per (robot, clip, window)
and reused across every variant of that clip. Retargeting (the SMPL pose fit) is
hardcoded to CUDA inside loco-mujoco, so this runs on a GPU box (WSL2).
"""

from __future__ import annotations

import argparse
import os
from copy import deepcopy
from pathlib import Path

# Retargeting runs the SMPL pose fit on torch/CUDA while loco-mujoco still pulls in
# jax. Stop jax from preallocating ~75% of the GPU so torch has room to share it.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from loco_mujoco.environments import LocoEnv
from loco_mujoco.smpl.retargeting import (
    extend_motion,
    load_robot_conf_file,
    retarget_traj_from_robot_to_robot,
)
from loco_mujoco.task_factories import ImitationFactory, LAFAN1DatasetConf
from loco_mujoco.trajectory import Trajectory

from morphology_deepmimic import (
    clamp_trajectory_to_floor,
    crop_trajectory,
    fitted_source_path,
    get_robot,
    min_floor_distance,
    prepare_variant,
    reference_path,
    resolve_window,
)


def configure_conf(conf, args: argparse.Namespace):
    conf = deepcopy(conf)
    # torch_device only affects fit_smpl_shape (CPU-only); the source pose fit is
    # hardcoded to cuda inside loco-mujoco regardless.
    conf.optimization_params.torch_device = args.torch_device
    conf.optimization_params.shape_iterations = args.shape_iterations
    conf.optimization_params.pose_iterations = args.pose_iterations
    conf.optimization_params.motion_iterations = args.motion_iterations
    conf.optimization_params.init_motion_iterations = args.init_motion_iterations
    return conf


def retarget_cell(
    robot_key: str,
    preset_name: str,
    clip: str,
    *,
    duration: float = 30.0,
    start_frame: int | None = None,
    cache_tag: str = "dance",
    force: bool = False,
    conf_args: argparse.Namespace | None = None,
) -> dict:
    """Retarget one (robot, preset, clip) cell and return its reference metadata."""
    robot = get_robot(robot_key)
    conf_args = conf_args or default_conf_args()

    # Load the stock-robot LAFAN1 clip and pick the shared highlight window.
    source_env = ImitationFactory.make(
        robot.cpu_env_name, lafan1_dataset_conf=LAFAN1DatasetConf([clip])
    )
    full_traj = source_env.th.traj
    frequency = float(full_traj.info.frequency)
    start, n_frames = resolve_window(full_traj, duration, start_frame)
    source_traj = crop_trajectory(full_traj, start, n_frames)

    ref_path = reference_path(robot.key, clip, preset_name, start, n_frames)
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    variant = prepare_variant(robot, preset_name, cache_tag)
    xml_path = variant["xml_path"]

    if ref_path.exists() and not force:
        target_traj = Trajectory.load(str(ref_path))
        print(f"[retarget] reuse cached reference: {ref_path}")
        return _cell_summary(robot, preset_name, clip, start, n_frames, frequency,
                             xml_path, ref_path, target_traj, None, None)

    # The source env (stock robot) needs to be registered under a stable name for
    # the retargeter; reuse the stock class directly.
    source_env_name = f"{robot.key.upper()}Src_{clip}_{cache_tag}"
    LocoEnv.registered_envs.setdefault(source_env_name, robot.base_cpu_cls)

    source_conf = configure_conf(load_robot_conf_file(robot.retarget_conf_name), conf_args)
    target_conf = configure_conf(load_robot_conf_file(robot.retarget_conf_name), conf_args)
    fitted_path = fitted_source_path(robot.key, clip, start, n_frames)
    fitted_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[retarget] {robot.key}/{clip}/{preset_name}: frames [{start}, {start + n_frames}) "
          f"@ {frequency:.0f}Hz -> {variant['cpu_env_name']}")
    target_traj = retarget_traj_from_robot_to_robot(
        source_env_name,
        source_traj,
        variant["cpu_env_name"],
        robot_conf_source=source_conf,
        robot_conf_target=target_conf,
        path_to_fitted_motion_source=str(fitted_path),
    )
    pre_min = min_floor_distance(xml_path, target_traj)
    target_traj, max_lift = clamp_trajectory_to_floor(xml_path, target_traj)
    target_traj = extend_motion(variant["mjx_env_name"], target_conf.env_params, target_traj)
    target_traj.save(str(ref_path))
    print(f"[retarget] saved {ref_path} (pre_floor={pre_min:.4f}m, lift={max_lift:.4f}m, "
          f"complete={target_traj.data.is_complete})")
    return _cell_summary(robot, preset_name, clip, start, n_frames, frequency,
                         xml_path, ref_path, target_traj, pre_min, max_lift)


def _cell_summary(robot, preset_name, clip, start, n_frames, frequency, xml_path,
                  ref_path, traj, pre_min, max_lift) -> dict:
    return {
        "robot": robot.key,
        "preset": preset_name,
        "clip": clip,
        "window_start_frame": int(start),
        "window_frames": int(n_frames),
        "frequency_hz": frequency,
        "xml_path": str(xml_path),
        "reference_path": str(ref_path),
        "reference_samples": int(traj.data.n_samples),
        "reference_complete": bool(traj.data.is_complete),
        "pre_correction_min_floor_distance_meters": pre_min,
        "max_floor_lift_meters": max_lift,
        "mjx_env_name": "Mjx" + robot.key.upper() + f"Var_{preset_name}",
    }


def default_conf_args() -> argparse.Namespace:
    return argparse.Namespace(
        torch_device="cpu",
        shape_iterations=1000,
        pose_iterations=400,
        motion_iterations=25,
        init_motion_iterations=1000,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", required=True, choices=["h1", "g1"])
    parser.add_argument("--preset", required=True)
    parser.add_argument("--clip", default="dance2_subject4")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--cache-tag", default="dance")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--torch-device", default="cpu")
    parser.add_argument("--shape-iterations", type=int, default=1000)
    parser.add_argument("--pose-iterations", type=int, default=400)
    parser.add_argument("--motion-iterations", type=int, default=25)
    parser.add_argument("--init-motion-iterations", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = retarget_cell(
        args.robot,
        args.preset,
        args.clip,
        duration=args.duration,
        start_frame=args.start_frame,
        cache_tag=args.cache_tag,
        force=args.force,
        conf_args=args,
    )
    print(summary)


if __name__ == "__main__":
    main()
