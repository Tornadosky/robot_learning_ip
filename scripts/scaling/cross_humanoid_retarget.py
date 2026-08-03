"""Build one motion reference cache across heterogeneous humanoid robots.

This is the repository-native equivalent of a GMR-style bridge.  ``direct``
uses LocoMuJoCo's already converted SMPL/LAFAN cache when it exists.  The more
general ``robot2robot`` mode fits the source motion to SMPL once, reuses that
canonical cache, and solves each target robot independently.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

WORKSPACE = Path(__file__).resolve().parents[2]
SCRIPTS = WORKSPACE / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from loco_mujoco.environments.humanoids import (  # noqa: E402
    Apollo,
    Atlas,
    BoosterT1,
    FourierGR1T2,
    MjxApollo,
    MjxAtlas,
    MjxBoosterT1,
    MjxFourierGR1T2,
    MjxTalos,
    MjxToddlerBot,
    MjxUnitreeG1,
    MjxUnitreeH1,
    MjxUnitreeH1v2,
    Talos,
    ToddlerBot,
    UnitreeG1,
    UnitreeH1,
    UnitreeH1v2,
)
from loco_mujoco.smpl.retargeting import (  # noqa: E402
    extend_motion,
    load_robot_conf_file,
    retarget_traj_from_robot_to_robot,
)
from loco_mujoco.task_factories import ImitationFactory, LAFAN1DatasetConf  # noqa: E402
from loco_mujoco.trajectory import Trajectory  # noqa: E402

from morphology_deepmimic import (  # noqa: E402
    clamp_trajectory_to_floor,
    crop_trajectory,
    fitted_source_path,
    min_floor_distance,
    resolve_window,
)


@dataclass(frozen=True)
class HumanoidSpec:
    key: str
    cpu_env_name: str
    mjx_env_name: str
    conf_name: str
    cpu_cls: type
    mjx_cls: type

    @property
    def xml_path(self) -> Path:
        return Path(self.cpu_cls.get_default_xml_file_path())


HUMANOIDS = {
    spec.key: spec
    for spec in (
        HumanoidSpec(
            "h1", "UnitreeH1", "MjxUnitreeH1", "UnitreeH1", UnitreeH1, MjxUnitreeH1
        ),
        HumanoidSpec(
            "g1", "UnitreeG1", "MjxUnitreeG1", "UnitreeG1", UnitreeG1, MjxUnitreeG1
        ),
        HumanoidSpec(
            "h1v2",
            "UnitreeH1v2",
            "MjxUnitreeH1v2",
            "UnitreeH1v2",
            UnitreeH1v2,
            MjxUnitreeH1v2,
        ),
        HumanoidSpec("atlas", "Atlas", "MjxAtlas", "Atlas", Atlas, MjxAtlas),
        HumanoidSpec("talos", "Talos", "MjxTalos", "Talos", Talos, MjxTalos),
        HumanoidSpec("apollo", "Apollo", "MjxApollo", "Apollo", Apollo, MjxApollo),
        HumanoidSpec(
            "booster_t1",
            "BoosterT1",
            "MjxBoosterT1",
            "BoosterT1",
            BoosterT1,
            MjxBoosterT1,
        ),
        HumanoidSpec(
            "toddlerbot",
            "ToddlerBot",
            "MjxToddlerBot",
            "ToddlerBot",
            ToddlerBot,
            MjxToddlerBot,
        ),
        HumanoidSpec(
            "fourier_gr1t2",
            "FourierGR1T2",
            "MjxFourierGR1T2",
            "FourierGR1T2",
            FourierGR1T2,
            MjxFourierGR1T2,
        ),
    )
}


def _configured_retarget_conf(spec: HumanoidSpec, args):
    conf = deepcopy(load_robot_conf_file(spec.conf_name))
    conf.optimization_params.torch_device = args.torch_device
    conf.optimization_params.shape_iterations = args.shape_iterations
    conf.optimization_params.pose_iterations = args.pose_iterations
    conf.optimization_params.motion_iterations = args.motion_iterations
    conf.optimization_params.init_motion_iterations = args.init_motion_iterations
    return conf


def _reference_summary(
    spec, trajectory, path, elapsed, cache_hit, floor_before=None, floor_lift=None
):
    return {
        "target": spec.key,
        "cpu_env_name": spec.cpu_env_name,
        "mjx_env_name": spec.mjx_env_name,
        "xml_path": str(spec.xml_path),
        "reference_path": str(path),
        "samples": int(trajectory.data.n_samples),
        "frequency_hz": float(trajectory.info.frequency),
        "duration_seconds": float(
            (int(trajectory.data.n_samples) - 1) / trajectory.info.frequency
        ),
        "qpos_dim": int(trajectory.data.qpos.shape[-1]),
        "qvel_dim": int(trajectory.data.qvel.shape[-1]),
        "complete": bool(trajectory.data.is_complete),
        "seconds": elapsed,
        "cache_hit": cache_hit,
        "floor_distance_before_meters": floor_before,
        "max_floor_lift_meters": floor_lift,
    }


def _direct_reference(spec, clip, start, frames, output_path, force):
    t0 = time.perf_counter()
    cache_hit = output_path.is_file() and not force
    if cache_hit:
        trajectory = Trajectory.load(str(output_path))
    else:
        env = ImitationFactory.make(
            spec.cpu_env_name,
            lafan1_dataset_conf=LAFAN1DatasetConf([clip]),
        )
        available = int(env.th.traj.data.n_samples)
        if start + frames > available:
            raise ValueError(
                f"{spec.key} has {available} frames for {clip}, but "
                f"[{start}, {start + frames}) was requested."
            )
        trajectory = crop_trajectory(env.th.traj, start, frames)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        trajectory.save(str(output_path))
    return _reference_summary(
        spec, trajectory, output_path, time.perf_counter() - t0, cache_hit
    )


def _robot2robot_reference(
    source_spec,
    source_trajectory,
    target_spec,
    fitted_path,
    output_path,
    args,
):
    t0 = time.perf_counter()
    cache_hit = output_path.is_file() and not args.force
    if cache_hit:
        target_trajectory = Trajectory.load(str(output_path))
        return _reference_summary(
            target_spec,
            target_trajectory,
            output_path,
            time.perf_counter() - t0,
            True,
        )

    source_conf = _configured_retarget_conf(source_spec, args)
    target_conf = _configured_retarget_conf(target_spec, args)
    target_trajectory = retarget_traj_from_robot_to_robot(
        source_spec.cpu_env_name,
        source_trajectory,
        target_spec.cpu_env_name,
        robot_conf_source=source_conf,
        robot_conf_target=target_conf,
        path_to_fitted_motion_source=str(fitted_path),
    )
    floor_before = min_floor_distance(target_spec.xml_path, target_trajectory)
    target_trajectory, floor_lift = clamp_trajectory_to_floor(
        target_spec.xml_path, target_trajectory
    )
    target_trajectory = extend_motion(
        target_spec.mjx_env_name, target_conf.env_params, target_trajectory
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    target_trajectory.save(str(output_path))
    return _reference_summary(
        target_spec,
        target_trajectory,
        output_path,
        time.perf_counter() - t0,
        False,
        floor_before,
        floor_lift,
    )


def main():
    args = parse_args()
    source_spec = HUMANOIDS[args.source]
    source_env = ImitationFactory.make(
        source_spec.cpu_env_name,
        lafan1_dataset_conf=LAFAN1DatasetConf([args.clip]),
    )
    source_full = source_env.th.traj
    start, frames = resolve_window(source_full, args.duration, args.start_frame)
    source_trajectory = crop_trajectory(source_full, start, frames)

    output_root = args.output_root / f"{args.source}_source" / args.clip
    fitted_path = fitted_source_path(args.source, args.clip, start, frames)
    run_started = time.perf_counter()
    targets = []
    for target_key in args.targets:
        target_spec = HUMANOIDS[target_key]
        output_path = (
            output_root / target_key / f"start{start}_{frames}f_{args.mode}.npz"
        )
        if args.mode == "direct":
            summary = _direct_reference(
                target_spec, args.clip, start, frames, output_path, args.force
            )
        else:
            summary = _robot2robot_reference(
                source_spec,
                source_trajectory,
                target_spec,
                fitted_path,
                output_path,
                args,
            )
        targets.append(summary)
        print(
            f"[cross] {target_key:>14s}: qpos={summary['qpos_dim']} "
            f"qvel={summary['qvel_dim']} samples={summary['samples']} "
            f"{summary['seconds']:.2f}s cache={summary['cache_hit']}",
            flush=True,
        )

    manifest = {
        "experiment": "cross_humanoid_reference_pipeline",
        "implementation": (
            "locomujoco_preconverted_smpl_cache"
            if args.mode == "direct"
            else "locomujoco_robot_to_smpl_to_robot"
        ),
        "mode": args.mode,
        "source": args.source,
        "clip": args.clip,
        "window_start_frame": int(start),
        "window_frames": int(frames),
        "frequency_hz": float(source_trajectory.info.frequency),
        "shared_fitted_source_path": str(fitted_path),
        "targets": targets,
        "total_seconds": time.perf_counter() - run_started,
    }
    manifest_path = output_root / f"manifest_start{start}_{frames}f_{args.mode}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"[cross] manifest -> {manifest_path}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=HUMANOIDS, default="h1")
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=HUMANOIDS,
        default=["h1", "g1", "atlas", "toddlerbot"],
    )
    parser.add_argument("--clip", default="dance2_subject4")
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--start-frame", type=int, default=19482)
    parser.add_argument("--mode", choices=["direct", "robot2robot"], default="direct")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--torch-device", default="cpu")
    parser.add_argument("--shape-iterations", type=int, default=1000)
    parser.add_argument("--pose-iterations", type=int, default=400)
    parser.add_argument("--motion-iterations", type=int, default=25)
    parser.add_argument("--init-motion-iterations", type=int, default=1000)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=WORKSPACE / "external_data" / "cross_humanoid",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
