from __future__ import annotations

import argparse
import shutil
from copy import deepcopy
from pathlib import Path

import jax.numpy as jnp
import mujoco
import numpy as np
from PIL import Image

from loco_mujoco.environments import LocoEnv, UnitreeH1
from loco_mujoco.smpl.retargeting import (
    extend_motion,
    load_robot_conf_file,
    retarget_traj_from_robot_to_robot,
)
from loco_mujoco.task_factories import DefaultDatasetConf, ImitationFactory
from loco_mujoco.trajectory import Trajectory, TrajectoryData


WORKSPACE = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retarget a short Unitree H1 squat clip onto a randomized H1 morphology."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-tag", default="proof")
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--scale-range", type=float, default=0.035)
    parser.add_argument("--shape-iterations", type=int, default=12)
    parser.add_argument("--pose-iterations", type=int, default=6)
    parser.add_argument("--motion-iterations", type=int, default=5)
    parser.add_argument("--init-motion-iterations", type=int, default=30)
    parser.add_argument("--no-floor-correction", action="store_true")
    return parser.parse_args()


def create_randomized_h1_xml(seed: int, scale_range: float) -> tuple[Path, np.ndarray]:
    source_xml = Path(UnitreeH1.get_default_xml_file_path())
    output_dir = WORKSPACE / "generated_variants" / f"randomized_h1_seed_{seed}"
    output_xml = output_dir / "h1.xml"
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_xml.parent / "assets", output_dir / "assets", dirs_exist_ok=True)

    rng = np.random.default_rng(seed)
    scale = rng.uniform(1.0 - scale_range, 1.0 + scale_range, size=3)
    average_scale = float(np.mean(scale))
    spec = mujoco.MjSpec.from_file(str(source_xml))

    # Keep inertias isotropic when exporting XML so every generated model remains valid.
    for body in list(spec.bodies)[1:]:
        body.mass *= average_scale**3
        body.inertia *= average_scale**5
        body.ipos *= scale
        body.pos *= scale

    for geom in list(spec.geoms)[1:]:
        geom.size *= average_scale
        geom.pos *= scale

    for site in spec.sites:
        site.pos *= scale

    for joint in list(spec.joints)[1:]:
        joint.pos *= scale

    output_xml.write_text(spec.to_xml(), encoding="utf-8")
    mujoco.MjModel.from_xml_path(str(output_xml))
    return output_xml, scale


def register_variant_env(env_name: str, xml_path: Path) -> None:
    def get_default_xml_file_path(cls) -> str:
        return str(xml_path)

    variant_cls = type(
        env_name,
        (UnitreeH1,),
        {"get_default_xml_file_path": classmethod(get_default_xml_file_path)},
    )
    LocoEnv.registered_envs[env_name] = variant_cls


def configure_for_cpu_proof(conf, args: argparse.Namespace):
    conf = deepcopy(conf)
    conf.optimization_params.torch_device = "cpu"
    conf.optimization_params.shape_iterations = args.shape_iterations
    conf.optimization_params.pose_iterations = args.pose_iterations
    conf.optimization_params.motion_iterations = args.motion_iterations
    conf.optimization_params.init_motion_iterations = args.init_motion_iterations
    return conf


def crop_first_trajectory(traj: Trajectory, start_frame: int, frames: int) -> Trajectory:
    available = int(traj.data.len_trajectory(0))
    if start_frame < 0 or frames < 3 or start_frame + frames > available:
        raise ValueError(
            f"Requested frames [{start_frame}, {start_frame + frames}) from a "
            f"trajectory with {available} samples. At least 3 frames are required."
        )
    data = TrajectoryData.dynamic_slice_in_dim(traj.data, 0, start_frame, frames, backend=jnp)
    return Trajectory(info=traj.info, data=data)


def render_trajectory_frame(xml_path: Path, traj: Trajectory, output_path: Path) -> None:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    frame = int(traj.data.n_samples) // 2
    data.qpos[:] = np.asarray(traj.data.qpos[frame])
    data.qvel[:] = np.asarray(traj.data.qvel[frame])
    mujoco.mj_forward(model, data)

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [0.0, 0.0, 0.9]
    camera.distance = 3.4
    camera.azimuth = 135
    camera.elevation = -12

    with mujoco.Renderer(model, height=480, width=640) as renderer:
        renderer.update_scene(data, camera=camera)
        pixels = renderer.render()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(output_path)


def lift_trajectory_above_floor(
    xml_path: Path, traj: Trajectory, clearance: float = 0.002
) -> tuple[Trajectory, float]:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    foot_ids = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_foot"),
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_foot"),
    }
    qpos = np.asarray(traj.data.qpos).copy()
    qvel = np.asarray(traj.data.qvel).copy()
    lifts = np.zeros(len(qpos))

    for frame in range(len(qpos)):
        data.qpos[:] = qpos[frame]
        data.qvel[:] = qvel[frame]
        mujoco.mj_forward(model, data)
        distances = [
            float(data.contact[i].dist)
            for i in range(data.ncon)
            if floor_id in (int(data.contact[i].geom1), int(data.contact[i].geom2))
            and (
                int(data.contact[i].geom1) in foot_ids
                or int(data.contact[i].geom2) in foot_ids
            )
        ]
        deepest_contact = min(distances, default=0.0)
        lifts[frame] = max(0.0, clearance - deepest_contact) if deepest_contact < 0.0 else 0.0

    qpos[:, 2] += lifts
    qvel[:, 2] = np.gradient(qpos[:, 2], 1.0 / float(traj.info.frequency))
    return traj.__class__(traj.info, traj.data.replace(qpos=jnp.asarray(qpos), qvel=jnp.asarray(qvel))), float(lifts.max())


def main() -> None:
    args = parse_args()
    source_env_name = f"ProofUnitreeH1Source_{args.cache_tag}"
    target_env_name = f"RandomizedUnitreeH1Seed{args.seed}_{args.cache_tag}"
    output_xml, scale = create_randomized_h1_xml(args.seed, args.scale_range)

    LocoEnv.registered_envs[source_env_name] = UnitreeH1
    register_variant_env(target_env_name, output_xml)

    source_conf = configure_for_cpu_proof(load_robot_conf_file("UnitreeH1"), args)
    target_conf = configure_for_cpu_proof(load_robot_conf_file("UnitreeH1"), args)

    source_env = ImitationFactory.make(
        "UnitreeH1", default_dataset_conf=DefaultDatasetConf(["squat"])
    )
    source_traj = crop_first_trajectory(source_env.th.traj, args.start_frame, args.frames)

    proof_dir = WORKSPACE / "external_data" / "retarget_proof"
    clip_id = f"{args.cache_tag}_start_{args.start_frame}_{args.frames}_frames"
    fitted_motion_path = proof_dir / f"h1_squat_{clip_id}_smpl.npz"
    target_traj_path = proof_dir / f"randomized_h1_seed_{args.seed}_squat_{clip_id}.npz"
    render_path = WORKSPACE / "images" / f"randomized_h1_seed_{args.seed}_squat_{clip_id}.png"

    target_traj = retarget_traj_from_robot_to_robot(
        source_env_name,
        source_traj,
        target_env_name,
        robot_conf_source=source_conf,
        robot_conf_target=target_conf,
        path_to_fitted_motion_source=str(fitted_motion_path),
    )
    max_floor_lift = 0.0
    if not args.no_floor_correction:
        target_traj, max_floor_lift = lift_trajectory_above_floor(output_xml, target_traj)
    target_traj = extend_motion(target_env_name, target_conf.env_params, target_traj)
    target_traj.save(str(target_traj_path))
    render_trajectory_frame(output_xml, target_traj, render_path)

    print("Retargeting proof complete")
    print(f"  morphology_scale_xyz: {np.array2string(scale, precision=6)}")
    print(f"  randomized_xml: {output_xml}")
    print(f"  fitted_source_motion: {fitted_motion_path}")
    print(f"  retargeted_trajectory: {target_traj_path}")
    print(f"  rendered_frame: {render_path}")
    print(f"  target_samples: {int(target_traj.data.n_samples)}")
    print(f"  target_qpos_shape: {tuple(target_traj.data.qpos.shape)}")
    print(f"  target_qvel_shape: {tuple(target_traj.data.qvel.shape)}")
    print(f"  trajectory_complete: {target_traj.data.is_complete}")
    print(f"  max_floor_lift_meters: {max_floor_lift:.6f}")


if __name__ == "__main__":
    main()
