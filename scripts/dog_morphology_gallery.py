"""Create Unitree Go2 (robot dog) morphology variants and render a static gallery.

Quadrupeds have no mocap/dance datasets in loco-mujoco (and no SMPL retargeting
support), so this gallery shows the morphology variants in their home pose only.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import mujoco
import numpy as np

from h1_morphology_variants import wrap_text

WORKSPACE = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Go2MorphologyPreset:
    name: str
    label: str
    leg_length_scale: float = 1.0
    body_length_scale: float = 1.0
    paw_scale: float = 1.0
    torso_mass_scale: float = 1.0

    @property
    def details(self) -> str:
        changes = []
        if self.leg_length_scale != 1.0:
            changes.append(f"legs {self.leg_length_scale:.2f}x")
        if self.body_length_scale != 1.0:
            changes.append(f"body {self.body_length_scale:.2f}x")
        if self.paw_scale != 1.0:
            changes.append(f"paws {self.paw_scale:.2f}x")
        if self.torso_mass_scale != 1.0:
            changes.append(f"torso mass {self.torso_mass_scale:.2f}x")
        return ", ".join(changes) or "standard Go2"


PRESETS = {
    preset.name: preset
    for preset in (
        Go2MorphologyPreset("nominal", "Nominal Go2"),
        Go2MorphologyPreset("long_legs", "Longer legs", leg_length_scale=1.35),
        Go2MorphologyPreset("short_legs", "Shorter legs (corgi)", leg_length_scale=0.70),
        Go2MorphologyPreset("long_body", "Longer body (dachshund)", body_length_scale=1.40),
        Go2MorphologyPreset("big_paws", "Bigger paws", paw_scale=1.70),
        Go2MorphologyPreset("heavy_torso", "Heavier torso", torso_mass_scale=2.0),
        Go2MorphologyPreset(
            "combined",
            "Combined: tall and stretched",
            leg_length_scale=1.25,
            body_length_scale=1.25,
            paw_scale=1.30,
            torso_mass_scale=1.30,
        ),
    )
}

LEG_PREFIXES = ("FL", "FR", "RL", "RR")
THIGH_MESH_STEMS = {"thigh_0", "thigh_1", "thigh_mirror_0", "thigh_mirror_1"}
CALF_MESH_STEMS = {"calf_0", "calf_1", "calf_mirror_0", "calf_mirror_1"}
BASE_MESH_STEMS = {"base_0", "base_1", "base_2", "base_3", "base_4"}


def _find_body(spec, name):
    return next(body for body in spec.bodies if body.name == name)


def _scale_meshes(spec, stems, scale_xyz) -> None:
    scale_xyz = np.asarray(scale_xyz, dtype=float)
    for mesh in spec.meshes:
        if Path(mesh.file).stem in stems:
            mesh.scale = np.asarray(mesh.scale) * scale_xyz


def go2_default_xml() -> Path:
    from loco_mujoco.environments.quadrupeds import UnitreeGo2

    return Path(UnitreeGo2.get_default_xml_file_path())


def create_go2_variant_xml(preset: Go2MorphologyPreset, output_root: Path | None = None) -> Path:
    source_xml = go2_default_xml()
    output_root = output_root or WORKSPACE / "generated_variants"
    output_dir = output_root / f"go2_morphology_{preset.name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_xml.parent / "assets", output_dir / "assets", dirs_exist_ok=True)
    spec = mujoco.MjSpec.from_file(str(source_xml))

    if preset.leg_length_scale != 1.0:
        s = preset.leg_length_scale
        for prefix in LEG_PREFIXES:
            calf = _find_body(spec, f"{prefix}_calf")
            calf.pos = np.asarray(calf.pos) * np.array([1.0, 1.0, s])
            for geom in calf.geoms:
                geom.pos = np.asarray(geom.pos) * np.array([1.0, 1.0, s])
            foot = next(geom for geom in calf.geoms if geom.name == f"{prefix}_foot")
            # class "foot" leaves pos inherited; pin it explicitly at the scaled calf end
            foot.pos = np.array([-0.002, 0.0, -0.213 * s])
            calf.ipos = np.asarray(calf.ipos) * np.array([1.0, 1.0, s])
            thigh = _find_body(spec, f"{prefix}_thigh")
            thigh.ipos = np.asarray(thigh.ipos) * np.array([1.0, 1.0, s])
        _scale_meshes(spec, THIGH_MESH_STEMS | CALF_MESH_STEMS, (1.0, 1.0, s))

    if preset.body_length_scale != 1.0:
        s = preset.body_length_scale
        for prefix in LEG_PREFIXES:
            hip = _find_body(spec, f"{prefix}_hip")
            hip.pos = np.asarray(hip.pos) * np.array([s, 1.0, 1.0])
        base = _find_body(spec, "base")
        base.ipos = np.asarray(base.ipos) * np.array([s, 1.0, 1.0])
        for geom in base.geoms:
            geom.pos = np.asarray(geom.pos) * np.array([s, 1.0, 1.0])
            if geom.type == mujoco.mjtGeom.mjGEOM_BOX:
                geom.size = np.asarray(geom.size) * np.array([s, 1.0, 1.0])
        _scale_meshes(spec, BASE_MESH_STEMS, (s, 1.0, 1.0))

    if preset.paw_scale != 1.0:
        s = preset.paw_scale
        for prefix in LEG_PREFIXES:
            calf = _find_body(spec, f"{prefix}_calf")
            foot = next(geom for geom in calf.geoms if geom.name == f"{prefix}_foot")
            foot.size = np.array([0.022 * s, 0.0, 0.0])
            if preset.leg_length_scale == 1.0:
                foot.pos = np.array([-0.002, 0.0, -0.213])

    if preset.torso_mass_scale != 1.0:
        base = _find_body(spec, "base")
        base.mass *= preset.torso_mass_scale
        base.inertia = np.asarray(base.inertia) * preset.torso_mass_scale

    xml_path = output_dir / "go2.xml"
    xml_path.write_text(spec.to_xml(), encoding="utf-8")
    mujoco.MjModel.from_xml_path(str(xml_path))
    return xml_path


def ground_pose(model: mujoco.MjModel, data: mujoco.MjData, clearance: float = 0.002) -> float:
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    foot_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{prefix}_foot")
        for prefix in LEG_PREFIXES
    ]
    mujoco.mj_forward(model, data)
    distances = [
        mujoco.mj_geomDistance(model, data, floor_id, foot_id, 10.0, None)
        for foot_id in foot_ids
    ]
    adjustment = clearance - min(distances)
    data.qpos[2] += adjustment
    mujoco.mj_forward(model, data)
    return float(adjustment)


def render_home_pose(xml_path: Path, width: int = 420, height: int = 360) -> tuple[np.ndarray, dict]:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    if model.nkey:
        data.qpos[:] = model.key_qpos[0]
    floor_adjustment = ground_pose(model, data)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [0.0, 0.0, float(data.qpos[2]) * 0.75]
    camera.distance = 0.95
    camera.azimuth = 140
    camera.elevation = -14
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        renderer.update_scene(data, camera=camera)
        image = renderer.render()
    return image, {
        "floor_adjustment_meters": floor_adjustment,
        "total_mass_kg": float(model.body_mass.sum()),
        "standing_height_m": float(data.qpos[2]),
    }


def draw_panel(image: np.ndarray, label: str, details: str) -> np.ndarray:
    panel = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    header = np.full((86, panel.shape[1], 3), (26, 29, 30), dtype=np.uint8)
    cv2.putText(header, label, (14, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (242, 242, 238), 2)
    for index, line in enumerate(wrap_text(details, panel.shape[1] - 28)):
        cv2.putText(header, line, (14, 55 + index * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (185, 202, 197), 1)
    return np.vstack((header, panel))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--presets", nargs="+", choices=PRESETS, default=list(PRESETS))
    parser.add_argument("--output", type=Path, default=WORKSPACE / "images" / "go2_morphology_gallery.png")
    args = parser.parse_args()

    panels = []
    summary = {}
    for name in args.presets:
        preset = PRESETS[name]
        xml_path = create_go2_variant_xml(preset)
        image, metrics = render_home_pose(xml_path)
        panels.append(draw_panel(image, preset.label, preset.details))
        summary[name] = {"xml_path": str(xml_path), **asdict(preset), **metrics}
        print(f"{name}: mass={metrics['total_mass_kg']:.2f} kg, height={metrics['standing_height_m']:.3f} m")

    columns = 4
    blank = np.full_like(panels[0], (26, 29, 30))
    rows = []
    for start in range(0, len(panels), columns):
        row = panels[start : start + columns]
        rows.append(np.hstack(row + [blank] * (columns - len(row))))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), np.vstack(rows))
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"gallery": str(args.output), "summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
