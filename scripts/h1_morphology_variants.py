"""Create explicit H1 morphology variants for retargeting experiments."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import mujoco
import numpy as np
from loco_mujoco.environments import UnitreeH1


WORKSPACE = Path(__file__).resolve().parents[1]
SOURCE_MODEL_DIR = Path(UnitreeH1.get_default_xml_file_path()).parent


@dataclass(frozen=True)
class H1MorphologyPreset:
    name: str
    label: str
    leg_length_scale: float = 1.0
    arm_length_scale: float = 1.0
    shoulder_width_scale: float = 1.0
    foot_scale_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0)
    torso_mass_scale: float = 1.0
    # Expanded dims (2026-08-09), mirroring online_h1.MORPHOLOGY_SPEC.
    torso_length_scale: float = 1.0
    total_mass_scale: float = 1.0
    damping_scale: float = 1.0
    armature_scale: float = 1.0
    strength_scale: float = 1.0
    friction_scale: float = 1.0
    torso_com_x_offset: float = 0.0

    @property
    def details(self) -> str:
        changes = []
        if self.leg_length_scale != 1.0:
            changes.append(f"legs {self.leg_length_scale:.2f}x")
        if self.arm_length_scale != 1.0:
            changes.append(f"arms {self.arm_length_scale:.2f}x")
        if self.shoulder_width_scale != 1.0:
            changes.append(f"shoulders {self.shoulder_width_scale:.2f}x")
        if self.foot_scale_xyz != (1.0, 1.0, 1.0):
            changes.append("feet " + "/".join(f"{value:.2f}" for value in self.foot_scale_xyz))
        if self.torso_mass_scale != 1.0:
            changes.append(f"torso mass {self.torso_mass_scale:.2f}x")
        if self.torso_length_scale != 1.0:
            changes.append(f"torso len {self.torso_length_scale:.2f}x")
        if self.total_mass_scale != 1.0:
            changes.append(f"total mass {self.total_mass_scale:.2f}x")
        if self.damping_scale != 1.0:
            changes.append(f"damping {self.damping_scale:.2f}x")
        if self.armature_scale != 1.0:
            changes.append(f"armature {self.armature_scale:.2f}x")
        if self.strength_scale != 1.0:
            changes.append(f"strength {self.strength_scale:.2f}x")
        if self.friction_scale != 1.0:
            changes.append(f"friction {self.friction_scale:.2f}x")
        if self.torso_com_x_offset != 0.0:
            changes.append(f"COM x {self.torso_com_x_offset*100:+.1f} cm")
        return ", ".join(changes) or "standard H1"


PRESETS = {
    preset.name: preset
    for preset in (
        H1MorphologyPreset("nominal", "Nominal H1"),
        H1MorphologyPreset("tall_legs", "Longer legs", leg_length_scale=1.12),
        H1MorphologyPreset("short_legs", "Shorter legs", leg_length_scale=0.90),
        H1MorphologyPreset("long_arms", "Longer arms", arm_length_scale=1.15),
        H1MorphologyPreset("broad_shoulders", "Broader shoulders", shoulder_width_scale=1.15),
        H1MorphologyPreset("big_feet", "Larger feet", foot_scale_xyz=(1.15, 1.10, 1.05)),
        H1MorphologyPreset("heavy_torso", "Heavier torso", torso_mass_scale=1.35),
        H1MorphologyPreset(
            "combined",
            "Combined moderate variation",
            leg_length_scale=1.08,
            arm_length_scale=0.94,
            shoulder_width_scale=1.08,
            foot_scale_xyz=(1.08, 1.05, 1.0),
            torso_mass_scale=1.15,
        ),
        H1MorphologyPreset(
            "extreme_tall_light",
            "Extreme: tall and light",
            leg_length_scale=1.50,
            arm_length_scale=0.72,
            shoulder_width_scale=0.72,
            foot_scale_xyz=(0.68, 0.68, 0.82),
            torso_mass_scale=0.55,
        ),
        H1MorphologyPreset(
            "extreme_short_heavy",
            "Extreme: short and heavy",
            leg_length_scale=0.68,
            arm_length_scale=1.55,
            shoulder_width_scale=1.55,
            foot_scale_xyz=(1.65, 1.45, 1.20),
            torso_mass_scale=2.40,
        ),
        H1MorphologyPreset(
            "extreme_combined",
            "Extreme: combined stress case",
            leg_length_scale=1.42,
            arm_length_scale=1.48,
            shoulder_width_scale=1.48,
            foot_scale_xyz=(1.60, 1.40, 1.18),
            torso_mass_scale=2.00,
        ),
    )
}


def _find_named(items, name):
    return next(item for item in items if item.name == name)


def _scale_inertial(body, scale_xyz) -> None:
    scale_xyz = np.asarray(scale_xyz, dtype=float)
    volume_scale = float(np.prod(scale_xyz))
    body.ipos = np.asarray(body.ipos) * scale_xyz
    body.mass *= volume_scale
    body.inertia = np.asarray(body.inertia) * volume_scale * float(np.mean(scale_xyz) ** 2)


def _scale_meshes(spec, stems, scale_xyz) -> None:
    scale_xyz = np.asarray(scale_xyz, dtype=float)
    for mesh in spec.meshes:
        if Path(mesh.file).stem in stems:
            mesh.scale = np.asarray(mesh.scale) * scale_xyz


def create_h1_variant_xml(
    preset: H1MorphologyPreset,
    output_root: Path | None = None,
) -> Path:
    """Write a self-contained H1 XML/assets directory for one named preset."""
    output_root = output_root or WORKSPACE / "generated_variants"
    output_dir = output_root / f"h1_morphology_{preset.name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SOURCE_MODEL_DIR / "assets", output_dir / "assets", dirs_exist_ok=True)
    xml_path = output_dir / "h1.xml"
    spec = mujoco.MjSpec.from_file(str(SOURCE_MODEL_DIR / "h1.xml"))

    if preset.leg_length_scale != 1.0:
        leg_scale = preset.leg_length_scale
        for side in ("left", "right"):
            for name in (f"{side}_knee_link", f"{side}_ankle_link"):
                body = _find_named(spec.bodies, name)
                body.pos = np.asarray(body.pos) * np.array([1.0, 1.0, leg_scale])
            _scale_inertial(_find_named(spec.bodies, f"{side}_hip_pitch_link"), (1, 1, leg_scale))
            _scale_inertial(_find_named(spec.bodies, f"{side}_knee_link"), (1, 1, leg_scale))
        _scale_meshes(
            spec,
            {"left_hip_pitch_link", "right_hip_pitch_link", "left_knee_link", "right_knee_link"},
            (1, 1, leg_scale),
        )

    if preset.arm_length_scale != 1.0:
        arm_scale = preset.arm_length_scale
        for side in ("left", "right"):
            upper_arm = _find_named(spec.bodies, f"{side}_shoulder_yaw_link")
            forearm = _find_named(spec.bodies, f"{side}_elbow_link")
            upper_arm.pos = np.asarray(upper_arm.pos) * np.array([1.0, 1.0, arm_scale])
            forearm.pos = np.asarray(forearm.pos) * arm_scale
            hand_site = _find_named(spec.sites, f"{side}_hand_mimic")
            hand_site.pos = np.asarray(hand_site.pos) * np.array([arm_scale, 1.0, 1.0])
            _scale_inertial(upper_arm, (1, 1, arm_scale))
            _scale_inertial(forearm, (arm_scale, 1, 1))
        _scale_meshes(spec, {"left_shoulder_yaw_link", "right_shoulder_yaw_link"}, (1, 1, arm_scale))
        _scale_meshes(spec, {"left_elbow_link", "right_elbow_link"}, (arm_scale, 1, 1))

    if preset.shoulder_width_scale != 1.0:
        width_scale = preset.shoulder_width_scale
        for side in ("left", "right"):
            body = _find_named(spec.bodies, f"{side}_shoulder_pitch_link")
            body.pos = np.asarray(body.pos) * np.array([1.0, width_scale, 1.0])

    if preset.foot_scale_xyz != (1.0, 1.0, 1.0):
        foot_scale = np.asarray(preset.foot_scale_xyz)
        _scale_meshes(spec, {"left_ankle_link", "right_ankle_link"}, foot_scale)
        for side in ("left", "right"):
            site = _find_named(spec.sites, f"{side}_foot_mimic")
            site.pos = np.asarray(site.pos) * foot_scale
            _scale_inertial(_find_named(spec.bodies, f"{side}_ankle_link"), foot_scale)

    if preset.torso_mass_scale != 1.0:
        torso = _find_named(spec.bodies, "torso_link")
        torso.mass *= preset.torso_mass_scale
        torso.inertia = np.asarray(torso.inertia) * preset.torso_mass_scale

    if preset.torso_length_scale != 1.0:
        length = preset.torso_length_scale
        for side in ("left", "right"):
            body = _find_named(spec.bodies, f"{side}_shoulder_pitch_link")
            body.pos = np.asarray(body.pos) * np.array([1.0, 1.0, length])
        _scale_inertial(_find_named(spec.bodies, "torso_link"), (1, 1, length))
        site = _find_named(spec.sites, "upper_body_mimic")
        site.pos = np.asarray(site.pos) * np.array([1.0, 1.0, length])
        _scale_meshes(spec, {"torso_link"}, (1, 1, length))

    if preset.total_mass_scale != 1.0:
        for body in spec.bodies:
            if body.mass > 0.0:
                body.mass *= preset.total_mass_scale
                body.inertia = np.asarray(body.inertia) * preset.total_mass_scale

    if preset.damping_scale != 1.0 or preset.armature_scale != 1.0:
        for joint in spec.joints:
            if joint.type != mujoco.mjtJoint.mjJNT_FREE:
                joint.damping *= preset.damping_scale
                joint.armature *= preset.armature_scale

    if preset.strength_scale != 1.0:
        for actuator in spec.actuators:
            gainprm = np.asarray(actuator.gainprm, dtype=float)
            gainprm[0] *= preset.strength_scale
            actuator.gainprm = gainprm

    if preset.friction_scale != 1.0:
        for geom in spec.geoms:
            friction = np.asarray(geom.friction, dtype=float)
            friction[0] *= preset.friction_scale
            geom.friction = friction

    if preset.torso_com_x_offset != 0.0:
        torso = _find_named(spec.bodies, "torso_link")
        ipos = np.asarray(torso.ipos, dtype=float)
        ipos[0] += preset.torso_com_x_offset
        torso.ipos = ipos

    xml_path.write_text(spec.to_xml(), encoding="utf-8")
    mujoco.MjModel.from_xml_path(str(xml_path))
    return xml_path


def ground_pose(model: mujoco.MjModel, data: mujoco.MjData, clearance: float = 0.002) -> float:
    """Place a pose just above the floor based on the clean model geometry."""
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    foot_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in ("left_foot", "right_foot")
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


def render_default_pose(xml_path: Path, width: int = 420, height: int = 390) -> tuple[np.ndarray, dict]:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    if model.nkey:
        data.qpos[:] = model.key_qpos[0]
    floor_adjustment = ground_pose(model, data)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = np.array([0.0, 0.0, 0.85])
    camera.distance = 2.8
    camera.azimuth = 145
    camera.elevation = -16
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        renderer.update_scene(data, camera=camera)
        image = renderer.render()
    return image, {
        "floor_adjustment_meters": floor_adjustment,
        "total_mass_kg": float(model.body_mass.sum()),
    }


def wrap_text(text: str, width: int, font_scale: float = 0.42) -> list[str]:
    lines = []
    current = []
    for word in text.split():
        candidate = " ".join((*current, word))
        if current and cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0][0] > width:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines[:2]


def _draw_panel(image: np.ndarray, preset: H1MorphologyPreset) -> np.ndarray:
    panel = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    header = np.full((86, panel.shape[1], 3), (26, 29, 30), dtype=np.uint8)
    cv2.putText(header, preset.label, (14, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (242, 242, 238), 2)
    for index, line in enumerate(wrap_text(preset.details, panel.shape[1] - 28)):
        cv2.putText(header, line, (14, 55 + index * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (185, 202, 197), 1)
    return np.vstack((header, panel))


def build_gallery(preset_names: list[str], output_path: Path) -> dict:
    panels = []
    summary = {}
    for name in preset_names:
        preset = PRESETS[name]
        xml_path = create_h1_variant_xml(preset)
        image, metrics = render_default_pose(xml_path)
        panels.append(_draw_panel(image, preset))
        summary[name] = {"xml_path": str(xml_path), **asdict(preset), **metrics}

    columns = 4
    rows = []
    blank = np.full_like(panels[0], (26, 29, 30))
    for start in range(0, len(panels), columns):
        row_panels = panels[start : start + columns]
        rows.append(np.hstack(row_panels + [blank] * (columns - len(row_panels))))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), np.vstack(rows))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--presets", nargs="+", choices=PRESETS, default=list(PRESETS))
    parser.add_argument("--output", type=Path, default=WORKSPACE / "images" / "h1_morphology_gallery.png")
    args = parser.parse_args()
    summary = build_gallery(args.presets, args.output)
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"gallery": str(args.output), "summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
