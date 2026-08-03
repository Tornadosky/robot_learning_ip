"""Create explicit Unitree G1 morphology variants and render a static pose gallery.

This is the G1 counterpart of `h1_morphology_variants.py`. Every preset only
scales link lengths / widths / masses, so the joint DOF layout is unchanged.
That property is what lets `dance_g1_morphology.py` replay the *same* recorded
joint trajectory on every variant (kinematic comparison, no SMPL needed).
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
from loco_mujoco.environments import UnitreeG1


WORKSPACE = Path(__file__).resolve().parents[1]
SOURCE_XML = Path(UnitreeG1.get_default_xml_file_path())
SOURCE_MODEL_DIR = SOURCE_XML.parent


@dataclass(frozen=True)
class G1MorphologyPreset:
    name: str
    label: str
    leg_length_scale: float = 1.0
    arm_length_scale: float = 1.0
    shoulder_width_scale: float = 1.0
    foot_scale_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0)
    torso_mass_scale: float = 1.0

    @property
    def details(self) -> str:
        # Build a short human-readable description of what changed vs nominal G1.
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
        return ", ".join(changes) or "standard G1"


def _tall_light(leg_scale: float, name: str, label: str) -> "G1MorphologyPreset":
    """A 'tall & light' body at a chosen leg length, interpolated from nominal
    toward the extreme_tall_light end-point (legs 1.55x / torso 0.55x etc).

    Used to map the controllability cliff: as the legs lengthen and the torso
    lightens, balance gets harder until G1 can no longer learn to track the dance.
    """
    t = (leg_scale - 1.0) / (1.55 - 1.0)  # 0 at nominal, 1 at extreme_tall_light
    lerp = lambda target: round(1.0 + t * (target - 1.0), 4)
    return G1MorphologyPreset(
        name, label,
        leg_length_scale=round(leg_scale, 4),
        arm_length_scale=lerp(0.70),
        shoulder_width_scale=lerp(0.70),
        foot_scale_xyz=(lerp(0.66), lerp(0.66), lerp(0.80)),
        torso_mass_scale=lerp(0.55),
    )


# Presets mirror the H1 set so the two robots can be compared on equal footing.
PRESETS = {
    preset.name: preset
    for preset in (
        G1MorphologyPreset("nominal", "Nominal G1"),
        G1MorphologyPreset("tall_legs", "Longer legs", leg_length_scale=1.14),
        G1MorphologyPreset("short_legs", "Shorter legs", leg_length_scale=0.88),
        G1MorphologyPreset("long_arms", "Longer arms", arm_length_scale=1.18),
        G1MorphologyPreset("broad_shoulders", "Broader shoulders", shoulder_width_scale=1.18),
        G1MorphologyPreset("big_feet", "Larger feet", foot_scale_xyz=(1.18, 1.12, 1.06)),
        G1MorphologyPreset("heavy_torso", "Heavier torso", torso_mass_scale=1.40),
        G1MorphologyPreset(
            "combined",
            "Combined moderate variation",
            leg_length_scale=1.10,
            arm_length_scale=0.92,
            shoulder_width_scale=1.10,
            foot_scale_xyz=(1.10, 1.06, 1.0),
            torso_mass_scale=1.18,
        ),
        G1MorphologyPreset(
            "extreme_tall_light",
            "Extreme: tall and light",
            leg_length_scale=1.55,
            arm_length_scale=0.70,
            shoulder_width_scale=0.70,
            foot_scale_xyz=(0.66, 0.66, 0.80),
            torso_mass_scale=0.55,
        ),
        G1MorphologyPreset(
            "extreme_short_heavy",
            "Extreme: short and heavy",
            leg_length_scale=0.66,
            arm_length_scale=1.55,
            shoulder_width_scale=1.55,
            foot_scale_xyz=(1.70, 1.45, 1.20),
            torso_mass_scale=2.40,
        ),
        G1MorphologyPreset(
            "extreme_combined",
            "Extreme: combined stress case",
            leg_length_scale=1.45,
            arm_length_scale=1.50,
            shoulder_width_scale=1.50,
            foot_scale_xyz=(1.62, 1.42, 1.18),
            torso_mass_scale=2.00,
        ),
        # Graduated tall&light family (controllability-cliff sweep): same axis as
        # extreme_tall_light (legs 1.55x) but milder, to find where G1 stops
        # being able to learn the dance.
        _tall_light(1.20, "tall_light_leg120", "Tall&light (legs 1.20x)"),
        _tall_light(1.35, "tall_light_leg135", "Tall&light (legs 1.35x)"),
        _tall_light(1.50, "tall_light_leg150", "Tall&light (legs 1.50x)"),
    )
}


def _find_named(items, name):
    return next(item for item in items if item.name == name)


def _scale_inertial(body, scale_xyz) -> None:
    # Scale the lumped inertial of a body consistently with a geometric stretch.
    scale_xyz = np.asarray(scale_xyz, dtype=float)
    volume_scale = float(np.prod(scale_xyz))
    body.ipos = np.asarray(body.ipos) * scale_xyz
    body.mass *= volume_scale
    body.inertia = np.asarray(body.inertia) * volume_scale * float(np.mean(scale_xyz) ** 2)


def _scale_meshes(spec, stems, scale_xyz) -> None:
    # Stretch the visual/collision meshes that belong to the given link stems.
    scale_xyz = np.asarray(scale_xyz, dtype=float)
    for mesh in spec.meshes:
        if Path(mesh.file).stem in stems:
            mesh.scale = np.asarray(mesh.scale) * scale_xyz


def create_g1_variant_xml(
    preset: G1MorphologyPreset,
    output_root: Path | None = None,
) -> Path:
    """Write a self-contained G1 XML/assets directory for one named preset."""
    output_root = output_root or WORKSPACE / "generated_variants"
    output_dir = output_root / f"g1_morphology_{preset.name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    # Copy the mesh/texture asset folders next to the generated XML so the file
    # is portable. The G1 model keeps meshes in "meshes" (+ a sibling "images"
    # texture dir); copy whichever asset folders exist.
    for asset_dir in ("meshes", "images", "assets"):
        source_assets = SOURCE_MODEL_DIR / asset_dir
        if source_assets.is_dir():
            shutil.copytree(source_assets, output_dir / asset_dir, dirs_exist_ok=True)
    xml_path = output_dir / "g1.xml"
    spec = mujoco.MjSpec.from_file(str(SOURCE_XML))

    # --- Legs: lengthen thigh (hip_yaw->knee) and shank (knee->ankle) along z. ---
    if preset.leg_length_scale != 1.0:
        leg_scale = preset.leg_length_scale
        for side in ("left", "right"):
            # Child body positions are relative to their parent, so scaling the
            # z offset stretches that bone segment.
            for name in (f"{side}_knee_link", f"{side}_ankle_pitch_link"):
                body = _find_named(spec.bodies, name)
                body.pos = np.asarray(body.pos) * np.array([1.0, 1.0, leg_scale])
            _scale_inertial(_find_named(spec.bodies, f"{side}_hip_yaw_link"), (1, 1, leg_scale))
            _scale_inertial(_find_named(spec.bodies, f"{side}_knee_link"), (1, 1, leg_scale))
        _scale_meshes(
            spec,
            {"left_hip_yaw_link", "right_hip_yaw_link", "left_knee_link", "right_knee_link"},
            (1, 1, leg_scale),
        )

    # --- Arms: lengthen upper arm (shoulder_yaw->elbow) and forearm (elbow->wrist). ---
    if preset.arm_length_scale != 1.0:
        arm_scale = preset.arm_length_scale
        for side in ("left", "right"):
            elbow = _find_named(spec.bodies, f"{side}_elbow_link")
            wrist = _find_named(spec.bodies, f"{side}_wrist_roll_rubber_hand")
            elbow.pos = np.asarray(elbow.pos) * arm_scale
            wrist.pos = np.asarray(wrist.pos) * arm_scale
            _scale_inertial(elbow, (arm_scale, 1, 1))
        _scale_meshes(spec, {"left_elbow_link", "right_elbow_link"}, (arm_scale, 1, 1))

    # --- Shoulders: widen the lateral offset of the shoulder mounting. ---
    if preset.shoulder_width_scale != 1.0:
        width_scale = preset.shoulder_width_scale
        for side in ("left", "right"):
            body = _find_named(spec.bodies, f"{side}_shoulder_pitch_link")
            body.pos = np.asarray(body.pos) * np.array([1.0, width_scale, 1.0])

    # --- Feet: scale the ankle_roll meshes/inertials and the foot mimic site. ---
    if preset.foot_scale_xyz != (1.0, 1.0, 1.0):
        foot_scale = np.asarray(preset.foot_scale_xyz)
        _scale_meshes(spec, {"left_ankle_roll_link", "right_ankle_roll_link"}, foot_scale)
        for side in ("left", "right"):
            site = _find_named(spec.sites, f"{side}_foot_mimic")
            site.pos = np.asarray(site.pos) * foot_scale
            _scale_inertial(_find_named(spec.bodies, f"{side}_ankle_roll_link"), foot_scale)

    # --- Torso: make the trunk heavier without changing its shape. ---
    if preset.torso_mass_scale != 1.0:
        torso = _find_named(spec.bodies, "torso_link")
        torso.mass *= preset.torso_mass_scale
        torso.inertia = np.asarray(torso.inertia) * preset.torso_mass_scale

    xml_path.write_text(spec.to_xml(), encoding="utf-8")
    # Compile once to guarantee every generated model is valid before returning.
    mujoco.MjModel.from_xml_path(str(xml_path))
    return xml_path


def ground_pose(model: mujoco.MjModel, data: mujoco.MjData, clearance: float = 0.002) -> float:
    """Drop the model so its lowest body geom sits just above the floor.

    Generic across morphologies: we don't rely on named foot geoms, we just look
    at the minimum world-z of every non-floor geom.
    """
    mujoco.mj_forward(model, data)
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    zs = [float(data.geom_xpos[g][2]) for g in range(model.ngeom) if g != floor_id]
    adjustment = clearance - min(zs)
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
    camera.lookat[:] = np.array([0.0, 0.0, 0.7])
    camera.distance = 2.4
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
    lines: list[str] = []
    current: list[str] = []
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


def _draw_panel(image: np.ndarray, preset: G1MorphologyPreset) -> np.ndarray:
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
        xml_path = create_g1_variant_xml(preset)
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--presets", nargs="+", choices=PRESETS, default=list(PRESETS))
    parser.add_argument("--output", type=Path, default=WORKSPACE / "images" / "g1_morphology_gallery.png")
    args = parser.parse_args()
    summary = build_gallery(args.presets, args.output)
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"gallery": str(args.output), "summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
