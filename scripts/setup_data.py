#!/usr/bin/env python3
"""Bootstrap datasets and LocoMuJoCo paths after cloning this repo."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
EXTERNAL_DATA = WORKSPACE / "external_data"
LOCO_MUJOCO_DIR = WORKSPACE / "loco-mujoco"

DATA_DIRS = (
    EXTERNAL_DATA / "amass",
    EXTERNAL_DATA / "smpl",
    EXTERNAL_DATA / "amass_converted",
    EXTERNAL_DATA / "amass_extreme",
    EXTERNAL_DATA / "retarget_proof",
    EXTERNAL_DATA / "morphology_sweep",
    WORKSPACE / "videos",
)

# Default clip used by retarget_amass_clip_randomized_h1.py and related scripts.
DEFAULT_AMASS_CLIP = (
    "DanceDB/20120911_TheodorosSourmelis/Capoeira_Theodoros_v2_C3D_poses"
)

SMPLH_SOURCES = (
    EXTERNAL_DATA / "smpl" / "smplh" / "neutral" / "model.npz",
    EXTERNAL_DATA / "smpl" / "mano_v1_2" / "models" / "MANO_LEFT.pkl",
    EXTERNAL_DATA / "smpl" / "mano_v1_2" / "models" / "MANO_RIGHT.pkl",
)
SMPLH_OUTPUT = EXTERNAL_DATA / "smpl" / "SMPLH_NEUTRAL.pkl"
GENERATE_SMPLH_SCRIPT = LOCO_MUJOCO_DIR / "loco_mujoco" / "smpl" / "generate_smplh_model.py"


def ensure_directories() -> None:
    for directory in DATA_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


def configure_loco_mujoco_paths() -> None:
    from loco_mujoco.utils.dataset import _set_path_in_yaml_conf
    import loco_mujoco

    cache_root = EXTERNAL_DATA / "amass_converted"
    variables = {
        "LOCOMUJOCO_AMASS_PATH": str(EXTERNAL_DATA / "amass"),
        "LOCOMUJOCO_SMPL_MODEL_PATH": str(EXTERNAL_DATA / "smpl"),
        "LOCOMUJOCO_CONVERTED_AMASS_PATH": str(cache_root / "AMASS"),
        "LOCOMUJOCO_CONVERTED_LAFAN1_PATH": str(cache_root / "LAFAN1"),
        "LOCOMUJOCO_CONVERTED_DEFAULT_PATH": str(cache_root / "DEFAULT"),
    }

    for name, path in variables.items():
        _set_path_in_yaml_conf(path, name, path_to_conf=loco_mujoco.PATH_TO_VARIABLES, quiet=True)

    print("Configured LocoMuJoCo paths in loco-mujoco/loco_mujoco/LOCOMUJOCO_VARIABLES.yaml")
    for name, path in variables.items():
        print(f"  {name} -> {path}")


def prefetch_huggingface_datasets(tasks: list[str]) -> None:
    from loco_mujoco.task_factories import DefaultDatasetConf, ImitationFactory

    for task in tasks:
        print(f"Prefetching default mocap dataset for UnitreeH1 / {task} ...")
        ImitationFactory.make(
            "UnitreeH1",
            default_dataset_conf=DefaultDatasetConf([task]),
        )
        print(f"  cached: {task}")


def amass_clip_path(clip_id: str) -> Path:
    return EXTERNAL_DATA / "amass" / f"{clip_id}.npz"


def check_amass(clip_id: str = DEFAULT_AMASS_CLIP) -> bool:
    path = amass_clip_path(clip_id)
    if path.is_file():
        print(f"AMASS clip found: {path}")
        return True
    print(f"AMASS clip missing: {path}")
    print(f"  See {EXTERNAL_DATA / 'amass' / 'README.md'} for download instructions.")
    return False


def check_smpl_sources() -> bool:
    missing = [path for path in SMPLH_SOURCES if not path.is_file()]
    if not missing:
        print("SMPL-H source files found.")
        return True
    print("SMPL-H source files missing:")
    for path in missing:
        print(f"  {path}")
    print(f"  See {EXTERNAL_DATA / 'smpl' / 'README.md'} for download instructions.")
    return False


def check_smplh_model() -> bool:
    if SMPLH_OUTPUT.is_file():
        print(f"SMPL-H combined model found: {SMPLH_OUTPUT}")
        return True
    print(f"SMPL-H combined model missing: {SMPLH_OUTPUT}")
    if check_smpl_sources():
        print("  Run: python scripts/setup_data.py --build-smplh")
    return False


def build_smplh_model() -> None:
    if not check_smpl_sources():
        sys.exit(1)

    configure_loco_mujoco_paths()

    if SMPLH_OUTPUT.is_file():
        print(f"SMPLH_NEUTRAL.pkl already exists at {SMPLH_OUTPUT}")
        return

    import loco_mujoco

    print("Building SMPLH_NEUTRAL.pkl (requires numpy<1.23 and chumpy) ...")
    try:
        subprocess.run(
            [
                sys.executable,
                str(GENERATE_SMPLH_SCRIPT),
                "--smpl-conf-file",
                str(loco_mujoco.PATH_TO_VARIABLES),
            ],
            check=True,
        )
    except subprocess.CalledProcessError:
        print(
            "\nAutomatic SMPL-H build failed. Create a Python 3.10 env with "
            "numpy<1.23 and chumpy, then run:\n"
            f"  python {GENERATE_SMPLH_SCRIPT} "
            f"--smpl-conf-file {loco_mujoco.PATH_TO_VARIABLES}"
        )
        sys.exit(1)

    if SMPLH_OUTPUT.is_file():
        print(f"Created {SMPLH_OUTPUT}")
    else:
        print("SMPL-H build finished but output file was not found.")
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Set up datasets and LocoMuJoCo paths for this workspace."
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only verify that manual datasets are present; do not download anything.",
    )
    parser.add_argument(
        "--build-smplh",
        action="store_true",
        help="Build SMPLH_NEUTRAL.pkl from manually downloaded SMPL/MANO source files.",
    )
    parser.add_argument(
        "--skip-prefetch",
        action="store_true",
        help="Skip prefetching default mocap clips from HuggingFace.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["squat"],
        help="Default mocap tasks to prefetch from HuggingFace (default: squat).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_directories()

    try:
        import loco_mujoco  # noqa: F401
    except ImportError:
        print(
            "loco-mujoco is not installed. From the repo root run:\n"
            "  pip install -e loco-mujoco\n"
            "For AMASS retargeting also install SMPL extras:\n"
            "  pip install torch --extra-index-url https://download.pytorch.org/whl/cpu\n"
            "  pip install -e 'loco-mujoco[smpl]'"
        )
        sys.exit(1)

    if args.build_smplh:
        build_smplh_model()
        return

    if args.check_only:
        ok = check_amass() and check_smplh_model()
        sys.exit(0 if ok else 1)

    configure_loco_mujoco_paths()

    if not args.skip_prefetch:
        prefetch_huggingface_datasets(args.tasks)

    print()
    print("Automatic setup complete.")
    print()
    print("Next steps (manual, license-gated):")
    print(f"  1. AMASS DanceDB -> {EXTERNAL_DATA / 'amass'}")
    print(f"     Guide: {EXTERNAL_DATA / 'amass' / 'README.md'}")
    print(f"  2. SMPL-H + MANO   -> {EXTERNAL_DATA / 'smpl'}")
    print(f"     Guide: {EXTERNAL_DATA / 'smpl' / 'README.md'}")
    print("  3. python scripts/setup_data.py --build-smplh")
    print("  4. python scripts/setup_data.py --check-only")
    print()
    check_amass()
    check_smplh_model()


if __name__ == "__main__":
    main()
