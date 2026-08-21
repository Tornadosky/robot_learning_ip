# Robot Learning IP

H1 morphology retargeting experiments built on [LocoMuJoCo](loco-mujoco/), [loco_mjx](loco_mjx/), and [RL-X](RL-X/).

Large datasets and generated artifacts are **not** committed to git. After cloning, run the setup script below.

## Quick start (new machine)

```bash
git clone <repo-url> robot_learning_ip
cd robot_learning_ip

# Create a virtual environment (recommended)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

# Install LocoMuJoCo (editable) and optional SMPL support for AMASS retargeting
pip install -e loco-mujoco
pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
pip install -e "loco-mujoco[smpl]"

# Auto-download default mocap clips from HuggingFace and configure local paths
python scripts/setup_data.py
```

## Manual datasets (license required)

AMASS and SMPL-H cannot be downloaded automatically. Place them under `external_data/`:

| Asset | Destination | Guide |
|-------|-------------|-------|
| AMASS DanceDB (SMPL-H G) | `external_data/amass/` | [external_data/amass/README.md](external_data/amass/README.md) |
| SMPL-H + MANO models | `external_data/smpl/` | [external_data/smpl/README.md](external_data/smpl/README.md) |

After copying SMPL source files:

```bash
python scripts/setup_data.py --build-smplh
python scripts/setup_data.py --check-only
```

## What gets ignored by git

- `external_data/amass/`, `external_data/smpl/` — licensed motion capture and body models
- `external_data/amass_converted/`, `external_data/morphology_sweep/` — retarget caches
- `videos/`, rendered `images/*.png` — generated media
- `.venv/`, `LOCOMUJOCO_VARIABLES.yaml` — machine-local config

Small source artifacts that **are** tracked: `scripts/`, `generated_variants/` (H1 XML morphologies), and gallery JSON metadata in `images/`.

## Example scripts

```bash
# Morphology variant gallery (uses default squat mocap from HuggingFace)
python scripts/h1_morphology_variants.py

# AMASS retarget (requires AMASS + SMPL setup above)
python scripts/retarget_amass_clip_randomized_h1.py

# Full morphology sweep retarget
python scripts/retarget_h1_morphology_sweep.py --task squat

# LAFAN1 dance2 on stock humanoids (auto-downloads from HuggingFace, no license needed)
python scripts/dance_lafan1_robots.py --robots UnitreeH1 UnitreeG1 ToddlerBot Atlas

# SMPL retarget of a LAFAN1 dance clip onto H1 morphology variants (requires SMPL setup)
python scripts/retarget_h1_dance_sweep.py --clip dance2_subject4

# Unitree Go2 ("robot dog") morphology gallery — static poses only;
# quadrupeds have no mocap datasets or retargeting support in loco-mujoco
python scripts/dog_morphology_gallery.py

# DeepMimic: train a PPO policy to physically track the dance (needs GPU JAX, e.g. WSL2;
# MjWarp backend is broken with mujoco 3.9.0 — script defaults to plain MJX)
python scripts/train_deepmimic_dance.py --output-dir external_data/deepmimic_dance

# Roll out the trained policy in plain MuJoCo and record a video (CPU JAX is fine)
python scripts/eval_deepmimic_dance.py --agent-path external_data/deepmimic_dance/PPOJax_saved.pkl --n-steps 3000
```

## Scaling docs

- [`SCALING_ROADMAP.md`](SCALING_ROADMAP.md) — 10-step path to thousands of randomized robots / multi-motion / cross-family control
- [`SCALING_PROGRESS_2026-08-01.md`](SCALING_PROGRESS_2026-08-01.md) — measured progress on parallel + online morphology training

## Subprojects

- `loco-mujoco/` — imitation learning benchmark and retargeting pipeline
- `loco_mjx/` — JAX/MJX locomotion experiments
- `RL-X/` — RL algorithm implementations
