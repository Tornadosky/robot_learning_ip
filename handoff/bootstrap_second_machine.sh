#!/usr/bin/env bash
# Bootstrap the robot_learning_ip training stack on a second machine (RTX 5080).
# Tested layout: Windows 11 + WSL2 Ubuntu (22.04/24.04), NVIDIA driver >= 570,
# repo checked out under /mnt/c/... (Windows filesystem, so Windows tools can
# read logs) -- exactly like the primary box. Run INSIDE WSL:
#
#   bash handoff/bootstrap_second_machine.sh /mnt/c/Users/<you>/Desktop/robot_learning_ip
#
# Steps: 1 clone + submodules from the upstream remotes, 2 fast-forward the
# submodules to our local commits from the bundles in handoff/, 3 create the
# two venvs (~/jaxgpu for training/eval, ~/locomjx for rendering), 4 print the
# data you still have to copy (see handoff/RUNBOOK_SECOND_MACHINE.md).
set -euo pipefail
DEST=${1:?target directory, e.g. /mnt/c/Users/you/Desktop/robot_learning_ip}
REPO=https://github.com/Tornadosky/robot_learning_ip.git

if [ ! -d "$DEST/.git" ]; then
  # shallow: the main repo history carries ~21 GB of committed data files
  git clone --depth 1 "$REPO" "$DEST"
fi
cd "$DEST"
git submodule update --init --recursive loco_mjx loco-mujoco RL-X || echo "(gitlink commits are local: fetched from the bundles next)"

# 2. our local submodule commits travel as bundles (the upstream remotes do not
#    have them); each bundle fast-forwards from the upstream commit it was cut at
git -C loco_mjx fetch ../handoff/loco_mjx_local.bundle HEAD && git -C loco_mjx checkout -q -B tracking-phase1 FETCH_HEAD
git -C loco-mujoco fetch ../handoff/loco_mujoco_local.bundle HEAD && git -C loco-mujoco checkout -q -B integration FETCH_HEAD
git -C RL-X fetch ../handoff/rlx_local.bundle HEAD && git -C RL-X checkout -q -B local FETCH_HEAD
for m in loco_mjx loco-mujoco RL-X; do echo "$m -> $(git -C $m rev-parse --short HEAD) $(git -C $m log -1 --format=%s | cut -c1-70)"; done

# 3. venvs (python 3.12; jax 0.7.1 + CUDA 12 plugin; mujoco 3.7.0)
#    Native Ubuntu works the same as WSL. Ubuntu 22.04 ships 3.10: add
#    `sudo add-apt-repository ppa:deadsnakes/ppa` first, or use a conda/uv
#    python 3.12 and replace python3.12 below.
sudo apt-get install -y -q python3.12 python3.12-venv python3.12-dev ffmpeg libegl1 libgl1 >/dev/null || echo "install python3.12 yourself (deadsnakes / conda) and rerun"
python3.12 -m venv ~/jaxgpu
~/jaxgpu/bin/pip install -q --upgrade pip
~/jaxgpu/bin/pip install -q -r handoff/requirements_jaxgpu.txt
~/jaxgpu/bin/pip install -q -e RL-X -e loco_mjx -e loco-mujoco 2>/dev/null || true
~/jaxgpu/bin/python -c "import jax; print('jax devices:', jax.devices())"
python3.12 -m venv ~/locomjx
~/locomjx/bin/pip install -q --upgrade pip
~/locomjx/bin/pip install -q -r handoff/requirements_locomjx_render.txt
~/locomjx/bin/pip install -q -e loco_mjx 2>/dev/null || true

# 4. WSL memory: /mnt/c/Users/<you>/.wslconfig on the WINDOWS side must contain
#    [wsl2] memory=24GB swap=16GB (a 3-robot compile OOM-kills the default 50 %).
cat <<'EOF'

NEXT: copy the data listed in handoff/RUNBOOK_SECOND_MACHINE.md (external_data,
clips, tokenizer), set the paths in the runbook, then run one smoke:
  NAME=smoke CLIPDIR=$DEST/experiments/fsq_khaendler/clips_3t_v2 LATENT=1 JLAT_CH=-1 \
  NR_ENVS=64 MINIBATCH=2048 TOTAL=40960 SAVE_EVERY=40960 PROJECT=smoke \
  bash scripts/scaling/wave6/local_train.sh
EOF
