#!/bin/bash
# Rebuild the Viper ROCm training env for urma2 (purged from /ptmp).
# Mirrors the known-good local WSL venv (jax 0.7.2-family, mujoco 3.7.0),
# with the ROCm plugin combo documented in viper_train.sbatch (jax 0.7.1).
set -uo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true
module load python-waterboa/2025.06
eval "$(conda shell.bash hook)"

ENVP=/ptmp/akalenik/urma/env
if [ ! -e "$ENVP/bin/python" ]; then
  conda create -y -q -p "$ENVP" python=3.11 2>&1 | tail -1
fi
conda activate "$ENVP"
pip install --upgrade -q pip
echo "== installing jax rocm =="
pip install -q "jax==0.7.1" "jax-rocm7-plugin" "jax-rocm7-pjrt" 2>&1 | tail -3
echo "== installing deps =="
pip install -q "mujoco==3.7.0" "mujoco-mjx==3.7.0" "flax==0.12.0" "optax==0.2.8" \
  "numpy<2.5" ml-collections imageio "etils[epath]" tensorboard pygame dm_control 2>&1 | tail -2
echo "== installing RL-X (no deps) =="
cd /ptmp/akalenik/urma/RL-X && pip install -q -e . --no-deps 2>&1 | tail -1
echo "== installing loco_mjx (no deps) =="
cd /ptmp/akalenik/urma/loco_mjx && pip install -q -e . --no-deps 2>&1 | tail -1
echo "== import check (CPU ok on login node) =="
taskset -c 0-7 python - <<'EOF'
import jax
print("jax", jax.__version__)
import mujoco, flax, optax
print("mujoco", mujoco.__version__)
import rl_x
print("rl_x ok")
EOF
echo SETUP_DONE
