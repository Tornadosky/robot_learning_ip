#!/usr/bin/env bash
# Render half of night_dashboard_renders.sh, retried with the ~/locomjx venv
# (render_ab.sh's proven EGL path) after jaxgpu's EGL bindings failed.
set -u
REPO=/mnt/c/Users/smirn/Desktop/robot_learning_ip
E=$REPO/experiments/fsq_khaendler
MEDIA=$E/night_media
DUMPS=$MEDIA/dumps
PY=~/locomjx/bin/python
export MUJOCO_GL=egl
render() {
  local base="$1" robot="$2" xdir="$3" out="$4"
  local npz="$DUMPS/${base}__${robot}.npz"
  [ -f "$npz" ] || { echo "  MISSING $npz"; return 1; }
  $PY "$E/rf_render_dance2.py" --npz "$npz" \
    --xml "$REPO/loco_mjx/loco_mjx/environments/robots/$xdir/data/plane.xml" \
    --out "$MEDIA/$out" --width 960 --height 640 --stride 2 --max_frames 450 \
    > "$DUMPS/render2_${out%.mp4}.log" 2>&1
  [ -s "$MEDIA/$out" ] && echo "  OK $out ($(stat -c%s "$MEDIA/$out") bytes)" || echo "  FAIL $out"
}
render l1 unitree_h1 unitree_h1 l1_3t_unitree_h1.mp4
render l1 unitree_g1 unitree_g1 l1_3t_unitree_g1.mp4
render l1 booster_t1 booster_t1 l1_3t_booster_t1.mp4
render h10ref unitree_h1 unitree_h1 h10_ref_unitree_h1.mp4
render h10ref unitree_g1 unitree_g1 h10_ref_unitree_g1.mp4
render h10tok unitree_h1 unitree_h1 h10_tok_unitree_h1.mp4
render h10tok unitree_g1 unitree_g1 h10_tok_unitree_g1.mp4
echo "RENDER2 DONE"
