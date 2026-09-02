#!/usr/bin/env bash
# Render the NIGHT3 local 3-robot pair (ln3_ref / ln3_tok) from the eval dumps.
# Renders need the ~/locomjx venv (jaxgpu's EGL is broken -- confirmed again
# tonight: eglDestroyContext error on every mp4), per night_dashboard_render2.sh.
set -u
REPO=/mnt/c/Users/smirn/Desktop/robot_learning_ip
PY=~/locomjx/bin/python
E=$REPO/experiments/fsq_khaendler
DUMPS=$REPO/experiments/local_3t/renders_ln3
MEDIA=$REPO/experiments/local_3t/media_ln3
mkdir -p "$MEDIA"
cd "$E"
export MUJOCO_GL=egl

render() {  # render <dumpbase> <robot> <xmlrobotdir> <outname>
  local base="$1" robot="$2" xdir="$3" out="$4"
  local npz="$DUMPS/${base}__${robot}.npz"
  [ -f "$npz" ] || { echo "  MISSING $npz"; return 1; }
  $PY rf_render_dance2.py --npz "$npz" \
    --xml "$REPO/loco_mjx/loco_mjx/environments/robots/$xdir/data/plane.xml" \
    --out "$MEDIA/$out" --width 960 --height 640 --stride 2 --max_frames 450 \
    > "$DUMPS/render_${out%.mp4}.log" 2>&1
  [ -s "$MEDIA/$out" ] && echo "  OK $out" || echo "  FAIL $out (see log)"
}
for arm in ln3_ref ln3_tok; do
  render "$arm" unitree_h1 unitree_h1 "${arm}_unitree_h1.mp4"
  render "$arm" unitree_g1 unitree_g1 "${arm}_unitree_g1.mp4"
  render "$arm" booster_t1 booster_t1 "${arm}_booster_t1.mp4"
done
echo "[ln3render] ALL DONE $(date -Is)"
ls -la "$MEDIA"/*.mp4 2>/dev/null
