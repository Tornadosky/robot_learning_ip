#!/usr/bin/env bash
# Fetch the night3/4 Viper render dumps and render the presentation set:
#   n3a4g_{ref,tok}   super-clip (5 dances) + gait dose, hold=1  -- "does it dance"
#   n3gb33h1_{ref,tok} single dance2_subject4 + gait dose, hold=1
#   n4air45_{ref,tok} air-time-coeff 45 attempt (latest gait try)
# locomjx venv + EGL (jaxgpu EGL broken).
set -u
REPO=/mnt/c/Users/smirn/Desktop/robot_learning_ip
PY=~/locomjx/bin/python
E=$REPO/experiments/fsq_khaendler
DUMPS=$REPO/experiments/local_3t/renders_n34
MEDIA=$REPO/experiments/local_3t/media_n34
mkdir -p "$DUMPS" "$MEDIA"

scp -q "viper11:/ptmp/akalenik/urma/renders_night3/*.npz" "$DUMPS/" || { echo FETCH FAILED; exit 1; }
ls "$DUMPS" | wc -l

cd "$E"
export MUJOCO_GL=egl
render() {  local base="$1" robot="$2" out="${1}_${2}.mp4"
  local npz="$DUMPS/${base}_s1_dump__${robot}.npz"
  [ -f "$npz" ] || npz="$DUMPS/${base}__${robot}.npz"
  [ -f "$npz" ] || { echo "  MISSING $base $robot"; return 1; }
  $PY rf_render_dance2.py --npz "$npz" \
    --xml "$REPO/loco_mjx/loco_mjx/environments/robots/$robot/data/plane.xml" \
    --out "$MEDIA/$out" --width 960 --height 640 --stride 2 --max_frames 450 \
    > "$DUMPS/render_${base}_${robot}.log" 2>&1
  [ -s "$MEDIA/$out" ] && echo "  OK $out" || echo "  FAIL $out"
}
for b in n3a4g_ref n3a4g_tok n3gb33h1_ref n3gb33h1_tok n4air45_ref n4air45_tok; do
  render "$b" unitree_h1
  render "$b" unitree_g1
done
echo "[n34render] DONE $(date -Is)"
