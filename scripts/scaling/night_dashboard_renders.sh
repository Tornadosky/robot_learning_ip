#!/usr/bin/env bash
# Renders for the 2026-08-30 overnight dashboard:
#   1. L1 l3t_fix        -- one policy, H1+G1+T1 (the C4 artifact)
#   2. v1h10_ref_s1      -- reference-only at hold=10 (collapses, 0.336)
#   3. v1h10_tok_s1      -- token at hold=10, tk4 routing (holds, 0.222)
# Dumps via crosseval_motion.py --dump_render (exact eval configs from the
# night), then rf_render_dance2.py -> mp4 (policy beside reference ghost).
set -u
REPO=/mnt/c/Users/smirn/Desktop/robot_learning_ip
PY=~/jaxgpu/bin/python
E=$REPO/experiments/fsq_khaendler
LAF=$REPO/external_data/amass_converted/LAFAN1
L3T=$REPO/external_data/amass_converted/LAFAN1_3t
TOKLOC=$E/tokentest_local
MEDIA=$E/night_media
DUMPS=$MEDIA/dumps
mkdir -p "$MEDIA" "$DUMPS"

# tokentest mirror: ORIGINAL npz + the kevin _zq the V1 campaign trained on.
for r in UnitreeH1 UnitreeG1; do
  mkdir -p "$TOKLOC/$r"
  cp -n "$LAF/$r/dance2_subject4.npz" "$TOKLOC/$r/"
  cp -n "$E/clips_kevin/$r/dance2_subject4_zq.npz" "$TOKLOC/$r/"
done
md5sum "$TOKLOC"/*/dance2_subject4_zq.npz

M_L1=$(ls -t "$REPO"/loco_mjx/experiments/runs/local_3t/l3t_fix/*/models/latest.model | head -1)
M_REF=$E/viper_models/v1h10_ref_s1/latest.model
M_TOK=$E/viper_models/v1h10_tok_s1/latest.model

cd "$E"
export PYTHONPATH="$REPO:$REPO/RL-X:$REPO/loco_mjx"
export MUJOCO_GL=disable XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
export JAX_ENABLE_COMPILATION_CACHE=true
export JAX_COMPILATION_CACHE_DIR=$REPO/.jax_cache_local

echo "[dump] L1 3-topology $(date -Is)"
$PY crosseval_motion.py \
  --model_path "$M_L1" \
  --clip_dir "$L3T" --clip dance2_subject4.npz --raw_clip_dir "$L3T" \
  --robots unitree_h1:unitree_g1:booster_t1 \
  --refbias 0.0 --anchor absolute --fitvariant False \
  --refroot True --refroot_floor True --refvel_obs False \
  --root_heading_obs False \
  --nr_envs 96 --steps 1000 --seed 0 \
  --dump_render "$DUMPS/l1" --record_envs 2 \
  --out "$DUMPS/l1_render_eval.json" > "$DUMPS/l1_dump.log" 2>&1
echo "  rc=$? $(ls "$DUMPS"/l1__*.npz 2>/dev/null | wc -l) dumps"

echo "[dump] v1h10_ref $(date -Is)"
$PY crosseval_motion.py \
  --model_path "$M_REF" \
  --clip_dir "$TOKLOC" --clip dance2_subject4.npz --raw_clip_dir "$LAF" \
  --robots unitree_h1:unitree_g1 \
  --refbias 0.0 --anchor absolute --fitvariant False \
  --refroot True --refroot_floor True --refvel_obs False \
  --reference_hold 10 \
  --nr_envs 96 --steps 1000 --seed 0 \
  --dump_render "$DUMPS/h10ref" --record_envs 2 \
  --out "$DUMPS/h10ref_render_eval.json" > "$DUMPS/h10ref_dump.log" 2>&1
echo "  rc=$? $(ls "$DUMPS"/h10ref__*.npz 2>/dev/null | wc -l) dumps"

echo "[dump] v1h10_tok $(date -Is)"
$PY crosseval_motion.py \
  --model_path "$M_TOK" \
  --clip_dir "$TOKLOC" --clip dance2_subject4.npz --raw_clip_dir "$LAF" \
  --robots unitree_h1:unitree_g1 \
  --refbias 0.0 --anchor absolute --fitvariant False \
  --refroot True --refroot_floor True --refvel_obs False \
  --reference_hold 10 \
  --latent --latent_dim 32 --latent_scope per_joint --latent_divisor 10.0 \
  --latent_replaces False --jlat_enc_dim 4 \
  --nr_envs 96 --steps 1000 --seed 0 \
  --dump_render "$DUMPS/h10tok" --record_envs 2 \
  --out "$DUMPS/h10tok_render_eval.json" > "$DUMPS/h10tok_dump.log" 2>&1
echo "  rc=$? $(ls "$DUMPS"/h10tok__*.npz 2>/dev/null | wc -l) dumps"

echo "[render] $(date -Is)"
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
render l1 unitree_h1 unitree_h1 l1_3t_unitree_h1.mp4
render l1 unitree_g1 unitree_g1 l1_3t_unitree_g1.mp4
render l1 booster_t1 booster_t1 l1_3t_booster_t1.mp4
render h10ref unitree_h1 unitree_h1 h10_ref_unitree_h1.mp4
render h10ref unitree_g1 unitree_g1 h10_ref_unitree_g1.mp4
render h10tok unitree_h1 unitree_h1 h10_tok_unitree_h1.mp4
render h10tok unitree_g1 unitree_g1 h10_tok_unitree_g1.mp4
echo "[render] ALL DONE $(date -Is)"
ls -la "$MEDIA"/*.mp4 2>/dev/null
