#!/usr/bin/env bash
# Videos for the W2 update of the dashboard:
#   w2rep    -- TOKEN-ONLY (no reference in obs) at hold=1
#   w2h20ref -- reference-only at hold=20 (collapses, 0.376 H1)
#   w2h20tok -- token at hold=20 (holds, 0.226 H1)
# All three trained on the practical recipe (heading OBSERVED -> obs width!)
# with the mc sidecars (clips/tokentest_mc), so the local eval mirrors both.
set -u
REPO=/mnt/c/Users/smirn/Desktop/robot_learning_ip
PY=~/jaxgpu/bin/python
E=$REPO/experiments/fsq_khaendler
LAF=$REPO/external_data/amass_converted/LAFAN1
TOKMC=$E/tokentest_local_mc
MEDIA=$E/night_media
DUMPS=$MEDIA/dumps
mkdir -p "$MEDIA" "$DUMPS"

for r in UnitreeH1 UnitreeG1; do
  mkdir -p "$TOKMC/$r"
  cp -n "$LAF/$r/dance2_subject4.npz" "$TOKMC/$r/"
  cp -n "$E/clips_kevin_mc/$r/dance2_subject4_zq.npz" "$TOKMC/$r/"
done
md5sum "$TOKMC"/*/dance2_subject4_zq.npz

cd "$E"
export PYTHONPATH="$REPO:$REPO/RL-X:$REPO/loco_mjx"
export MUJOCO_GL=disable XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
export JAX_ENABLE_COMPILATION_CACHE=true
export JAX_COMPILATION_CACHE_DIR=$REPO/.jax_cache_local

dump() {  # dump <tag> <model-dir> <hold> <latentmode: none|tok|rep>
  local tag="$1" mdir="$2" hold="$3" lm="$4"
  local args=(
    --model_path "$E/viper_models/$mdir/latest.model"
    --clip_dir "$TOKMC" --clip dance2_subject4.npz --raw_clip_dir "$LAF"
    --robots unitree_h1:unitree_g1
    --refbias 0.0 --anchor absolute --fitvariant False
    --refroot True --refroot_floor True --refvel_obs False
    --root_heading_obs True --reference_hold "$hold"
    --nr_envs 96 --steps 1000 --seed 0
    --dump_render "$DUMPS/$tag" --record_envs 2
    --out "$DUMPS/${tag}_render_eval.json"
  )
  case "$lm" in
    tok) args+=(--latent --latent_dim 32 --latent_scope per_joint --latent_divisor 10.0 --latent_replaces False --jlat_enc_dim 4);;
    rep) args+=(--latent --latent_dim 32 --latent_scope per_joint --latent_divisor 10.0 --latent_replaces True  --jlat_enc_dim 4);;
  esac
  echo "[dump] $tag $(date -Is)"
  $PY crosseval_motion.py "${args[@]}" > "$DUMPS/${tag}_dump.log" 2>&1
  echo "  rc=$? $(ls "$DUMPS/${tag}"__*.npz 2>/dev/null | wc -l) dumps"
}
dump w2rep    w2rep_h1_s1   1  rep
dump w2h20ref w2h20_ref_s1  20 none
dump w2h20tok w2h20_tok_s1  20 tok

echo "[render] $(date -Is)"
RPY=~/locomjx/bin/python
export MUJOCO_GL=egl
render() {
  local base="$1" robot="$2" out="$3"
  local npz="$DUMPS/${base}__${robot}.npz"
  [ -f "$npz" ] || { echo "  MISSING $npz"; return 1; }
  $RPY "$E/rf_render_dance2.py" --npz "$npz" \
    --xml "$REPO/loco_mjx/loco_mjx/environments/robots/$robot/data/plane.xml" \
    --out "$MEDIA/$out" --width 960 --height 640 --stride 2 --max_frames 450 \
    > "$DUMPS/render2_${out%.mp4}.log" 2>&1
  [ -s "$MEDIA/$out" ] && echo "  OK $out ($(stat -c%s "$MEDIA/$out") bytes)" || echo "  FAIL $out"
}
render w2rep    unitree_h1 w2_tokenonly_unitree_h1.mp4
render w2rep    unitree_g1 w2_tokenonly_unitree_g1.mp4
render w2h20ref unitree_h1 w2_h20_ref_unitree_h1.mp4
render w2h20tok unitree_h1 w2_h20_tok_unitree_h1.mp4
echo "W2 RENDERS DONE $(date -Is)"
