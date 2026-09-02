# Second machine (RTX 5080) — runbook

Everything the training/eval/render stack needs, in the order to do it.
The primary box is Windows 11 + WSL2 Ubuntu with the repo under
`/mnt/c/Users/<you>/Desktop/robot_learning_ip`; keep that layout (the launch
scripts default to it; override with `REPO=... PY=...`).

## 1. Code (5 min)

```
git clone https://github.com/Tornadosky/robot_learning_ip.git
cd robot_learning_ip
bash handoff/bootstrap_second_machine.sh $PWD      # inside WSL
```
This clones the three submodules from their upstream remotes and fast-forwards
them to our local commits from `handoff/*.bundle` (the upstreams do not have
them): `loco_mjx` → `tracking-phase1` (wave-6 aux head, co-training, leg
kernel, routing fix), `loco-mujoco` → `integration` (khaendler's autoencoder +
the retarget-feasibility configs), `RL-X` → `local`. Then it creates
`~/jaxgpu` (train/eval; jax 0.7.1 + CUDA 12 plugin, mujoco 3.7.0, flax 0.12.0)
and `~/locomjx` (render, `MUJOCO_GL=egl`) from `handoff/requirements_*.txt`.

Windows side: `C:\Users\<you>\.wslconfig` must contain
```
[wsl2]
memory=24GB
swap=16GB
```
(a 3-robot compile OOM-kills WSL at the default 50 % of RAM). Never run two
JAX processes at once inside WSL: three concurrent compiles restarted the VM.

## 2. Data (copy over LAN or a USB disk; ~3.4 GB minimum)

| path (relative to repo) | size | needed for |
|---|---|---|
| `external_data/amass_converted/LAFAN1/` | 812 M | source clips WITH foot sites (H1 11, G1 10, T1/Atlas/Talos/ToddlerBot/H1v2 few) |
| `external_data/amass_converted/LAFAN1_3t/` | 277 M | every 3-robot run (H1 + G1 from LAFAN1, T1 from LAFAN1_fixed) |
| `external_data/amass_converted/LAFAN1_fixed/` | 78 M | feasibility-fixed T1/Atlas/Talos/ToddlerBot clips (no foot sites) |
| `external_data/amass_converted/smpl_source_fits/`, `AMASS/` | 35 M | retargeting (skips the torch source fits) |
| `external_data/smpl/` | 591 M | retargeting only |
| `experiments/fsq_khaendler/tokenizer_3t_v2/` (+ other `tokenizer*`) | 80 M | codec: `params.msgpack`, `config.json`, `descriptions.npz` |
| `experiments/fsq_khaendler/clips_3t_v2/` | 463 M | dance2_subject4 + walk1 for H1/G1/T1 with `_zq` and `_win` sidecars |
| `experiments/fsq_khaendler/clips_super/` | 583 M | five-dance super-clip (H1/G1) with `_zq` and `_win` |
| `viper_mirror/crosseval/` (optional) | 40 M | every Viper cross-eval JSON, for the ledger |

`loco-mujoco/loco_mujoco/LOCOMUJOCO_VARIABLES.yaml` holds machine paths; rerun
`python scripts/setup_data.py` or edit it. Retargeting (`reissue_clips.py`,
`khaendler_fsq_clip.py`) runs in the Windows `.venv` with torch CUDA; see
`docs/ONBOARDING.md` §2.

## 3. Run one arm (the only launcher you need)

`scripts/scaling/wave6/local_train.sh` — every knob is an environment variable.
```
cd /mnt/c/Users/<you>/Desktop/robot_learning_ip
export REPO=$PWD
# reference-only control, H1+G1, 19.66M steps (≈ 50 min on a 4060 Ti at 768 envs)
NAME=ref_s1 CLIPDIR=$REPO/experiments/fsq_khaendler/clips_3t_v2 LATENT=0 SEED=1 bash scripts/scaling/wave6/local_train.sh
# precomputed token with its own projection
NAME=split_s1 CLIPDIR=... LATENT=1 JLAT_CH=-1 SEED=1 bash scripts/scaling/wave6/local_train.sh
# SONIC-style co-training (encoder + FSQ inside the policy, init from the tokenizer)
NAME=cot_s1 CLIPDIR=... LATENT=1 LATENT_DIM=44 LATENT_DIVISOR=1.0 SIDECAR=_win \
  COTRAIN_ROWS=11 COTRAIN_CH=4 COTRAIN_INIT=$REPO/experiments/fsq_khaendler/tokenizer_3t_v2/params.msgpack \
  SEED=1 bash scripts/scaling/wave6/local_train.sh
# three topologies (local only; Viper faults on >2), 192 envs, minibatch must divide by the robot count
NAME=cot3_s1 ROBOTS=unitree_h1:unitree_g1:booster_t1 NR_ENVS=192 MINIBATCH=6144 CLIPDIR=... <token vars> bash scripts/scaling/wave6/local_train.sh
```
Other knobs: `HOLD` (reference hold during training), `TOTAL` (steps),
`MORPH_MODE=schedule MORPH_COEFF=0.7 MORPH_START=0.2 MORPH_RAMP=15000000`
(body randomization; default here is `fixed 0.0`, Viper's default is the
schedule), `LEGW` (leg kernel), `AUX_COEFF/AUX_HORIZON`, `CLIP=<file>`, `EXTRA`.
Checkpoints: `loco_mjx/experiments/runs/<PROJECT>/<NAME>/<ts>/models/latest.model`.
Logs are console tables; `grep nr_env_steps` counts iterations (400 = 19.66M).

## 4. Evaluate (cross-eval) and render

```
cd experiments/fsq_khaendler && export PYTHONPATH=$REPO:$REPO/RL-X:$REPO/loco_mjx MUJOCO_GL=disable \
  XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_command_buffer= \
  JAX_ENABLE_COMPILATION_CACHE=true JAX_COMPILATION_CACHE_DIR=$REPO/.jax_cache_local
~/jaxgpu/bin/python crosseval_motion.py --model_path <latest.model> --clip_dir <CLIPDIR> --raw_clip_dir <CLIPDIR> \
  --clip dance2_subject4.npz --refbias 0.0 --anchor absolute --fitvariant False --refroot True --refroot_floor True \
  --refvel_obs False --root_heading_obs True --contact_timeconst 0.004 --robots unitree_h1:unitree_g1 \
  --nr_envs 64 --steps 1000 --seed 0 --reference_hold 1 --out out.json \
  [token: --latent --latent_dim 32 --latent_divisor 10.0 --jlat_enc_dim 4]  [co-training: --latent --latent_dim 44 --latent_divisor 1.0 --latent_sidecar _win --jlat_enc_dim 4]
```
The eval MUST mirror the arm's observation layout (these flags), or it scores
garbage silently. Add `--dump_render <dir> --record_envs 2` to dump, then
`MUJOCO_GL=egl ~/locomjx/bin/python rf_render_dance2.py --npz <dump>__unitree_g1.npz --xml $REPO/loco_mjx/loco_mjx/environments/robots/unitree_g1/data/plane.xml --out x.mp4 --width 800 --height 534 --stride 2 --max_frames 1100`.
`scripts/scaling/wave6/local_queue_w6.sh` and `media_w6.sh` are the worked
examples (sequential queue with cross-evals; dump + render batch).

## 5. Tokenizer and sidecars (when the clip set changes)

```
.venv/Scripts/python.exe scripts/scaling/khaendler_fsq_clip.py train --robots UnitreeH1 UnitreeG1 BoosterT1 \
   --clip <clips...> --heldout-clips <held-out clips> --clip-dir external_data/amass_converted/LAFAN1_3t \
   --out experiments/fsq_khaendler/tokenizer_<name> --foot-channels --epochs 250 --lookahead 10 --latent-dim 32 --fsq-levels 8
.venv/Scripts/python.exe scripts/scaling/khaendler_fsq_clip.py reconstruct --tokenizer <tok> --clip-dir <dir> --robots ... --clip ... --out <dir>_rec   # writes _zq
python scripts/scaling/wave6/emit_window_sidecars.py --tokenizer <tok> --clip-dir <dir> --robots ... --clip ...                           # writes _win
python experiments/fsq_khaendler/build_superclip.py --clip-dir <dir> --out-dir <superdir> --name superN.npz --robots ... --clips ...       # multi-motion clip
```
Gate a tokenizer on held-out joint-angle RMSE (`reconstruction_report.json`),
never on its training loss (97 % velocity).
