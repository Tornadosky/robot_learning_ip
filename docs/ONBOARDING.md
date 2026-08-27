# Onboarding — from `git clone` to a trained policy

Audience: a colleague joining the FSQ / URMA2 DeepMimic work (state as of 2026-08-27).
The experiment reports are under `experiments/fsq_khaendler/REPORT_*.md` (read
`REPORT_FSQ_WAVE2.md` last-first — each report supersedes recipe details of the previous one).

## 1. What this repo actually is

This is a **superproject**: the experiment record, launchers, tokenizers and reports live
here; the training stack itself lives in submodules.

| path | what it is |
|---|---|
| `loco_mjx/` (submodule, branch `tracking-phase1`) | **The real training stack.** URMA2 (topology-agnostic policy) on MJX, plus our additions: the DeepMimic tracking port (`.../urma2/mjx/command_functions/tracking_clip.py`, `reward_functions/tracking.py`, `clip_reference.py`) and the FSQ token seam (`tracking_clip_latent_obs`). Nothing trains without this submodule at the pinned commit. |
| `loco-mujoco/` (submodule, group repo, branch `integration`) | Used for **offline retargeting** of mocap onto robots. NOTE: the `integration` branch currently tracks no files under `loco_mujoco/trajectory/`, so local retargeting scripts do not import — get clips from a teammate/cluster instead (see §3). |
| `RL-X/` (submodule) | The RL framework URMA2 runs on. |
| `experiments/urma2_h1g1/run.sh` | **The main launcher.** Stage-based (`smoke`, `mmtrain`, `mmsplit`, …); every reward/env knob is an env var with a documented default. |
| `experiments/fsq_khaendler/` | The FSQ campaign: per-arm launchers (`run_arm.sh`), Viper sbatch files, crossevals (`ce_*/`), reports, dashboards, tokenizer outputs. |
| `scripts/scaling/khaendler_fsq_clip.py`, `canonical_fsq_clip.py` | The two tokenizer trainers (per-joint / canonical). `canonical_zq_sidecar.py`, `scramble_zq.py`, `stitch_novel_clip.py`, `derive_clip_signs.py` are the supporting tools. |
| `docs/notes/` | **Gitignored.** Local design notes, goal docs and supervisor summaries — not in the clone. |
| `external_data/`, `videos/`, `experiments/*/media` | **Gitignored.** Clips, checkpoints and renders are not in git. |

Older top-level scripts (`run_multimotion.sh`, `run_overnight.sh`, `scripts/overnight/`)
belong to the pre-URMA2 loco-mujoco trainer — historical, do not start there.

## 2. Clone + environment

```bash
git clone --recurse-submodules https://github.com/Tornadosky/robot_learning_ip.git
cd robot_learning_ip
```

You need read access to the submodule remotes (`loco_mjx`, `loco-mujoco` group repo, `RL-X`).
`loco_mjx` must sit on the pinned commit of branch `tracking-phase1` — if
`git submodule status` shows a different commit, run `git submodule update --init`.

Environment (everything below assumes **Linux or WSL**; every launcher is bash and the
training boxes are WSL + CUDA or the cluster + ROCm):

```bash
python3.12 -m venv ~/locomjx        # plain venv. On Windows do NOT use conda —
source ~/locomjx/bin/activate       # conda's mujoco DLL shadows the pip one.
pip install -e RL-X                 # RL framework
pip install -e loco_mjx             # the stack (see loco_mjx/README.md)
pip install -U "jax[cuda12]"        # match your CUDA; MJX needs GPU jax
```

Sanity check: `python -c "import jax; print(jax.default_backend())"` must say `gpu`.

## 3. Data — you cannot train without it

Training needs **retargeted clips** at
`external_data/amass_converted/LAFAN1/<Robot>/<clip>.npz`
(Robot ∈ `UnitreeH1`, `UnitreeG1`, `BoosterT1`, …). These are gitignored. Options:

1. **Copy them** from a teammate or from the cluster (`clips/LAFAN1/` on Viper). This is
   the normal path — the set is a few hundred MB.
2. Retarget yourself via loco-mujoco — currently broken on the `integration` branch
   (see §1), so treat this as unavailable until that branch carries the trajectory code.

The multi-motion references are concatenated "super-clips" (`super5dance.npz` = 5 dances,
`superM9.npz` = 4 dances + 5 walks) living in the same directory layout. FSQ arms
additionally need the token sidecars (`<clip>_zq.npz`) produced by the tokenizer (§5).

**Adding a new robot family?** Run `scripts/scaling/derive_clip_signs.py` against its clip
FIRST. Three families have burned a training run each on silently reversed joint axes
(13/19 joints on H1, 15/23 on G1, 14/23 on T1); the script screens a family in a minute
and derives its `CLIP_SIGNS` table. Also add the robot to `tracking_clip_robot_map` in
`urma2/mjx/default_config.py` (short name: `t1`, not `booster_t1`).

## 4. Training — your options

### Option A: local GPU (WSL), single arm — start here

`experiments/urma2_h1g1/run.sh` is the launcher. **Edit the two hardcoded paths at the
top first** (`REPO=/mnt/c/...`, `VENV=$HOME/locomjx`) — they are machine-specific.

```bash
bash experiments/urma2_h1g1/run.sh smoke     # 4 envs, seconds: proves import/compile/GPU
```

Then a real DeepMimic arm. `mmtrain` is the stage used by every recent campaign; all
knobs are env vars (read the case block in `run.sh` — each var is documented inline).
The current best-known recipe (the "ladder" recipe, 2026-08-25) on one H1:

```bash
ROBOTS_LIST=unitree_h1 \
CLIP_FILE=dance2_subject4.npz \
ANCHOR=absolute FITVARIANT=False REFBIAS=0.0 \
TRACK_TEMP=0.05 DEVRATIO=0.5 \
GAITMODE=floor GAITCOEFF=0.25 GROUNDPEN=1000 FOOTSLIP=20 POSTCONTACT=True \
NR_ENVS=1024 TOTAL_STEPS=98304000 EXP_NAME=my_first_arm \
bash experiments/urma2_h1g1/run.sh mmtrain
```

Why those: `ANCHOR=absolute` (the 08-23 anchor fix — `centered` scores against a
mis-anchored reference), `TRACK_TEMP=0.05` (the one lever that improves leg tracking;
the old "never below 0.25" comment in the config is measured-false), `GAITMODE=floor`
+ the high contact dose (without it foot terms multiply to zero — the 08-24 bug).
`experiments/fsq_khaendler/run_arm.sh` wraps exactly this with a GPU lock for queueing
several arms; `viper_submit_wave2.sh` records the exact env of every wave-2 arm.

Budget feel: ~11.6M steps/hour on an RTX 4060 Ti; the standard arm is 98.3M ≈ 8.5 h.
First compile of a topology is ~4–10 min; the launcher enables the JAX persistent
compilation cache so identical re-runs skip it.

### Option B: the MPCDF Viper cluster (MI300A, ROCm)

`experiments/urma2_h1g1/viper_train.sbatch` + `experiments/fsq_khaendler/viper_submit_*.sh`.
ROCm specifics that will each cost you a run if unknown:

- Submit from viper11–13 only; `MUJOCO_GL=disable`.
- ≤768 envs (1024 dies); long runs need `--xla_gpu_enable_command_buffer=` in flags.
- `/ptmp` is **per-node** (viper01 sees a different ptmp than viper11).
- `minibatch_size % nr_train_robots == 0` is asserted — the default 8192 breaks at 3
  robots; use `MINIBATCH=6144` (divides 1,2,3,4,6). This assert masqueraded as a
  "3-topology memory wall" for weeks.
- Verify a completed job by its artifacts, not its exit code; drive ssh by piping a
  script file over stdin (`tr -d '\r'`), never inline `$(...)` (it evaluates locally).

### Option C: FSQ arms (token-conditioned policies)

1. Train a tokenizer, or reuse the shipped fits under `experiments/fsq_khaendler/`
   (`tokenizer*/`, `canonical_tokenizer*/`):
   `python scripts/scaling/khaendler_fsq_clip.py train --robots UnitreeH1 UnitreeG1 --clip <clip>.npz ...`
2. `reconstruct` writes reconstructed clips + `_zq.npz` token sidecars **to a new
   directory** — never point it at the original clip dir (it used to overwrite
   references in place; fixed, but keep the habit).
3. Launch the same `mmtrain` with the latent seam switched on
   (`tracking_clip_latent_obs`: reference / token / both). The wave-2 submit scripts
   (`viper_submit_wave2.sh`) are the reference for the exact flags of every arm family
   (`M9_ref` / `M9_z` / `M9_both`, holds, canonical).

## 5. Evaluation — the part people get wrong

Training curves **cannot** distinguish the arms (return/ep-len saturate identically).
The only meaningful number is the offline crosseval against the raw clip:

```bash
python experiments/fsq_khaendler/crosseval_motion.py <checkpoint> ...   # see ce_* dirs for examples
```

Rules (each one is a lesson that invalidated an earlier claim):

1. **n ≥ 4 rollout seeds.** A single seed moves the score by up to 4.95% — the size of
   every FSQ effect we study. Seed via `CE_SEED`; a crosseval is ~15 min.
2. **Always run the zero-action floor** (same eval, robot does nothing) as denominator.
3. Compare like with like: local and Viper crossevals are not interchangeable; when
   pooling JSONs, filter on `eval_condition["clip"]` (arm names alone mix clips).
4. Self-referenced metrics (return, ep-len vs the arm's *own* smoothed reference) flatter
   FSQ arms; only executed-vs-raw-clip RMSE counts.
5. Known blind spots of all current numbers: heading is unscored by reward and was
   until 08-27 unscored by the crosseval (every arm ends ~82° off), and foot-height
   tracking is off by default.

Aggregation/plots: `experiments/fsq_khaendler/plot_wave2.py`, `build_*dashboard.py`,
`tools/agg_wave2.py` (Viper side); curves are scraped to `curves_all.csv` (local/cluster
artifact, not in git).

## 6. Where to read what happened so far

Chronological: `experiments/urma_fsq_overnight_20260813/RESULTS.md` → `experiments/fsq_khaendler/REPORT_2026-08-22.md` →
`REPORT_overnight_fix.md` → `REPORT_feet_fsq.md` → `REPORT_ladder.md` →
`REPORT_FSQ_SCALE.md` → `REPORT_FSQ_WAVE2.md` (newest).
