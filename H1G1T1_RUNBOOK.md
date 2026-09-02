# H1 + G1 + T1 shared-policy diagnostic runbook

This overlay launches one `urma2.mjx` policy for Unitree H1, Unitree G1, and Booster T1. Each family tracks its retarget of the same `dance2_subject4.npz` motion. The run enables reset-time seen-body morphology randomization, DeepMimic-style pose/velocity/body/foot rewards, and an explicit observable root-heading target.

The supplied project archive is a diagnostic snapshot: it omits the full robot assets, full motion files, installed environment, and GPU runtime. Install this overlay into the full repository you already use for training.

## Command to run

From the extracted diagnostic bundle:

```bash
REPO=/mnt/c/Users/smirn/Desktop/robot_learning_ip \
PY=$HOME/jaxgpu/bin/python \
bash run.sh full
```

`run.sh` backs up any replaced files, installs the overlay, runs preflight checks, trains, evaluates, generates plots, and creates a compact return ZIP.

The default geometry is:

- 576 total environments, exactly 192 per family;
- 64 simulation steps per rollout;
- 36,864 samples per PPO batch;
- 6,144 samples per minibatch, preserving all three families in every minibatch;
- 5 PPO epochs;
- 117,964,800 total global environment steps;
- a checkpoint every 11,796,480 steps, for 10 checkpoint intervals.

Morphology magnitude ramps from 0.0 to 0.3 during the first 39,321,600 global steps and then stays at 0.3. Randomized bodies are sampled at episode reset. `sampling_probability=0.0` prevents mid-episode body mutation.

The trainer is pinned to one visible accelerator by default because this monolithic `urma2.mjx` path has device-sensitive batch arithmetic. Override `CUDA_VISIBLE_DEVICES` only when deliberately testing another single device.

## What to return

At the end, the launcher prints:

```text
RESULT_PACKAGE=/.../experiments/h1g1t1_debug/<RUN_ID>_diagnostics.zip
```

Upload that ZIP. Do not upload the entire repository. If preflight, training, evaluation, or reporting fails during `full`, the exit trap still creates a partial archive and prints `FAILURE_RESULT_PACKAGE=...`; upload that file instead. The compact package includes:

- the exact effective train and evaluation commands;
- recipe JSON, Python version, package freeze, GPU report, OS report, Git commit/status/diff when available;
- full-clip integrity checks for all three retargets;
- the complete console log and TensorBoard event files;
- raw parsed metrics plus derived joint RMSE, qvel RMSE, and heading error in degrees;
- per-family joint, velocity, heading, DeepMimic body, foot, morphology, optimizer, and secondary return/survival plots;
- nominal-policy crosseval, morphology 0.15 crosseval, morphology 0.30 crosseval, and a nominal zero-action baseline;
- absolute and mean-centered joint RMSE, per-joint RMSE, reference-versus-raw convention floor, heading mean/p95, and alive fraction;
- nominal rollout NPZ files for visual inspection or rendering;
- checkpoint inventory, effective checkpoint config, training progress, sizes, and SHA-256 checksums.

Model weights are excluded by default. For a later policy-level investigation, package only the latest model with:

```bash
REPO=/mnt/c/Users/smirn/Desktop/robot_learning_ip \
PY=$HOME/jaxgpu/bin/python \
RUN_ID=<same run id> INCLUDE_MODEL=1 \
bash run.sh package
```

## Useful stages

```bash
bash run.sh dry-run     # print exact configuration and validated arithmetic
bash run.sh preflight   # check repository, dependencies, clips, T1 sign fix, arithmetic
bash run.sh smoke       # compile and execute a small 24,576-step three-family run
bash run.sh train       # preflight, train, verify exact final step and final checkpoint
bash run.sh evaluate    # nominal, morphology 0.15/0.30, and zero-action crosseval
bash run.sh report      # parsed CSV/JSON/Markdown diagnostics and plots
bash run.sh package     # compact result ZIP
bash run.sh full        # train + evaluate + report + package
```

To rerun `evaluate`, `report`, or `package`, set the original `RUN_ID` so the launcher resolves the same runner and result directories.

## Lower-memory geometries

The default reproduces the effective 36,864-step increment in the archived run while making the meaning explicit: 576 total environments, 192 per family.

A lower-memory 192-environment run is valid:

```bash
REPO=... PY=... \
NR_ENVS=192 MINIBATCH=6144 \
SAVE_EVERY=9830400 TOTAL=98304000 \
bash run.sh full
```

This gives 64 environments per family, a 12,288-sample batch, two minibatches per epoch, 800 PPO updates per checkpoint interval, and 10 intervals.

A smaller diagnostic run is:

```bash
REPO=... PY=... \
NR_ENVS=96 MINIBATCH=3072 \
SAVE_EVERY=4915200 TOTAL=49152000 \
bash run.sh full
```

This gives 32 environments per family and retains exact divisibility. Do not use 128 or 256 total environments for three families: neither divides evenly by three. The preflight rejects invalid environment, minibatch, checkpoint, and total-step arithmetic before JAX compilation.

## How to judge the run

Return and episode length are secondary context. The primary verdict is based on:

1. per-family online joint RMSE, qvel RMSE, and root-heading error;
2. policy foot-airborne fraction against the reference fraction;
3. nominal absolute crosseval RMSE versus the zero-action baseline and reference-versus-raw floor;
4. degradation from nominal to morphology 0.15 and 0.30;
5. per-joint failure concentration, especially fresh Booster T1 results after the sign fix;
6. PPO approximate KL, clip fraction, update-rejection rate, gradient norms, policy standard deviation, and value explained variance;
7. the nominal rollout NPZ/reference trajectory rather than a return-only conclusion.

The first diagnostic recipe deliberately leaves `tracking_deviation_ratio=0.0`. Earlier gated starts in this project collapsed episodes before a three-family randomized policy had learned the motion. This run isolates whether the shared policy can learn the observable target under scheduled morphology. A later production arm can add a tracking-deviation gate, typically starting with the repository's previously viable ratio near 1.0, after the returned diagnostics establish a stable bootstrap.
