# H1 + G1 + T1 single-policy dance training bundle

This is an overlay for the supplied `robot_learning_ip` repository. It does not replace the repository. It installs a validated H1/G1/T1 launcher, a small heading-reward patch, preflight checks, metric parsing, plots, crossevaluation, and diagnostics packaging.

## Run the diagnostic probe

```bash
unzip h1g1t1_single_policy_training_bundle.zip
cd h1g1t1_training_bundle

REPO=/mnt/c/Users/smirn/Desktop/robot_learning_ip \
PYTHON=$HOME/jaxgpu/bin/python \
PROFILE=probe \
bash install_and_run.sh
```

The default motion location is:

```text
<REPO>/external_data/amass_converted/LAFAN1
```

To override it:

```bash
REPO=/path/to/robot_learning_ip \
PYTHON=/path/to/gpu/python \
CLIP_DIR=/absolute/path/to/LAFAN1 \
CLIP=dance2_subject4.npz \
PROFILE=probe \
bash install_and_run.sh
```

Use one visible GPU. If `CUDA_VISIBLE_DEVICES` is not set, the launcher selects device `0`.

## First do an installation-only check when the checkout has local edits

```bash
REPO=/path/to/robot_learning_ip \
PYTHON=/path/to/gpu/python \
INSTALL_ONLY=1 \
bash install_and_run.sh
```

The installer verifies hashes for the two existing source files it patches. It accepts the exact uploaded version or a previously installed copy. A mismatch stops installation and preserves your checkout. Do not use `FORCE_INSTALL=1` without reviewing the reported file.

Files replaced during installation are backed up under:

```text
<REPO>/.h1g1t1_bundle_backups/<timestamp>/
```

## Profiles

| Profile | Purpose | Total envs | Envs/family | Aggregate samples | Samples/family |
|---|---|---:|---:|---:|---:|
| `smoke` | compilation/integration check, not a learning result | 96 | 32 | 614,400 | 204,800 |
| `probe` | diagnostic training run | 576 | 192 | 29,491,200 | 9,830,400 |
| `full` | larger follow-up after probe analysis | 576 | 192 | 147,456,000 | 49,152,000 |

The profile uses a rollout length of 64 and balanced minibatches. For the probe, each rollout contains 36,864 samples and each 6,144-sample minibatch contains 2,048 samples from each family.

## Task semantics

The run trains one shared URMA2 policy on `unitree_h1`, `unitree_g1`, and `booster_t1`. It uses the family-specific offline retargets of one motion and the repository’s topology-aware joint descriptions.

The policy is a residual controller around the reference (`tracking_reference_action_bias=1.0`). DeepMimic-style terms score joint pose, normalized joint velocity, relative body position, body orientation, root height, root heading, and foot height. The root-heading delta is observable by the policy. The heading term uses a broad cosine kernel, avoiding the near-zero recovery signal of the previous narrow exponential term at large errors.

Seen-body morphology ramps from 0 to 0.3. The existing randomization stack remains active, with exact inertia rescaling and fourth-power torque scaling. Absolute joint anchoring is retained and `tracking_clip_fit_per_variant=False`, so the dance is not silently rescaled into a different joint-space motion for every randomized body.

## Evidence produced

Each run is written to:

```text
<REPO>/experiments/h1g1t1_runs/<RUN_ID>/
```

It contains:

- the exact train command and resolved configuration;
- preflight results, Python packages, JAX devices, GPU snapshot, Git revisions, and environment variables;
- complete console and GPU telemetry logs;
- TensorBoard event files in the RL-X run directory;
- wide and long metric CSV files;
- joint RMSE, normalized qvel RMSE, heading error, body-pose error, foot error, morphology, PPO-health, optimization, episode-return, and episode-length plots;
- nominal and morphology-randomized crossevaluations;
- zero-residual reference-feedforward controls under the same semantics;
- per-joint RMSE tables and worst-joint summaries;
- explicit verification that the requested final step and `latest.model` were reached.

Episode return and episode length are retained as context, but the primary analysis is based on motion-reproduction and optimization metrics.

## What to send back

At the end the launcher prints:

```text
[h1g1t1] return this file: <REPO>/experiments/h1g1t1_runs/<RUN_ID>/<RUN_ID>_diagnostics_return.zip
```

Send that ZIP. It excludes model weights by default but records every model’s filename, byte size, and SHA-256.

To recollect the same run with checkpoint weights:

```bash
cd /path/to/robot_learning_ip
RUN_ID=<run-id> \
PROFILE=probe \
INCLUDE_MODEL=1 \
PYTHON=$HOME/jaxgpu/bin/python \
bash scripts/h1g1t1/run_experiment.sh collect
```

## Run stages separately

```bash
REPO=/path/to/robot_learning_ip PYTHON=/path/to/python PROFILE=probe \
  bash scripts/h1g1t1/run_experiment.sh preflight

REPO=/path/to/robot_learning_ip PYTHON=/path/to/python PROFILE=probe RUN_ID=<same-id> \
  bash scripts/h1g1t1/run_experiment.sh train
```

The supported stages are `preflight`, `train`, `analyze`, `evaluate`, `collect`, and `all`. Keep the same `RUN_ID` when invoking stages separately.

Evaluation is deliberately substantial because a single stochastic rollout can hide family-specific failure. By default, the probe evaluates policy seeds 0 and 1 on nominal and randomized bodies, plus nominal/randomized reference-feedforward controls. Set `EVAL_SEEDS="0 1 2 3"` for four policy seeds or `SKIP_EVAL=1` to collect training diagnostics first.

## Full profile

Run this only after the probe diagnostics have been reviewed:

```bash
cd /path/to/robot_learning_ip
REPO=$PWD PYTHON=$HOME/jaxgpu/bin/python PROFILE=full \
  bash scripts/h1g1t1/run_experiment.sh all
```

A successful process exit alone is not accepted. The verifier scans for fatal log patterns, requires the exact configured final step, and requires a nonempty `latest.model`.
