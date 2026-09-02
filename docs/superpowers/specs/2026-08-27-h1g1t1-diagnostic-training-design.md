# H1+G1+T1 Diagnostic Training Design

## Goal

Provide a reversible overlay that runs one observable, randomized, DeepMimic-style URMA2 policy across H1, G1, and T1 and returns enough motion-quality and optimizer evidence to diagnose failure without transferring the entire repository or every checkpoint.

## Architecture

The existing monolithic trainer remains the execution path. A shell orchestrator validates assets and arithmetic, records the effective recipe, launches training, verifies exact completion, runs condition-stamped crossevals, generates dependency-light reports, and packages a compact artifact. Targeted patches repair PPO rejection semantics and evaluation checkpoint/randomization parity; no broad trainer refactor is introduced.

## Components

- `scripts/h1g1t1/run_h1g1t1.sh`: stage orchestration and exact recipe.
- `scripts/h1g1t1/preflight.py`: repository, dependency, full-clip, T1 convention, and arithmetic validation.
- `scripts/h1g1t1/diagnostics.py`: console/CSV parsing, derived RMSE/degree channels, health checks, plots, JSON/Markdown summary.
- `scripts/h1g1t1/collect_results.py`: checkpoint inventory and compact archive.
- `experiments/fsq_khaendler/crosseval_motion.py`: observation parity, reset-only morphology conditions, condition stamping, bounded sanitized checkpoint copy.
- `loco_mjx/.../update_guard.py`: isolated PPO candidate acceptance decision.
- `loco_mjx/.../urma2.py`: use the guard and log rejection.
- `RUN_H1G1T1.sh` and bundle-level `run.sh`: stable entry points.

## Data flow

The launcher writes recipe/command/environment manifests before compilation. Training emits console and TensorBoard data. Exact completion is checked from the parsed final step and checkpoint presence. Crosseval reads the final policy and emits nominal, randomized, and zero-action JSON plus nominal rollout NPZ data. The reporter retains raw values, derives joint/qvel RMSE and heading degrees, generates per-family plots, and combines crosseval evidence. The collector inventories all checkpoints but includes model bytes only when explicitly requested.

## Error handling

Preflight aborts on missing files, invalid clip structure/non-finite data, absent T1 sign fix, missing dependencies, or invalid batch/checkpoint arithmetic. Training aborts on nonzero process status, fatal log markers, missing final checkpoint, or final-step mismatch. Evaluations require both a log and JSON result. The bundle installer backs up replaced files and refuses a target that lacks repository sentinels.

## Test strategy

Dependency-light tests cover parser fragmentation, RMSE conversion, arithmetic, clip checks, launcher flags, compact model exclusion, crosseval settings/checkpoint sanitation, and PPO guard semantics. Completion verification includes Python byte-compilation, shell syntax, clean-overlay installation into a fresh archive copy, dry-run command construction, report regeneration, ZIP integrity, manifest checks, and SHA-256 generation. Full GPU behavior remains a user-machine validation because the snapshot lacks assets and runtime dependencies.
