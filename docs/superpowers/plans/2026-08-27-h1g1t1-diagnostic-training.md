# H1+G1+T1 Diagnostic Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reversible training/evaluation overlay that diagnoses one shared H1+G1+T1 DeepMimic policy under morphology randomization.

**Architecture:** Retain the existing monolithic URMA2 path, add an orchestrator and dependency-light diagnostics around it, and patch only confirmed PPO/evaluation defects. Package commands, provenance, metrics, plots, crossevals, and checkpoint metadata in one compact ZIP.

**Tech Stack:** Bash, Python 3, NumPy, pandas, matplotlib, JAX/Flax/Optax, MuJoCo/MJX, pytest.

**Spec:** `docs/superpowers/specs/2026-08-27-h1g1t1-diagnostic-training-design.md`

## Global Constraints

- One policy must train all three families on the same motion.
- Environment and minibatch counts must divide evenly by three.
- Heading reward must have a matching policy observation.
- Morphology changes occur at reset, not mid-episode.
- Return and episode length are secondary, not the imitation verdict.
- The result archive excludes model weights unless explicitly requested.
- The uploaded snapshot cannot be treated as a full GPU-testable repository.

---

### Task 1: Reproduce and characterize the archived failure

**Files:**
- Read: `scripts/scaling/local_3t.sh`
- Read: `experiments/fsq_khaendler/curves_3t.csv`
- Read: `experiments/local_3t/ce_local_3t_s0.json`
- Create: `analysis/root_cause_report.md`

**Interfaces:**
- Consumes: supplied logs/configuration/results.
- Produces: measured root-cause report and constraints for the replacement recipe.

- [x] Compare requested objective with effective launcher flags.
- [x] Convert logged squared tracking errors to RMSE and heading radians to degrees.
- [x] identify stale T1 reference-convention evidence.
- [x] Trace family batching and heading observation/reward data flow.
- [x] Document confirmed faults separately from unresolved runtime hypotheses.

### Task 2: Add dependency-light diagnostic parsing

**Files:**
- Create: `scripts/h1g1t1/diagnostics.py`
- Test: `tests/test_h1g1t1_diagnostics.py`

**Interfaces:**
- Consumes: RL-X console log or metric CSV plus crosseval JSON files.
- Produces: `raw_metrics.csv`, `tracking_metrics.csv`, `summary.json`, `summary.md`, plots, and combined crosseval JSON.

- [x] Write failing tests for console blocks, duplicate-step fragments, RMSE conversion, recipe arithmetic, and motion-focused summaries.
- [x] Verify each test fails for the missing behavior.
- [x] Implement parsing, duplicate coalescing, aliases, derived metrics, health checks, plotting, and summaries.
- [x] Run the focused tests to green.

### Task 3: Repair PPO rejection semantics

**Files:**
- Create: `loco_mjx/loco_mjx/algorithms/urma2/mjx/update_guard.py`
- Modify: `loco_mjx/loco_mjx/algorithms/urma2/mjx/urma2.py`
- Test: `tests/test_urma2_update_guard.py`

**Interfaces:**
- Consumes: current policy state, candidate policy state, approximate KL values, KL threshold.
- Produces: selected state and numeric rejection flag.

- [x] Write failing tests for accepted, high-KL, non-finite-KL, and zero-KL cases.
- [x] Implement a JAX-compatible guard that keeps current state on unsafe KL.
- [x] Wire it into URMA2 and log `policy/update_rejected`.
- [x] Run guard tests to green.

### Task 4: Make crosseval reproduce training conditions

**Files:**
- Modify: `experiments/fsq_khaendler/crosseval_motion.py`
- Test: `tests/test_crosseval_h1g1t1_cli.py`

**Interfaces:**
- Consumes: final model and explicit heading/morphology evaluation flags.
- Produces: condition-stamped JSON and optional rollout NPZ files.

- [x] Write failing tests for heading observation, reset-only morphology, nominal disabling, and selected-checkpoint sanitation.
- [x] Add CLI/config helpers and morphology metadata.
- [x] Rewrite only the selected model ZIP and strip resume-only members.
- [x] Run crosseval helper tests to green.

### Task 5: Build the reproducible launcher and collector

**Files:**
- Create: `scripts/h1g1t1/preflight.py`
- Create: `scripts/h1g1t1/collect_results.py`
- Create: `scripts/h1g1t1/run_h1g1t1.sh`
- Create: `RUN_H1G1T1.sh`
- Create: `H1G1T1_RUNBOOK.md`
- Test: `tests/test_h1g1t1_launcher.py`

**Interfaces:**
- Consumes: full repository path, Python interpreter, full retarget directory, optional environment overrides.
- Produces: verified train/eval/report stages and `<RUN_ID>_diagnostics.zip`.

- [x] Write failing tests for three clips, suspicious full-clip segments, required recipe flags, exact-step verification, and model exclusion.
- [x] Implement preflight, single-device recipe, provenance capture, strict completion checks, four crossevals, report generation, and compact packaging.
- [x] Document exact commands, lower-memory arithmetic, and return artifact.
- [x] Run launcher tests to green.

### Task 6: Package and verify the overlay

**Files:**
- Create: bundle `README.md`, `install_overlay.sh`, `run.sh`, `PATCH.diff`, `MANIFEST.json`, `SHA256SUMS`.
- Create: `sample_diagnostics/`.

**Interfaces:**
- Consumes: modified snapshot and a clean target repository.
- Produces: one downloadable overlay ZIP.

- [x] Run the complete focused test suite.
- [x] Byte-compile all modified Python files and syntax-check all shell scripts.
- [x] Regenerate reports from supplied historical artifacts.
- [x] Install the overlay into a fresh copy and run the dry-run stage.
- [x] Verify archive contents, absence of caches/model weights, ZIP integrity, and SHA-256 checksums.
