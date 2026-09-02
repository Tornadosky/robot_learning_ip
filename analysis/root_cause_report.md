# H1 + G1 + T1 training failure analysis

## Executive diagnosis

The archived run did learn some joint-pose structure, but it did not execute the requested experiment. The historical launcher trained nominal bodies only, disabled the heading reward, did not expose the heading target to the policy, disabled TensorBoard, retained only the latest checkpoint by default, and used a weak completion test. Consequently, high return and long episodes could coexist with poor dance imitation and could not demonstrate morphology robustness.

The multi-family PPO reshape is not the primary fault. The batch retains a robot axis and normalizes advantages per family. With a total environment count and minibatch divisible by three, H1, G1, and T1 are represented in every update.

## Evidence from the supplied run

The historical curve export ends at 19,169,280 global environment steps. Its fixed step increment is 36,864, which equals 576 environments times a 64-step rollout. The supplied launcher defaults to 192 environments and 19,660,800 total steps. Since the effective command and resolved config were not saved, the archive cannot establish whether `NR_ENVS=576` and a different total were intentional overrides or whether the process ended early.

Final online metrics from the 520-row curve export were:

| Family | Joint RMSE start → final | Heading error start → final | Policy / reference airborne | Return / episode length |
|---|---:|---:|---:|---:|
| Unitree H1 | 0.578 → 0.284 rad | 11.2° → 89.2° | 0.085 / 0.117 | 1510.7 / 922.7 |
| Unitree G1 | 0.514 → 0.195 rad | 24.9° → 86.8° | 0.065 / 0.297 | 1658.6 / 919.7 |
| Booster T1 | 0.881 → 0.347 rad | 8.5° → 82.4° | 0.019 / 0.111 | 1196.2 / 792.3 |

This is partial pose learning, not a complete three-family dance result. All three policies ended approximately sideways or opposite the reference heading, and G1/T1 substantially under-produced reference foot lift.

The archived nominal crosseval reports:

| Family | Policy shape RMSE | Policy absolute RMSE | Reference-vs-raw absolute floor | Heading mean / p95 |
|---|---:|---:|---:|---:|
| Unitree H1 | 0.2147 | 0.2829 | 0.0298 | 88.1° / 172.5° |
| Unitree G1 | 0.1722 | 0.1955 | 0.0052 | 69.4° / 165.4° |
| Booster T1 | 0.8882 | 1.3453 | 1.4425 | 42.8° / 141.9° |

The T1 reference floor is itself 1.44 rad, so that old T1 result is convention-invalid. The repository notes identify reversed Booster T1 clip axes and add `T1_CLIP_SIGNS`; crosseval must be regenerated after that fix. The new preflight refuses to proceed if the sign-fix symbol is absent.

## Confirmed configuration failures

### No morphology randomization

`scripts/scaling/local_3t.sh` sets:

```text
morphology_coeff_mode=fixed
morphology_coeff_value=0.0
```

Therefore the archived policy was trained on nominal H1, G1, and T1 bodies, not randomized members of those families. No conclusion about within-family morphology generalization is supported by that run.

### Heading was both disabled and unobservable

The launcher sets `root_heading_tracking_weight_ratio=0.0` and does not enable `tracking_clip_observe_root_heading`. The environment computes and logs absolute wrapped heading error regardless, but only appends the actionable `(sin(error), cos(error))` target when the observation flag is true. Historical heading-weight sweeps without that observation tested a target the policy could not perceive.

The replacement enables both:

```text
tracking_clip_observe_root_heading=True
root_heading_tracking_weight_ratio=0.75
root_heading_tracking_temperature=2.0
```

The broader temperature avoids an effectively dead exponential reward at large initial errors while the explicit observation makes the objective Markov with respect to the reference turn.

### Completion and provenance were insufficient

The old script declared success when the process exit status was zero and more than ten `nr_env_steps` strings were present. RL-X can log an uncaught exception while returning zero, and more than ten blocks does not prove the requested total was reached. It also did not save the effective command, environment overrides, package state, GPU state, or checkpoint config.

The replacement scans fatal markers, parses the console log, requires the final step to equal the requested total exactly, requires a non-empty `latest.model`, records every effective command and override, and inventories checkpoint contents and hashes.

### Evaluation did not reproduce the training observation/randomization condition

The original crosseval helper did not expose root-heading observation parity or a morphology coefficient. It also copied every checkpoint before each evaluation and attempted to remove resume files only from the directory, although resume members can be inside the model ZIP.

The patched evaluator sets observation parity, supports reset-only morphology at 0.15 and 0.30, stamps the condition inside every JSON, rewrites only the selected checkpoint, and removes resume-only ZIP members.

## Latent PPO update-rejection defect

The original PPO guard rejected an update when mean KL exceeded the threshold **or** clip fraction was exactly zero. A zero clip fraction is a valid small update. More importantly, rejection selected `prev_policy_state`, which after an accepted update is one policy older than the current state. A rejected candidate could therefore roll the policy back by two update positions instead of preserving the policy that generated the candidate.

The patch keeps the current policy when mean KL is too high or non-finite, accepts zero-clip-fraction updates, and logs `policy/update_rejected`. This defect is real, but the supplied artifacts do not prove it was the dominant cause of the historical run; the added metric will show whether it matters in the new run.

## What appears structurally correct

The monolithic URMA2 trainer reshapes rollout data to `(samples_per_family, family, ...)`, uses the same sampled indices across the robot axis, and normalizes advantages along the sample axis separately for each family. With valid divisibility, the shared policy receives balanced family data rather than silently starving one topology.

The replacement makes that arithmetic explicit and preflighted. The default uses 576 total environments, 192 per family, a 36,864-sample rollout batch, and a 6,144-sample minibatch.

## Reference-motion risk that still requires the full repository

The uploaded archive contains only 120-frame samples, not the full motion assets or meshes. Those samples are finite and have plausible root heights and moderate velocity ranges, but they cannot certify the full clips. Project notes mention earlier full T1 retargets with near-ground root height and extreme qvel spikes. The new preflight scans every frame of each full retarget, records duration/root-height/quaternion/qvel statistics, and emits explicit warnings for suspicious segments before GPU compilation.

## Replacement experiment

The run uses one shared policy, one motion, all three families, reset-only morphology scheduled from 0.0 to 0.3, a visible heading target, DeepMimic pose/qvel/body/foot terms, TensorBoard, ten checkpoint intervals, and four post-training crossevals:

1. nominal policy;
2. policy at morphology 0.15;
3. policy at morphology 0.30;
4. nominal zero-action baseline.

This first diagnostic arm leaves the tracking-deviation termination gate off so early three-family instability cannot terminate the reference before the policy has learned. The returned evidence will determine whether the next arm should add the repository's previously viable loose gate near 1.0, reduce morphology magnitude, adjust reward temperatures, or isolate a specific family/retarget.

## Diagnostic acceptance criteria

These are investigation gates, not guaranteed final-performance numbers:

- each family's policy absolute RMSE must be materially below its zero-action baseline;
- the reference-versus-raw floor must be low enough that policy RMSE is interpretable, especially for T1;
- heading error should trend down rather than converge near 90 degrees;
- morphology 0.15/0.30 should degrade gradually rather than collapse one family;
- joint RMSE and qvel RMSE should improve together, not trade static pose accuracy for uncontrolled motion;
- policy foot-airborne fraction should move toward the reference distribution;
- approximate KL, gradients, policy standard deviation, explained variance, and update-rejection rate must remain finite and stable;
- rollout/reference inspection must agree with the numerical verdict.

## Verification boundary

The patch has dependency-light unit tests, shell syntax checks, Python compilation checks, report generation against supplied artifacts, and an overlay-install/dry-run test. A full GPU/MuJoCo training result cannot be produced from the uploaded snapshot because it omits full assets and the required training environment. The compact result ZIP generated on the user's machine is the required next evidence.
