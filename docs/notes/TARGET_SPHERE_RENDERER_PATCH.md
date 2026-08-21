# Cross-topology target-sphere video patch

## Why this patch is required

`render_cross_topology_policy.py` currently logs rollout `qpos` and replays it on
a nominal CPU model. That is useful for policy behavior, but it does not prove
that the randomized body, reward target, trajectory phase, and visual target
markers are aligned.

The target markers must be generated from the same topology-specific trajectory,
the same `(traj_no, subtraj_step_no)`, and the same sampled morphology that the
reward used. A nominal-body marker overlay would be misleading.

## Required command-line additions

Add these options to `scripts/scaling/render_cross_topology_policy.py`:

```text
--show-targets
--show-achieved-sites
--target-env-index 0
--target-sidecar <json-path>
--fail-target-mismatch-m 1e-5
```

## Rollout data that must be logged

For the selected environment in each topology group, log every rendered control
step:

```text
qpos
traj_no
subtraj_step_no
morphology[4]
morphology_generation
absorbing
reset_happened
```

Do not reconstruct phase from the rendered frame number. Asynchronous resets
make frame number and trajectory timestamp diverge.

## CPU replay model

For each family and morphology generation:

1. Load that topology's nominal MuJoCo CPU model.
2. Apply the same NumPy morphology mutation used by
   `verify_fk_targets.py` / `eval_fk_tracking.py`.
3. Replay the achieved `qpos` on that mutated model.
4. Read the topology-specific reference `qpos` at the exact logged trajectory
   and phase.
5. Replay the reference `qpos` on the same mutated model.
6. Extract the exact five `sites_for_mimic` used by `MorphMimicReward`.

A model may be cached by `(family, morphology[4])` for rendering. Do not cache a
target by phase alone.

## Visual convention

Use non-colliding scene decorations rather than physical geoms:

```text
target upper body: white sphere
target left hand/foot: green spheres
target right hand/foot: magenta spheres
achieved sites: small cyan spheres
```

Draw a thin line from each achieved site to its target. Burn the following text
into each panel:

```text
family
morphology values
trajectory id
reference phase
generation
mean target error
```

## Sidecar JSON

Write one row per output frame and family:

```json
{
  "frame": 120,
  "family": "g1",
  "traj_no": 0,
  "subtraj_step_no": 431,
  "morphology": [1.08, 1.21, 0.84, 1.10],
  "morphology_generation": 7,
  "absorbing": false,
  "target_provider_max_error_m": 4.8e-7
}
```

The final JSON must also contain:

```text
max_target_provider_error_m
number_of_resets
number_of_morphology_changes
frames_per_family
reference_file_per_family
```

## Mandatory acceptance tests

1. At nominal morphology, visual target sites agree with stored trajectory sites
   to the repository's nominal tolerance.
2. At low/high corner bodies, visual target sites agree with
   `MorphMimicReward._traj_site_quantities` within `1e-5 m`.
3. The same frame rendered with phase `p+1` must not pass the phase-`p` check.
4. A morphology reset changes marker positions and the sidecar generation.
5. H1 and G1 use their own retargeted joint trajectories.
6. The renderer exits nonzero when `target_provider_max_error_m` exceeds the
   configured threshold.

## Status: implemented

Landed in `scripts/scaling/render_cross_topology_policy.py`; acceptance tests in
`tests/test_target_renderer.py`. Two deviations from the text above, both
deliberate:

* `target_provider_max_error_m` is the disagreement between the CPU
  body-correct provider and the **production MJX provider** at the same logged
  (body, trajectory, phase) — measured at ~2e-7 m across nominal and both
  corner bodies, so the 1e-5 m default threshold is a real gate rather than a
  formality. Comparing the drawn marker against the array it was drawn from
  would have been a tautology.
* Markers are anchored on the main mimic site rather than drawn at raw
  reference world coordinates. The reward scores *relative* site positions; the
  world-frame root offset is scored separately (by the goal's root-error block
  and the deviation terminal criterion), so overlaying raw world targets would
  have made a well-tracking policy look broken.

`eval_contact_quality.py` remains the non-visual body-correct evidence and is
still produced by the pipeline's `evaluate` stage.
