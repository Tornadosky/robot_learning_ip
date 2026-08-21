# Required no-FSQ production fixes before a 60M/300M claim

## Implementation status

All P0 and P1 items below are implemented. Where the work landed:

| Item | Files |
| --- | --- |
| P0 shared reference provider | `scripts/scaling/body_correct_reference.py`; `scripts/h1md/morph_mimic_reward.py` now consumes it |
| P0 body-correct goal | `scripts/scaling/body_correct_goal.py` (`MorphGoalTrajMimicRootErr`), `--goal-type` in `parallel_cross_humanoid_train.py`; recorded in the manifest as `goal_type` |
| P0 terminal semantics | `--terminal-handler` / `--max-root-deviation` in `parallel_cross_humanoid_train.py`; recorded as `terminal_handler`, `max_root_pos_deviation` and `root_deviation_check` (the threshold measured against the clip's own root travel) |
| P0 acceptance tests | `tests/test_body_correct_reference.py` (36 cases: H1 + G1 x nominal/low/high x four phases) |
| P1 admission accounting | `scripts/scaling/morphology_admission.py`, wired into `scripts/scaling/family_morphology.py`; recorded as `body_admission`; `tests/test_morphology_admission.py` |
| P1 target renderer | `--show-targets` etc. in `scripts/scaling/render_cross_topology_policy.py`; `tests/test_target_renderer.py` |

Two consequences worth knowing before reading a checkpoint:

* `MorphGoalTrajMimicRootErr` is **three dimensions wider** than `GoalTrajMimic`
  (the heading-frame root error). The evaluators/renderer now carry `goal_type`
  over from the manifest, so a checkpoint is replayed under the goal it trained
  with; a stock-goal checkpoint is unaffected.
* The goal mimics the environment's full 15-site list while the reward scores
  the 5-site H1/G1 subset. The subset relation is asserted at build time, and
  the acceptance test compares the reward's rows of the goal block directly.

---

The current H1/G1 cross-family trainer can run an integration smoke without
FSQ, but its stock no-FSQ information contract is not yet internally consistent
for randomized geometry:

- `GoalTrajMimic` exposes trajectory-side relative site features stored in the
  topology reference, which were generated on the nominal topology model.
- `MorphMimicReward` scores relative site features recomputed by FK on the
  currently sampled morphology.

Thus the actor can be commanded toward one spatial target while the reward
scores another. The runner blocks long runs by default until this is corrected.

## P0 — one shared body-correct reference provider

Create a pure helper outside vendored `loco-mujoco`, for example:

```text
scripts/scaling/body_correct_reference.py
```

Its production API should accept:

```text
env
model / sampled body model
trajectory state: traj_no + subtraj_step_no
carry.morphology
backend: numpy or jax.numpy
```

and return one immutable bundle:

```text
reference_qpos_clamped
reference_qvel
relative_site_position
relative_site_orientation
relative_site_velocity
```

Required semantics:

1. Load the topology-specific H1 or G1 joint trajectory at the exact current
   `(traj_no, subtraj_step_no)`.
2. Apply the same per-sampled-body joint-range clamp used by
   `MorphMimicReward`.
3. Build the same sampled MJX model used by physics:

   ```python
   body_model = env._apply_morphology(env.sys, carry.morphology)
   ```

4. Run FK once on `reference_qpos_clamped` using `body_model`.
5. Compute the exact five shared mimic-site quantities using the same site,
   body, and root indices as the reward.
6. Return the bundle to all consumers. Do not duplicate the FK formula in the
   goal, reward, evaluator, and renderer.

Refactor `MorphMimicReward` to consume this provider rather than owning a
separate implementation.

## P0 — body-correct no-FSQ goal

Add a goal adapter such as:

```text
MorphGoalTrajMimicRootErr
```

It should preserve the standard DeepMimic current-state and reference joint
features, but obtain the trajectory-side site quantities from the shared
provider above. Append the existing heading-frame root position error so that
world/root tracking scored by the reward is observable.

Register it without modifying vendored framework code and expose it through:

```text
parallel_cross_humanoid_train.py --goal-type MorphGoalTrajMimicRootErr
```

The training manifest must record the goal type.

## P0 — terminal semantics for randomized height

Forward these options through the cross-family trainer to every family env:

```text
--terminal-handler MorphologyAwareRootPoseTrajTerminalStateHandler
--max-root-deviation 0.5
```

The deviation threshold is an experiment setting, not a universal constant;
verify it against the selected clip. Record it in the manifest. The morphology-
aware height interval is required before widening leg-length randomization.

## P0 acceptance tests

Add tests covering H1 and G1, nominal/low/high morphologies, and multiple phases:

1. Goal target site block equals reward target bundle within `1e-5 m`.
2. Jitted and eager providers agree.
3. Independent CPU MuJoCo FK agrees within existing verifier tolerances.
4. Phase `p` does not accidentally match `p+1`.
5. Low/high morphology targets differ from nominal where geometry changes.
6. Reference qpos used by joint reward and FK target is the same clamped array.
7. Reset observation, reward, and trajectory cursor use the same post-reset
   phase.
8. A tall valid body is not absorbing at reset solely because nominal height
   bounds were used.

## P1 — body admission and rejection accounting

The present conservative four-dimensional H1/G1 sampler uses every draw and has
no skip/rejection counters. Before adding joint-axis, joint-range, foot-size, or
more extreme geometry randomization, add bounded admission checks at reset:

```text
draws_total
accepted_total
rejected_total
resamples_total
resample_exhausted_total
rejections_by_reason
per-family morphology min/max/quantiles
```

Suggested rejection reasons:

```text
nonfinite_model
invalid_mass_or_inertia
invalid_joint_range
reference_limit_violation
initial_penetration
initial_absorbing
fk_nonfinite
reference_screen_failed
```

Use a fixed maximum number of resamples. Exceeding it must fail closed rather
than silently substituting a nominal body. For current conservative bounds,
zero rejected bodies is a measurement to log, not an assumption.

## P1 — body-correct target renderer

Implement `TARGET_SPHERE_RENDERER_PATCH.md`. The renderer must log the exact
trajectory cursor and morphology generation from the rollout, reconstruct the
same randomized CPU model, use the shared reference provider, and emit a JSON
sidecar. Nominal-mesh replay is not sufficient evidence for randomized-body
reference correctness.

## Promotion rule

Only after all P0 tests pass should the runner be invoked with:

```bash
export GOAL_TYPE=MorphGoalTrajMimicRootErr
export TERMINAL_HANDLER=MorphologyAwareRootPoseTrajTerminalStateHandler
export MAX_ROOT_DEVIATION=0.5
bash run_h1g1_urma_pipeline.sh train60m
```

`ALLOW_STOCK_GOAL_BASELINE=1` exists solely to reproduce and explicitly label
an integration baseline with the known observation/reward mismatch. It must not
be used for the production claim.
