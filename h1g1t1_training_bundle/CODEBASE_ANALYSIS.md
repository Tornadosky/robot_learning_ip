# H1/G1/T1 codebase analysis and experiment rationale

## What the existing implementation already gets right

The multi-robot training environment concatenates H1, G1, and T1 populations while using one shared URMA2 policy. Per-joint descriptions and action padding allow the same policy parameters to operate across different action dimensions and topologies. The clip loader maps each robot family to its own offline retarget, and the current source contains the corrected T1 clip-sign table.

The reward implementation already exposes useful non-return diagnostics: joint tracking error, normalized qvel tracking error, root-heading error, relative body-position and body-orientation errors, foot-height and contact metrics, PPO KL, clipping fraction, policy variance, gradient norms, and losses.

## Why the previous run was not the intended experiment

The archived launcher fixed `morphology_coeff_value=0.0`. It trained three nominal topologies, not three randomized robot families.

The archived run also disabled both the root-heading observation and the heading reward. The remaining DeepMimic body terms are intentionally heading-free, while the clip-derived yaw-rate command is capped. Consequently, the policy could obtain substantial imitation reward while facing the wrong global direction. At the last archived row, mean heading errors were approximately 89.2° for H1, 86.8° for G1, and 82.4° for T1.

The historical heading term used `exp(-error² / temperature)` with temperature 0.25. At 90° error its score is about 0.000052, so after a large drift it provides almost no useful recovery gradient. This bundle preserves the exponential mode for compatibility and adds an explicitly selected cosine mode whose score remains informative throughout the wrapped heading range.

The current `create_env.py` interprets `environment.nr_envs` as the total per device across all train robots. Therefore `nr_envs=192` means only 64 environments per family. The archived tensor shapes show that the old completed process actually ran under earlier semantics with 192 per family. The new launcher makes the current semantics explicit by setting 576 total on one visible GPU.

The previous run reached 19,169,280 aggregate samples, or roughly 6.39 million per family. Its final derived joint RMSE was approximately 0.284 rad for H1, 0.195 rad for G1, and 0.347 rad for T1. Episode return and length were much less diagnostic: they remained comparatively high despite those pose errors and the near-90° heading failure.

The old success check accepted process exit zero plus a count of console blocks. RL-X catches training exceptions, logs them, and may allow the outer process to exit normally. The replacement verifier requires the exact final step, scans for fatal patterns, and checks the actual model artifact.

The old crossevaluation helper copied every sibling checkpoint and attempted to remove resume files beside the `.model`; those files are members inside the ZIP-formatted `.model`. The replacement evaluator repacks only the selected checkpoint and strips the three resume-only members without altering policy or critic weights.

## Experimental hypothesis

A shared policy should be able to reproduce one retargeted dance across H1, G1, and T1 with moderate family randomization when:

1. every family contributes equally to each PPO minibatch;
2. motion phase is observable through pose and velocity references;
3. absolute heading is both observable and rewarded;
4. the action is learned as feedback around the reference feed-forward target;
5. morphology difficulty ramps instead of being applied at full strength immediately;
6. evaluation exactly matches action and observation semantics;
7. success is judged by joint, velocity, heading, body, and foot metrics, not return alone.

The probe is evidence-gathering, not proof of final convergence. Its diagnostics are designed to distinguish policy optimization instability, family imbalance, reference/retarget mismatch, heading failure, foot/contact failure, and morphology-ramp failure before spending the larger full-profile budget.
