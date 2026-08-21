# Scaling Roadmap — Morphology-Independent Multi-Skill Control

**End goal:** one (or a small family of) policies that control thousands of
randomly morphologized robots — eventually across families — on one motion,
then many, without hand-tuning per body.

**Related progress log:** [`SCALING_PROGRESS_2026-08-01.md`](SCALING_PROGRESS_2026-08-01.md)
(implementation status, measured numbers, caveats).
**Cross-embodiment results (steps 7–8), 2026-08-03:**
[`CROSS_EMBODIMENT_RESULTS.md`](CROSS_EMBODIMENT_RESULTS.md).

Status column reflects the repo as of the progress log above:
`done` / `partial` / `todo`.

---

## 10 steps toward the goal

| # | Step | What you do | Tools / papers | Unlocks | Depends on | Status |
|---|---|---|---|---|---|---|
| **1** | **Parallel multi-morph training** | Replace round-robin with simultaneous rollouts (e.g. several bodies × envs → one PPO update) | `scripts/scaling/parallel_multimorph_train.py`; URMAv2 training recipe as reference | Honest scaling of bodies per wall-clock; gradients from all morphs each update | — | **done** (grouped-static) |
| **2** | **Online same-DOF embodiment randomization** | Sample leg/arm/mass/foot scales every reset (or curriculum), fixed joint layout; no new XML compile per sample | `scripts/scaling/online_h1_train.py`; URMAv2 “% of nominal” ER | Path from ~6 presets → **10²–10³+** virtual H1s without retarget | 1 helps | **done** (H1 online path) |
| **3** | **Same-family scale stress test (1 motion)** | Train one dance/walk policy on large online-sampled H1 sets; hold out extreme / OOD bodies | DeepMimic OK for **1** clip; raw grounded refs; `evaluate_online_policy.py` | Supervisor curve: how many morphologies until zero-shot breaks? | 1 + 2 | **partial** (100M online MLP; not yet full dance convergence / extrapolation) |
| **4** | **Leave DeepMimic for multi-motion** | Same robot, many LAFAN1 clips; adversarial / style objective | **AMP** in loco-mujoco (`AMPJax`); `parallel_multimorph_amp_train.py`; MimicKit (ASE/ADD) if needed | Breaks the ~140-step multi-clip DeepMimic plateau | Independent of 1–3 | **partial** (AMP trainer exists; not yet the proven plateau-breaker at scale) |
| **5** | **Multi-body × multi-motion (one family)** | Online H1 morphs + many dance/walk clips, shared policy | AMP + steps 1–2; optional morph descriptor | First “randomized body + several skills” result in-family | 3 + 4 | **partial** (static multi-body×multi-clip AMP; not yet combined with online morph) |
| **6** | **Cross-humanoid reference pipeline** | Fast, high-quality human→H1/G1/Atlas/… refs for the same mocap | **GMR** ([Retargeting Matters](https://arxiv.org/abs/2510.02252)); `cross_humanoid_retarget.py`; loco-mujoco SMPL as baseline | Dance refs across **different DOF humanoids** without per-cell SMPL pain | Before true cross-robot train | **partial** (pipeline present; quality vs GMR still open) |
| **7** | **Embodiment-aware policy backbone** | Per-joint describe → attend → decode actions; variable obs/act length | **URMA / URMAv2**; `--backbone urma\|urmav2` in online trainer; RL-X / [one_policy…](https://github.com/nico-bohlinger/one_policy_to_run_them_all) | Architecture that can own H1+G1 (+ later others) in one net | Start walk-only; compare to descriptor MLP | **done (answered, negatively)** — URMA finally given genuine cross-topology input via `joint_descriptions.py` + `CrossTopologyURMAPPO`; matched comparison says the masked MLP is **at least as good everywhere and much better at low optimizer budget**. Exception worth chasing: URMA wins Atlas (most joints) 2.7× at the higher budget. See [`CROSS_EMBODIMENT_RESULTS.md`](CROSS_EMBODIMENT_RESULTS.md) §4.3–4.4 |
| **8** | **Cross-family humanoids, 1–few motions** | URMA/masked policy + cross-robot refs + AMP/tracker on H1+G1 (+1 more), online morph DR per family | `parallel_cross_humanoid_train.py` + GMR/AMP | Morphology-**and**-robot-independent tracking of dance/walk | 5 + 6 + 7 | **done for 1 motion** — one policy on H1(19)+G1(23)+Atlas(27) in one padded PPO update; **every robot individually beats its exact-reset zero-action baseline** (h1 +141, g1 +172, atlas +30 for URMA; +143/+174/+11 for the MLP), all three in every update. Still **partial** overall: single clip, stock bodies (no per-family morphology randomisation), and zero-shot transfer to a held-out topology (ToddlerBot, 30 joints) is **negative** |
| **9** | **Supersize the motion side** | Scale data/model for many skills; shared motion latent / tokens; less reward engineering | **SONIC** ([GEAR-SONIC](https://nvlabs.github.io/GEAR-SONIC/)); **BeyondMimic**; MimicKit ASE | Hundreds of motions; teleop/VLA-ready command interface; skill composition | 4–5 proven; 8 preferred | **todo** |
| **10** | **Foundation controller** | Thousands of morphs × several families × many actions; task interface on top of tracker | Merge: URMAv2 ER + URMA policy + GMR/SONIC motion front-end + BeyondMimic/ASE skills | End state: random robot in, motion/task command in → robust whole-body behavior | 1–9 | **todo** |

---

## How the pieces fit (don’t confuse roles)

| Piece | Job in this goal |
|---|---|
| **Online ER + parallel envs** | Scale *bodies* (thousands of morphs) |
| **URMA / URMAv2** | Scale across *families / DOF layouts* |
| **GMR / retargeting** | Put the *same human motion* on those bodies/robots |
| **AMP → ASE / BeyondMimic** | Scale *number of motions* past DeepMimic |
| **SONIC** | Scale *motion foundation* (data/model/compute + universal command tokens) |
| **BeyondMimic (diffusion head)** | Turn tracking into *many task actions* at test time |
| **DeepMimic** | Keep for early single-motion ablations — not the end stack |

---

## Milestone checkpoints

| After step | Claim you can make |
|---|---|
| **3** | “We train ~100+ random H1 morphologies jointly on one motion; holdouts work in-distribution.” |
| **5** | “One in-family policy: random morph + several motions.” |
| **8** | “One policy across H1+G1 (different DOF), randomized morphs, few motions.” — *the different-DOF half is now evidenced (H1+G1+Atlas, one motion, every robot beating its own baseline); the randomized-morph half is not* |
| **10** | “Morphology-independent multi-skill controller at foundation scale.” |

---

## Sequencing rule

1. Finish hardening **infra (1–2)** and **objective (4)** — they unblock everything and stay closest to the current stack.
2. Pull **URMA (7)** and **GMR (6)** hard when leaving the same-DOF family; until then a descriptor MLP may win on fixed-topology H1.
3. Pull **SONIC / BeyondMimic (9)** when *motion diversity*, not embodiment count, is the bottleneck.

---

## Research references

- URMA: <https://proceedings.mlr.press/v270/bohlinger25a.html> · [project](https://nico-bohlinger.github.io/one_policy_to_run_them_all_website/) · [code](https://github.com/nico-bohlinger/one_policy_to_run_them_all)
- URMAv2: <https://arxiv.org/abs/2509.02815>
- GMR / Retargeting Matters: <https://arxiv.org/abs/2510.02252> · [code](https://github.com/YanjieZe/GMR)
- SONIC: <https://nvlabs.github.io/GEAR-SONIC/> · <https://arxiv.org/abs/2511.07820>
- BeyondMimic: <https://beyondmimic.github.io/>
- MimicKit (AMP/ASE/…): <https://github.com/xbpeng/MimicKit>
- Prior experiment reports:
  [`experiments/goal_20260710_223728/report.md`](experiments/goal_20260710_223728/report.md),
  [`experiments/frontier_20260714_224204/report.md`](experiments/frontier_20260714_224204/report.md),
  [`experiments/overnight_shared_policy_20260629_094141/report.md`](experiments/overnight_shared_policy_20260629_094141/report.md)
