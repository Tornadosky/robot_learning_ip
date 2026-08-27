# FSQ: what we tried, what we got, where it stands

Plain-language summary, 2026-08-26. Numbers from
`experiments/fsq_khaendler/REPORT_FSQ_SCALE.md` and the 63+ crossevals in
`ce_fsqscale/`.

## What the metric means

Unless stated otherwise, **"error" means joint-angle tracking error in radians**:
the RMSE between the joint angles the robot *actually executed* and the motion
capture clip's angles at the same point in the motion, mean-centred per joint so
it measures motion SHAPE rather than rest-pose convention. Measured over 64
parallel robots x 1000 control steps, only on steps where the episode was still
alive. **It is not reward, not return, and not episode length.** When we say
"the token costs 2 %", we mean this RMSE is 2 % larger.

Two supporting numbers appear as well: **alive fraction** (share of samples where
the episode had not terminated) and **foot penetration** (metres the foot sinks
through the floor). The older B100 baseline used different metrics; they are
named explicitly wherever it is quoted.

## How we represented FSQ — two different designs

Both are finite scalar quantisation (FSQ) autoencoders trained on the same
retargeted clips, and both feed the policy through the same environment seam
(`tracking_clip_latent_obs`).

1. **Per-joint tokens (the colleague's / Kevin's design).** Each joint gets its
   own 32-number quantised code, so a whole-body frame is 19 x 32 = 608 numbers
   on H1 and 23 x 32 on G1. Faithful, but the width depends on the robot.
2. **Canonical tokens (ours).** ONE code per frame for the whole body, computed
   from task-space features (root height/attitude/velocity + end-effector
   positions), decoded per robot through a decoder conditioned on that robot's
   joint descriptions. Body-independent by construction — the same numbers reach
   H1 and G1.

The comparison in every experiment is against the **explicit reference**: the
plain per-joint target angle the policy normally receives.

## The experiments, in order

**B100 (2026-08-22) — the baseline the FSQ work sits on.** One policy, three
robots (H1, G1, booster_t1), 98.3 M steps, 8 h 35 wall. Reported as
*tracking-gated episode length* out of 1000 (an episode is cut when joint error
exceeds half the no-reference baseline, so length is itself a quality measure):
**454 / 827 / 438**. Joint error 0.034 / 0.020 / 0.077 rad^2, heading error
1.33 / 1.05 / 1.49 rad. This established that one policy can dance on three
bodies, and that heading was the weakest channel.

**FSQ A/B/C (08-22).** Compared the explicit reference, per-joint tokens, and
canonical tokens on a single clip. Result: token reconstruction ties the
baseline at best; canonical was already limited to ~0.3 rad reconstruction.

**The ladder (08-25).** Cleaned up the baseline and measured the FSQ penalty with
matched controls: tokens lost about **5 %** at multi-motion and zero-shot.

**FSQ-SCALE (overnight 08-25/26).** 20 training runs plus ~90 evaluations, to ask
whether that 5 % shrinks when one policy must cover many motions.

## What we found

**1. The 5 % was mostly measurement noise and a weak baseline.** Re-running the
same comparison at the improved recipe gives **+2.97 %**, and — the important
part — **a single evaluation seed moves any arm by up to 4.95 %**. Every FSQ
number ever reported on this project was a single-seed measurement of the same
size as its own noise. We now run four seeds per point as standard.

**2. Replacing the reference with tokens costs a small, constant 2 %.** At four
motion-set sizes (1, 4, 5 and 9 dances), the penalty is +3.3 / +1.4 / +3.0 /
+1.1 %. **It does not shrink as the motion set grows, and it does not grow.** The
main hypothesis of the campaign — that tokens get relatively better with more
motions, because a token spans a window while a target angle is a single instant
— is falsified across a 9x range.

**3. Adding tokens ON TOP of the reference IMPROVES tracking.** This is the first
positive result. Every previous experiment forced the token to *replace* the
reference, which is the one setting where it can only lose.

| | reference only | reference + token | change |
|---|---|---|---|
| 9 dances | 0.1314 | **0.1268** | **-3.5 %** |
| 5 dances | 0.1422 | **0.1395** | **-1.9 %** |

Both motion-set sizes improve, at four seeds each.

**4. Two robots cost more than one, but the learning is fine.** With one policy
on H1+G1 the token penalty roughly triples (about 6 %), the *reference* channel
alone costs H1 9-16 %, and G1 ends up tracking better than H1. The compute cost
is the real issue: the environment SPLITS a fixed number of parallel simulations
across robots, so a second robot halves each one's experience — matching
per-robot data costs about 2.7x, not 2x.

**5. Scaling past two robots is blocked by DATA, not compute.** Runs at 3, 4 and
6 robots fail immediately with "No clip mapping for robot t1" — booster_t1 has no
retargeted clip in the multi-motion set. One offline retarget per family unblocks
it.

**6. The body-independent (canonical) token does not work yet, and we know why
it is not three things.** Driving both robots from one shared stream tracks
*worse than sending no commands at all*. Its reconstruction error is 0.37 rad
against the per-joint design's 0.05. We eliminated, in order:

| suspected cause | test | result |
|---|---|---|
| codebook too small | 32x bigger book | 6 % better — not it |
| encoder cannot see knees/elbows | 14 body sites instead of 4 | **0 % better — not it** |
| code too narrow | 4 -> 32 numbers per frame | **32 % better — partly it** |

At 32 numbers per frame the code is nearly unique per frame (43 000 distinct
codes for 45 000 frames) and reconstruction is *still* 0.25 rad. So the remaining
bottleneck is the **shared decoder**, not the code.

**7. Token rate.** The policy can be driven at 20 tokens/second slightly BETTER
than at 40 (-3.7 %); 8/s costs +9 %, 4/s costs +24 %, 2/s costs +68 %. Making
the token span a longer window did **not** rescue the low rates.

## Where this leaves us

- **Tokens as a replacement for the reference: closed.** A constant ~2 % tax, no
  scaling benefit. The comparison was always rigged — the token is computed from
  the reference, so it cannot carry more than the reference does.
- **Tokens as an addition to the reference: open and positive.** ~2-3.5 % better
  than the reference alone, at two motion-set sizes.
- **Body-independent tokens: not working, cause narrowed to the decoder.**
- **Multi-robot scaling: blocked on one retarget per family**, which is a day of
  data work, not a research problem.
