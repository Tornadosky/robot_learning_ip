"""Shared helpers for the multi-motion experiment (one policy, several motions).

A single DeepMimic policy is trained on several LAFAN1 clips at once: loco-mujoco's
trajectory handler samples a random clip per episode and the GoalTrajMimic goal
feeds the *current* clip's reference into the observation, so the policy must read
the goal and track whichever motion it was dropped into. Body is the stock G1
(== the nominal variant) with the 3x PD gains that made the single-clip nominal
expert balance.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE / "scripts"))

from morphology_deepmimic import MIMIC_REWARD_PARAMS, GOAL_PARAMS, PD_GAINS  # noqa: E402

# MM_ROBOT=h1 switches the experiment to stock H1 with its default torque control
# (H1 experts never needed PD); default is the original G1 + 3x PD setup.
MM_ROBOT = os.environ.get("MM_ROBOT", "g1").lower()
BODY_ENV = "MjxUnitreeH1" if MM_ROBOT == "h1" else "MjxUnitreeG1"
TRAIN_CLIPS = ["dance2_subject1", "dance2_subject2", "dance2_subject3", "dance2_subject4"]
HELDOUT_CLIP = "dance2_subject5"
PD_SCALE = 0.0 if MM_ROBOT == "h1" else 3.0   # h1: torque control, no PD scaling

# Difficulty-weighted clip list (a.k.a. adaptive clip sampling, static form): the
# handler samples trajectories uniformly, so listing a clip multiple times raises
# its sampling probability. Weights come from the strengthened run's per-clip
# tracking fidelity (subject1=0.41, subject2=0.47 were the worst -> 2x; the easier
# subject3/subject4 -> 1x), so the policy spends more capacity on the clips it
# tracks worst. Eval still uses the 4 DISTINCT clips + the held-out one.
WEIGHTED_TRAIN_CLIPS = ["dance2_subject1", "dance2_subject1",
                        "dance2_subject2", "dance2_subject2",
                        "dance2_subject3", "dance2_subject4"]


def pd_control_params(scale: float = PD_SCALE) -> dict:
    return dict(control_type="PDControl",
                control_params=dict(p_gain=[g * scale for g in PD_GAINS["g1"]["p_gain"]],
                                    d_gain=[g * scale for g in PD_GAINS["g1"]["d_gain"]]))


def make_multiclip_env(clips, pd_scale: float = PD_SCALE, **overrides):
    """Build an MJX multi-clip DeepMimic env (shared mimic reward/goal)."""
    from loco_mujoco.task_factories import ImitationFactory, LAFAN1DatasetConf
    params = dict(
        lafan1_dataset_conf=LAFAN1DatasetConf(list(clips)),
        goal_type="GoalTrajMimic", goal_params=GOAL_PARAMS,
        reward_type="MimicReward", reward_params=MIMIC_REWARD_PARAMS,
        headless=True,
    )
    if MM_ROBOT != "h1":                 # h1 keeps the env's default torque control
        params.update(pd_control_params(pd_scale))
    params.update(overrides)
    return ImitationFactory.make(BODY_ENV, **params)
