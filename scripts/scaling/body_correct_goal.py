"""``MorphGoalTrajMimicRootErr`` -- the no-FSQ goal that commands the body the
reward actually scores.

What was wrong
--------------
``GoalTrajMimic`` builds its trajectory-side site block out of
``traj_data_single.site_xpos`` / ``site_xmat``.  Those arrays were produced once,
at load time, on the *nominal* topology model.  ``MorphMimicReward`` recomputes
the same quantities by forward kinematics on the *sampled* body.  Under online
morphology the actor is therefore commanded toward one point in space and paid
for reaching another, and no metric in the run can see the gap.

What this class changes
-----------------------
Exactly one thing: the trajectory-side site block now comes from
``scaling.body_correct_reference.body_correct_reference`` -- the same provider,
the same cursor, the same sampled body, the same site/body/root indices the
reward uses.  Everything else is inherited:

* the current-state site block (already computed from live ``data``, so it was
  always body-correct);
* the reference joint positions/velocities (body-independent, apart from the
  joint-range clamp which the provider now applies to both consumers from one
  array);
* the heading-frame root position error appended by ``GoalTrajMimicRootErr``,
  which is what makes the world-frame position ``MimicReward``'s qpos term
  scores observable at all (Finding 41).

Site sets
---------
The goal mimics the environment's full ``sites_for_mimic`` list (15 on the
LocoMuJoCo humanoids); the reward is configured with the 5-site subset shared by
H1 and G1.  Both lists start with ``upper_body_mimic``, so the reward's
relative-site rows are a row-subset of the goal's -- ``reward_site_rows``
computes the mapping and ``__init__`` fails loudly if the subset relation ever
breaks.  Widening the goal to the reward's targets rather than narrowing the
goal to 5 sites keeps the reference information the policy already had.

Registration is additive: ``Goal.registered[name] = cls``.  No vendored
``loco-mujoco`` file is modified.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
from typing import Any, Dict, List

import numpy as np

_SCRIPTS = str(_Path(__file__).resolve().parents[1])
if _SCRIPTS not in _sys.path:
    _sys.path.insert(0, _SCRIPTS)
_H1MD = str(_Path(__file__).resolve().parents[1] / "h1md")
if _H1MD not in _sys.path:
    _sys.path.insert(0, _H1MD)

from goal_rooterr import GoalTrajMimicRootErr  # noqa: E402

from scaling.body_correct_reference import (  # noqa: E402
    body_correct_reference,
    reward_site_rows,
)

#: The name the trainer, the manifest and the pipeline runner all use.
GOAL_NAME = "MorphGoalTrajMimicRootErr"


class MorphGoalTrajMimicRootErr(GoalTrajMimicRootErr):
    """Body-correct DeepMimic goal + heading-frame root error.

    Args:
        info_props: standard LocoMuJoCo info properties.
        rel_body_names: forwarded to ``GoalTrajMimic``.
        include_z: forwarded to ``GoalTrajMimicRootErr`` (3 vs 2 root-error dims).
        reward_sites_for_mimic: the reward's site-name list, used only to assert
            at construction time that the reward's targets really are a subset
            of what this goal exposes.  ``None`` skips the check.
    """

    def __init__(self, info_props: Dict, rel_body_names: List[str] = None,
                 include_z: bool = True,
                 reward_sites_for_mimic: List[str] = None,
                 root_frame: str = "absolute",
                 **kwargs):
        super().__init__(info_props, rel_body_names=rel_body_names,
                         include_z=include_z, **kwargs)
        # Must match the reward's convention, or the actor is shown a root error
        # measured in a different frame than the one it is paid for.
        self._root_frame = str(root_frame)
        self._reward_site_names = (
            None if reward_sites_for_mimic is None else list(reward_sites_for_mimic)
        )
        #: rows of this goal's relative-site block that the reward scores.
        #: Filled in _init_from_mj, once site ids exist.
        self.reward_site_rows = None

    def _init_from_mj(self, env: Any, model, data, current_obs_size: int):
        super()._init_from_mj(env, model, data, current_obs_size)

        # NOTE ON ORDERING: MuJoCoBase builds the goal (line ~135) BEFORE the
        # reward (line ~169), so `env._reward_function` does not exist yet.  The
        # subset check therefore runs here only when the caller named the
        # reward's sites explicitly; otherwise it is deferred to
        # resolve_reward_site_rows(), which the tests and the renderer call once
        # the environment is fully built.
        if self._reward_site_names is None:
            return

        import mujoco

        reward_ids = np.asarray(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
                for name in self._reward_site_names
            ]
        )
        if (reward_ids < 0).any():
            missing = [
                name for name, i in zip(self._reward_site_names, reward_ids) if i < 0
            ]
            raise ValueError(f"Unknown mimic site(s) for the reward: {missing}")

        # Raises if the reward scores a site this goal never exposes, or if the
        # two blocks are relative to different main sites.
        self.reward_site_rows = reward_site_rows(self._rel_site_ids, reward_ids)

    def resolve_reward_site_rows(self, env: Any) -> np.ndarray:
        """Rows of this goal's site block that the reward scores.

        Host-side only.  Raises if the reward's site set is not a subset of the
        goal's, which is the failure mode that would silently reintroduce the
        command/score mismatch this class exists to remove.
        """
        if self.reward_site_rows is None:
            reward = getattr(env, "_reward_function", None)
            reward_ids = getattr(reward, "_rel_site_ids", None)
            if reward_ids is None:
                raise ValueError(
                    "Environment exposes no mimic reward site ids; cannot map "
                    "the goal's site block onto the reward's targets."
                )
            self.reward_site_rows = reward_site_rows(
                self._rel_site_ids, np.asarray(reward_ids).ravel()
            )
        return self.reward_site_rows

    def get_obs_and_update_state(self, env, model, data, carry, backend):
        """Same layout as ``GoalTrajMimicRootErr``, body-correct targets.

        The parent is deliberately not called: its trajectory-side block is the
        thing being replaced.  Only the root-error tail is re-derived here, from
        the same trajectory sample, so the two classes stay in step.
        """
        if backend == np:
            from scipy.spatial.transform import Rotation as R
        else:
            from jax.scipy.spatial.transform import Rotation as R

        rel_site_ids = self._rel_site_ids
        rel_body_ids = self._site_bodyid[rel_site_ids]

        # --- trajectory side: the shared provider, on the sampled body.
        # Velocities are included because the goal block carries site_rvel; that
        # is what pays for the com_pos/com_vel passes here but not in the reward.
        bundle = body_correct_reference(
            env, model, data, carry, backend,
            rel_site_ids=rel_site_ids,
            rel_body_ids=rel_body_ids,
            body_rootid=self._body_rootid,
            root_frame=getattr(self, "_root_frame", "absolute"),
            include_site_velocity=True,
        )

        traj_goal_obs = backend.concatenate([
            bundle.reference_qpos_clamped[self._qpos_ind],
            bundle.reference_qvel[self._qvel_ind],
            backend.ravel(bundle.relative_site_position),
            backend.ravel(bundle.relative_site_orientation),
            backend.ravel(bundle.relative_site_velocity),
        ])

        if self.visualize_goal:
            # Inherited, and still nominal-body: set_visuals reads the
            # trajectory's stored site_xpos. It is off everywhere in this
            # pipeline (GOAL_PARAMS sets visualize_goal=False) and the supported
            # visual path is render_cross_topology_policy.py --show-targets,
            # which uses this same provider.
            carry = self.set_visuals(env, model, data, carry, backend)

        # --- current state side: already body-correct (computed from live data)
        if len(rel_site_ids) > 0:
            from loco_mujoco.core.utils.math import calculate_relative_site_quatities

            site_rpos, site_rangles, site_rvel = calculate_relative_site_quatities(
                data, rel_site_ids, rel_body_ids, self._body_rootid, backend)
            goal = backend.concatenate([
                backend.ravel(site_rpos),
                backend.ravel(site_rangles),
                backend.ravel(site_rvel),
                backend.ravel(traj_goal_obs),
            ])
        else:
            goal = traj_goal_obs

        # --- heading-frame root position error (GoalTrajMimicRootErr)
        # Read from the bundle rather than re-fetching the trajectory: the root
        # is a free joint, which the joint-range clamp never touches, so this is
        # the same array the parent would have used -- with no second chance for
        # the two to end up on different cursors.
        err_world = bundle.reference_qpos_clamped[:3] - data.qpos[:3]
        quat = data.qpos[3:7]
        rot = R.from_quat(backend.concatenate([quat[1:], quat[:1]])).as_matrix()
        err_local = rot.T @ err_world
        if not self._include_z:
            err_local = err_local[:2]

        return backend.concatenate([goal, err_local]), carry


def goal_block_slices(goal) -> Dict[str, slice]:
    """Where each block lives inside the emitted goal observation.

    Derived from the goal's own index arrays rather than hard-coded widths, so a
    change of site set or topology cannot silently desynchronise a consumer that
    slices the observation (tests, renderer sidecar, offline probes).

    Layout, in emission order::

        current_site_position | current_site_orientation | current_site_velocity
        reference_qpos | reference_qvel
        target_site_position | target_site_orientation | target_site_velocity
        root_position_error
    """
    n_rel = len(np.asarray(goal._rel_site_ids)) - 1
    qpos_len = len(np.asarray(goal._qpos_ind))
    qvel_len = len(np.asarray(goal._qvel_ind))

    out: Dict[str, slice] = {}
    cursor = 0
    for name, width in (
        ("current_site_position", 3 * n_rel),
        ("current_site_orientation", 3 * n_rel),
        ("current_site_velocity", 6 * n_rel),
        ("reference_qpos", qpos_len),
        ("reference_qvel", qvel_len),
        ("target_site_position", 3 * n_rel),
        ("target_site_orientation", 3 * n_rel),
        ("target_site_velocity", 6 * n_rel),
        ("root_position_error", goal._n_root_err),
    ):
        out[name] = slice(cursor, cursor + width)
        cursor += width
    if cursor != goal.dim:
        raise ValueError(
            f"goal_block_slices computed {cursor} dims but the goal advertises "
            f"{goal.dim}; the emission order and this map have diverged."
        )
    return out


def register(name: str = GOAL_NAME) -> None:
    """Make the goal selectable by string, without touching vendored code.

    ``MuJoCoBase._setup_goal`` resolves ``goal_type`` through ``Goal.registered``.
    """
    from loco_mujoco.core.observations.goals import Goal

    Goal.registered[name] = MorphGoalTrajMimicRootErr


register()
