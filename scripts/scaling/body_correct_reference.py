"""One body-correct reference provider, shared by goal, reward, eval and render.

The bug this closes
-------------------
``GoalTrajMimic`` reads its trajectory-side site block straight out of
``traj_data.site_xpos``/``site_xmat``, which the loader computed once on the
**nominal** topology model.  ``MorphMimicReward`` recomputes the same site block
by forward kinematics on the **sampled** body.  With online morphology the two
disagree: the actor is commanded toward the nominal robot's hand/foot positions
while the reward scores the sampled robot's.  Nothing in the training loop can
notice, because neither side ever sees the other's numbers.

This module is the single place where "where should this body be, right now"
is answered.  Every consumer -- goal observation, reward, FK evaluator, target
renderer, verifier -- calls :func:`body_correct_reference` and gets the same
immutable :class:`ReferenceBundle`.  There is deliberately no second copy of the
FK formula anywhere else.

Why the pieces are shaped the way they are
------------------------------------------
* **The clamp comes first.**  A sampled body may be physically unable to reach
  part of the shared reference's joint trajectory.  One clipped ``qpos`` array
  feeds both the joint term and the FK pass, so the two views of "the reference"
  cannot drift apart.  (Pipeline-v3 verified the clamp is bit-for-bit a no-op
  while ``joint_range_shift`` stays pinned; it stops being one the moment leg
  randomization widens.)
* **The model is rebuilt, never taken from the caller.**  ``loco-mujoco`` hands
  observations the *CPU* ``MjModel`` at reset and the sampled MJX model during a
  step (``mujoco_mjx.mjx_reset`` line ~118 vs ``mjx_step`` line ~214).  Relying
  on the ``model`` argument would silently give nominal targets on the reset
  step.  So the body is always rebuilt from ``env._apply_morphology(env.sys,
  morphology)``.
* **Velocities need more than kinematics.**  ``calc_site_velocities`` reads
  ``data.cvel`` and ``data.subtree_com``, which ``mjx.kinematics`` does not
  fill.  Asking for :attr:`ReferenceBundle.relative_site_velocity` therefore
  runs ``kinematics -> com_pos -> com_vel``.  Callers that only score positions
  and orientations (the reward) pass ``include_site_velocity=False`` and pay for
  kinematics alone; the field is then ``None`` rather than a stale array, so a
  wrong number can never be read by accident.

Index provenance
----------------
Site, body and root indices are *passed in* by the caller and are expected to be
exactly the reward's ``_rel_site_ids`` / ``_rel_body_ids`` /
``model.body_rootid``.  :func:`reward_site_rows` maps the reward's relative-site
rows onto a wider consumer's rows (the goal mimics 15 sites, the reward 5) so a
test can compare them without either side re-deriving an ordering.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import Any, Optional, Sequence

import numpy as np

#: Static ``(qpos indices, joint ids)`` of clampable joints, keyed by topology.
#: Module-level on purpose: a reward/goal instance is part of the traced pytree,
#: so caching jax arrays on ``self`` changes its flattened structure between the
#: AOT compile and the call ("Computation compiled for 901 inputs but called with
#: 1").  Numpy index arrays are static and index jax arrays fine.
_CLAMP_IDX_CACHE: dict = {}


@dataclass(frozen=True)
class ReferenceBundle:
    """What the reference says this body should be doing, right now.

    Immutable on purpose: every consumer reads the same object, nobody patches
    one field and hands it on.

    Attributes:
        reference_qpos_clamped: full-length reference ``qpos`` after the current
            body's joint-range clamp.  This exact array is what the FK pass ran
            on, so the joint reward and the site targets cannot disagree.
        reference_qvel: full-length reference ``qvel`` (body-independent).
        relative_site_position: ``(n_sites - 1, 3)`` positions relative to the
            main mimic site.
        relative_site_orientation: ``(n_sites - 1, 3)`` rotation vectors
            relative to the main mimic site.
        relative_site_velocity: ``(n_sites - 1, 6)`` local-frame relative site
            velocities, or ``None`` when ``include_site_velocity=False``.
        site_position_world / site_orientation_world: the same sites before the
            relative transform, ``(n_sites, 3)`` and ``(n_sites, 9)``.  The
            reward and goal never look at these; the renderer needs somewhere to
            put a marker, and taking them from this same FK pass is what keeps
            the marker and the scored target the same computation.
        traj_no / subtraj_step_no: the exact trajectory cursor this bundle was
            read at.  Carried so a renderer or sidecar never has to reconstruct
            phase from a frame counter.
        body_correct: ``True`` when the site block came from FK on the sampled
            body; ``False`` when it fell back to the trajectory's stored
            nominal-body site data (no morphology on this env).
    """

    reference_qpos_clamped: Any
    reference_qvel: Any
    relative_site_position: Any
    relative_site_orientation: Any
    relative_site_velocity: Optional[Any]
    site_position_world: Any
    site_orientation_world: Any
    traj_no: Any
    subtraj_step_no: Any
    body_correct: bool


# --------------------------------------------------------------------------- #
# joint-range clamp
# --------------------------------------------------------------------------- #
def clampable_qpos_index(sys) -> tuple[np.ndarray, np.ndarray]:
    """``(qpos indices, joint ids)`` of limited single-dof joints.  Static."""
    import mujoco

    jnt_type = np.asarray(sys.jnt_type)
    jnt_limited = np.asarray(sys.jnt_limited)
    jnt_qposadr = np.asarray(sys.jnt_qposadr)
    key = (int(jnt_type.shape[0]), jnt_type.tobytes(), jnt_limited.tobytes())
    cached = _CLAMP_IDX_CACHE.get(key)
    if cached is not None:
        return cached
    single_dof = (jnt_type == mujoco.mjtJoint.mjJNT_HINGE) | (
        jnt_type == mujoco.mjtJoint.mjJNT_SLIDE
    )
    jids = np.nonzero(single_dof & (jnt_limited != 0))[0]
    cached = (jnt_qposadr[jids].astype(np.int32), jids.astype(np.int32))
    _CLAMP_IDX_CACHE[key] = cached
    return cached


def clamp_reference_qpos(qpos_traj_full, jnt_range, sys, backend: ModuleType):
    """Clip the reference joint angles into THIS body's joint ranges."""
    qpos_idx, jids = clampable_qpos_index(sys)
    lo = jnt_range[jids, 0]
    hi = jnt_range[jids, 1]
    vals = backend.clip(qpos_traj_full[qpos_idx], lo, hi)
    if backend is np:
        out = np.array(qpos_traj_full, copy=True)
        out[np.asarray(qpos_idx)] = vals
        return out
    return qpos_traj_full.at[qpos_idx].set(vals)


# --------------------------------------------------------------------------- #
# the sampled body
# --------------------------------------------------------------------------- #
def sampled_body_model(env: Any, morphology, backend: ModuleType):
    """The MJX model physics is actually stepping, or ``None``.

    ``None`` means "this env has no per-env morphology" (a plain single-body
    build, a CPU env, or ``--morphology`` unset), in which case the trajectory's
    own stored site data is already correct for the body and no FK is needed.
    """
    patch = getattr(env, "_apply_morphology", None)
    if morphology is None or patch is None or backend is np:
        return None
    return patch(env.sys, morphology)


def morphology_of(carry: Any):
    """``carry.morphology`` if this is an online-morphology carry, else None."""
    return getattr(carry, "morphology", None)


# --------------------------------------------------------------------------- #
# the provider
# --------------------------------------------------------------------------- #
def body_correct_reference(
    env: Any,
    model: Any,
    data: Any,
    carry: Any,
    backend: ModuleType,
    *,
    rel_site_ids,
    rel_body_ids,
    body_rootid,
    morphology=None,
    body_model=None,
    traj_no=None,
    subtraj_step_no=None,
    subtraj_step_no_init=None,
    root_frame: str = "absolute",
    include_site_velocity: bool = True,
) -> ReferenceBundle:
    """Body-correct reference bundle at the current trajectory cursor.

    Args:
        env: the LocoMuJoCo environment (needs ``th``, ``sys``, and, for the
            body-correct path, ``_apply_morphology``).
        model: the model the *caller* was handed.  Used only for the CPU-MuJoCo
            path and for ``mujoco.MjModel`` detection -- the sampled body is
            always rebuilt, never taken from here (see module docstring).
        data: an MJX ``Data`` (or ``MjData``) to reuse as the FK scratch buffer.
            Only ``qpos``/``qvel`` are overwritten.
        carry: the environment carry.  ``carry.traj_state`` supplies the cursor
            and ``carry.morphology`` the body, unless overridden below.
        backend: ``numpy`` or ``jax.numpy``.
        rel_site_ids: the reward's ``_rel_site_ids`` (main mimic site first).
        rel_body_ids: the reward's ``_rel_body_ids``.
        body_rootid: ``model.body_rootid`` of the *nominal* model -- morphology
            never changes the kinematic tree, so this is topology-static.
        morphology: override for ``carry.morphology``.
        body_model: an already-built sampled model, to avoid rebuilding it when
            the caller has one (the reward builds it for the clamp).
        traj_no / subtraj_step_no: override the cursor, for a verifier or a
            renderer replaying a logged step.
        include_site_velocity: run ``com_pos``/``com_vel`` so
            ``relative_site_velocity`` is filled.  ``False`` leaves it ``None``
            and costs one kinematics pass only.

    Returns:
        ReferenceBundle: see the class docstring.
    """
    from loco_mujoco.core.utils.math import calculate_relative_site_quatities

    traj_state = getattr(carry, "traj_state", None)
    if traj_no is None:
        traj_no = traj_state.traj_no
    if subtraj_step_no is None:
        subtraj_step_no = traj_state.subtraj_step_no

    traj_single = env.th.traj.data.get(traj_no, subtraj_step_no, backend)

    # --- root frame -------------------------------------------------------
    # The reset places the robot at world XY (0, 0), and
    # RootPoseTrajTerminalStateHandler compares against the reference's
    # DISPLACEMENT from the episode's own start frame. Upstream MimicReward and
    # GoalTrajMimicRootErr instead read the reference's ABSOLUTE root XY, so on
    # a random start phase a perfectly-posed robot is charged for the clip's
    # absolute position at that phase -- up to 1.34 m on dance2_subject4, which
    # is 100% of the qpos error at reset. The reward then rises for walking to a
    # point the terminal criterion terminates the episode for reaching.
    #
    # "episode_start" re-bases reference root XY the same way the terminal does,
    # so reward, goal and termination finally agree. Height is deliberately NOT
    # re-based: the robot stands on a floor, not on its start height. Site
    # targets are unaffected either way -- they are relative to the main mimic
    # site, so a translation of the whole reference cancels.
    if root_frame == "episode_start":
        if subtraj_step_no_init is None:
            subtraj_step_no_init = traj_state.subtraj_step_no_init
        initial = env.th.traj.data.get(traj_no, subtraj_step_no_init, backend)
        rebased = traj_single.qpos[:2] - initial.qpos[:2]
        if backend is np:
            qpos = np.array(traj_single.qpos, copy=True)
            qpos[:2] = rebased
        else:
            qpos = traj_single.qpos.at[:2].set(rebased)
        traj_single = traj_single.replace(qpos=qpos)
    elif root_frame != "absolute":
        raise ValueError(
            f"root_frame must be 'absolute' or 'episode_start', got {root_frame!r}"
        )

    if morphology is None:
        morphology = morphology_of(carry)
    if body_model is None:
        body_model = sampled_body_model(env, morphology, backend)

    # --- one clamped reference array, shared by the joint term and the FK pass
    if body_model is not None:
        reference_qpos = clamp_reference_qpos(
            traj_single.qpos, body_model.jnt_range, env.sys, backend
        )
    else:
        reference_qpos = traj_single.qpos
    reference_qvel = traj_single.qvel

    if body_model is not None:
        from mujoco import mjx

        ref = data.replace(qpos=reference_qpos, qvel=reference_qvel)
        ref = mjx.kinematics(body_model, ref)
        if include_site_velocity:
            # calc_site_velocities reads cvel and subtree_com, which kinematics
            # alone does not fill; these two passes are what make them real.
            ref = mjx.com_pos(body_model, ref)
            ref = mjx.com_vel(body_model, ref)
        body_correct = True
    else:
        import mujoco

        if isinstance(model, mujoco.MjModel) and backend is np:
            # CPU path: the caller is responsible for having mutated `model`
            # (see cpu_morphology_model), so mj_forward is body-correct here.
            d = mujoco.MjData(model)
            d.qpos[:] = np.asarray(reference_qpos)
            d.qvel[:] = np.asarray(reference_qvel)
            mujoco.mj_forward(model, d)
            ref = d
            body_correct = True
        else:
            # No per-env morphology: the trajectory's own site data already
            # describes this body.  Stock behaviour, and the only branch that
            # does not run FK.
            ref = traj_single
            body_correct = False

    rpos, rangles, rvel = calculate_relative_site_quatities(
        ref, rel_site_ids, rel_body_ids, body_rootid, backend
    )

    return ReferenceBundle(
        reference_qpos_clamped=reference_qpos,
        reference_qvel=reference_qvel,
        relative_site_position=rpos,
        relative_site_orientation=rangles,
        relative_site_velocity=rvel if include_site_velocity else None,
        site_position_world=ref.site_xpos[rel_site_ids],
        site_orientation_world=ref.site_xmat[rel_site_ids],
        traj_no=traj_no,
        subtraj_step_no=subtraj_step_no,
        body_correct=body_correct,
    )


# --------------------------------------------------------------------------- #
# CPU-side helpers (renderer, verifier, offline audits)
# --------------------------------------------------------------------------- #
def cpu_morphology_model(model, family_key: str, morphology) -> Any:
    """A CPU ``MjModel`` carrying the same mutation as ``_apply_morphology``.

    The MJX patch is a jax function over ``env.sys``; a renderer or an
    independent verifier needs the same body as a plain ``mujoco.MjModel``.  The
    numbers below mirror ``FamilyMorphMixin._apply_morphology`` term for term.

    Only ``body_pos`` affects forward kinematics, so a site-target check would
    pass with the leg scaling alone -- the mass/inertia/damping/actuator terms
    are applied anyway so that a caller who *steps* this model gets the same
    body the GPU stepped, not a kinematic look-alike.
    """
    import copy

    import mujoco

    from scaling.family_morphology import FAMILY_BODIES

    out = copy.deepcopy(model)
    bodies = FAMILY_BODIES[family_key]
    morphology = np.asarray(morphology, dtype=np.float64)
    leg_scale, torso_mass_scale, damping_scale, strength_scale = morphology[:4]

    def body_id(name: str) -> int:
        index = mujoco.mj_name2id(out, mujoco.mjtObj.mjOBJ_BODY, name)
        if index < 0:
            raise ValueError(f"{family_key}: no body named {name!r}")
        return index

    leg_ids = np.asarray(
        [body_id(n) for n in (*bodies.knee_bodies, *bodies.ankle_bodies)],
        dtype=np.int64,
    )
    parents = np.asarray(out.body_parentid)
    inertial_ids = np.asarray(
        sorted({int(parents[i]) for i in leg_ids}), dtype=np.int64
    )
    torso_id = body_id(bodies.torso_body)

    out.body_pos[leg_ids, 2] *= leg_scale

    # z-stretch => volume ~ s, inertia ~ s * mean(scale)^2 (the MJX convention)
    body_scale = np.ones_like(out.body_ipos)
    body_scale[inertial_ids, 2] = leg_scale
    volume_scale = np.prod(body_scale, axis=1)
    inertia_scale = volume_scale * np.square(np.mean(body_scale, axis=1))
    out.body_ipos[:] = out.body_ipos * body_scale
    out.body_mass[:] = out.body_mass * volume_scale
    out.body_inertia[:] = out.body_inertia * inertia_scale[:, None]
    out.body_mass[torso_id] *= torso_mass_scale
    out.body_inertia[torso_id] *= torso_mass_scale

    out.dof_damping[:] = out.dof_damping * damping_scale
    out.actuator_gainprm[:, 0] *= strength_scale
    out.actuator_forcerange[:] = out.actuator_forcerange * strength_scale
    return out


class CpuModelCache:
    """Mutated CPU models, keyed by morphology, with a bounded footprint.

    Every offline consumer (renderer, FK evaluator, contact plots) needs "the
    CPU model for this body" and would otherwise either deep-copy per frame or
    mutate one model in place -- the latter being how a stale body silently
    leaks into the next frame's numbers.

    Keyed by morphology alone, never by phase: a target cached by phase would
    make a moving reference look stationary.
    """

    def __init__(self, base_model, family_key: str, max_entries: int = 64):
        self._base = base_model
        self._family = str(family_key)
        self._max = int(max_entries)
        self._models: dict = {}

    @staticmethod
    def key(morphology) -> tuple:
        # 9 decimals is far below any morphology difference that moves a site,
        # and stable against float32/float64 round-tripping.
        return tuple(round(float(v), 9) for v in np.asarray(morphology).ravel())

    def get(self, morphology):
        key = self.key(morphology)
        model = self._models.get(key)
        if model is None:
            model = (
                self._base
                if not key
                else cpu_morphology_model(self._base, self._family, np.asarray(key))
            )
            if len(self._models) >= self._max:
                # FIFO: rollouts revisit a body for a whole episode, so the
                # oldest entry is the one least likely to come back.
                self._models.pop(next(iter(self._models)))
            self._models[key] = model
        return model

    def __len__(self) -> int:
        return len(self._models)


def cpu_reference_bundle(
    env: Any,
    cpu_model,
    traj_no: int,
    subtraj_step_no: int,
    *,
    rel_site_ids,
    rel_body_ids,
    include_site_velocity: bool = True,
) -> ReferenceBundle:
    """The same bundle, from the C engine, on an already-mutated CPU model.

    Used by the renderer and the acceptance tests as the independent check: it
    shares the clamp and the site-quantity call with the production path but
    reaches them through ``mujoco.mj_forward`` rather than MJX.
    """
    from types import SimpleNamespace

    return body_correct_reference(
        env,
        cpu_model,
        None,
        SimpleNamespace(traj_state=None, morphology=None),
        np,
        rel_site_ids=rel_site_ids,
        rel_body_ids=rel_body_ids,
        body_rootid=cpu_model.body_rootid,
        traj_no=traj_no,
        subtraj_step_no=subtraj_step_no,
        include_site_velocity=include_site_velocity,
    )


def reward_site_rows(
    consumer_site_ids: Sequence[int], reward_site_ids: Sequence[int]
) -> np.ndarray:
    """Rows of a wider consumer's relative-site block that the reward scores.

    ``calculate_relative_site_quatities`` drops row 0 (the main mimic site) and
    expresses everything relative to it, so a row index is
    ``position_in_site_list - 1``.  Both lists must start with the same main
    site or the two blocks are not comparable at all.
    """
    consumer = [int(i) for i in np.asarray(consumer_site_ids).ravel()]
    reward = [int(i) for i in np.asarray(reward_site_ids).ravel()]
    if not consumer or not reward:
        raise ValueError("Both site lists must be non-empty.")
    if consumer[0] != reward[0]:
        raise ValueError(
            f"Main mimic site differs: consumer {consumer[0]} vs reward "
            f"{reward[0]}; the relative-site blocks are not comparable."
        )
    missing = [i for i in reward[1:] if i not in consumer]
    if missing:
        raise ValueError(
            f"Reward sites {missing} are not part of the consumer's site set "
            f"{consumer}; the goal cannot expose what the reward scores."
        )
    return np.asarray([consumer.index(i) - 1 for i in reward[1:]], dtype=np.int64)
