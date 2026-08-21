"""Render a trained cross-topology policy driving every robot it was trained on.

The existing renderers under ``scripts/`` target the static-XML morphology
pipeline and cannot load a padded cross-topology checkpoint.  This one rolls the
policy out inside the same ``ParallelMorphVecEnv`` the evaluator uses, records
each robot's ``qpos`` per step, and replays those trajectories through MuJoCo's
offscreen renderer.

Rendering is decoupled from simulation on purpose: MJX runs on the GPU and has
no renderer, so we capture state during the rollout and draw it afterwards on
the CPU model.  Run with ``MUJOCO_GL=osmesa``.

.. warning::
   **Do not run this under WSL.** Constructing a second ``mujoco.Renderer`` in
   one process segfaults there on both the ``egl`` and ``osmesa`` backends
   (mujoco 3.11) -- and a two-topology render needs one renderer per family,
   because H1 and G1 have different geom counts. It is not the GL backend, not
   CUDA/EGL coexistence (it reproduces with ``JAX_PLATFORMS=cpu``), and not
   renderer close/recreate. The Windows ``wgl`` path handles the same sequence
   without complaint, so render there: the rollout is only a few hundred steps
   with one environment per topology and runs acceptably on CPU JAX.

Body-correct target markers (``--show-targets``)
------------------------------------------------
Replaying rollout ``qpos`` on a *nominal* CPU model shows policy behaviour but
proves nothing about whether the randomized body, the reward target, the
trajectory phase and the visual markers agree -- a nominal-body marker overlay
would in fact be misleading.  With ``--show-targets`` the renderer:

* logs the exact trajectory cursor and morphology generation from the rollout,
  per control step, rather than reconstructing phase from the frame number
  (asynchronous resets make those two diverge);
* rebuilds the same randomized body as a CPU ``MjModel`` and replays the
  achieved ``qpos`` on *that*;
* takes the target sites from ``scaling.body_correct_reference`` -- the same
  provider ``MorphMimicReward`` scores and the goal commands;
* cross-checks the CPU provider against the production MJX provider at every
  logged frame and exits non-zero if they diverge beyond
  ``--fail-target-mismatch-m``;
* writes a JSON sidecar so the claim is auditable without re-watching the video.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

WORKSPACE = Path(__file__).resolve().parents[2]
SCRIPTS = WORKSPACE / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from scaling.evaluate_cross_humanoid_policy import (  # noqa: E402
    _env_args,
    _find_manifest,
    _resolve_checkpoint,
)
from scaling.parallel_cross_humanoid_train import (  # noqa: E402
    build_cross_humanoid_env,
    trainer_for,
)

#: Marker colours, per TARGET_SPHERE_RENDERER_PATCH.md. Keyed by mimic-site
#: name so a family with a different site order still gets the right colour.
TARGET_COLORS = {
    "upper_body_mimic": (1.0, 1.0, 1.0, 0.85),
    "left_hand_mimic": (0.15, 0.9, 0.25, 0.85),
    "left_foot_mimic": (0.15, 0.9, 0.25, 0.85),
    "right_hand_mimic": (0.95, 0.15, 0.85, 0.85),
    "right_foot_mimic": (0.95, 0.15, 0.85, 0.85),
}
DEFAULT_TARGET_COLOR = (1.0, 0.85, 0.2, 0.85)
ACHIEVED_COLOR = (0.2, 0.9, 0.95, 0.9)
TARGET_RADIUS = 0.045
ACHIEVED_RADIUS = 0.022


@dataclass
class GroupRollout:
    """Everything the target overlay needs, logged per control step.

    Phase is logged, never inferred: with asynchronous resets the rendered frame
    index and the trajectory timestamp diverge the first time an episode ends.
    """

    qpos: list = field(default_factory=list)
    traj_no: list = field(default_factory=list)
    subtraj_step_no: list = field(default_factory=list)
    morphology: list = field(default_factory=list)
    morphology_generation: list = field(default_factory=list)
    absorbing: list = field(default_factory=list)
    reset_happened: list = field(default_factory=list)

    def as_arrays(self) -> dict:
        return {
            "qpos": np.stack(self.qpos),
            "traj_no": np.asarray(self.traj_no, dtype=np.int64),
            "subtraj_step_no": np.asarray(self.subtraj_step_no, dtype=np.int64),
            "morphology": np.stack(self.morphology)
            if self.morphology
            else np.zeros((0, 0)),
            "morphology_generation": np.asarray(
                self.morphology_generation, dtype=np.int64
            ),
            "absorbing": np.asarray(self.absorbing, dtype=bool),
            "reset_happened": np.asarray(self.reset_happened, dtype=bool),
        }


def _group_scalar(value, index: int):
    array = np.asarray(value)
    return array[index] if array.ndim else array


def rollout(
    env,
    agent_conf,
    agent_state,
    steps: int,
    seed: int,
    zero_action: bool,
    buffer=None,
    switch_step: int | None = None,
    force_motion: int | None = None,
    target_env_index: int = 0,
):
    """Step the padded env, returning per-group qpos tracks (+ action log + log).

    ``buffer`` is the motion-latent table (fake or learned); when
    ``switch_step`` is set, the commanded motion id flips to the next
    trajectory (same phase) from that step on — the mid-rollout command
    switch of Gate 6.  Actions are logged so the discontinuity around the
    switch can be measured rather than eyeballed.
    """
    keys = jax.random.split(jax.random.PRNGKey(seed), env.num_envs)
    observation, state = jax.jit(env.reset)(keys)
    variables = {
        "params": agent_state.train_state.params,
        "run_stats": agent_state.train_state.run_stats,
    }
    step = jax.jit(env.step)

    def act(obs, z):
        if zero_action:
            return jnp.zeros((obs.shape[0], env.max_action_dim), dtype=obs.dtype)
        if z is None:
            (policy, _), _ = agent_conf.network.apply(
                variables, obs, mutable=["run_stats"]
            )
        else:
            (policy, _), _ = agent_conf.network.apply(
                variables, obs, z, mutable=["run_stats"]
            )
        return policy.mean()

    tracks = [[] for _ in env.groups]
    logs = [GroupRollout() for _ in env.groups]
    action_log = []
    # `done` from the PREVIOUS step: env.step auto-resets inside itself, so the
    # state we are about to log is already the post-reset one on a done step.
    # That pairing is what makes reset_happened mean "this frame is a fresh
    # episode", which is exactly when the morphology generation changes.
    previous_done = np.zeros((env.num_envs,), dtype=bool)
    previous_absorbing = np.zeros((env.num_envs,), dtype=bool)
    for t in range(steps):
        for gi, group_state in enumerate(state.group_states):
            index = min(int(target_env_index), env.groups[gi].size - 1)
            flat = env.groups[gi].start + index
            inner = group_state.env_state
            tracks[gi].append(np.asarray(inner.data.qpos[index]))
            carry = inner.additional_carry
            log = logs[gi]
            log.qpos.append(np.asarray(inner.data.qpos[index]))
            log.traj_no.append(int(_group_scalar(carry.traj_state.traj_no, index)))
            log.subtraj_step_no.append(
                int(_group_scalar(carry.traj_state.subtraj_step_no, index))
            )
            morphology = getattr(carry, "morphology", None)
            log.morphology.append(
                np.zeros(0) if morphology is None
                else np.asarray(morphology[index], dtype=np.float64)
            )
            generation = getattr(carry, "morphology_generation", None)
            log.morphology_generation.append(
                -1 if generation is None else int(_group_scalar(generation, index))
            )
            log.absorbing.append(bool(previous_absorbing[flat]))
            log.reset_happened.append(bool(previous_done[flat]))
        z = None
        if buffer is not None:
            ts = state.additional_carry.traj_state
            if force_motion is not None:
                # identical canonical command (motion, time) for EVERY env —
                # the same-z-across-families demonstration
                n = ts.traj_no.shape[0]
                traj_no = jnp.full((n,), force_motion, dtype=jnp.int32)
                step_no = jnp.full(
                    (n,), min(t, 799), dtype=jnp.int32
                )
                z = buffer.get(traj_no, step_no)
            else:
                traj_no = ts.traj_no
                if switch_step is not None and t >= switch_step:
                    traj_no = (traj_no + 1) % buffer.num_trajectories
                z = buffer.get(traj_no, ts.subtraj_step_no)
        action = act(observation, z)
        action_log.append(np.asarray(action))
        observation, _, absorbing, done, _, state = step(state, action)
        previous_done = np.asarray(done, dtype=bool)
        previous_absorbing = np.asarray(absorbing, dtype=bool)
    return (
        [np.stack(t) for t in tracks],
        np.stack(action_log),
        [log.as_arrays() for log in logs],
    )


class BodyCorrectTargets:
    """Per-frame target sites, computed on the body the rollout actually used.

    One instance per family. Models and reference bundles are cached by the
    morphology vector -- never by phase alone, which would make a moving target
    look stationary.
    """

    def __init__(self, env, family: str, reference_path: str):
        from scaling.body_correct_reference import CpuModelCache

        self.env = env
        self.family = family
        self.reference_path = reference_path
        self.reward = env._reward_function
        self.site_ids = np.asarray(self.reward._rel_site_ids).ravel()
        self.body_ids = np.asarray(self.reward._rel_body_ids).ravel()
        self.site_names = [
            mujoco.mj_id2name(env._model, mujoco.mjtObj.mjOBJ_SITE, int(i))
            for i in self.site_ids
        ]
        self._models = CpuModelCache(env._model, family)
        self._bundles: dict = {}
        self._mjx_provider = None
        self.max_provider_error_m = 0.0

    def _key(self, morphology) -> tuple:
        from scaling.body_correct_reference import CpuModelCache

        return CpuModelCache.key(morphology)

    def model_for(self, morphology):
        return self._models.get(morphology)

    def targets(self, morphology, traj_no: int, subtraj_step_no: int):
        """``(site_xpos, site_xmat, relative_position)`` for one logged frame."""
        from scaling.body_correct_reference import cpu_reference_bundle

        key = (self._key(morphology), int(traj_no), int(subtraj_step_no))
        if key not in self._bundles:
            model = self.model_for(morphology)
            bundle = cpu_reference_bundle(
                self.env, model, int(traj_no), int(subtraj_step_no),
                rel_site_ids=self.site_ids,
                rel_body_ids=self.body_ids,
                include_site_velocity=False,
            )
            self._bundles[key] = (
                np.asarray(bundle.site_position_world),
                np.asarray(bundle.site_orientation_world),
                np.asarray(bundle.relative_site_position),
            )
        return self._bundles[key]

    def provider_error(self, morphology, traj_no: int, subtraj_step_no: int) -> float:
        """CPU provider vs the production MJX provider, in metres.

        This is the number that decides the renderer's exit code: the markers
        are only evidence if the body-correct CPU path agrees with the MJX path
        the reward actually scored.
        """
        _, _, cpu_rpos = self.targets(morphology, traj_no, subtraj_step_no)
        mjx_rpos = np.asarray(
            self._jitted_mjx_provider()(
                jnp.asarray(np.asarray(morphology, dtype=np.float32)),
                jnp.asarray(int(traj_no)),
                jnp.asarray(int(subtraj_step_no)),
            )
        )
        error = float(np.abs(mjx_rpos - cpu_rpos).max())
        self.max_provider_error_m = max(self.max_provider_error_m, error)
        return error

    def _jitted_mjx_provider(self):
        """The production MJX provider, jitted once and reused per frame.

        Eager MJX kinematics per frame dominates an 800-frame render; the
        morphology and cursor are traced arguments, so a single compile serves
        every frame regardless of how often the body changes.
        """
        if self._mjx_provider is None:
            from mujoco import mjx

            from scaling.body_correct_reference import body_correct_reference

            data0 = mjx.make_data(self.env.sys)

            def compute(morphology, traj_no, subtraj_step_no):
                carry = SimpleNamespace(
                    morphology=morphology,
                    traj_state=SimpleNamespace(
                        traj_no=traj_no, subtraj_step_no=subtraj_step_no
                    ),
                )
                return body_correct_reference(
                    self.env, self.env._model, data0, carry, jnp,
                    rel_site_ids=self.site_ids,
                    rel_body_ids=self.body_ids,
                    body_rootid=self.env.sys.body_rootid,
                    include_site_velocity=False,
                ).relative_site_position

            self._mjx_provider = jax.jit(compute)
        return self._mjx_provider


def _add_sphere(scene, position, radius, color):
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        int(mujoco.mjtGeom.mjGEOM_SPHERE),
        np.array([radius, 0.0, 0.0]),
        np.asarray(position, dtype=np.float64),
        np.eye(3).flatten(),
        np.asarray(color, dtype=np.float32),
    )
    scene.ngeom += 1


def _add_line(scene, start, end, color, width=2.5):
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_connector(
        scene.geoms[scene.ngeom],
        int(mujoco.mjtGeom.mjGEOM_LINE),
        width,
        np.asarray(start, dtype=np.float64),
        np.asarray(end, dtype=np.float64),
    )
    scene.geoms[scene.ngeom].rgba[:] = np.asarray(color, dtype=np.float32)
    scene.ngeom += 1


def render_track_with_targets(
    targets: BodyCorrectTargets,
    log: dict,
    width: int,
    height: int,
    distance: float,
    elevation: float,
    show_targets: bool,
    show_achieved: bool,
):
    """Replay achieved qpos on the SAMPLED body, drawing body-correct markers.

    Returns ``(frames, rows)`` where ``rows`` is one sidecar record per frame.
    """
    frames, rows = [], []
    camera = mujoco.MjvCamera()
    camera.azimuth, camera.elevation, camera.distance = 135.0, elevation, distance

    # ONE renderer for the whole rollout, bound to the NOMINAL model and never
    # recreated.
    #
    # Creating a mujoco.Renderer per sampled body (or closing and recreating one
    # when the body changes) segfaults inside Renderer.render() on both the EGL
    # and osmesa paths -- it is the context churn, not the backend.
    #
    # Binding to the nominal model is exact here rather than an approximation:
    # the morphology mutation touches body_pos/ipos/mass/inertia/damping/
    # actuators and leaves geom_type and geom_size untouched, and mjv_updateScene
    # takes geom POSES from data (computed by mj_forward on the mutated model)
    # and only sizes/types from the model. Verified pixel-wise: a frame drawn
    # this way differs from one drawn by a renderer bound to the mutated model
    # by at most 1/255 on a corner body.
    renderer = mujoco.Renderer(targets.env._model, height, width)
    datas: dict = {}

    n = log["qpos"].shape[0]
    for t in range(n):
        morphology = log["morphology"][t]
        model = targets.model_for(morphology)
        key = targets._key(morphology)
        if key not in datas:
            # MjData is cheap and holds no GL resources, so one per body is fine.
            if len(datas) > 8:
                datas.pop(next(iter(datas)))
            datas[key] = mujoco.MjData(model)
        data = datas[key]

        data.qpos[:] = log["qpos"][t]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        camera.lookat[:] = [data.qpos[0], data.qpos[1], 0.75]
        renderer.update_scene(data, camera)

        traj_no = int(log["traj_no"][t])
        step_no = int(log["subtraj_step_no"][t])
        error = targets.provider_error(morphology, traj_no, step_no)
        target_xpos, _, _ = targets.targets(morphology, traj_no, step_no)

        # The reference is expressed in the trajectory's own world frame; the
        # rollout root has drifted from it. Anchoring the marker set on the main
        # mimic site keeps the shape comparison honest without pretending the
        # world-frame offset is zero (that offset is what the root-error goal
        # block and the deviation terminal criterion score).
        achieved_xpos = np.asarray(data.site_xpos[targets.site_ids])
        anchor = achieved_xpos[0] - target_xpos[0]
        drawn_targets = target_xpos + anchor

        scene = renderer.scene
        for i, name in enumerate(targets.site_names):
            if show_targets:
                _add_sphere(
                    scene, drawn_targets[i], TARGET_RADIUS,
                    TARGET_COLORS.get(name, DEFAULT_TARGET_COLOR),
                )
            if show_achieved:
                _add_sphere(scene, achieved_xpos[i], ACHIEVED_RADIUS,
                            ACHIEVED_COLOR)
            if show_targets and show_achieved:
                _add_line(scene, achieved_xpos[i], drawn_targets[i],
                          ACHIEVED_COLOR)

        frames.append(renderer.render().copy())
        rows.append({
            "frame": t,
            "family": targets.family,
            "traj_no": traj_no,
            "subtraj_step_no": step_no,
            "morphology": [float(v) for v in np.asarray(morphology).ravel()],
            "morphology_generation": int(log["morphology_generation"][t]),
            "absorbing": bool(log["absorbing"][t]),
            "reset_happened": bool(log["reset_happened"][t]),
            "target_provider_max_error_m": error,
            "mean_target_error_m": float(
                np.linalg.norm(achieved_xpos - drawn_targets, axis=-1).mean()
            ),
        })

    if renderer is not None:
        renderer.close()
    return frames, rows


def enforce_provider_agreement(max_error_m: float, threshold_m: float) -> None:
    """Refuse to present the overlay as evidence when the providers disagree.

    A video is not a measurement; it becomes one only because the markers were
    generated by the same computation the reward scored. If that stops being
    true the run must fail loudly, not ship a plausible-looking MP4.
    """
    if max_error_m > threshold_m:
        print(
            "[render] FAIL: body-correct CPU targets disagree with the "
            f"production MJX provider by {max_error_m:.2e} m > "
            f"{threshold_m:.2e} m; the overlay is not evidence.",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(3)


def render_track(model, qpos, width, height, distance, elevation):
    """Replay one qpos trajectory offscreen with a camera tracking the root."""
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height, width)
    camera = mujoco.MjvCamera()
    camera.azimuth, camera.elevation, camera.distance = 135.0, elevation, distance
    frames = []
    for row in qpos:
        data.qpos[:] = row
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        camera.lookat[:] = [data.qpos[0], data.qpos[1], 0.75]
        renderer.update_scene(data, camera)
        frames.append(renderer.render().copy())
    renderer.close()
    return frames


def label(frame, text, sub):
    """Burn a caption strip into the top of a frame (no font deps: cv2 is present).

    cv2's Hershey fonts are ASCII-only: any non-ASCII character is drawn as '?'.
    Both lines are therefore transliterated and then trimmed to the measured
    pixel width, so a caption is never silently cut off at the panel edge.
    """
    import cv2

    def fit(value: str, scale: float, font: int, width: int) -> str:
        value = (value.replace("·", "|").replace("—", "-")
                 .replace("–", "-").replace("×", "x"))
        value = value.encode("ascii", "replace").decode("ascii")
        while value and cv2.getTextSize(value, font, scale, 1)[0][0] > width:
            value = value[:-1]
        return value

    out = frame.copy()
    width = out.shape[1]
    cv2.rectangle(out, (0, 0), (width, 34), (16, 22, 32), -1)
    cv2.putText(out, fit(text, 0.42, cv2.FONT_HERSHEY_DUPLEX, width - 14),
                (7, 15), cv2.FONT_HERSHEY_DUPLEX, 0.42, (238, 242, 247), 1,
                cv2.LINE_AA)
    cv2.putText(out, fit(sub, 0.34, cv2.FONT_HERSHEY_SIMPLEX, width - 14),
                (7, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (150, 168, 188), 1,
                cv2.LINE_AA)
    return out


def main():
    args = parse_args()
    args.checkpoint = _resolve_checkpoint(args.checkpoint)
    _, manifest = _find_manifest(args.checkpoint)
    backbone = str(manifest.get("backbone", "masked_mlp"))
    env_args = _env_args(
        SimpleNamespace(
            robots=None,
            reference_root=args.reference_root,
            use_mjwarp=False,
            envs_per_robot=1,
        ),
        manifest,
    )
    if args.morphology_override is not None:
        # e.g. 'nominal' renders the stock body through the morph env class,
        # keeping the checkpoint's 26-dim description layout intact
        env_args.morphology = args.morphology_override
    if args.morphology_catalog_file is not None:
        env_args.morphology_catalog_file = str(args.morphology_catalog_file)
    agent_conf, agent_state = trainer_for(backbone).load_agent(args.checkpoint)
    env, build_meta = build_cross_humanoid_env(env_args)
    buffer = getattr(agent_conf, "actor_latent_buffer", None)
    if args.latent_codes is not None:
        from scaling.fsq_motion import buffer_from_codes_npz

        buffer = buffer_from_codes_npz(args.latent_codes)
        print(f"[render] hot-swapped latent codes from {args.latent_codes}",
              flush=True)
    print(f"[render] {backbone} · {list(env.names)} · {args.steps} steps "
          f"· latent={'none' if buffer is None else buffer.values.shape}",
          flush=True)

    tracks, action_log, rollout_logs = rollout(
        env, agent_conf, agent_state, args.steps, args.seed, args.zero_action,
        buffer=buffer, switch_step=args.switch_step,
        force_motion=args.force_motion,
        target_env_index=args.target_env_index,
    )
    if args.switch_step is not None:
        delta = np.abs(np.diff(action_log, axis=0)).mean(axis=(1, 2))
        window = slice(max(0, args.switch_step - 5), args.switch_step + 5)
        report = {
            "switch_step": args.switch_step,
            "mean_abs_action_delta_overall": float(delta.mean()),
            "mean_abs_action_delta_at_switch": float(
                delta[args.switch_step - 1]
            ),
            "action_delta_window": delta[window].tolist(),
        }
        metrics_path = args.output.with_suffix(".switch_metrics.json")
        metrics_path.write_text(__import__("json").dumps(report, indent=2))
        print(f"[render] switch metrics -> {metrics_path}: "
              f"at-switch {report['mean_abs_action_delta_at_switch']:.4f} vs "
              f"overall {report['mean_abs_action_delta_overall']:.4f}", flush=True)

    overlay = bool(args.show_targets or args.show_achieved_sites)
    sidecar_rows: list = []
    sidecar_summary: dict = {}
    max_provider_error = 0.0

    panels = []
    for gi, group in enumerate(env.groups):
        raw_env = env._raw_envs[gi]
        model = raw_env._model
        joints = int(env.group_action_dims[gi])
        tag = "zero action" if args.zero_action else f"{backbone} policy"
        log = rollout_logs[gi]

        if overlay:
            reference = str(
                build_meta.get("references", {})
                .get(group.name, {})
                .get("path", "")
            )
            targets = BodyCorrectTargets(raw_env, group.name, reference)
            frames, rows = render_track_with_targets(
                targets, log, args.width, args.height, args.distance,
                args.elevation, args.show_targets, args.show_achieved_sites,
            )
            sidecar_rows.extend(rows)
            max_provider_error = max(max_provider_error,
                                     targets.max_provider_error_m)
            sidecar_summary[group.name] = {
                "frames": len(frames),
                "reference_file": reference,
                "number_of_resets": int(log["reset_happened"].sum()),
                "number_of_morphology_changes": int(
                    np.count_nonzero(np.diff(log["morphology_generation"]))
                ),
                "max_target_provider_error_m": targets.max_provider_error_m,
            }
            # Burned-in text carries family, morphology, trajectory id, phase,
            # generation and mean target error -- everything a viewer needs to
            # check the overlay against the sidecar without trusting the video.
            panels.append([
                label(
                    frame,
                    (
                        f"{group.name.upper()} {joints}j  "
                        f"m=[{','.join(f'{v:.2f}' for v in row['morphology'])}]"
                    ),
                    (
                        f"traj{row['traj_no']} ph{row['subtraj_step_no']} "
                        f"gen{row['morphology_generation']} "
                        f"err {row['mean_target_error_m']:.2f}m"
                        + ("  RESET" if row["reset_happened"] else "")
                        + ("  ABSORB" if row["absorbing"] else "")
                    ),
                )
                for frame, row in zip(frames, rows)
            ])
        else:
            frames = render_track(model, tracks[gi], args.width, args.height,
                                  args.distance, args.elevation)
            panels.append([label(f, f"{group.name.upper()}  ·  {joints} joints", tag)
                           for f in frames])
        print(f"[render] {group.name}: {len(frames)} frames", flush=True)

    if overlay:
        sidecar_path = args.target_sidecar or args.output.with_suffix(
            ".targets.json"
        )
        payload = {
            "checkpoint": str(args.checkpoint),
            "seed": args.seed,
            "target_env_index": args.target_env_index,
            "fail_target_mismatch_m": args.fail_target_mismatch_m,
            "max_target_provider_error_m": max_provider_error,
            "frames_per_family": {
                name: block["frames"] for name, block in sidecar_summary.items()
            },
            "reference_file_per_family": {
                name: block["reference_file"]
                for name, block in sidecar_summary.items()
            },
            "number_of_resets": sum(
                block["number_of_resets"] for block in sidecar_summary.values()
            ),
            "number_of_morphology_changes": sum(
                block["number_of_morphology_changes"]
                for block in sidecar_summary.values()
            ),
            "per_family": sidecar_summary,
            "rows": sidecar_rows,
        }
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar_path.write_text(
            json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
        )
        print(
            f"[render] target sidecar -> {sidecar_path} "
            f"(max provider error {max_provider_error:.2e} m)",
            flush=True,
        )
        enforce_provider_agreement(
            max_provider_error, args.fail_target_mismatch_m
        )

    import cv2

    grid = [np.hstack([p[i] for p in panels]) for i in range(len(panels[0]))]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw = args.output.with_suffix(".raw.mp4")
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
                             (grid[0].shape[1], grid[0].shape[0]))
    for frame in grid:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"[render] wrote {raw}", flush=True)

    if args.ffmpeg:
        # A Windows ffmpeg.exe invoked from WSL cannot resolve /mnt/c paths.
        def host(path):
            text = str(path)
            if args.ffmpeg.endswith(".exe") and text.startswith("/mnt/"):
                return text[5].upper() + ":" + text[6:].replace("/", "\\")
            return text

        subprocess.run([args.ffmpeg, "-y", "-loglevel", "error", "-i", host(raw),
                        "-c:v", "libx264", "-crf", "32", "-preset", "slow", "-an",
                        "-movflags", "+faststart", host(args.output)], check=True)
        raw.unlink()
        print(f"[render] encoded {args.output} "
              f"({args.output.stat().st_size // 1024} KB)", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=240)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--height", type=int, default=400)
    p.add_argument("--distance", type=float, default=3.4)
    p.add_argument("--elevation", type=float, default=-12.0)
    p.add_argument("--seed", type=int, default=20260801)
    p.add_argument("--zero-action", action="store_true")
    p.add_argument(
        "--latent-codes", type=Path, default=None,
        help="npz token cache to hot-swap in place of the checkpoint's buffer",
    )
    p.add_argument(
        "--switch-step", type=int, default=None,
        help="flip the commanded motion id at this step (Gate 6 switch video)",
    )
    p.add_argument(
        "--morphology-override", default=None,
        help="override the manifest morphology mode (e.g. 'nominal')",
    )
    p.add_argument(
        "--morphology-catalog-file", type=Path, default=None,
        help="JSON (N, 4) catalog pinning the bodies to render, so a specific "
        "corner of the morphology box can be shown rather than whatever the "
        "rollout happened to sample.",
    )
    p.add_argument(
        "--force-motion", type=int, default=None,
        help="command this canonical motion id to every env with a shared "
        "global clock (same-z-across-families video)",
    )
    p.add_argument(
        "--show-targets", action="store_true",
        help="Draw the body-correct reference sites as spheres, replaying the "
        "achieved qpos on the SAMPLED CPU body rather than the nominal mesh.",
    )
    p.add_argument(
        "--show-achieved-sites", action="store_true",
        help="Draw the achieved mimic sites (small cyan spheres) and a line to "
        "each target.",
    )
    p.add_argument(
        "--target-env-index", type=int, default=0,
        help="Which environment of each topology group the overlay follows.",
    )
    p.add_argument(
        "--target-sidecar", type=Path, default=None,
        help="JSON sidecar path; defaults to <output>.targets.json.",
    )
    p.add_argument(
        "--fail-target-mismatch-m", type=float, default=1e-5,
        help="Exit non-zero when the body-correct CPU targets disagree with the "
        "production MJX provider by more than this. The overlay is only "
        "evidence while the two agree.",
    )
    p.add_argument("--ffmpeg", default=None, help="ffmpeg binary for web encoding")
    p.add_argument("--reference-root", type=Path,
                   default=WORKSPACE / "external_data" / "cross_humanoid")
    return p.parse_args()


if __name__ == "__main__":
    main()
