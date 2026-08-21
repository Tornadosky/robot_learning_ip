"""Train one padded shared PPO policy across several humanoid topologies.

This is a bridge between the cross-humanoid reference cache and the fully
embodiment-aware policy work.  Each robot remains a homogeneous MJX vmap group;
observations/actions are padded to the largest robot and a robot one-hot is
appended before all groups enter one PPO rollout/update.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

WORKSPACE = Path(__file__).resolve().parents[2]
SCRIPTS = WORKSPACE / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from loco_mujoco.trajectory import Trajectory  # noqa: E402

from morphology_deepmimic import make_mimic_env  # noqa: E402
from scaling.cross_humanoid_retarget import HUMANOIDS  # noqa: E402
from scaling.cross_topology_urma import CrossTopologyURMAPPO  # noqa: E402
from scaling.joint_descriptions import build_joint_block_spec  # noqa: E402
from scaling.masked_mlp import MaskedParallelPPO  # noqa: E402
from scaling.parallel_env import (  # noqa: E402
    ParallelMorphVecEnv,
    balanced_group_sizes,
    describe_layout,
)

BACKBONES = ("masked_mlp", "urma", "urmav2")


def trainer_for(backbone: str):
    return MaskedParallelPPO if backbone == "masked_mlp" else CrossTopologyURMAPPO


def _largest_divisor_at_most(value: int, limit: int) -> int:
    for candidate in range(min(value, limit), 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


def build_config(args, total_envs: int, actual_timesteps: int):
    batch_size = args.num_steps * total_envs
    minibatches = _largest_divisor_at_most(batch_size, args.num_minibatches)
    updates = max(1, actual_timesteps // batch_size)
    return OmegaConf.create(
        {
            "experiment": {
                "hidden_layers": list(args.hidden),
                # URMAPPO reads these; the masked MLP ignores them.
                "backbone": "urma" if args.backbone == "urma" else "urmav2",
                "urma_activation": "elu",
                "urma_latent_slots": args.urma_latent_slots,
                "urma_joint_value_dim": args.urma_joint_value_dim,
                # separate motion-command latent (0 = no latent, pre-existing
                # behaviour and checkpoint format); getattr keeps pre-latent
                # callers (tests) working
                "actor_latent_dim": getattr(args, "actor_latent_dim", 0),
                "fake_latent_seed": getattr(args, "fake_latent_seed", 0),
                "fake_latent_scale": getattr(args, "fake_latent_scale", 1.0),
                "motion_latent_embed_dim": getattr(
                    args, "motion_latent_embed_dim", 64
                ),
                "lr": args.lr,
                "num_envs": total_envs,
                "num_steps": args.num_steps,
                "total_timesteps": float(actual_timesteps),
                "update_epochs": args.update_epochs,
                "proportion_env_reward": 0.0,
                "num_minibatches": minibatches,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_eps": 0.2,
                "init_std": args.init_std,
                "learnable_std": bool(args.learnable_std),
                "ent_coef": 0.0,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
                "activation": "tanh",
                "anneal_lr": False,
                "weight_decay": 0.0,
                "normalize_env": not args.no_normalize_reward,
                "debug": False,
                "n_seeds": 1,
                "vmap_across_seeds": True,
                "validation": {
                    "active": False,
                    "num_steps": 100,
                    "num_envs": min(100, total_envs),
                    "num": min(10, updates),
                },
            }
        }
    )


def clip_windows(args) -> list[tuple[str, int, int]]:
    """Canonical motion list: [(clip, start_frame, frames)], same for every
    robot, in order — trajectory index therefore IS the canonical motion id.

    Attribute access is getattr-tolerant so pre-latent callers (tests,
    evaluator, renderer) keep working with their narrower namespaces."""
    specs = getattr(args, "clip_windows", None)
    if specs:
        windows = []
        for spec in specs:
            clip, start, frames = spec.split(":")
            windows.append((clip, int(start), int(frames)))
        return windows
    return [(args.clip, args.start_frame, args.frames)]


def reference_path(args, robot: str, window=None) -> Path:
    clip, start, frames = window if window else (
        args.clip, args.start_frame, args.frames
    )
    tag = f"start{start}_{frames}f_{args.reference_mode}.npz"
    return args.reference_root / f"{args.source}_source" / clip / robot / tag


def _mimic_env_name(args, robot: str, plain: bool = False) -> str:
    """Environment name to instantiate — the family-morph subclass when
    online morphology randomization is requested.  ``plain`` skips the morph
    wrapper (reserved robots are only probed for their padded dimensions and
    have no family morphology spec)."""
    spec = HUMANOIDS[robot]
    from loco_mujoco.environments.base import LocoEnv

    from scaling.family_morphology import make_family_morph_env_class
    from scaling.foot_contact_fix import BOX_FOOT, make_box_foot_mixin

    # Upstream's MJX H1 stands on two centre-line capsules whose lowest points
    # differ by 9 mm, so only the toe roller bears load and the support polygon
    # has no fore-aft extent at all -- H1 cannot hold a pose under PD at any
    # gain. `--foot-model box` swaps in a flat box foot of the same footprint
    # and underside height. Off by default: it changes the contact model, so it
    # is an experiment arm, not a silent correction.
    box_foot = getattr(args, "foot_model", "stock") == "box" and robot in BOX_FOOT
    base_cls = spec.mjx_cls
    prefix = ""
    if box_foot:
        base_cls = type(f"BoxFoot{spec.mjx_env_name}",
                        (make_box_foot_mixin(robot), spec.mjx_cls), {})
        prefix = "BoxFoot"

    if plain or not args.morphology:
        if not box_foot:
            return spec.mjx_env_name
        name = f"BoxFoot{spec.mjx_env_name}"
        LocoEnv.registered_envs.setdefault(name, base_cls)
        return name

    name = f"FamilyMorph{prefix}{spec.mjx_env_name}"
    if name not in LocoEnv.registered_envs:
        LocoEnv.registered_envs[name] = make_family_morph_env_class(
            base_cls, robot
        )
    return name


def _morph_env_params(args, robot: str, plain: bool = False) -> dict:
    if plain or not args.morphology:
        return {}
    params = dict(
        family_key=robot,
        resample_morphology_on_episode_reset=args.morphology == "continuous",
    )
    admission = getattr(args, "admission_config", None)
    if admission is None and getattr(args, "no_admission_checks", False):
        from scaling.morphology_admission import AdmissionConfig

        # Reproduces the pre-admission use-every-draw sampler. Exists as a
        # bisection lever: when a run dies on an accelerator, being able to
        # remove one in-graph subsystem without editing code is the difference
        # between a 20-minute and a 2-hour diagnosis.
        admission = AdmissionConfig(enabled=False)
    if admission is not None:
        params["admission"] = admission
    catalog_file = getattr(args, "morphology_catalog_file", None)
    if catalog_file:
        # Explicit (N, 4) catalog, e.g. a held-out OUT-OF-BOX body for
        # evaluation. Overrides the morphology mode's own sampling.
        catalog = np.asarray(
            json.loads(Path(catalog_file).read_text()), dtype=np.float32
        )
        params["resample_morphology_on_episode_reset"] = False
        params["morphology_catalog"] = catalog
        return params
    if args.morphology == "nominal":
        # frozen nominal body through the SAME morph env class, so a
        # morphology-trained checkpoint (26-dim descriptions) can be rendered
        # on the stock body for the nominal-vs-randomized comparison
        from scaling.family_morphology import FAMILY_MORPHOLOGY_DIM

        params["morphology_catalog"] = np.ones(
            (2, FAMILY_MORPHOLOGY_DIM), dtype=np.float32
        )
        return params
    if args.morphology.startswith("catalog"):
        from scaling.family_morphology import (
            FAMILY_MORPHOLOGY_HIGH,
            FAMILY_MORPHOLOGY_LOW,
        )

        n = int(args.morphology[len("catalog"):])
        if n == 2:
            # two deterministic bodies per family: short/weak vs tall/strong
            catalog = np.stack([FAMILY_MORPHOLOGY_LOW, FAMILY_MORPHOLOGY_HIGH])
        else:
            # deterministic sample of the same box — M changes the DATA, not
            # the number of compiled topology branches (Gate 4 evidence)
            rng = np.random.default_rng(1234)
            catalog = rng.uniform(
                FAMILY_MORPHOLOGY_LOW, FAMILY_MORPHOLOGY_HIGH,
                size=(n, FAMILY_MORPHOLOGY_LOW.shape[0]),
            ).astype(np.float32)
        params["morphology_catalog"] = catalog
    return params


def _reward_env_params(args) -> dict:
    """Reward class override for every family env; default is the shared
    stock ``MimicReward`` from morphology_deepmimic.

    ``MorphMimicReward`` (scripts/h1md) recomputes the reference SITE targets
    by forward kinematics on the body actually being simulated — the sampled
    per-env morphology model — instead of the nominal body the trajectory was
    computed on.  On envs without per-env morphology (plain/reserved builds,
    ``--morphology`` unset) it falls through to the trajectory's own site data,
    i.e. stock behaviour.
    """
    from morphology_deepmimic import MIMIC_REWARD_PARAMS

    name = str(getattr(args, "reward_type", "MimicReward"))
    if name == "MorphMimicReward":
        h1md = str(WORKSPACE / "scripts" / "h1md")
        if h1md not in sys.path:
            sys.path.insert(0, h1md)
        import morph_mimic_reward  # noqa: F401  (registers the class)

        params = {"reward_type": name}
        root_frame = str(getattr(args, "root_frame", "absolute"))
        if root_frame != "absolute":
            params["reward_params"] = dict(MIMIC_REWARD_PARAMS,
                                           root_frame=root_frame)
        return params
    if name == "Urma2CoreReward":
        h1md = str(WORKSPACE / "scripts" / "h1md")
        if h1md not in sys.path:
            sys.path.insert(0, h1md)
        import scaling.urma2_core_reward  # noqa: F401  (registers the class)

        params = {"reward_type": name}
        root_frame = str(getattr(args, "root_frame", "absolute"))
        if root_frame != "absolute":
            params["reward_params"] = dict(MIMIC_REWARD_PARAMS,
                                           root_frame=root_frame)
        return params
    if name != "MimicReward":
        raise ValueError(f"Unknown reward type {name!r}.")
    return {}


def _goal_env_params(args) -> dict:
    """Goal class override for every family env; default is stock ``GoalTrajMimic``.

    ``MorphGoalTrajMimicRootErr`` is the production no-FSQ goal: its
    trajectory-side site block comes from the same body-correct provider the
    reward scores, and it appends the heading-frame root position error so the
    world-frame tracking the reward charges for is observable.  Without it the
    actor is commanded toward nominal-body targets while being paid for
    sampled-body ones (see PRODUCTION_FIXES_BEFORE_LONG_RUN.md).
    """
    name = str(getattr(args, "goal_type", "GoalTrajMimic") or "GoalTrajMimic")
    if name == "GoalTrajMimic":
        return {}
    if name == "MorphGoalTrajMimicRootErr":
        from morphology_deepmimic import GOAL_PARAMS, MIMIC_SITES

        import scaling.body_correct_goal  # noqa: F401  (registers the class)

        params = dict(GOAL_PARAMS)
        # The goal mimics the env's full site list; naming the reward's subset
        # makes the "goal exposes what the reward scores" check run at build
        # time rather than at first use.
        params["reward_sites_for_mimic"] = list(MIMIC_SITES)
        params["root_frame"] = str(getattr(args, "root_frame", "absolute"))
        return {"goal_type": name, "goal_params": params}
    if name == "GoalTrajMimicRootErr":
        h1md = str(WORKSPACE / "scripts" / "h1md")
        if h1md not in sys.path:
            sys.path.insert(0, h1md)
        import goal_rooterr

        goal_rooterr.register()
        return {"goal_type": name}
    raise ValueError(f"Unknown goal type {name!r}.")


def _terminal_env_params(args) -> dict:
    """Terminal-state handler override, forwarded to every family env.

    ``RootPoseTrajTerminalStateHandler`` derives its root-height window from the
    reference trajectory alone, so a legitimately taller sampled body is
    declared absorbing for standing higher than the human-derived motion. The
    morphology-aware subclass judges each body against its own standing height,
    and is required before leg-length randomization widens.

    ``max_root_pos_deviation`` is an experiment setting, not a constant: upstream
    defaults it to 1e6 (never terminate), which makes drifting free.
    """
    handler = getattr(args, "terminal_handler", None)
    deviation = getattr(args, "max_root_deviation", None)
    rot_margin = getattr(args, "root_rot_margin_degrees", None)
    tilt = getattr(args, "terminal_tilt_degrees", None)
    params: dict = {}
    if handler:
        import scaling.morphology_terminal  # noqa: F401  (registers the handler)

        params["terminal_state_type"] = handler
    ts_params: dict = {}
    if deviation is not None:
        ts_params["max_root_pos_deviation"] = float(deviation)
    if rot_margin is not None:
        ts_params["root_rot_margin_degrees"] = float(rot_margin)
    if tilt is not None:
        if handler != "GravityRootPoseTrajTerminalStateHandler":
            raise ValueError(
                "--terminal-tilt-degrees requires --terminal-handler "
                "GravityRootPoseTrajTerminalStateHandler")
        ts_params["max_tilt_degrees"] = float(tilt)
    if ts_params:
        params["terminal_state_params"] = ts_params
    return params


def _control_env_params(args, robot: str) -> dict:
    """Per-family control override.

    ``morphology_deepmimic`` records that ``DefaultControl`` (direct torque) is
    "very hard to learn on the smaller, 23-DOF G1" and ships PD gains for it,
    but ``control_config`` was never wired into this cross-family trainer, so
    both topologies have always trained on raw torque here. ``--pd-control g1``
    makes that lever testable instead of assumed.
    """
    families = [f.lower() for f in (getattr(args, "pd_control", None) or [])]
    if robot not in families:
        return {}
    from morphology_deepmimic import PD_GAINS

    if robot not in PD_GAINS:
        raise ValueError(
            f"--pd-control {robot}: no PD gains defined for this family in "
            "morphology_deepmimic.PD_GAINS."
        )
    scale = float(getattr(args, "pd_gain_scale", 1.0) or 1.0)
    gains = PD_GAINS[robot]
    # Upstream PDControl maps action [-1,1] across the joint's ENTIRE travel and
    # offsets by the range midpoint, so a=0 commands the midpoint rather than the
    # nominal pose and a=1 commands the limit. Measured here: 1.92 rad median on
    # G1 (max 2.88) against the 0.25-0.75 rad per-actuator `scaling_factors` that
    # loco_mjx/urma2 uses across all ~50 of its robots (G1 0.5, H1 0.75).
    # --pd-action-scale switches to that residual convention.
    action_scale = getattr(args, "pd_action_scale", None)
    if action_scale:
        from scaling.residual_pd_control import register as _register_residual_pd

        _register_residual_pd()
        return dict(
            control_type="ResidualPDControl",
            control_params=dict(
                p_gain=[g * scale for g in gains["p_gain"]],
                d_gain=[g * scale for g in gains["d_gain"]],
                action_scale=float(action_scale),
            ),
        )
    return dict(
        control_type="PDControl",
        control_params=dict(
            p_gain=[g * scale for g in gains["p_gain"]],
            d_gain=[g * scale for g in gains["d_gain"]],
        ),
    )


def _build_robot_env(args, robot: str, plain: bool = False):
    trajectories = []
    for window in clip_windows(args):
        path = reference_path(args, robot, window)
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {robot} reference {path}. Run "
                "scripts/scaling/cross_humanoid_retarget.py first."
            )
        trajectories.append(Trajectory.load(str(path)))
    trajectory = (
        trajectories[0]
        if len(trajectories) == 1
        else Trajectory.concatenate(trajectories)
    )
    grounding = str(getattr(args, "reference_grounding", "none") or "none")
    grounding_stats = {"mode": "none"}
    if grounding != "none":
        # The cross-humanoid references are ungrounded: on dance2_subject4 the
        # feet never touch the floor in ANY frame, so tracking the reference and
        # standing up are in conflict. Ground before the env is built, so reward,
        # goal, terminal and FK targets all see the corrected clip.
        import mujoco

        from scaling.reference_grounding import apply as apply_grounding

        probe = mujoco.MjModel.from_xml_path(str(HUMANOIDS[robot].xml_path))
        trajectory, grounding_stats = apply_grounding(probe, trajectory, grounding)
        print(f"[cross-train] {robot} reference grounding={grounding}: "
              f"floating>2cm before "
              f"{grounding_stats.get('pct_frames_floating_over_2cm_before', 0):.0f}%"
              + (f", within 2cm after "
                 f"{grounding_stats['pct_frames_within_2cm_after']:.0f}%"
                 if "pct_frames_within_2cm_after" in grounding_stats else ""),
              flush=True)

    substeps = getattr(args, "n_substeps", None)
    horizon = getattr(args, "env_horizon", None)
    env = make_mimic_env(
        _mimic_env_name(args, robot, plain),
        trajectory,
        use_mjwarp=args.use_mjwarp,
        nconmax=7000,
        headless=True,
        **({"n_substeps": int(substeps)} if substeps else {}),
        **({"horizon": int(horizon)} if horizon else {}),
        **_reward_env_params(args),
        **_control_env_params(args, robot),
        **_goal_env_params(args),
        **_terminal_env_params(args),
        **_morph_env_params(args, robot, plain),
    )
    return env, trajectory


def _root_deviation_report(args, trajectories: dict) -> dict:
    """Check ``--max-root-deviation`` against the clip it will be applied to.

    ``RootPoseTrajTerminalStateHandler`` terminates on
    ``||root_xy - (traj_root_xy - traj_init_root_xy)||``, i.e. the deviation
    from the reference's own displacement -- not from the origin. So the clip's
    travel does not bound the threshold directly, but it is the scale a reader
    needs to judge one: 0.5 m means something quite different for a clip that
    travels 0.2 m than for one that travels 6 m.

    Two conditions are worth failing on, and both are recorded rather than
    assumed:

    * a non-positive threshold terminates every episode at reset;
    * a threshold larger than the clip's entire travel can never fire before
      the episode ends by other means, so it is not an early-termination
      mechanism at all -- reported as a warning, since a deliberately inert
      threshold is a legitimate (if pointless) experiment setting.
    """
    deviation = getattr(args, "max_root_deviation", None)
    report = {
        "max_root_pos_deviation": None if deviation is None else float(deviation),
        "per_robot_reference_travel_m": {},
        "verified": deviation is not None,
        "warnings": [],
    }
    for robot, trajectory in trajectories.items():
        root_xy = np.asarray(trajectory.data.qpos)[:, :2]
        displacement = root_xy - root_xy[0]
        report["per_robot_reference_travel_m"][robot] = {
            "max_displacement_from_start": float(
                np.linalg.norm(displacement, axis=-1).max()
            ),
            "path_length": float(
                np.linalg.norm(np.diff(root_xy, axis=0), axis=-1).sum()
            ),
        }
    if deviation is None:
        report["warnings"].append(
            "no max-root-deviation set: upstream defaults to 1e6, i.e. the root "
            "may drift arbitrarily far from the reference without terminating"
        )
        return report
    if float(deviation) <= 0.0:
        raise ValueError(
            f"--max-root-deviation must be positive; got {deviation}. A "
            "non-positive threshold terminates every episode at reset."
        )
    travel = max(
        block["max_displacement_from_start"]
        for block in report["per_robot_reference_travel_m"].values()
    )
    if float(deviation) > travel:
        report["warnings"].append(
            f"threshold {float(deviation):.3f} m exceeds the clip's own maximum "
            f"root displacement {travel:.3f} m; it will effectively never fire"
        )
    for warning in report["warnings"]:
        print(f"[cross-train] WARNING root-deviation: {warning}", flush=True)
    return report


def _reserved_dims(args):
    """Observation/action widths to keep free for robots we do not train on.

    A fixed-width network can only be replayed on a held-out topology if that
    topology's slots existed at training time.  Reserving them costs a few
    always-zero, always-masked inputs and nothing else.
    """
    reserve = [robot for robot in args.reserve_robots if robot not in args.robots]
    if not reserve:
        return 0, 0, 0, {}
    observation_dim = 0
    action_dim = 0
    details = {}
    for robot in reserve:
        env, _ = _build_robot_env(args, robot, plain=True)
        details[robot] = {
            "observation_dim": int(env.info.observation_space.shape[0]),
            "action_dim": int(env.info.action_space.shape[0]),
        }
        observation_dim = max(observation_dim, details[robot]["observation_dim"])
        action_dim = max(action_dim, details[robot]["action_dim"])
        print(
            f"[cross-train] reserved slots for {robot}: "
            f"obs={details[robot]['observation_dim']} "
            f"act={details[robot]['action_dim']}",
            flush=True,
        )
        del env
    return observation_dim, action_dim, len(args.robots) + len(reserve), details


def _ensure_latent_defaults(args):
    """Default the motion-latent/morphology fields for pre-latent callers."""
    for field, default in (
        ("clip_windows", None),
        ("morphology", None),
        ("blank_goal", False),
        ("goal_for_critic", False),
        ("actor_latent_dim", 0),
        ("latent_codes", None),
        ("reward_type", "MimicReward"),
        ("goal_type", "GoalTrajMimic"),
        ("terminal_handler", None),
        ("max_root_deviation", None),
        ("root_rot_margin_degrees", None),
        ("terminal_tilt_degrees", None),
        ("n_substeps", None),
        ("env_horizon", None),
        ("joint_target_obs", False),
        ("admission_census_resets", 0),
        ("admission_config", None),
        ("no_admission_checks", False),
        ("root_frame", "absolute"),
        ("reference_grounding", "none"),
        ("foot_model", "stock"),
        ("pd_action_scale", None),
        ("segments", 1),
        ("init_checkpoint", None),
        ("pd_control", None),
        ("pd_gain_scale", 1.0),
        ("morphology_catalog_file", None),
    ):
        if not hasattr(args, field):
            setattr(args, field, default)
    return args


def build_cross_humanoid_env(args):
    _ensure_latent_defaults(args)
    envs = []
    joint_block_specs = []
    references = {}
    trajectories = {}
    build_started = time.perf_counter()
    for robot in args.robots:
        env, trajectory = _build_robot_env(args, robot)
        envs.append(env)
        trajectories[robot] = trajectory
        joint_block_specs.append(
            build_joint_block_spec(
                env, robot,
                include_reference_targets=bool(
                    getattr(args, "joint_target_obs", False)),
            )
        )
        references[robot] = {
            "path": str(reference_path(args, robot)),
            "samples": int(trajectory.data.n_samples),
            "frequency_hz": float(trajectory.info.frequency),
            "observation_dim": int(env.info.observation_space.shape[0]),
            "action_dim": int(env.info.action_space.shape[0]),
        }
        print(
            f"[cross-train] built {robot:>12s}: "
            f"obs={env.info.observation_space.shape[0]} "
            f"act={env.info.action_space.shape[0]} "
            f"samples={trajectory.data.n_samples}",
            flush=True,
        )

    reserved_obs, reserved_act, reserved_slots, reserved_detail = _reserved_dims(args)

    group_sizes = (
        tuple([args.envs_per_robot] * len(envs))
        if args.envs_per_robot is not None
        else balanced_group_sizes(args.total_envs, len(envs))
    )
    parallel_env = ParallelMorphVecEnv(
        envs,
        group_sizes,
        names=args.robots,
        history_length=1,
        pad_to_max_shapes=True,
        append_group_one_hot=bool(args.robot_one_hot),
        append_action_mask=True,
        joint_block_specs=joint_block_specs if args.append_joint_features else None,
        reserved_observation_dim=reserved_obs,
        reserved_action_dim=reserved_act,
        reserved_group_slots=reserved_slots,
        blank_goal_observation=bool(args.blank_goal),
        append_goal_for_critic=bool(args.goal_for_critic),
        route_reset_observation_on_done=args.actor_latent_dim > 0
        or args.latent_codes is not None,
    )
    layout = parallel_env.urma_input_layout
    metadata = {
        "source_robot": args.source,
        "clip": args.clip,
        "clip_windows": [
            f"{clip}:{start}:{frames}" for clip, start, frames in clip_windows(args)
        ],
        "blank_goal_observation": bool(args.blank_goal),
        "goal_for_critic": bool(args.goal_for_critic),
        "reward_type": str(getattr(args, "reward_type", "MimicReward")),
        # The audit tool and the pipeline runner gate a long run on these three:
        # a 60M/300M claim is only meaningful if the goal the actor saw and the
        # terminal criterion the episode ended on are on the record.
        "goal_type": str(getattr(args, "goal_type", "GoalTrajMimic")),
        "root_frame": str(getattr(args, "root_frame", "absolute")),
        "reference_grounding": str(getattr(args, "reference_grounding", "none")),
        "foot_model": str(getattr(args, "foot_model", "stock")),
        "pd_action_scale": getattr(args, "pd_action_scale", None),
        "pd_control": list(getattr(args, "pd_control", None) or []),
        "pd_gain_scale": float(getattr(args, "pd_gain_scale", 1.0) or 1.0),
        "admission_checks_enabled": not bool(
            getattr(args, "no_admission_checks", False)
        ),
        "terminal_handler": getattr(args, "terminal_handler", None),
        "root_rot_margin_degrees": getattr(args, "root_rot_margin_degrees", None),
        "terminal_tilt_degrees": getattr(args, "terminal_tilt_degrees", None),
        "n_substeps": getattr(args, "n_substeps", None),
        "env_horizon": getattr(args, "env_horizon", None),
        "joint_target_obs": bool(getattr(args, "joint_target_obs", False)),
        "max_root_pos_deviation": (
            None
            if getattr(args, "max_root_deviation", None) is None
            else float(args.max_root_deviation)
        ),
        "root_deviation_check": _root_deviation_report(args, trajectories),
        "morphology": args.morphology,
        "joint_description_dim": int(parallel_env.joint_description_dim),
        "reference_mode": args.reference_mode,
        "window_start_frame": args.start_frame,
        "window_frames": args.frames,
        "references": references,
        "environment_build_seconds": time.perf_counter() - build_started,
        "group_sizes": list(group_sizes),
        "group_observation_dims": list(parallel_env.group_observation_dims),
        "group_action_dims": list(parallel_env.group_action_dims),
        "padded_observation_dim": parallel_env.output_observation_dim,
        "padded_action_dim": parallel_env.max_action_dim,
        "robot_one_hot": bool(args.robot_one_hot),
        "robot_one_hot_dim": parallel_env.one_hot_dim,
        "action_mask_observation_start": parallel_env.action_mask_observation_start,
        "append_joint_features": bool(parallel_env.append_joint_features),
        "joint_feature_start": parallel_env.joint_feature_start,
        "num_joint_slots": parallel_env.num_joint_slots,
        "reserved_robots": list(reserved_detail),
        "reserved_robot_dims": reserved_detail,
        "joint_counts": {
            spec.name: spec.num_joints for spec in (joint_block_specs or ())
        },
        "urma_input_layout": None if layout is None else vars(layout),
    }
    return parallel_env, metadata


def run_admission_census(env, args) -> dict:
    """Body admission/rejection accounting for the manifest.

    P1 of PRODUCTION_FIXES_BEFORE_LONG_RUN.md: before widening randomization the
    run must be able to say how many bodies were drawn, how many were rejected
    and why -- and zero rejected has to be a measurement, not the absence of a
    counter. Skipped when ``--admission-census-resets 0``.
    """
    resets = int(getattr(args, "admission_census_resets", 0) or 0)
    if resets <= 0:
        return {"supported": False, "reason": "census disabled (0 resets)"}
    from scaling.morphology_admission import admission_census

    per_family = max(1, min(64, min(env.group_sizes)))
    started = time.perf_counter()
    payload = admission_census(
        env._raw_envs, env.names, args.seed, resets, per_family
    )
    payload["census_seconds"] = time.perf_counter() - started
    total = payload.get("total", {})
    print(
        f"[cross-train] body admission census: draws={total.get('draws_total')} "
        f"rejected={total.get('rejected_bodies')} "
        f"resamples={total.get('resamples_total')} "
        f"exhausted={total.get('resample_exhausted_total')}",
        flush=True,
    )
    return payload


def run_preflight(env, seed: int):
    keys = jax.random.split(jax.random.PRNGKey(seed), env.num_envs)
    started = time.perf_counter()
    observation, state = jax.jit(env.reset)(keys)
    jax.block_until_ready(observation)
    reset_seconds = time.perf_counter() - started
    action = jnp.zeros((env.num_envs, env.max_action_dim), dtype=observation.dtype)
    started = time.perf_counter()
    next_observation, reward, _, done, _, _ = jax.jit(env.step)(state, action)
    jax.block_until_ready(next_observation)
    step_seconds = time.perf_counter() - started
    result = {
        "observation_shape": list(observation.shape),
        "action_shape": list(action.shape),
        "reward_shape": list(reward.shape),
        "done_shape": list(done.shape),
        "reset_compile_and_run_seconds": reset_seconds,
        "step_compile_and_run_seconds": step_seconds,
        "finite_observations": bool(np.isfinite(np.asarray(next_observation)).all()),
        "finite_rewards": bool(np.isfinite(np.asarray(reward)).all()),
    }
    print(json.dumps(result, indent=2), flush=True)
    return result


def _finite_or_none(value):
    value = float(value)
    return value if np.isfinite(value) else None


def _device_memory_stats():
    stats = jax.devices()[0].memory_stats()
    if not stats:
        return {}
    return {
        key: int(value)
        for key, value in stats.items()
        if isinstance(value, (int, np.integer))
    }


def main():
    args = parse_args()
    if len(set(args.robots)) != len(args.robots):
        raise ValueError("--robots must not contain duplicates.")
    env, build_metadata = build_cross_humanoid_env(args)
    print(
        f"[cross-train] backend={jax.default_backend()} robots={len(args.robots)} "
        f"total_envs={env.num_envs} "
        f"layout={describe_layout(env.names, env.group_sizes)} "
        f"padded_obs={env.output_observation_dim} "
        f"padded_act={env.max_action_dim}",
        flush=True,
    )
    if args.preflight_only:
        preflight = run_preflight(env, args.seed)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "experiment": "parallel_cross_humanoid_preflight",
            "implementation": "grouped_static_mjx_padded_observation_action_masked_shared_ppo",
            "robots": list(args.robots),
            "num_robot_topologies": len(args.robots),
            "total_envs": env.num_envs,
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in jax.devices()],
            "preflight": preflight,
            "body_admission": run_admission_census(env, args),
            "device_memory_stats": _device_memory_stats(),
            **build_metadata,
        }
        (args.output_dir / "preflight.json").write_text(
            json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
        )
        return

    # Run the census BEFORE compiling the train function: it is the evidence the
    # long-run promotion rule depends on, and it must not be skipped because a
    # training compile failed later.
    admission = run_admission_census(env, args)
    if admission.get("supported") and not admission.get("fail_closed", True):
        raise RuntimeError(
            "Body admission census found episodes that exhausted the resample "
            "budget; the sampler is producing inadmissible bodies. Refusing to "
            f"train. Census: {json.dumps(admission.get('total'))}"
        )

    steps_per_update = args.num_steps * env.num_envs
    # Segmented training. A 300M ROCm run costs ~36 min of compile plus hours of
    # rollout; a single jitted call means a Slurm timeout or a node fault loses
    # all of it. Splitting into segments and saving between them costs nothing
    # (the resume function is compiled ONCE and reused) and caps the loss at one
    # segment. --init-checkpoint then restarts from the last one.
    segments = max(1, int(getattr(args, "segments", 1) or 1))
    total_updates = max(segments, int(args.total_timesteps // steps_per_update))
    updates_per_segment = max(1, total_updates // segments)
    num_updates = updates_per_segment
    segment_timesteps = updates_per_segment * steps_per_update
    actual_timesteps = segment_timesteps * segments
    config = build_config(args, env.num_envs, segment_timesteps)
    trainer = trainer_for(args.backbone)
    actor_latent_buffer = None
    if args.latent_codes is not None:
        from scaling.fsq_motion import buffer_from_codes_npz

        actor_latent_buffer = buffer_from_codes_npz(args.latent_codes)
        print(
            f"[cross-train] latent codes {args.latent_codes}: "
            f"{actor_latent_buffer.values.shape}",
            flush=True,
        )
    if args.backbone == "masked_mlp":
        if args.actor_latent_dim > 0 or actor_latent_buffer is not None:
            raise ValueError(
                "The motion latent is wired for the URMA backbones only."
            )
        if args.goal_for_critic:
            raise ValueError(
                "--goal-for-critic needs the URMA critic_extra_indices path; "
                "the masked MLP actor would see the appended reference."
            )
        agent_conf = trainer.init_agent_conf(env, config)
    else:
        agent_conf = trainer.init_agent_conf(env, config, actor_latent_buffer)
    initial_state = None
    resume_from = getattr(args, "init_checkpoint", None)
    if resume_from is not None:
        # Only the agent state carries over; the environment and optimiser are
        # rebuilt for this stage, exactly as online_h1_train.py does.
        # save_agent names the pickle after the trainer class, so accept either
        # the file or the directory holding it.
        from scaling.evaluate_cross_humanoid_policy import _resolve_checkpoint

        resume_from = _resolve_checkpoint(Path(resume_from))
        _, initial_state = trainer.load_agent(str(resume_from))
        print(f"[cross-train] warm start from {resume_from}", flush=True)

    resume_fn = jax.jit(trainer.build_resume_train_fn(env, agent_conf, mh=None))
    train_fn = jax.jit(trainer.build_train_fn(env, agent_conf, mh=None))
    key = jax.random.PRNGKey(args.seed)

    print(
        f"[cross-train] lowering/compiling segments={segments} x "
        f"updates={updates_per_segment} steps/update={steps_per_update:,} "
        f"segment={segment_timesteps:,} total={actual_timesteps:,}",
        flush=True,
    )
    started = time.perf_counter()
    if initial_state is None:
        executable = train_fn.lower(key).compile()
    else:
        executable = None
    compile_seconds = time.perf_counter() - started
    print(f"[cross-train] compile complete in {compile_seconds:.1f}s", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    returns_all, lengths_all = [], []
    agent_state = initial_state
    training_seconds = 0.0
    segment_records = []
    for segment in range(segments):
        segment_key = jax.random.fold_in(key, segment)
        started = time.perf_counter()
        if agent_state is None:
            output = executable(segment_key)
        else:
            output = resume_fn(segment_key, agent_state)
        jax.block_until_ready(output["agent_state"])
        elapsed = time.perf_counter() - started
        training_seconds += elapsed
        agent_state = output["agent_state"]
        returns_all.extend(
            np.asarray(output["training_metrics"].mean_episode_return).tolist())
        lengths_all.extend(
            np.asarray(output["training_metrics"].mean_episode_length).tolist())

        # Save after EVERY segment, so an interrupted run resumes from here.
        checkpoint_dir = args.output_dir / (
            "checkpoint_final" if segment == segments - 1
            else f"checkpoint_segment_{segment + 1:02d}")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        agent_path = trainer.save_agent(str(checkpoint_dir), agent_conf, agent_state)
        record = {
            "segment": segment + 1,
            "steps_completed": int(segment_timesteps * (segment + 1)),
            "seconds": elapsed,
            "checkpoint": str(checkpoint_dir),
            "mean_episode_length_last": _finite_or_none(lengths_all[-1]),
            "mean_episode_return_last": _finite_or_none(returns_all[-1]),
        }
        segment_records.append(record)
        (args.output_dir / "segments.json").write_text(
            json.dumps(segment_records, indent=2, allow_nan=False), encoding="utf-8")
        print(
            f"[cross-train] segment {segment + 1}/{segments} done in "
            f"{elapsed:.0f}s ({segment_timesteps / elapsed / 1e6 * 60:.2f}M/min) "
            f"len={record['mean_episode_length_last']} -> {checkpoint_dir.name}",
            flush=True,
        )
    throughput = actual_timesteps / max(training_seconds, 1e-9)
    returns = np.asarray(returns_all)
    lengths = np.asarray(lengths_all)
    manifest = {
        "experiment": "parallel_cross_humanoid",
        "implementation": "grouped_static_mjx_padded_observation_action_masked_shared_ppo",
        "backbone": args.backbone,
        "robots": list(args.robots),
        "num_robot_topologies": len(args.robots),
        "total_envs": env.num_envs,
        "num_steps": args.num_steps,
        "num_updates": num_updates,
        "num_minibatches_requested": args.num_minibatches,
        "num_minibatches_actual": int(config.experiment.num_minibatches),
        "update_epochs": args.update_epochs,
        "total_timesteps": int(actual_timesteps),
        "segments": int(segments),
        "segment_timesteps": int(segment_timesteps),
        "segment_records": segment_records,
        "init_checkpoint": (None if resume_from is None else str(resume_from)),
        "seed": args.seed,
        "hidden": list(args.hidden),
        "lr": args.lr,
        "compile_seconds": compile_seconds,
        "training_seconds": training_seconds,
        "steps_per_second": throughput,
        "steps_per_minute": throughput * 60.0,
        "mean_episode_return_last": _finite_or_none(returns[-1]),
        "mean_episode_length_last": _finite_or_none(lengths[-1]),
        "return_curve": [_finite_or_none(value) for value in returns],
        "length_curve": [_finite_or_none(value) for value in lengths],
        "agent_path": str(agent_path),
        "actor_latent_dim": int(args.actor_latent_dim),
        "latent_codes": None if args.latent_codes is None else str(args.latent_codes),
        "fake_latent_seed": int(args.fake_latent_seed),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "device_memory_stats": _device_memory_stats(),
        "action_mask_counts": [
            int(np.asarray(env.action_mask[group.start]).sum()) for group in env.groups
        ],
        "body_admission": admission,
        **build_metadata,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(
        f"[cross-train] trained {actual_timesteps:,} steps in "
        f"{training_seconds:.1f}s ({throughput / 1e6 * 60:.2f}M steps/min) "
        f"-> {args.output_dir}",
        flush=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robots", nargs="+", choices=HUMANOIDS, default=["h1", "g1", "atlas"]
    )
    parser.add_argument("--source", choices=HUMANOIDS, default="h1")
    parser.add_argument("--clip", default="dance2_subject4")
    parser.add_argument("--start-frame", type=int, default=19482)
    parser.add_argument("--frames", type=int, default=800)
    parser.add_argument(
        "--reference-mode", choices=["direct", "robot2robot"], default="direct"
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=WORKSPACE / "external_data" / "cross_humanoid",
    )
    parser.add_argument("--backbone", choices=BACKBONES, default="masked_mlp")
    parser.add_argument(
        "--robot-one-hot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append the robot index. Disable to test structural generalisation.",
    )
    parser.add_argument(
        "--append-joint-features",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Emit the padded per-joint block. Defaults on for URMA backbones.",
    )
    parser.add_argument(
        "--reserve-robots",
        nargs="*",
        choices=list(HUMANOIDS),
        default=[],
        help="Robots to keep padded slots for without training on them.",
    )
    parser.add_argument("--urma-latent-slots", type=int, default=64)
    parser.add_argument("--urma-joint-value-dim", type=int, default=4)
    parser.add_argument(
        "--clip-windows",
        nargs="+",
        default=None,
        metavar="CLIP:START:FRAMES",
        help="Canonical motion list shared by every robot (K>=2 motions). "
        "Overrides --clip/--start-frame/--frames.",
    )
    parser.add_argument(
        "--morphology",
        default=None,
        help="Online within-family morphology randomization: 'continuous' "
        "resamples the 4-dim morphology box per episode; 'catalogN' pins N "
        "deterministic bodies per family (catalog2 = box corners, the "
        "correctness proof; larger N = Gate-4 M-scaling evidence).",
    )
    parser.add_argument(
        "--reward-type",
        choices=["MimicReward", "MorphMimicReward", "Urma2CoreReward"],
        default="MimicReward",
        help="MorphMimicReward projects the reference site targets onto the "
        "SAMPLED per-env body via in-graph forward kinematics (body-correct "
        "task-space targets); MimicReward scores the nominal-body targets.",
    )
    parser.add_argument(
        "--goal-type",
        choices=["GoalTrajMimic", "GoalTrajMimicRootErr",
                 "MorphGoalTrajMimicRootErr"],
        default="GoalTrajMimic",
        help="MorphGoalTrajMimicRootErr is the production no-FSQ goal: its "
        "trajectory-side site block comes from the same body-correct provider "
        "MorphMimicReward scores, plus the heading-frame root position error. "
        "GoalTrajMimic (stock) commands nominal-body targets while the reward "
        "scores sampled-body ones -- an integration baseline only.",
    )
    parser.add_argument(
        "--terminal-handler",
        default=None,
        help="Terminal-state handler override, forwarded to every family env. "
        "Randomized leg length needs "
        "MorphologyAwareRootPoseTrajTerminalStateHandler, or a legitimately "
        "taller body is declared absorbing at reset for standing higher than "
        "the reference motion.",
    )
    parser.add_argument(
        "--joint-target-obs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Route each actuated joint's REFERENCE target angle into that "
        "joint's URMA block (urma2-style per-joint conditioning; O2 "
        "experiment). Requires the MorphGoalTrajMimicRootErr goal, whose "
        "reference_qpos block supplies the values. Changes the per-joint "
        "feature width, so it is incompatible with checkpoints trained "
        "without it.",
    )
    parser.add_argument(
        "--env-horizon",
        type=int,
        default=None,
        help="Episode horizon in control steps (stock make_mimic_env: 1000). "
        "2000 at 100 Hz matches urma2's 20 s (arm A5).",
    )
    parser.add_argument(
        "--n-substeps",
        type=int,
        default=None,
        help="Physics substeps per control step (dt 0.002): 5 = 100 Hz control "
        "(the stock value), 10 = 50 Hz to match loco_mjx/urma2.",
    )
    parser.add_argument(
        "--root-rot-margin-degrees",
        type=float,
        default=None,
        help="Margin added to the clip's max centroid angular distance in the "
        "rotation terminal. A huge value (1e6) disables the rotation check "
        "entirely (arm A9).",
    )
    parser.add_argument(
        "--terminal-tilt-degrees",
        type=float,
        default=None,
        help="GravityRootPoseTrajTerminalStateHandler only: terminate when the "
        "root z-axis tilts more than this from world-up (arm A8).",
    )
    parser.add_argument(
        "--max-root-deviation",
        type=float,
        default=None,
        help="Terminate when the root strays this far (m) from the reference. "
        "Upstream defaults to 1e6 (never), which makes drifting free. This is "
        "an experiment setting, not a constant: verify it against the clip. "
        "Initial experiment value: 0.5.",
    )
    parser.add_argument(
        "--blank-goal",
        action="store_true",
        help="Zero the mimic goal block in the observation so the motion "
        "latent is the only future command (critic loses it too unless "
        "--goal-for-critic).",
    )
    parser.add_argument(
        "--goal-for-critic",
        action="store_true",
        help="Privileged critic: re-append the unblanked reference block to "
        "the observation where only the value head reads it. Requires "
        "--blank-goal.",
    )
    parser.add_argument(
        "--actor-latent-dim",
        type=int,
        default=0,
        help="Width of the separate motion latent. 0 = pre-existing "
        "no-latent behaviour.",
    )
    parser.add_argument("--fake-latent-seed", type=int, default=0)
    parser.add_argument("--fake-latent-scale", type=float, default=1.0)
    parser.add_argument("--motion-latent-embed-dim", type=int, default=64)
    parser.add_argument(
        "--latent-codes",
        type=Path,
        default=None,
        help="npz token cache (codes+split_points) from train_fsq_motion.py "
        "encode; replaces the fake table.",
    )
    parser.add_argument(
        "--admission-census-resets",
        type=int,
        default=4,
        help="Reset rounds per family used to measure body admission "
        "(draws/rejections/reasons/quantiles) into the manifest. 0 disables it, "
        "which also removes the only evidence that zero bodies were rejected.",
    )
    parser.add_argument(
        "--no-admission-checks",
        action="store_true",
        help="Disable the in-graph body admission checks (counters stay). "
        "Reproduces the pre-admission sampler; a bisection lever, not a "
        "production setting.",
    )
    parser.add_argument(
        "--root-frame",
        choices=["absolute", "episode_start"],
        default="absolute",
        help="Frame the reference root XY is expressed in. 'absolute' is "
        "upstream behaviour and disagrees with both the reset pose (world "
        "origin) and RootPoseTrajTerminalStateHandler (displacement from the "
        "episode start), so a perfectly-posed robot is charged for the clip's "
        "absolute position. 'episode_start' makes reward, goal and termination "
        "share one frame.",
    )
    parser.add_argument(
        "--pd-control",
        nargs="*",
        default=None,
        metavar="ROBOT",
        help="Families to drive with PD position control instead of raw "
        "torque, e.g. --pd-control g1. The repo records that torque is very "
        "hard to learn on G1 and that its dance needs ~3x stiffer gains.",
    )
    parser.add_argument(
        "--pd-gain-scale", type=float, default=1.0,
        help="Multiplier on the PD gains (repo finding for G1 dance: 3.0).",
    )
    parser.add_argument(
        "--pd-action-scale",
        type=float,
        default=None,
        help="Radians a PD action of 1.0 commands away from the nominal pose. "
        "Upstream PDControl instead spans the whole joint range (1.92 rad median "
        "on G1) and centres on the range midpoint; loco_mjx/urma2 uses 0.25-0.75 "
        "rad per actuator across ~50 robots. Unset keeps upstream behaviour.",
    )
    parser.add_argument(
        "--foot-model",
        choices=["stock", "box"],
        default="stock",
        help="Foot contact geometry. 'stock' keeps upstream's two centre-line "
        "capsules, whose 9 mm height offset leaves H1 standing on a single "
        "lateral roller with zero fore-aft support -- it topples under PD hold "
        "at every gain from 1x to 30x. 'box' installs a flat 0.24x0.09 m foot "
        "with the same underside height. No effect on G1, whose feet are "
        "already four spheres in a rectangle.",
    )
    parser.add_argument(
        "--reference-grounding",
        choices=["none", "constant", "per_frame"],
        default="none",
        help="Put the reference's feet on the floor. The cross-humanoid "
        "references are ungrounded -- on dance2_subject4 100%% of frames have "
        "no ground contact on either robot -- so the policy is asked to imitate "
        "an airborne motion. 'per_frame' is what the single-body H1 results "
        "that did not fall were trained on; 'constant' cannot fix per-frame "
        "floating and is the control.",
    )
    parser.add_argument(
        "--segments", type=int, default=1,
        help="Split training into N segments, saving a checkpoint after each. "
        "The resume function is compiled once and reused, so segmenting is "
        "free; it caps what a timeout or node fault can destroy.",
    )
    parser.add_argument(
        "--init-checkpoint", type=Path, default=None,
        help="Warm start from a saved checkpoint (e.g. the last segment of an "
        "interrupted run). Parameters carry over; env and optimiser are rebuilt.",
    )
    parser.add_argument("--total-envs", type=int, default=384)
    parser.add_argument("--envs-per-robot", type=int, default=None)
    parser.add_argument("--total-timesteps", type=float, default=1e6)
    parser.add_argument("--num-steps", type=int, default=32)
    parser.add_argument("--num-minibatches", type=int, default=12)
    parser.add_argument("--update-epochs", type=int, default=1)
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 128])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--init-std", type=float, default=0.2)
    parser.add_argument("--learnable-std", action="store_true")
    parser.add_argument("--no-normalize-reward", action="store_true")
    parser.add_argument("--use-mjwarp", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--run-tag", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.morphology is not None and args.morphology not in (
        "continuous",
        "nominal",
    ):
        if not (
            args.morphology.startswith("catalog")
            and args.morphology[len("catalog"):].isdigit()
            and int(args.morphology[len("catalog"):]) >= 2
        ):
            parser.error("--morphology must be 'continuous' or 'catalogN' (N>=2)")
    if args.append_joint_features is None:
        args.append_joint_features = args.backbone != "masked_mlp"
    if args.backbone != "masked_mlp" and not args.append_joint_features:
        parser.error("URMA backbones require --append-joint-features.")
    if args.output_dir is None:
        args.output_dir = (
            WORKSPACE / "experiments" / "scaling_cross_topology" / args.run_tag
        )
    return args


if __name__ == "__main__":
    main()
