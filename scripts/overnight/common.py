"""Shared backbone for the overnight shared-policy experiment.

Research question: can ONE shared policy control 5 modified G1 variants, instead of
needing one expert per variant?

Everything below runs on the Windows CPU venv (CPU JAX for the loco-mujoco experts,
torch+CUDA for behaviour cloning). The expensive primitives validated before writing
this module:
  * PPOJax.load_agent(pkl) -> (agent_conf, agent_state)         (CPU JAX)
  * ImitationFactory mimic env on a registered *variant* CPU env (obs=450, act=23)
  * env.step(action) -> (obs, reward, absorbing, done, info)    (~100 steps/s/env)

All five chosen variants are G1 (23 DoF), so obs/action dims are identical across
them -- that is what makes "drop expert E's brain into robot V" cross-evaluation
meaningful at all.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[2]
SCRIPTS = WORKSPACE / "scripts"
sys.path.insert(0, str(SCRIPTS))

# --------------------------------------------------------------------------- #
# Fixed experiment configuration
# --------------------------------------------------------------------------- #
ROBOT_KEY = "g1"
CLIP = "dance2_subject4"             # the single shared motion every G1 expert tracks
CACHE_TAG = "dance"
DEEPMIMIC_ROOT = WORKSPACE / "external_data" / "deepmimic_morphology" / ROBOT_KEY / CLIP
RESEARCH_CSV = WORKSPACE / "images" / "g1_morphology_research_summary.csv"
HORIZON = 1000                       # make_mimic_env horizon (a survived episode hits this)

# The 5 variants: base + three single-axis variations + one extreme multi-axis body.
# extreme_tall_light is intentionally the most different -> used as the held-out
# variant for the morphology-generalization probe.
SELECTED_VARIANTS = ["nominal", "short_legs", "long_arms", "heavy_torso", "extreme_tall_light"]
HELDOUT_VARIANT = "extreme_tall_light"

# Morphology descriptor: the raw per-variant scale parameters (the only knobs the
# variant generator exposes). foot_scale_xyz is unpacked into 3 features.
DESCRIPTOR_KEYS = [
    "leg_length_scale", "arm_length_scale", "shoulder_width_scale",
    "foot_scale_x", "foot_scale_y", "foot_scale_z", "torso_mass_scale",
]


# --------------------------------------------------------------------------- #
# Selection: best expert cell + checkpoint per variant
# --------------------------------------------------------------------------- #
def _research_rows() -> dict:
    with open(RESEARCH_CSV, newline="") as f:
        return {r["preset"]: r for r in csv.DictReader(f)}


def best_checkpoint_for(preset: str) -> dict:
    """Resolve the best (cell, checkpoint pkl) for a variant.

    Uses the research summary's ``best_cell`` column, then picks the
    highest-mean-return checkpoint inside that cell. Falls back to scanning every
    cell of the preset if the recorded best_cell is missing. Manifest agent_paths
    are WSL paths, so the pkl is always resolved relative to the cell dir.
    """
    rows = _research_rows()
    candidates: list[Path] = []
    if preset in rows and rows[preset].get("best_cell"):
        candidates.append(DEEPMIMIC_ROOT / rows[preset]["best_cell"])
    # fallback: any cell dir starting with "<preset>" (preset or preset__suffix)
    for d in sorted(DEEPMIMIC_ROOT.glob(f"{preset}*")):
        if d.is_dir() and (d / "manifest.json").exists() and d not in candidates:
            candidates.append(d)

    best = None
    for cell in candidates:
        man_path = cell / "manifest.json"
        if not man_path.exists():
            continue
        man = json.loads(man_path.read_text(encoding="utf-8"))
        ref = cell / "reference.npz"
        if not ref.exists():
            continue
        for ck in man.get("checkpoints", []):
            reldir = Path(ck["agent_path"]).parts[-2]
            pkl = cell / "checkpoints" / reldir / "PPOJax_saved.pkl"
            if not pkl.exists():
                continue
            ret = float(ck["mean_episode_return"])
            cand = {
                "preset": preset,
                "cell": cell.name,
                "cell_dir": str(cell),
                "agent_pkl": str(pkl),
                "reference_npz": str(ref),
                "checkpoint_steps": int(ck["cumulative_steps"]),
                "expert_mean_return": ret,
                "expert_mean_length": float(ck.get("mean_episode_length", 0.0)),
                "control": man.get("control", "torque"),
                "control_params_resolved": man.get("control_params_resolved"),
                "pd_gain_scale": man.get("pd_gain_scale", 1.0),
                "window_start_frame": man.get("window_start_frame"),
                "window_frames": man.get("window_frames"),
            }
            if best is None or ret > best["expert_mean_return"]:
                best = cand
    if best is None:
        raise FileNotFoundError(f"No usable expert checkpoint found for preset {preset!r}")
    return best


def descriptor_for(preset: str) -> dict:
    """Raw morphology scale descriptor for a variant (foot xyz unpacked)."""
    from g1_morphology_variants import PRESETS
    p = PRESETS[preset]
    fx, fy, fz = p.foot_scale_xyz
    return {
        "leg_length_scale": float(p.leg_length_scale),
        "arm_length_scale": float(p.arm_length_scale),
        "shoulder_width_scale": float(p.shoulder_width_scale),
        "foot_scale_x": float(fx), "foot_scale_y": float(fy), "foot_scale_z": float(fz),
        "torso_mass_scale": float(p.torso_mass_scale),
        "label": p.label,
    }


def load_selection(outdir: Path) -> dict:
    return json.loads((Path(outdir) / "selection.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Env + expert construction (cached per variant within a process)
# --------------------------------------------------------------------------- #
_ENV_CACHE: dict = {}
_AGENT_CACHE: dict = {}


def build_env(variant: dict):
    """Build (or reuse) the CPU mimic env for a variant cell.

    The variant body owns the PD controller, so an env is keyed purely by the
    variant's own resolved gains + reference -- exactly the "robot V" we drop a
    brain into.
    """
    preset = variant["preset"]
    if preset in _ENV_CACHE:
        return _ENV_CACHE[preset]
    from loco_mujoco.trajectory import Trajectory
    from morphology_deepmimic import prepare_variant, get_robot, make_mimic_env, control_config

    robot = get_robot(ROBOT_KEY)
    var = prepare_variant(robot, preset, CACHE_TAG)
    traj = Trajectory.load(variant["reference_npz"])

    # Resolve the exact control modality the policy was trained under (mirrors
    # render_morphology_deepmimic.rollout_to_mp4). A PD policy outputs target joint
    # angles; a torque env would misread them. Some older cells stored only the
    # gain *scale*, not the resolved gains (e.g. nominal: 3x), so reconstruct here.
    resolved = variant.get("control_params_resolved")
    control = variant.get("control", "torque")
    if control == "pd" and resolved:
        ctrl_params = dict(control_type="PDControl",
                           control_params={k: list(v) for k, v in resolved.items()})
    else:
        ctrl_params = control_config(robot.key, control)
        scale = float(variant.get("pd_gain_scale", 1.0) or 1.0)
        if control == "pd" and scale != 1.0:
            cp = ctrl_params["control_params"]
            cp["p_gain"] = [g * scale for g in cp["p_gain"]]
            cp["d_gain"] = [g * scale for g in cp["d_gain"]]

    env = make_mimic_env(var["cpu_env_name"], traj, headless=True, **ctrl_params)
    _ENV_CACHE[preset] = env
    return env


def load_expert(variant: dict):
    """Load an expert agent and return a deterministic+stochastic action fn.

    Returns ``(act_mean, act_sample)`` where each maps raw obs (np (obs,) or
    (1,obs)) -> action np (act,). Handles multi-seed train states (takes seed 0).
    """
    pkl = variant["agent_pkl"]
    if pkl in _AGENT_CACHE:
        return _AGENT_CACHE[pkl]
    import jax
    import jax.numpy as jnp
    from loco_mujoco.algorithms import PPOJax

    agent_conf, agent_state = PPOJax.load_agent(pkl)
    ts = agent_state.train_state
    n_seeds = int(agent_conf.config.experiment.n_seeds)
    if n_seeds > 1:
        ts = jax.tree.map(lambda x: x[0], ts)
    params, run_stats = ts.params, ts.run_stats
    network = agent_conf.network

    # NOTE: deliberately EAGER (no jax.jit). A jitted apply was verified to perturb
    # the global JAX RNG that the CPU env's keyless reset() draws from, which
    # silently wrecks the initial-state distribution (episodes collapse to ~30
    # steps) even though the per-step action values are bit-identical. The eager
    # path reproduces the reported expert lengths (~800). env.step dominates wall
    # time, so the eager apply costs nothing measurable.
    def _forward(obs):
        y, _ = network.apply({"params": params, "run_stats": run_stats},
                             jnp.atleast_2d(jnp.asarray(obs)), mutable=["run_stats"])
        pi, _v = y
        return pi

    def act_mean(obs):
        return np.asarray(_forward(obs).mean())

    def act_sample(obs, rng):
        pi = _forward(obs)
        mean, std = np.asarray(pi.mean()), np.asarray(pi.stddev())
        return mean + np.asarray(jax.random.normal(rng, mean.shape)) * std

    out = (act_mean, act_sample)
    _AGENT_CACHE[pkl] = out
    return out


# --------------------------------------------------------------------------- #
# Rollout
# --------------------------------------------------------------------------- #
@dataclass
class EpisodeResult:
    steps: int
    ret: float
    fell: bool                # absorbing terminal reached before horizon
    termination: str          # "fell" | "horizon" | "stopped"
    mean_reward: float        # ret / steps (== mean mimic tracking reward/step)


def run_episode(env, act_fn, *, max_steps=HORIZON, collect=False, key=None):
    """Run one episode. Returns (EpisodeResult, frames|None).

    ``frames`` (when collect=True) is a dict of stacked arrays:
    obs, action, reward, done, step_idx.

    The CPU env's reset is not bit-reproducible even when keyed (it carries an
    internal RNG that advances each call), so callers rely on episode COUNT for
    stable statistics, not on per-episode reproducibility. ``key`` is forwarded
    when given purely to decorrelate parallel phases.
    """
    import numpy as _np
    obs = env.reset() if key is None else env.reset(key=key)
    buf_obs, buf_act, buf_rew, buf_done, buf_step = [], [], [], [], []
    ret = 0.0
    fell = False
    termination = "horizon"
    steps = 0
    for t in range(max_steps):
        a = act_fn(obs)
        a2 = _np.atleast_2d(a)
        nobs, reward, absorbing, done, _info = env.step(a2)
        r = float(_np.asarray(reward).reshape(-1)[0])
        d = bool(_np.asarray(done).reshape(-1)[0])
        ab = bool(_np.asarray(absorbing).reshape(-1)[0])
        if collect:
            buf_obs.append(_np.asarray(obs).reshape(-1).copy())
            buf_act.append(_np.asarray(a).reshape(-1).copy())
            buf_rew.append(r)
            buf_done.append(d)
            buf_step.append(t)
        ret += r
        steps += 1
        obs = nobs
        if d:
            fell = ab
            termination = "fell" if ab else "horizon"
            break
    res = EpisodeResult(steps=steps, ret=ret, fell=fell, termination=termination,
                        mean_reward=ret / max(steps, 1))
    frames = None
    if collect:
        frames = {
            "obs": _np.asarray(buf_obs, dtype=_np.float32),
            "action": _np.asarray(buf_act, dtype=_np.float32),
            "reward": _np.asarray(buf_rew, dtype=_np.float32),
            "done": _np.asarray(buf_done, dtype=_np.bool_),
            "step_idx": _np.asarray(buf_step, dtype=_np.int32),
        }
    return res, frames


def aggregate(results: list[EpisodeResult]) -> dict:
    n = len(results)
    if n == 0:
        return dict(n_episodes=0)
    rets = np.array([r.ret for r in results])
    lens = np.array([r.steps for r in results])
    nonfall = np.array([0.0 if r.fell else 1.0 for r in results])
    mrew = np.array([r.mean_reward for r in results])
    return dict(
        n_episodes=n,
        mean_return=float(rets.mean()), std_return=float(rets.std()),
        mean_length=float(lens.mean()),
        nonfall_rate=float(nonfall.mean()),
        mean_tracking_reward=float(mrew.mean()),
        survived_full=int((lens >= HORIZON).sum()),
    )


# --------------------------------------------------------------------------- #
# Phase state helpers (resumable driver)
# --------------------------------------------------------------------------- #
def state_dir(outdir: Path) -> Path:
    d = Path(outdir) / "state"
    d.mkdir(parents=True, exist_ok=True)
    return d


def phase_done(outdir: Path, phase: str) -> bool:
    return (state_dir(outdir) / f"{phase}.done").exists()


def mark_done(outdir: Path, phase: str, info: dict | None = None) -> None:
    (state_dir(outdir) / f"{phase}.done").write_text(
        json.dumps(info or {"ts": time.time()}, indent=2), encoding="utf-8")


def write_json(path: Path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
