"""GPU (MJX) rollout core for the shared-policy experiment.

On WSL with GPU JAX, MJX vmapped rollouts run ~250x faster than the CPU MuJoCo
single-env path (~25-30k steps/s vs ~100). Everything here is pure functional JAX
(explicit keys), so it sidesteps the CPU-only jit/RNG-poisoning gotcha entirely.

Two policy regimes:
  * JAX expert policy  -> fully-jitted lax.scan (metrics + dataset collection)
  * torch BC policy    -> host loop (torch can't run inside a jitted jax scan), but
                          the env step is still jitted+vmapped over N envs.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

import common as C


def gpu_available() -> bool:
    try:
        return jax.default_backend() == "gpu"
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Env + expert (MJX)
# --------------------------------------------------------------------------- #
_MJX_ENV_CACHE: dict = {}
_JAX_POLICY_CACHE: dict = {}


def build_mjx_env(variant: dict):
    preset = variant["preset"]
    if preset in _MJX_ENV_CACHE:
        return _MJX_ENV_CACHE[preset]
    from loco_mujoco.trajectory import Trajectory
    from loco_mujoco.core.wrappers.mjx import LocoMjxWrapper
    from morphology_deepmimic import prepare_variant, get_robot, make_mimic_env, control_config

    robot = get_robot(C.ROBOT_KEY)
    var = prepare_variant(robot, preset, C.CACHE_TAG)
    traj = Trajectory.load(variant["reference_npz"])
    resolved = variant.get("control_params_resolved")
    control = variant.get("control", "torque")
    if control == "pd" and resolved:
        ctrl = dict(control_type="PDControl",
                    control_params={k: list(v) for k, v in resolved.items()})
    else:
        ctrl = control_config(robot.key, control)
        scale = float(variant.get("pd_gain_scale", 1.0) or 1.0)
        if control == "pd" and scale != 1.0:
            cp = ctrl["control_params"]
            cp["p_gain"] = [g * scale for g in cp["p_gain"]]
            cp["d_gain"] = [g * scale for g in cp["d_gain"]]
    env = make_mimic_env(var["mjx_env_name"], traj, headless=True, **ctrl)
    wenv = LocoMjxWrapper(env)
    _MJX_ENV_CACHE[preset] = wenv
    return wenv


def load_jax_policy(variant: dict):
    """Return a vmappable deterministic policy fn: single obs (450,) -> action (23,)."""
    return load_jax_policy_pkl(variant["agent_pkl"])


def load_jax_policy_pkl(pkl: str):
    """Vmappable deterministic policy fn from a saved PPOJax pkl path."""
    if pkl in _JAX_POLICY_CACHE:
        return _JAX_POLICY_CACHE[pkl]
    from loco_mujoco.algorithms import PPOJax
    agent_conf, agent_state = PPOJax.load_agent(pkl)
    ts = agent_state.train_state
    if int(agent_conf.config.experiment.n_seeds) > 1:
        ts = jax.tree.map(lambda x: x[0], ts)
    params, run_stats = ts.params, ts.run_stats
    net = agent_conf.network

    def policy(obs):  # obs: (obs_dim,)  -> (act_dim,)
        y, _ = net.apply({"params": params, "run_stats": run_stats}, obs,
                         mutable=["run_stats"])
        return y[0].mean()

    _JAX_POLICY_CACHE[pkl] = policy
    return policy


# --------------------------------------------------------------------------- #
# Pure-JAX rollouts (expert policy)
# --------------------------------------------------------------------------- #
def _episode_stats(rec_ret, rec_len, rec_done, rec_ab):
    """From (T,N) per-step masked records -> dict of completed-episode aggregates."""
    done = np.asarray(rec_done).reshape(-1)
    rets = np.asarray(rec_ret).reshape(-1)[done]
    lens = np.asarray(rec_len).reshape(-1)[done]
    ab = np.asarray(rec_ab).reshape(-1)[done]
    if len(rets) == 0:
        return dict(n_episodes=0, mean_return=0.0, std_return=0.0, mean_length=0.0,
                    nonfall_rate=0.0, mean_tracking_reward=0.0, survived_full=0)
    nonfall = 1.0 - ab.astype(np.float64)
    return dict(
        n_episodes=int(len(rets)),
        mean_return=float(rets.mean()), std_return=float(rets.std()),
        mean_length=float(lens.mean()),
        nonfall_rate=float(nonfall.mean()),
        mean_tracking_reward=float((rets / np.maximum(lens, 1)).mean()),
        survived_full=int((lens >= C.HORIZON).sum()),
    )


def expert_metrics(wenv, policy, n_envs, n_steps, key):
    """Roll out a JAX policy over N envs for T steps (auto-reset) -> episode aggregates."""
    reset = jax.jit(jax.vmap(wenv.reset))
    step = jax.jit(jax.vmap(wenv.step))
    pol = jax.vmap(policy)
    obs, state = reset(jax.random.split(key, n_envs))

    def body(carry, _):
        obs, state, cret, clen = carry
        a = pol(obs)
        nobs, r, ab, dn, info, nstate = step(state, a)
        cret = cret + r
        clen = clen + 1
        rec = (jnp.where(dn, cret, 0.0), jnp.where(dn, clen, 0),
               dn, jnp.where(dn, ab, False))
        cret = jnp.where(dn, 0.0, cret)
        clen = jnp.where(dn, 0, clen)
        return (nobs, nstate, cret, clen), rec

    init = (obs, state, jnp.zeros(n_envs), jnp.zeros(n_envs, dtype=jnp.int32))
    _carry, recs = jax.lax.scan(body, init, None, length=n_steps)
    jax.block_until_ready(recs[0])
    return _episode_stats(*recs)


def collect_dataset(wenv, policy, n_envs, chunk_steps, target, key, max_chunks=200):
    """Collect (obs, action, reward, done) transitions until ``target`` samples.

    Runs in chunks (each a jitted scan) and offloads to host between chunks to
    bound device memory. Returns concatenated host numpy arrays.
    """
    reset = jax.jit(jax.vmap(wenv.reset))
    step = jax.jit(jax.vmap(wenv.step))
    pol = jax.vmap(policy)

    def body(carry, _):
        obs, state = carry
        a = pol(obs)
        nobs, r, ab, dn, info, nstate = step(state, a)
        return (nobs, nstate), (obs, a, r, dn)

    @jax.jit
    def run_chunk(obs, state):
        return jax.lax.scan(body, (obs, state), None, length=chunk_steps)

    obs, state = reset(jax.random.split(key, n_envs))
    O, A, R, D = [], [], [], []
    n = 0
    for _ in range(max_chunks):
        (obs, state), (o, a, r, d) = run_chunk(obs, state)
        # (T,N,...) -> (T*N,...) on host
        O.append(np.asarray(o, np.float32).reshape(-1, o.shape[-1]))
        A.append(np.asarray(a, np.float32).reshape(-1, a.shape[-1]))
        R.append(np.asarray(r, np.float32).reshape(-1))
        D.append(np.asarray(d).reshape(-1))
        n += O[-1].shape[0]
        if n >= target:
            break
    obs_arr = np.concatenate(O)[:target]
    return dict(obs=obs_arr, action=np.concatenate(A)[:target],
                reward=np.concatenate(R)[:target], done=np.concatenate(D)[:target],
                step_idx=np.zeros(len(obs_arr), np.int32))


# --------------------------------------------------------------------------- #
# torch-policy rollout (BC eval / DAgger) -- host loop, jitted vmapped env step
# --------------------------------------------------------------------------- #
def torch_rollout(wenv, act_batch, n_envs, n_steps, key, collect=False):
    """Roll out a torch policy (act_batch: (N,obs)->(N,act) numpy) over N MJX envs.

    Returns (episode_stats, frames|None). frames (collect=True): obs (M,obs),
    action (M,act) host numpy from the steps actually taken.
    """
    reset = jax.jit(jax.vmap(wenv.reset))
    step = jax.jit(jax.vmap(wenv.step))
    obs, state = reset(jax.random.split(key, n_envs))

    cret = np.zeros(n_envs); clen = np.zeros(n_envs, np.int64)
    ep_ret, ep_len, ep_ab = [], [], []
    fobs, fact = [], []
    prev_a = None
    smooth = []
    for _t in range(n_steps):
        obs_np = np.asarray(obs, np.float32)
        a_np = act_batch(obs_np)                      # (N, act)
        if collect:
            fobs.append(obs_np.copy()); fact.append(a_np.copy())
        if prev_a is not None:
            smooth.append(float(np.abs(a_np - prev_a).mean()))
        prev_a = a_np
        obs, r, ab, dn, info, state = step(state, jnp.asarray(a_np))
        r_np = np.asarray(r); dn_np = np.asarray(dn).astype(bool); ab_np = np.asarray(ab).astype(bool)
        cret += r_np; clen += 1
        if dn_np.any():
            ep_ret.extend(cret[dn_np].tolist())
            ep_len.extend(clen[dn_np].tolist())
            ep_ab.extend(ab_np[dn_np].tolist())
            cret[dn_np] = 0.0; clen[dn_np] = 0
    rets = np.array(ep_ret); lens = np.array(ep_len); ab = np.array(ep_ab, dtype=bool)
    if len(rets) == 0:
        stats = dict(n_episodes=0, mean_return=0.0, std_return=0.0, mean_length=0.0,
                     nonfall_rate=0.0, mean_tracking_reward=0.0, survived_full=0)
    else:
        stats = dict(n_episodes=int(len(rets)), mean_return=float(rets.mean()),
                     std_return=float(rets.std()), mean_length=float(lens.mean()),
                     nonfall_rate=float((~ab).mean()),
                     mean_tracking_reward=float((rets / np.maximum(lens, 1)).mean()),
                     survived_full=int((lens >= C.HORIZON).sum()))
    stats["action_smoothness"] = float(np.mean(smooth)) if smooth else None
    frames = None
    if collect:
        frames = dict(obs=np.concatenate(fobs), action=np.concatenate(fact))
    return stats, frames
