"""Step-by-step trace of one episode per family, from a trained checkpoint.

Why this exists: G1 sits at 24-34 steps in every arm run so far — 60M and 300M,
alone and shared, PD gain 1x/3x/6x, dance clip and stand-still clip — against a
zero-torque baseline of 26 steps. Aggregate metrics cannot say whether the
policy's actions reach the actuators at all, whether control saturates
immediately, whether the PD target is in sane units, or which terminal fires
first. This walks one episode and dumps those quantities per step, for the
policy and for zero action in the same environment, so the failure can be read
instead of inferred.

It reuses the evaluator's own env/agent construction (`_env_args`,
`build_cross_humanoid_env`, `trainer.load_agent`) so the traced environment is
byte-for-byte the one the reported numbers came from — a bespoke env would be a
different experiment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
for _p in (str(WORKSPACE / "scripts"), str(WORKSPACE / "scripts" / "h1md")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from scaling.evaluate_cross_humanoid_policy import (  # noqa: E402
    _env_args,
    _find_manifest,
    _resolve_checkpoint,
)
from scaling.parallel_cross_humanoid_train import (  # noqa: E402
    build_cross_humanoid_env,
    trainer_for,
)


def _np(x):
    return np.asarray(jax.device_get(x))


def _group_data(group_state):
    """The MJX ``Data`` inside one group's env state, wherever it lives."""
    for attr in ("data", "mjx_data", "_data"):
        d = getattr(group_state, attr, None)
        if d is not None and hasattr(d, "qpos"):
            return d
    raise AttributeError(
        f"no MJX Data on {type(group_state).__name__}; "
        f"fields: {getattr(group_state, '__dataclass_fields__', {}).keys()}")


def _actuator_limits(model):
    nu = int(model.nu)
    ctrl_hi = np.abs(np.asarray(model.actuator_ctrlrange)[:nu, 1]).astype(float)
    ctrl_hi[ctrl_hi == 0] = 1.0
    force = []
    for a in range(nu):
        f = float(np.max(np.abs(np.asarray(model.actuator_forcerange)[a])))
        if not bool(np.asarray(model.actuator_forcelimited)[a]) or f <= 0:
            f = (abs(float(np.asarray(model.actuator_gear)[a, 0]))
                 * abs(float(np.asarray(model.actuator_gainprm)[a, 0]))
                 * float(ctrl_hi[a]))
        force.append(f if f > 0 else np.inf)
    qpos_adr, dof_adr = [], []
    for a in range(nu):
        j = int(np.asarray(model.actuator_trnid)[a, 0])
        qpos_adr.append(int(np.asarray(model.jnt_qposadr)[j]))
        dof_adr.append(int(np.asarray(model.jnt_dofadr)[j]))
    return ctrl_hi, np.array(force), np.array(qpos_adr), np.array(dof_adr)


def trace(env, agent_conf, variables, horizon: int, seed: int, zero_action: bool,
          env_index: int = 0):
    keys = jax.random.split(jax.random.PRNGKey(seed), env.num_envs)
    obs, state = jax.jit(env.reset)(keys)

    def choose(o):
        if zero_action:
            return jnp.zeros((o.shape[0], env.max_action_dim), dtype=o.dtype)
        (policy, _), _ = agent_conf.network.apply(variables, o, mutable=["run_stats"])
        return policy.mean()

    step_fn = jax.jit(env.step)
    action_fn = jax.jit(choose)

    limits = {}
    for gi, group in enumerate(env.groups):
        model = env.envs[gi].sys if hasattr(env, "envs") else None
        if model is None:
            model = getattr(state.group_states[gi], "model", None)
        limits[group.name] = _actuator_limits(model) if model is not None else None

    per_group = {g.name: [] for g in env.groups}
    alive = {g.name: True for g in env.groups}
    for t in range(horizon):
        action = action_fn(obs)
        obs, reward, absorbing, done, _, state = step_fn(state, action)
        a = _np(action)
        for gi, group in enumerate(env.groups):
            if not alive[group.name]:
                continue
            i = group.start + min(env_index, group.size - 1)
            gs = state.group_states[gi]
            d = _group_data(gs)
            qpos = _np(d.qpos)[min(env_index, group.size - 1)]
            ctrl = _np(d.ctrl)[min(env_index, group.size - 1)]
            tau = _np(d.qfrc_actuator)[min(env_index, group.size - 1)]
            lim = limits[group.name]
            rec = {
                "step": t,
                "reward": float(_np(reward)[i]),
                "absorbing": bool(_np(absorbing)[i]),
                "done": bool(_np(done)[i]),
                "action_absmax": float(np.max(np.abs(a[i]))),
                "action_absmean": float(np.mean(np.abs(a[i]))),
                "root_z": float(qpos[2]),
            }
            if lim is not None:
                ctrl_hi, force, qpos_adr, dof_adr = lim
                nu = len(ctrl_hi)
                sat = np.abs(ctrl[:nu]) / ctrl_hi
                rec.update({
                    "ctrl_absmax": float(np.max(np.abs(ctrl[:nu]))),
                    "ctrl_sat_frac": float(np.mean(sat > 0.99)),
                    "ctrl_sat_max": float(sat.max()),
                    "torque_absmax_Nm": float(np.max(np.abs(tau[dof_adr]))),
                    "torque_over_limit_frac": float(
                        np.mean(np.abs(tau[dof_adr]) > force * 0.99)),
                    "joint_absmax_rad": float(np.max(np.abs(qpos[qpos_adr]))),
                })
            per_group[group.name].append(rec)
            if rec["absorbing"] or rec["done"]:
                alive[group.name] = False
        if not any(alive.values()):
            break
    return per_group


def _summarise(series):
    if not series:
        return {}
    keys = [k for k in series[0] if isinstance(series[0][k], (int, float))
            and k != "step"]
    out = {}
    for k in keys:
        v = np.array([s[k] for s in series], dtype=float)
        out[k] = {"first": float(v[0]), "median": float(np.median(v)),
                  "max": float(v.max()), "last": float(v[-1])}
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--robots", nargs="+", default=None)
    p.add_argument("--envs-per-robot", type=int, default=4)
    p.add_argument("--horizon", type=int, default=150)
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--use-mjwarp", action="store_true")
    p.add_argument("--morphology-override", default="nominal")
    p.add_argument("--reference-root", type=Path,
                   default=WORKSPACE / "external_data" / "cross_humanoid")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    args.checkpoint = _resolve_checkpoint(args.checkpoint)
    _, manifest = _find_manifest(args.checkpoint)
    env_args = _env_args(args, manifest)
    if args.morphology_override:
        env_args.morphology = args.morphology_override
    trainer = trainer_for(str(manifest.get("backbone", "urmav2")))
    agent_conf, agent_state = trainer.load_agent(args.checkpoint)
    variables = {"params": agent_state.train_state.params,
                 "run_stats": agent_state.train_state.run_stats}
    env, _ = build_cross_humanoid_env(env_args)

    report = {"checkpoint": str(args.checkpoint),
              "morphology": env_args.morphology, "robots": {}}
    for label, zero in (("policy", False), ("zero_action", True)):
        traces = trace(env, agent_conf, variables, args.horizon, args.seed, zero)
        for name, recs in traces.items():
            entry = report["robots"].setdefault(name, {})
            entry[label] = {"steps_survived": len(recs),
                            "summary": _summarise(recs),
                            "per_step": recs}
            print(f"[trace] {name}/{label}: {len(recs)} steps", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"[trace] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
