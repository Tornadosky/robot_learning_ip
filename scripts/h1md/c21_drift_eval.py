"""C21 -- does the `dance_root` preset actually reduce world-frame drift?

C18/C19 established that policies trained with the stock DeepMimic weights end up
0.67-2.75 m from where the motion goes, because every spatial reward term is
upper-body-relative and root position survives only as 3 of 22 entries inside
`qpos`'s mean-square. C20 trained the `dance_root` preset, which restricts
`joints_for_mimic` to the root free joint so that term becomes pure root pose.

This measures the thing the fix is supposed to move: root XY error against the
reference at the matching phase. Episode length and return cannot answer it --
the two arms optimise different rewards, so their returns are not comparable.

Rollouts use a pinned start phase so step index == reference frame index
(C10 Finding 19: with the default random RSI this comparison is meaningless).
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE / "scripts"))
sys.path.insert(0, str(WORKSPACE / "scripts" / "scaling"))

from loco_mujoco.core.wrappers import LogWrapper, VecEnv  # noqa: E402
from online_h1_train import build_online_env  # noqa: E402


def eval_checkpoint(ckpt: Path, horizon: int, num_envs: int, seed: int) -> dict:
    from loco_mujoco.algorithms import PPOJax

    manifest = json.loads((ckpt.parent.parent / "manifest.json").read_text(encoding="utf-8"))
    freq = float(manifest["frequency_hz"])
    args = SimpleNamespace(
        clip=str(manifest["clip"]),
        duration=float(manifest["window_frames"]) / freq,
        start_frame=int(manifest["window_start_frame"]),
        run_tag=f"c21_{ckpt.parent.parent.name}",
        use_mjwarp=False,
        backbone=str(manifest.get("backbone", "mlp")),
        resample_per_episode=False,
        morph_low=list(manifest["morphology_low"]),
        morph_high=list(manifest["morphology_high"]),
        catalog=Path(manifest["catalog_path"]) if manifest.get("catalog_path") else None,
        catalog_mode=str(manifest.get("catalog_mode", "continuous")),
        catalog_stride=1,
        keep_morph_bounds=True,
        reward_weights=str(manifest.get("reward_weights_preset", "dance")),
        terminal_handler=manifest.get("terminal_handler"),
        # the goal type changes the observation width (441 vs 438), so the env
        # must be rebuilt with the same one the checkpoint was trained under
        goal_type=str(manifest.get("goal_type", "GoalTrajMimic")),
    )
    env, _meta = build_online_env(args)
    trajectory = env.th.traj

    agent_conf, agent_state = PPOJax.load_agent(ckpt)
    ts = agent_state.train_state
    if agent_conf.config.experiment.n_seeds > 1:
        ts = jax.tree.map(lambda x: x[0], ts)
    variables = {"params": ts.params, "run_stats": ts.run_stats}

    def act(obs):
        (pi, _), _ = agent_conf.network.apply(variables, obs, mutable=["run_stats"])
        return pi.mean()

    # same wrapping the repo's own online evaluator uses, so obs normalisation
    # and the (obs, state) interface match training
    wrapped = VecEnv(LogWrapper(env))
    keys = jax.random.split(jax.random.PRNGKey(seed), num_envs)
    obs, state = jax.jit(wrapped.reset)(keys)

    def step(carry, _):
        obs, state, alive = carry
        a = act(obs)
        obs2, _, _, done, _, state2 = wrapped.step(state, a)
        alive = alive & ~done.astype(jnp.bool_)
        # Record the trajectory PHASE alongside the root, not just the step
        # index: reference-state initialisation starts each env at a random
        # phase, so indexing the reference from frame 0 measures the phase
        # offset rather than the policy. Harmless on a clip whose root barely
        # moves; catastrophic on one that walks 10 m.
        phase = state2.env_state.additional_carry.traj_state.subtraj_step_no
        return (obs2, state2, alive), (state2.env_state.data.qpos[:, :3], alive, phase)

    init = (obs, state, jnp.ones((num_envs,), dtype=jnp.bool_))
    _, (roots, alives, phases) = jax.jit(lambda c: jax.lax.scan(step, c, None, horizon))(init)
    return np.asarray(roots), np.asarray(alives), np.asarray(phases), trajectory


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--horizon", type=int, default=400)
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20260809)
    ap.add_argument("--only", nargs="+", default=None,
                    help="substring filter on arm labels; evaluates only matching arms")
    ap.add_argument("--checkpoint-root", type=Path,
                    default=WORKSPACE / "experiments" / "h1_morphology_deepmimic_20260808" / "checkpoints")
    args = ap.parse_args()

    arms = {
        "dance (baseline)": ["c16_clean_nominal", "c17_single_seed1", "c17_single_seed2"],
        "dance_root": ["c20_rootterm_seed0", "c20_rootterm_seed1", "c20_rootterm_seed2"],
        "dance_root_heavy": ["c22_rootheavy_seed0", "c22_rootheavy_seed1", "c22_rootheavy_seed2"],
        "root_only (diagnostic)": ["c23_rootonly_seed0", "c23_rootonly_seed1", "c23_rootonly_seed2"],
        "dance_root @60M": ["c24_budget60m_seed0", "c24_budget60m_seed1"],
        "dance_root @100M": ["c24_budget100m_seed0"],
        "WALK dance_root @20M": ["c26_walk20m_seed0", "c26_walk20m_seed1", "c26_walk20m_seed2"],
        "COMBINED heavy@60M single": ["c28_heavy60m_seed0", "c28_heavy60m_seed1"],
        "COMBINED heavy@60M continuous": ["c28_heavy60m_continuous_seed0"],
        "ROOTERR observable @20M": ["c29_rooterr_seed0", "c29_rooterr_seed1", "c29_rooterr_seed2"],
        "WALK+ROOTERR @20M": ["c30_walkrooterr_seed0", "c30_walkrooterr_seed1", "c30_walkrooterr_seed2"],
        # continuous morphology, dance weights. All arms are evaluated WITHOUT
        # deviation termination so the measuring instrument is identical; the
        # arms differ only in how they were trained.
        "MULTIBODY no-deviation": ["c13_spatial", "c34_nodev_seed1"],
        "MULTIBODY deviation 0.5m": ["c34_dev05_seed0", "c34_dev05_seed1"],
    }
    if args.only:
        arms = {k: v for k, v in arms.items() if any(o.lower() in k.lower() for o in args.only)}
    out = {}
    for label, tags in arms.items():
        per_seed = []
        for tag in tags:
            ckpt = args.checkpoint_root / tag / "checkpoint_final" / "PPOJax_saved.pkl"
            if not ckpt.exists():
                print(f"  {tag}: MISSING")
                continue
            roots, alives, phases, traj = eval_checkpoint(ckpt, args.horizon, args.num_envs, args.seed)
            ref_all = np.asarray(traj.data.qpos)[:, :2]
            # index the reference by each env's OWN phase, not by step number
            idx = np.clip(phases, 0, len(ref_all) - 1)
            ref = ref_all[idx]                       # (T, E, 2)
            err = np.linalg.norm(roots[:, :, :2] - ref, axis=-1)
            m = float(err[alives].mean())
            fin = float(err[-1][alives[-1]].mean()) if alives[-1].any() else float("nan")
            per_seed.append(m)
            print(f"  {tag}: root XY err mean {m:.2f} m, final {fin:.2f} m, "
                  f"alive at end {float(alives[-1].mean()):.2f}")
        if per_seed:
            out[label] = {"per_seed_mean_root_err_m": per_seed,
                          "mean": st.mean(per_seed),
                          "sd": st.stdev(per_seed) if len(per_seed) > 1 else 0.0}
            print(f"{label}: {out[label]['mean']:.2f} +/- {out[label]['sd']:.2f} m\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "component": "C21_drift_eval",
        "metric": "root XY error vs the reference at the matching phase, averaged over live envs",
        "horizon": args.horizon, "num_envs": args.num_envs,
        "arms": out,
    }, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
