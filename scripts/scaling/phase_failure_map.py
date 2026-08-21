"""T0.6 -- where in the clip do episodes die?

If failure is uniform over ``dance2_subject4`` the problem is the method; if it
concentrates in one passage, one infeasible segment of this clip is killing
every episode and the fix is clip selection.  The distinction is only visible
against the RSI **start phase**, which no existing evaluator records.

Method.  One long rollout per arm, with the environment's own auto-reset left
running, so every environment supplies many episodes rather than one: each
episode is a segment of the ``done`` array, tagged with the
``subtraj_step_no_init`` the trajectory handler drew for it.  The start phase is
therefore sampled by the production RSI path, not pinned by a patched reset --
pinning would have to bypass ``TrajectoryHandler.reset_state``, which is part of
what is under test.  Coverage is reported (distinct phases, samples per bin) so
the sampling is auditable.

Termination cause is attributed exactly as ``diagnose_termination`` does, from
the state logged at the top of the ending step (the post-step data is already
the auto-reset episode): root height / root rotation / root deviation, each
scored against its own limit, plus the benign "reference clip ran out" case.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
for _p in (str(WORKSPACE / "scripts"), str(WORKSPACE / "scripts" / "h1md")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from loco_mujoco.core.utils.math import quat_scalarfirst2scalarlast  # noqa: E402

from scaling.crosstopo_eval_common import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_REFERENCE_ROOT,
    build_env,
    group_tables,
    load_manifest,
    load_policy,
    rollout,
)

CAUSES = ("fell_height", "fell_rotation", "drifted", "clip_exhausted", "unexplained")


def _criteria(table, pre):
    """(deviation, height, rotation) per (step, env), from the logged pre-step pose."""
    handler = table.terminal
    qpos = pre["qpos"]                                   # (steps, envs, nq)
    step_no = pre["step_no"]
    step_init = pre["step_no_init"]
    root_xy = np.asarray(handler.root_xy)

    ref_xy = table.ref_qpos[:, root_xy]                  # (clip, 2)
    deviation = np.linalg.norm(
        qpos[:, :, root_xy] - (ref_xy[step_no] - ref_xy[step_init]), axis=-1
    )

    height = qpos[:, :, int(handler.root_height_ind)].copy()
    offset_fn = getattr(table.raw, "root_height_offset", None)
    if offset_fn is not None and pre["morph"].shape[-1] > 0:
        flat = pre["morph"].reshape(-1, pre["morph"].shape[-1])
        offsets = np.asarray(jax.vmap(offset_fn)(jnp.asarray(flat)))
        height = height - offsets.reshape(height.shape)

    quat = quat_scalarfirst2scalarlast(qpos[:, :, np.asarray(handler.root_quat_ind)])
    quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
    rotation = 2 * np.arccos(
        np.clip(quat @ np.asarray(handler._centroid_quat), -1, 1)
    )
    return deviation, height, rotation


def _episodes(table, record, min_length: int = 1):
    """Every COMPLETED episode in the rollout, with its start phase and cause."""
    pre = record["pre"][table.index]
    done = record["done"][:, table.start : table.stop]
    reward = record["reward"][:, table.start : table.stop]
    deviation, height, rotation = _criteria(table, pre)

    handler = table.terminal
    low, high = [float(v) for v in handler.root_height_range]
    limit = float(handler.max_root_pos_deviation)
    threshold = float(handler._valid_threshold)
    clip_length = table.clip_length

    steps, n_envs = done.shape
    episodes = []
    for e in range(n_envs):
        start = 0
        for t in np.nonzero(done[:, e])[0]:
            t = int(t)
            length = t - start + 1
            if length < min_length:
                start = t + 1
                continue
            sev_hgt = abs(height[t, e] - 0.5 * (low + high)) / max(
                0.5 * (high - low), 1e-9
            )
            sev_rot = rotation[t, e] / max(threshold, 1e-9)
            sev_dev = deviation[t, e] / max(limit, 1e-9)
            phase_end = int(pre["step_no"][t, e])
            if sev_hgt >= 0.9:
                cause = "fell_height"
            elif sev_rot >= 0.9:
                cause = "fell_rotation"
            elif sev_dev >= 0.9:
                cause = "drifted"
            elif phase_end >= clip_length - 3:
                cause = "clip_exhausted"
            else:
                cause = "unexplained"
            episodes.append(
                {
                    "env": e,
                    "start_phase": int(pre["step_no_init"][start, e]),
                    "end_phase": phase_end,
                    "steps_survived": length,
                    "cause": cause,
                    "return": float(reward[start : t + 1, e].sum()),
                    "severity_height": float(sev_hgt),
                    "severity_rotation": float(sev_rot),
                    "severity_deviation": float(sev_dev),
                }
            )
            start = t + 1
    return episodes


def _bin_table(episodes, clip_length, n_bins):
    width = int(np.ceil(clip_length / n_bins))
    bins = {}
    for ep in episodes:
        b = min(ep["start_phase"] // width, n_bins - 1)
        bins.setdefault(b, []).append(ep)
    rows = []
    for b in range(n_bins):
        items = bins.get(b, [])
        lengths = np.array([e["steps_survived"] for e in items], dtype=float)
        causes = Counter(e["cause"] for e in items)
        rows.append(
            {
                "bin": b,
                "start_phase_lo": b * width,
                "start_phase_hi": min((b + 1) * width, clip_length) - 1,
                "n_episodes": len(items),
                "steps_survived_mean": float(lengths.mean()) if items else None,
                "steps_survived_median": float(np.median(lengths)) if items else None,
                "steps_survived_p10": (
                    float(np.percentile(lengths, 10)) if items else None
                ),
                "steps_survived_p90": (
                    float(np.percentile(lengths, 90)) if items else None
                ),
                "causes": {c: int(causes.get(c, 0)) for c in CAUSES},
            }
        )
    return rows, width


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--envs-per-robot", type=int, default=64)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--bins", type=int, default=40)
    parser.add_argument(
        "--morphologies", nargs="*", default=["nominal", "continuous"],
        help="'nominal' = the stock body; 'continuous' = the production "
        "randomized-body sampler, i.e. one distinct body per environment.",
    )
    parser.add_argument("--root-frame", default="episode_start",
                        choices=("episode_start", "absolute"))
    parser.add_argument("--reference-root", type=Path,
                        default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint, manifest_path, manifest = load_manifest(args.checkpoint)
    result = {
        "probe": "T0_6_phase_failure_map",
        "checkpoint": str(checkpoint),
        "training_manifest": str(manifest_path),
        "root_frame_used": args.root_frame,
        "steps": args.steps,
        "envs_per_robot": args.envs_per_robot,
        "seed": args.seed,
        "bins": args.bins,
        "clip": manifest.get("clip"),
        "clip_window": manifest.get("clip_windows"),
        "note": "start phases are RSI-sampled by the production reset path and "
                "binned; every completed episode in the rollout is used.",
        "arms": {},
    }

    for morphology in args.morphologies:
        env, _env_args, _meta = build_env(
            manifest,
            envs_per_robot=args.envs_per_robot,
            reference_root=args.reference_root,
            morphology=morphology,
            root_frame=args.root_frame,
        )
        tables = group_tables(env)
        _conf, _state, policy_mean = load_policy(checkpoint, manifest)
        record = rollout(
            env, tables, lambda i, obs, st, pre: policy_mean(obs),
            args.steps, args.seed,
        )
        arm = {}
        for t in tables:
            episodes = _episodes(t, record)
            rows, width = _bin_table(episodes, t.clip_length, args.bins)
            lengths = np.array([e["steps_survived"] for e in episodes], dtype=float)
            occupied = [r for r in rows if r["n_episodes"] > 0]
            means = np.array(
                [r["steps_survived_mean"] for r in occupied], dtype=float
            )
            causes = Counter(e["cause"] for e in episodes)
            phases = np.array([e["start_phase"] for e in episodes])
            arm[t.name] = {
                "clip_length": t.clip_length,
                "bin_width_frames": width,
                "n_episodes": len(episodes),
                "n_distinct_start_phases": int(np.unique(phases).size),
                "n_bins_occupied": len(occupied),
                "steps_survived_mean": float(lengths.mean()) if len(lengths) else None,
                "steps_survived_median": (
                    float(np.median(lengths)) if len(lengths) else None
                ),
                "cause_counts": {c: int(causes.get(c, 0)) for c in CAUSES},
                "bin_mean_spread": {
                    "min": float(means.min()) if len(means) else None,
                    "max": float(means.max()) if len(means) else None,
                    "std": float(means.std()) if len(means) else None,
                    "coefficient_of_variation": (
                        float(means.std() / means.mean()) if len(means) else None
                    ),
                    "max_over_min": (
                        float(means.max() / max(means.min(), 1e-9))
                        if len(means) else None
                    ),
                },
                "worst_bins": sorted(
                    occupied, key=lambda r: r["steps_survived_mean"]
                )[:5],
                "best_bins": sorted(
                    occupied, key=lambda r: -r["steps_survived_mean"]
                )[:5],
                "bins": rows,
                "end_phase_histogram": {
                    str(int(k)): int(v)
                    for k, v in sorted(
                        Counter(
                            (e["end_phase"] // width) * width for e in episodes
                        ).items()
                    )
                },
            }
            print(
                f"[t06] {morphology:>10s} {t.name:>3s}: {len(episodes)} episodes, "
                f"mean {arm[t.name]['steps_survived_mean']:.1f} steps, "
                f"bin-mean CV {arm[t.name]['bin_mean_spread']['coefficient_of_variation']:.3f}, "
                f"causes {arm[t.name]['cause_counts']}",
                flush=True,
            )
        result["arms"][morphology] = arm
        del env

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[t06] -> {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
