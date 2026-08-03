"""Turn cross-embodiment manifests and evaluations into the results tables.

Every number printed here comes from a file on disk; nothing is recomputed from
memory.  Per-robot rows are never collapsed into an aggregate, because an
aggregate hid a real per-robot failure in the earlier smoke test.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = WORKSPACE / "experiments" / "cross_embodiment"

BACKBONE_LABEL = {"urmav2": "URMAv2", "urma": "URMA", "masked_mlp": "masked MLP"}


def _load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def collect(root: Path):
    runs = {}
    for manifest_path in sorted(root.glob("*/manifest.json")):
        manifest = _load(manifest_path)
        if manifest is None:
            continue
        tag = manifest_path.parent.name
        runs[tag] = {
            "tag": tag,
            "manifest": manifest,
            "per_robot": _load(root / "evaluations" / f"{tag}_per_robot.json"),
            "heldout": _load(root / "evaluations" / f"{tag}_toddlerbot_heldout.json"),
        }
    return runs


def _mean_ci(values):
    """Mean and half-width of a 95% normal CI; None when n < 2."""
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, None
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, 1.96 * math.sqrt(variance / len(values))


def _fmt(value, digits=2):
    return "n/a" if value is None else f"{value:.{digits}f}"


def per_robot_table(evaluation) -> list[str]:
    lines = [
        "| Robot | Joints | Policy return | Zero-action return | Δ return | Welch t | "
        "Policy length | Zero length | Non-fall (policy) | Non-fall (zero) | Beats zero |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for robot, entry in evaluation["per_robot"].items():
        lines.append(
            f"| {robot} | {entry.get('num_joints', '?')} | "
            f"{_fmt(entry['policy_return']['mean'])} | "
            f"{_fmt(entry['zero_action_return']['mean'])} | "
            f"{_fmt(entry['mean_return_improvement'])} | "
            f"{_fmt(entry.get('return_improvement_welch_t'))} | "
            f"{_fmt(entry['policy_length']['mean'], 1)} | "
            f"{_fmt(entry['zero_action_length']['mean'], 1)} | "
            f"{_fmt(entry.get('policy_non_fall_rate'))} | "
            f"{_fmt(entry.get('zero_action_non_fall_rate'))} | "
            f"{'yes' if entry.get('beats_zero_action_on_mean_return') else 'NO'} |"
        )
    return lines


def render(runs) -> str:
    out: list[str] = []
    out.append("## Runs on disk\n")
    out.append(
        "| Run | Backbone | Robots | One-hot | Seed | Steps | Steps/min | "
        "Group sizes | Final train return | Evaluated |"
    )
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for tag, run in runs.items():
        m = run["manifest"]
        out.append(
            f"| `{tag}` | {BACKBONE_LABEL.get(m.get('backbone'), m.get('backbone'))} | "
            f"{'+'.join(m['robots'])} | {'yes' if m.get('robot_one_hot') else 'no'} | "
            f"{m.get('seed')} | {m.get('total_timesteps', 0):,} | "
            f"{m.get('steps_per_minute', 0) / 1e6:.2f}M | "
            f"{m.get('group_sizes')} | {_fmt(m.get('mean_episode_return_last'))} | "
            f"{'yes' if run['per_robot'] else 'no'} |"
        )

    for tag, run in runs.items():
        if not run["per_robot"]:
            continue
        m = run["manifest"]
        out.append(
            f"\n### `{tag}` — {BACKBONE_LABEL.get(m.get('backbone'), m.get('backbone'))}"
            f", seed {m.get('seed')}, one-hot "
            f"{'on' if m.get('robot_one_hot') else 'off'}\n"
        )
        out.extend(per_robot_table(run["per_robot"]))
        verdict = run["per_robot"].get("every_robot_beats_zero_action")
        out.append(
            f"\nEvery robot beats its exact-reset zero-action baseline: "
            f"**{'yes' if verdict else 'no'}**."
        )
        if run["heldout"]:
            out.append(
                "\nTopology held out (ToddlerBot, 30 joints, never trained on):\n"
            )
            out.extend(per_robot_table(run["heldout"]))

    # Runs are grouped by (robot set, optimizer epochs, step budget) before any
    # averaging.  Pooling a 1-epoch run with a 4-epoch one would compare
    # architectures across incomparable budgets and hide the epoch effect.
    out.append("\n## URMA vs masked MLP, within matched budgets\n")
    out.append(
        "Only runs sharing a robot set, step budget and optimizer-epoch count "
        "are averaged together. Compare rows *within* a block, never across.\n"
    )
    blocks: dict[tuple, dict[str, list]] = {}
    for run in runs.values():
        if not run["per_robot"]:
            continue
        m = run["manifest"]
        key = (
            "+".join(m["robots"]),
            int(m.get("update_epochs", 0)),
            int(m.get("total_timesteps", 0)),
        )
        entry = blocks.setdefault(key, {})
        entry.setdefault(str(m.get("backbone")), []).append(
            run["per_robot"]["overall_mean_return_improvement"]
        )

    for (robots, epochs, steps), arms in sorted(blocks.items()):
        out.append(f"\n**{robots} — {steps:,} steps, {epochs} optimizer epoch(s)**\n")
        out.append("| Backbone | Seeds | Mean Δ return vs zero action | Seed spread |")
        out.append("|---|---|---|---|")
        for backbone, deltas in sorted(arms.items()):
            mean, _ = _mean_ci(deltas)
            spread = max(deltas) - min(deltas) if len(deltas) > 1 else None
            out.append(
                f"| {BACKBONE_LABEL.get(backbone, backbone)} | {len(deltas)} | "
                f"{_fmt(mean)} | {_fmt(spread)} |"
            )
    return "\n".join(out) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    runs = collect(args.root)
    text = render(runs)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
