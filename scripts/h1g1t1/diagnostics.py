#!/usr/bin/env python3
"""Dependency-light URMA2 log parser and motion-quality reporter."""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

TABLE_RE = re.compile(r"│\s*(?P<key>[^│]+?)\s*│\s*(?P<value>[^│]+?)\s*│")
STEP_KEY = "steps/nr_env_steps"
ROBOT_ORDER = ("h1", "g1", "t1")
ROBOT_LABELS = {"h1": "Unitree H1", "g1": "Unitree G1", "t1": "Booster T1",
                "unitree_h1": "Unitree H1", "unitree_g1": "Unitree G1",
                "booster_t1": "Booster T1"}


def _scalar(text: str) -> float | str:
    value = text.strip().replace(",", "")
    if value.lower() in {"true", "false"}:
        return float(value.lower() == "true")
    try:
        return float(value)
    except ValueError:
        return value


def parse_console_log(path: str | Path) -> pd.DataFrame:
    """Parse RL-X box-table output into one row per global step."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    current: dict[str, Any] = {}

    def flush() -> None:
        if STEP_KEY in current:
            rows.append(dict(current))
        current.clear()

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "┌" in line:
            flush()
            continue
        if "└" in line:
            flush()
            continue
        match = TABLE_RE.search(line)
        if not match:
            continue
        key = match.group("key").strip()
        # Console blocks normally have box borders.  This fallback also handles
        # stripped/plain logs: after a step and its trailing steps/* fields, the
        # first non-step metric starts the next block.
        if STEP_KEY in current and not key.startswith("steps/"):
            flush()
        current[key] = _scalar(match.group("value"))
    flush()
    if not rows:
        raise ValueError(f"No complete metric blocks containing {STEP_KEY!r} in {path}")
    frame = pd.DataFrame(rows)
    for column in frame:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=[STEP_KEY]).copy()
    frame[STEP_KEY] = frame[STEP_KEY].astype(np.int64)
    frame = frame.sort_values(STEP_KEY, kind="stable")
    # Console captures can contain complementary fragments for the same step
    # (for example, a full metric block followed by an RL-X footer or a pasted
    # tail excerpt). GroupBy.last selects the last *non-null* value per column,
    # preserving all channels while still preferring the newest duplicate.
    return frame.groupby(STEP_KEY, as_index=False, sort=True).last().reset_index(drop=True)


def load_metric_frame(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() != ".csv":
        return parse_console_log(path)
    frame = pd.read_csv(path)
    if STEP_KEY not in frame:
        for candidate in ("step", "steps", "nr_env_steps"):
            if candidate in frame:
                frame = frame.rename(columns={candidate: STEP_KEY})
                break
    if STEP_KEY not in frame:
        raise ValueError(f"CSV {path} has no {STEP_KEY!r} column")
    # The supplied historical 3-topology curve export used compact column
    # names. Normalize it into the same schema as a fresh console parse.
    legacy = {}
    for robot in ROBOT_ORDER:
        legacy.update({
            f"joint_{robot}": f"env_info/joint_tracking_error/{robot}",
            f"head_{robot}": f"env_info/root_heading_error/{robot}",
            f"air_{robot}": f"env_info/foot_airborne/{robot}",
            f"refair_{robot}": f"env_info/ref_foot_airborne/{robot}",
            f"eplen_{robot}": f"episode/length/{robot}",
            f"ret_{robot}": f"episode/return/{robot}",
        })
    frame = frame.rename(columns={key: value for key, value in legacy.items() if key in frame})
    return frame.sort_values(STEP_KEY, kind="stable").reset_index(drop=True)


def _robots(columns: Iterable[str]) -> list[str]:
    found = {str(c).rsplit("/", 1)[-1].lower() for c in columns}
    return [robot for robot in ROBOT_ORDER if robot in found]


def _alias(frame: pd.DataFrame, source: str, target: str) -> None:
    if source in frame and target not in frame:
        frame[target] = pd.to_numeric(frame[source], errors="coerce")


def derive_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Retain raw metrics and add RMSE/degree channels with stable names."""
    out = frame.copy()
    for robot in _robots(frame.columns):
        joint = f"env_info/joint_tracking_error/{robot}"
        if joint in out:
            out[f"tracking/joint_rmse_rad/{robot}"] = np.sqrt(
                pd.to_numeric(out[joint], errors="coerce").clip(lower=0))
            out[f"tracking/joint_rmse_deg/{robot}"] = np.degrees(
                out[f"tracking/joint_rmse_rad/{robot}"])
        qvel = f"env_info/qvel_tracking_error/{robot}"
        if qvel in out:
            out[f"tracking/qvel_rmse/{robot}"] = np.sqrt(
                pd.to_numeric(out[qvel], errors="coerce").clip(lower=0))
        heading = f"env_info/root_heading_error/{robot}"
        if heading in out:
            out[f"tracking/root_heading_error_deg/{robot}"] = np.degrees(
                pd.to_numeric(out[heading], errors="coerce"))
        aliases = {
            f"env_info/rpos_tracking_error/{robot}": f"tracking/body_position_error/{robot}",
            f"env_info/rquat_tracking_error/{robot}": f"tracking/body_orientation_error/{robot}",
            f"env_info/foot_height_error/{robot}": f"feet/height_error_m2/{robot}",
            f"env_info/foot_airborne/{robot}": f"feet/airborne_fraction/{robot}",
            f"env_info/ref_foot_airborne/{robot}": f"feet/reference_airborne_fraction/{robot}",
            f"env_info/foot_penetration_m/{robot}": f"feet/penetration_m/{robot}",
            f"env_info/foot_clearance_m/{robot}": f"feet/clearance_m/{robot}",
            f"env_info/foot_slip_speed_sq/{robot}": f"feet/slip_speed_sq/{robot}",
            f"env_info/foot_z_speed_sq/{robot}": f"feet/z_speed_sq/{robot}",
            f"env_curriculum/morphology_coeff/{robot}": f"randomization/morphology_coeff/{robot}",
        }
        for source, target in aliases.items():
            _alias(out, source, target)
        for prefix, stable in (("episode/return", "episode/return"),
                               ("episode/length", "episode/length"),
                               ("rollout/episode_return", "episode/return"),
                               ("rollout/episode_length", "episode/length")):
            _alias(out, f"{prefix}/{robot}", f"{stable}/{robot}")
    for source, target in {
        "policy_ratio/approx_kl": "optimizer/approx_kl",
        "policy_ratio/clip_fraction": "optimizer/clip_fraction",
        "policy/update_rejected": "optimizer/update_rejected",
        "policy/grad_norm": "optimizer/policy_grad_norm",
        "gradients/policy_grad_norm": "optimizer/policy_grad_norm",
        "critic/grad_norm": "optimizer/critic_grad_norm",
        "gradients/critic_grad_norm": "optimizer/critic_grad_norm",
        "policy/std": "optimizer/policy_std",
        "policy/std_dev": "optimizer/policy_std",
        "critic/explained_variance": "optimizer/value_explained_variance",
        "v_value/explained_variance": "optimizer/value_explained_variance",
    }.items():
        _alias(out, source, target)
    return out


def validate_recipe(nr_envs: int, nr_steps: int, minibatch_size: int,
                    save_every: int, total_timesteps: int, n_robots: int) -> dict[str, int]:
    """Validate divisibility before an expensive JAX compile."""
    values = (nr_envs, nr_steps, minibatch_size, save_every, total_timesteps, n_robots)
    if any(int(v) <= 0 for v in values):
        raise ValueError("All recipe values must be positive integers")
    if nr_envs % n_robots:
        raise ValueError(f"nr_envs ({nr_envs}) must be divisible by robot count ({n_robots})")
    if minibatch_size % n_robots:
        raise ValueError(f"minibatch_size ({minibatch_size}) must be divisible by robot count ({n_robots})")
    batch_size = nr_envs * nr_steps
    if batch_size % minibatch_size:
        raise ValueError(f"batch_size ({batch_size}) must be divisible by minibatch_size ({minibatch_size})")
    if save_every % batch_size:
        raise ValueError(f"save_every ({save_every}) must be divisible by batch_size ({batch_size})")
    if total_timesteps % save_every:
        raise ValueError(f"total_timesteps ({total_timesteps}) must be divisible by save_every ({save_every})")
    return {
        "nr_envs": nr_envs, "nr_steps": nr_steps, "minibatch_size": minibatch_size,
        "save_every": save_every, "total_timesteps": total_timesteps,
        "n_robots": n_robots, "envs_per_robot": nr_envs // n_robots,
        "batch_size": batch_size, "minibatches_per_epoch": batch_size // minibatch_size,
        "updates_per_save": save_every // batch_size,
        "save_intervals": total_timesteps // save_every,
    }


def _last(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.iloc[-1]) if len(values) else None


def _first(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.iloc[0]) if len(values) else None


def summarize_final(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        raise ValueError("Cannot summarize an empty metric frame")
    robots: dict[str, Any] = {}
    flags: dict[str, Any] = {}
    for robot in _robots(frame.columns):
        joint = _last(frame, f"tracking/joint_rmse_rad/{robot}")
        heading = _last(frame, f"tracking/root_heading_error_deg/{robot}")
        airborne = _last(frame, f"feet/airborne_fraction/{robot}")
        ref_airborne = _last(frame, f"feet/reference_airborne_fraction/{robot}")
        robots[robot] = {
            "label": ROBOT_LABELS.get(robot, robot),
            "joint_rmse_rad": joint,
            "joint_rmse_deg": math.degrees(joint) if joint is not None else None,
            "joint_rmse_rad_initial": _first(frame, f"tracking/joint_rmse_rad/{robot}"),
            "qvel_rmse": _last(frame, f"tracking/qvel_rmse/{robot}"),
            "root_heading_error_deg": heading,
            "foot_airborne_fraction": airborne,
            "reference_airborne_fraction": ref_airborne,
            "episode_return": _last(frame, f"episode/return/{robot}"),
            "episode_length": _last(frame, f"episode/length/{robot}"),
        }
        flags[robot] = {
            "heading_error_over_45_deg": bool(heading is not None and heading > 45),
            "joint_rmse_over_0_35_rad": bool(joint is not None and joint > .35),
            "airborne_mismatch_over_0_10": bool(
                airborne is not None and ref_airborne is not None and abs(airborne - ref_airborne) > .10),
        }
    return {
        "final_step": int(pd.to_numeric(frame[STEP_KEY], errors="coerce").dropna().iloc[-1]),
        "rows": int(len(frame)), "robots": robots, "quality_flags": flags,
    }


def load_crosseval_files(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for item in paths:
        path = Path(item)
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["_source_file"] = str(path)
        payload.setdefault("eval_condition", {}).setdefault("morphology_coeff", 0.0)
        for metrics in payload.get("robots", {}).values():
            floor = metrics.get("reference_vs_raw_rmse_rad_absolute",
                                metrics.get("reference_vs_raw_rmse_rad"))
            metrics["reference_convention_suspect"] = bool(
                floor is not None and float(floor) > .20)
        results.append(payload)
    return results


def scan_log_health(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file() or path.suffix.lower() == ".csv":
        return {"source": str(path), "uncaught_exception": False,
                "traceback": False, "nan_lines": []}
    text = path.read_text(encoding="utf-8", errors="replace")
    nan_lines = [line.strip() for line in text.splitlines()
                 if re.search(r"(?<![A-Za-z])(?:nan|inf)(?![A-Za-z])", line, re.I)]
    return {"source": str(path), "uncaught_exception": "Uncaught exception" in text,
            "traceback": "Traceback (most recent call last)" in text,
            "nan_lines": nan_lines[-50:]}


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _family_plot(frame: pd.DataFrame, prefix: str, title: str,
                 ylabel: str, path: Path) -> bool:
    columns = [c for c in frame if c.startswith(prefix + "/")]
    if not columns:
        return False
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(9, 5.2))
    x = frame[STEP_KEY] / 1_000_000
    for column in columns:
        robot = column.rsplit("/", 1)[-1]
        ax.plot(x, frame[column], label=ROBOT_LABELS.get(robot, robot))
    ax.set(title=title, xlabel="Global environment steps (millions)", ylabel=ylabel)
    ax.grid(True, alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def generate_plots(frame: pd.DataFrame, output_dir: str | Path,
                   crossevals: Sequence[Mapping[str, Any]] = ()) -> list[str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []
    specs = (
        ("tracking/joint_rmse_rad", "Joint tracking RMSE", "RMSE (rad)", "tracking_joint_rmse.png"),
        ("tracking/root_heading_error_deg", "Root-heading tracking error", "Mean absolute error (deg)", "tracking_heading_error.png"),
        ("tracking/qvel_rmse", "Joint-velocity tracking RMSE", "RMSE (logged units)", "tracking_qvel_rmse.png"),
        ("tracking/body_position_error", "DeepMimic body-position error", "Logged error", "tracking_body_position.png"),
        ("tracking/body_orientation_error", "DeepMimic body-orientation error", "Logged error", "tracking_body_orientation.png"),
        ("randomization/morphology_coeff", "Morphology-randomization schedule", "Coefficient", "randomization_curriculum.png"),
    )
    for prefix, title, ylabel, filename in specs:
        if _family_plot(frame, prefix, title, ylabel, output_dir / filename):
            generated.append(filename)

    foot = [c for c in frame if c.startswith("feet/airborne_fraction/")
            or c.startswith("feet/reference_airborne_fraction/")]
    if foot:
        plt = _pyplot()
        fig, ax = plt.subplots(figsize=(9, 5.2))
        x = frame[STEP_KEY] / 1_000_000
        for column in foot:
            robot = column.rsplit("/", 1)[-1]
            kind = "reference" if "reference_" in column else "policy"
            ax.plot(x, frame[column], label=f"{ROBOT_LABELS.get(robot, robot)} {kind}")
        ax.set(title="Foot airborne fraction versus reference",
               xlabel="Global environment steps (millions)", ylabel="Fraction")
        ax.grid(True, alpha=.25)
        ax.legend(ncol=2)
        fig.tight_layout()
        filename = "feet_airborne_vs_reference.png"
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)
        generated.append(filename)

    optimizer = {
        "Approx. KL": ("optimizer/approx_kl", "policy_ratio/approx_kl"),
        "Clip fraction": ("optimizer/clip_fraction", "policy_ratio/clip_fraction"),
        "Update rejected": ("optimizer/update_rejected", "policy/update_rejected"),
        "Policy grad norm": ("optimizer/policy_grad_norm", "policy/grad_norm"),
        "Critic grad norm": ("optimizer/critic_grad_norm", "critic/grad_norm"),
        "Policy std": ("optimizer/policy_std", "policy/std"),
        "Explained variance": ("optimizer/value_explained_variance", "critic/explained_variance"),
    }
    present = [(label, next((c for c in options if c in frame), None))
               for label, options in optimizer.items()]
    present = [(label, column) for label, column in present if column]
    if present:
        plt = _pyplot()
        fig, ax = plt.subplots(figsize=(9, 5.2))
        x = frame[STEP_KEY] / 1_000_000
        for label, column in present:
            ax.plot(x, frame[column], label=label)
        ax.set(title="PPO optimizer health", xlabel="Global environment steps (millions)",
               ylabel="Logged value")
        ax.grid(True, alpha=.25)
        ax.legend()
        fig.tight_layout()
        filename = "optimizer_health.png"
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)
        generated.append(filename)

    context = [c for c in frame if c.startswith("episode/return/")
               or c.startswith("episode/length/")]
    if context:
        plt = _pyplot()
        fig, ax = plt.subplots(figsize=(9, 5.2))
        x = frame[STEP_KEY] / 1_000_000
        for column in context:
            ax.plot(x, frame[column], label=column.replace("episode/", ""))
        ax.set(title="Return and episode length (secondary context only)",
               xlabel="Global environment steps (millions)", ylabel="Logged value")
        ax.grid(True, alpha=.25)
        ax.legend(ncol=2)
        fig.tight_layout()
        filename = "returns_survival_context.png"
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)
        generated.append(filename)

    rows: list[dict[str, Any]] = []
    for result in crossevals:
        condition = result.get("eval_condition", {})
        coeff = float(condition.get("morphology_coeff", 0) or 0)
        kind = "zero" if condition.get("zero_action") else "policy"
        for robot, metrics in result.get("robots", {}).items():
            rows.append({"robot": robot, "condition": f"{kind} m={coeff:g}",
                         "rmse": metrics.get("raw_rmse_rad_absolute", metrics.get("raw_rmse_rad")),
                         "floor": metrics.get("reference_vs_raw_rmse_rad_absolute",
                                              metrics.get("reference_vs_raw_rmse_rad"))})
    cdf = pd.DataFrame(rows)
    if not cdf.empty and cdf["rmse"].notna().any():
        cdf = cdf.dropna(subset=["rmse"])
        plt = _pyplot()
        fig, ax = plt.subplots(figsize=(10, 5.5))
        x = np.arange(len(cdf))
        ax.bar(x, cdf["rmse"])
        if cdf["floor"].notna().any():
            ax.scatter(x, cdf["floor"], marker="x", label="reference-vs-raw floor")
            ax.legend()
        labels = [f"{ROBOT_LABELS.get(r.robot, r.robot)}\n{r.condition}" for r in cdf.itertuples()]
        ax.set_xticks(x, labels, rotation=35, ha="right")
        ax.set(title="Crosseval policy quality by morphology condition",
               ylabel="Absolute joint RMSE (rad)")
        ax.grid(True, axis="y", alpha=.25)
        fig.tight_layout()
        filename = "crosseval_summary.png"
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)
        generated.append(filename)
    return generated


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}" if value is not None and np.isfinite(float(value)) else "n/a"
    except (TypeError, ValueError):
        return "n/a"


def render_summary_markdown(summary: Mapping[str, Any], crossevals: Sequence[Mapping[str, Any]],
                            plots: Sequence[str], health: Mapping[str, Any]) -> str:
    lines = ["# H1+G1+T1 training diagnostic report", "",
             f"Final logged step: **{int(summary['final_step']):,}**.", "",
             "## Online motion-quality metrics", "",
             "| Robot | Joint RMSE (rad) | Joint RMSE (deg) | Heading error (deg) | qvel RMSE | Airborne / reference |",
             "|---|---:|---:|---:|---:|---:|"]
    for robot, metrics in summary.get("robots", {}).items():
        air = f"{_fmt(metrics.get('foot_airborne_fraction'), 3)} / {_fmt(metrics.get('reference_airborne_fraction'), 3)}"
        lines.append(f"| {metrics.get('label', robot)} | {_fmt(metrics.get('joint_rmse_rad'))} | "
                     f"{_fmt(metrics.get('joint_rmse_deg'), 2)} | "
                     f"{_fmt(metrics.get('root_heading_error_deg'), 2)} | "
                     f"{_fmt(metrics.get('qvel_rmse'))} | {air} |")
    lines += ["", "Return and episode length are retained only as context; they are not used as imitation-quality verdicts."]
    if crossevals:
        lines += ["", "## Crosseval", "",
                  "| Condition | Robot | Policy absolute RMSE | Reference-vs-raw floor | Heading mean / p95 | Alive |",
                  "|---|---|---:|---:|---:|---:|"]
        for result in crossevals:
            condition = result.get("eval_condition", {})
            label = ("zero-action" if condition.get("zero_action") else "policy") + \
                    f", morphology={float(condition.get('morphology_coeff', 0) or 0):g}"
            for robot, metrics in result.get("robots", {}).items():
                floor = metrics.get("reference_vs_raw_rmse_rad_absolute",
                                    metrics.get("reference_vs_raw_rmse_rad"))
                warning = " (suspect convention)" if metrics.get("reference_convention_suspect") else ""
                lines.append(f"| {label} | {ROBOT_LABELS.get(robot, robot)} | "
                             f"{_fmt(metrics.get('raw_rmse_rad_absolute', metrics.get('raw_rmse_rad')))} | "
                             f"{_fmt(floor)}{warning} | "
                             f"{_fmt(metrics.get('heading_error_deg_mean'), 1)} / "
                             f"{_fmt(metrics.get('heading_error_deg_p95'), 1)} | "
                             f"{_fmt(metrics.get('alive_fraction'), 3)} |")
    lines += ["", "## Run health", "",
              f"Uncaught exception marker: **{bool(health.get('uncaught_exception'))}**.",
              f"Traceback marker: **{bool(health.get('traceback'))}**.",
              f"NaN/Inf log lines retained: **{len(health.get('nan_lines', []))}**."]
    if plots:
        lines += ["", "## Generated plots", ""] + [f"- `{p}`" for p in plots]
    return "\n".join(lines) + "\n"


def generate_report(metric_source: str | Path, output_dir: str | Path,
                    crosseval_paths: Sequence[str | Path] = (),
                    recipe: Mapping[str, Any] | None = None) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = load_metric_frame(metric_source)
    derived = derive_metrics(raw)
    crossevals = load_crosseval_files(crosseval_paths)
    summary = summarize_final(derived)
    health = scan_log_health(metric_source)
    plots = generate_plots(derived, output_dir, crossevals)
    raw.to_csv(output_dir / "raw_metrics.csv", index=False)
    derived.to_csv(output_dir / "tracking_metrics.csv", index=False)
    (output_dir / "crosseval_combined.json").write_text(
        json.dumps(crossevals, indent=2, allow_nan=False), encoding="utf-8")
    report = {**summary, "metric_source": str(Path(metric_source)),
              "recipe": dict(recipe or {}), "run_health": health,
              "plots": plots, "crosseval_count": len(crossevals)}
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    (output_dir / "summary.md").write_text(
        render_summary_markdown(summary, crossevals, plots, health), encoding="utf-8")
    return report


def _expand(items: Sequence[str]) -> list[Path]:
    paths: list[Path] = []
    for item in items:
        candidate = Path(item)
        if any(ch in item for ch in "*?["):
            paths.extend(sorted(candidate.parent.glob(candidate.name)))
        elif candidate.is_dir():
            paths.extend(sorted(candidate.glob("*.json")))
        else:
            paths.append(candidate)
    return paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    parse = sub.add_parser("parse-log")
    parse.add_argument("--log", required=True)
    parse.add_argument("--out-dir", required=True)
    report = sub.add_parser("report")
    report.add_argument("--log", required=True, help="console log or metrics CSV")
    report.add_argument("--out-dir", required=True)
    report.add_argument("--crosseval", action="append", default=[])
    report.add_argument("--recipe-json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    if args.command == "parse-log":
        raw = parse_console_log(args.log)
        raw.to_csv(output / "raw_metrics.csv", index=False)
        derive_metrics(raw).to_csv(output / "tracking_metrics.csv", index=False)
        print(output / "tracking_metrics.csv")
        return 0
    recipe = {}
    if args.recipe_json:
        recipe = json.loads(Path(args.recipe_json).read_text(encoding="utf-8"))
    report = generate_report(args.log, output, _expand(args.crosseval), recipe)
    print(json.dumps({"final_step": report["final_step"], "out_dir": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
