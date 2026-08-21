#!/usr/bin/env python3
"""Audit an H1+G1 URMA single-motion experiment directory.

This is intentionally conservative. It distinguishes:
- system errors: malformed/missing evidence or a configuration mismatch;
- learning warnings: policy quality has not met the requested gate;
- unsupported evidence: the current sampler has no rejection/skip accounting.

It does not import the research repository and is safe to run on a copied
experiment directory.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable


EXPECTED_ROBOTS = ["h1", "g1"]
EXPECTED_MASKS = [19, 23]
VALID_BACKBONES = {"urma", "urmav2"}


@dataclass
class Finding:
    level: str
    check: str
    message: str
    path: str | None = None


def walk_values(value: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from walk_values(child, child_prefix)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from walk_values(child, f"{prefix}[{index}]")
    else:
        yield prefix, value


def find_key(value: Any, wanted: str) -> list[Any]:
    matches: list[Any] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key == wanted:
                matches.append(child)
            matches.extend(find_key(child, wanted))
    elif isinstance(value, list):
        for child in value:
            matches.extend(find_key(child, wanted))
    return matches


def load_json(path: Path, findings: list[Finding]) -> Any | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        findings.append(Finding("error", "json_parse", f"{exc}", str(path)))
        return None

    for key, scalar in walk_values(payload):
        if isinstance(scalar, float) and not math.isfinite(scalar):
            findings.append(
                Finding("error", "finite_json", f"non-finite value at {key}: {scalar}", str(path))
            )
    return payload


def normalize_robots(value: Any) -> list[str] | None:
    if not isinstance(value, list):
        return None
    return [str(item).lower() for item in value]


def check_common_config(
    payload: dict[str, Any], path: Path, findings: list[Finding], *, manifest: bool
) -> None:
    robots = normalize_robots(payload.get("robots"))
    if robots != EXPECTED_ROBOTS:
        findings.append(
            Finding("error", "robots", f"expected {EXPECTED_ROBOTS}, got {robots}", str(path))
        )

    if int(payload.get("num_robot_topologies", len(robots or []))) != 2:
        findings.append(Finding("error", "topology_count", "expected exactly two topologies", str(path)))

    backbone = payload.get("backbone")
    if manifest and backbone not in VALID_BACKBONES:
        findings.append(
            Finding("error", "backbone", f"expected URMA/URMAv2, got {backbone!r}", str(path))
        )

    reward = payload.get("reward_type")
    if reward is not None and reward != "MorphMimicReward":
        findings.append(
            Finding("error", "reward_type", f"expected MorphMimicReward, got {reward!r}", str(path))
        )

    one_hot = payload.get("robot_one_hot")
    if one_hot is True:
        findings.append(Finding("warning", "robot_one_hot", "robot one-hot is enabled", str(path)))

    reserved = payload.get("reserved_robots")
    if isinstance(reserved, list) and reserved:
        findings.append(
            Finding("warning", "reserved_robots", f"unexpected reserved slots: {reserved}", str(path))
        )

    blank = payload.get("blank_goal_observation")
    if blank is True:
        findings.append(
            Finding("error", "no_fsq_goal_contract", "goal is blanked in a no-FSQ experiment", str(path))
        )

    latent_dim = payload.get("actor_latent_dim")
    if latent_dim not in (None, 0):
        findings.append(
            Finding("error", "no_fsq_goal_contract", f"actor_latent_dim={latent_dim}", str(path))
        )

    latent_codes = payload.get("latent_codes")
    if latent_codes not in (None, "", False):
        findings.append(
            Finding("error", "no_fsq_goal_contract", f"latent_codes={latent_codes!r}", str(path))
        )

    masks = payload.get("action_mask_counts")
    if masks is not None and list(masks) != EXPECTED_MASKS:
        findings.append(
            Finding("error", "action_masks", f"expected {EXPECTED_MASKS}, got {masks}", str(path))
        )

    if manifest:
        total_timesteps = int(payload.get("total_timesteps", 0) or 0)
        if total_timesteps > 8_000_000:
            goal_type = payload.get("goal_type")
            if goal_type != "MorphGoalTrajMimicRootErr":
                findings.append(
                    Finding(
                        "error",
                        "production_goal_contract",
                        f"long run requires MorphGoalTrajMimicRootErr; got {goal_type!r}",
                        str(path),
                    )
                )
            terminal_handler = payload.get("terminal_handler", payload.get("terminal_state_type"))
            if terminal_handler != "MorphologyAwareRootPoseTrajTerminalStateHandler":
                findings.append(
                    Finding(
                        "error",
                        "production_terminal_contract",
                        "long run does not record MorphologyAwareRootPoseTrajTerminalStateHandler",
                        str(path),
                    )
                )
            root_dev = payload.get("max_root_pos_deviation", payload.get("max_root_deviation"))
            if root_dev is None:
                findings.append(
                    Finding(
                        "error",
                        "production_terminal_contract",
                        "long run does not record a maximum root-deviation threshold",
                        str(path),
                    )
                )


def per_robot_mapping(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get("per_robot")
    if isinstance(value, dict):
        return value
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="default: <experiment>/metrics/pipeline_audit.json",
    )
    parser.add_argument(
        "--strict-learning",
        action="store_true",
        help="return nonzero when every trained family does not beat zero action",
    )
    args = parser.parse_args()

    root = args.experiment.resolve()
    output = args.output or root / "metrics" / "pipeline_audit.json"
    findings: list[Finding] = []

    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 2

    json_paths = sorted(root.rglob("*.json"))
    if not json_paths:
        findings.append(Finding("error", "evidence", "no JSON evidence found", str(root)))

    payloads: dict[Path, Any] = {}
    for path in json_paths:
        payload = load_json(path, findings)
        if payload is not None:
            payloads[path] = payload

    preflights = [(p, v) for p, v in payloads.items() if p.name == "preflight.json"]
    manifests = [(p, v) for p, v in payloads.items() if p.name == "manifest.json"]

    if not preflights:
        findings.append(Finding("error", "preflight", "no preflight.json found", str(root)))
    for path, payload in preflights:
        if isinstance(payload, dict):
            check_common_config(payload, path, findings, manifest=False)
            masks = payload.get("action_mask_counts")
            if masks is None:
                nested = find_key(payload, "action_mask_counts")
                masks = nested[0] if nested else None
            if masks is not None and list(masks) != EXPECTED_MASKS:
                findings.append(
                    Finding("error", "action_masks", f"expected {EXPECTED_MASKS}, got {masks}", str(path))
                )
        else:
            findings.append(Finding("error", "preflight", "preflight root is not an object", str(path)))

    if not manifests:
        findings.append(Finding("error", "training_manifest", "no training manifest.json found", str(root)))

    production: tuple[Path, dict[str, Any]] | None = None
    for path, payload in manifests:
        if not isinstance(payload, dict):
            findings.append(Finding("error", "training_manifest", "manifest root is not an object", str(path)))
            continue
        check_common_config(payload, path, findings, manifest=True)
        steps = int(payload.get("total_timesteps", 0) or 0)
        if production is None or steps > int(production[1].get("total_timesteps", 0) or 0):
            production = (path, payload)

    fk_files = [
        (path, payload)
        for path, payload in payloads.items()
        if "fk_verification" in path.name.lower()
    ]
    if not fk_files:
        findings.append(
            Finding("error", "fk_verification", "fk_verification.json is missing", str(root))
        )
    for path, payload in fk_files:
        for key, scalar in walk_values(payload):
            key_l = key.lower()
            if (
                isinstance(scalar, (int, float))
                and ("error" in key_l or "difference" in key_l or "mismatch" in key_l)
                and ("jax" in key_l or "cpu" in key_l or "independent" in key_l)
                and float(scalar) > 1e-5
            ):
                findings.append(
                    Finding(
                        "error",
                        "fk_verification",
                        f"{key}={scalar} exceeds 1e-5",
                        str(path),
                    )
                )
        for passed in find_key(payload, "passed"):
            if passed is False:
                findings.append(Finding("error", "fk_verification", "reported passed=false", str(path)))

    policy_evals = [
        (p, v) for p, v in payloads.items() if "policy_vs_zero" in p.name.lower()
    ]
    if not policy_evals:
        findings.append(Finding("warning", "policy_vs_zero", "no policy-vs-zero evaluation found", str(root)))
    for path, payload in policy_evals:
        per_robot = per_robot_mapping(payload)
        if per_robot is None:
            findings.append(Finding("error", "policy_vs_zero", "missing per_robot object", str(path)))
            continue
        missing = [robot for robot in EXPECTED_ROBOTS if robot not in per_robot]
        if missing:
            findings.append(Finding("error", "policy_vs_zero", f"missing robots: {missing}", str(path)))
        verdicts = find_key(payload, "every_robot_beats_zero_action")
        if verdicts and not all(bool(v) for v in verdicts):
            level = "error" if args.strict_learning else "warning"
            findings.append(
                Finding(level, "learning_gate", "not every robot beats exact-reset zero action", str(path))
            )

    tracking_evals = [
        (p, v) for p, v in payloads.items() if "fk_tracking" in p.name.lower()
    ]
    if not tracking_evals:
        findings.append(Finding("warning", "fk_tracking", "no reward-independent FK tracking file found", str(root)))
    for path, payload in tracking_evals:
        per_robot = per_robot_mapping(payload)
        if per_robot is None:
            findings.append(Finding("error", "fk_tracking", "missing per_robot object", str(path)))
            continue
        missing = [robot for robot in EXPECTED_ROBOTS if robot not in per_robot]
        if missing:
            findings.append(Finding("error", "fk_tracking", f"missing robots: {missing}", str(path)))

    contact_evals = [
        (p, v) for p, v in payloads.items() if p.name.lower().startswith("contact_")
    ]
    if not contact_evals:
        findings.append(Finding("warning", "contact", "no contact-quality evaluation found", str(root)))

    media = sorted(
        p for p in (root / "media").glob("*")
        if p.suffix.lower() in {".mp4", ".png", ".webm", ".gif"}
    ) if (root / "media").is_dir() else []
    if not media:
        findings.append(Finding("warning", "media", "no videos or overlay images found", str(root / "media")))

    # The current continuous cross-family sampler does not reject invalid draws.
    rejection_keys = []
    for path, payload in payloads.items():
        for key, scalar in walk_values(payload):
            if any(term in key.lower() for term in ("rejected_bodies", "skipped_bodies", "reject_reasons")):
                rejection_keys.append((path, key, scalar))
    if not rejection_keys:
        findings.append(
            Finding(
                "unsupported",
                "body_rejection_accounting",
                "no rejected/skipped-body counters exist; zero skipped must not be inferred",
                str(root),
            )
        )

    if production is not None:
        path, payload = production
        curves = payload.get("length_curve")
        if isinstance(curves, list) and len(curves) >= 2:
            first = next((float(v) for v in curves if isinstance(v, (int, float))), None)
            last = next((float(v) for v in reversed(curves) if isinstance(v, (int, float))), None)
            if first is not None and last is not None and last <= first:
                findings.append(
                    Finding(
                        "warning",
                        "training_curve",
                        f"episode length did not improve ({first:.3g} -> {last:.3g})",
                        str(path),
                    )
                )

    counts = {
        level: sum(f.level == level for f in findings)
        for level in ("error", "warning", "unsupported")
    }
    report = {
        "experiment": str(root),
        "json_files_examined": len(payloads),
        "preflight_files": [str(p) for p, _ in preflights],
        "training_manifests": [str(p) for p, _ in manifests],
        "production_manifest": str(production[0]) if production else None,
        "media_files": [str(p) for p in media],
        "counts": counts,
        "findings": [asdict(f) for f in findings],
        "systems_pass": counts["error"] == 0,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")

    print(f"[audit] experiment: {root}")
    print(f"[audit] JSON files: {len(payloads)}")
    print(f"[audit] errors={counts['error']} warnings={counts['warning']} unsupported={counts['unsupported']}")
    for finding in findings:
        where = f" [{finding.path}]" if finding.path else ""
        print(f"[{finding.level.upper()}] {finding.check}: {finding.message}{where}")
    print(f"[audit] wrote {output}")

    return 0 if counts["error"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
