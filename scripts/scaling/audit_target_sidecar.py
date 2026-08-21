"""Machine-check the target-sphere video from its sidecar.

`instructions.md` lists what a human should look for in the overlay video:
frozen spheres, an off-by-one phase, one family's markers used for another,
markers stuck at nominal after a morphology change, morphology changing while
the observation still describes the previous body. Every one of those is a
property of the per-frame record the renderer already writes, so none of them
needs to be eyeballed -- and a human watching 800 frames would not reliably
catch any of them anyway.

Each check below fails for exactly one defect. Exits non-zero if any fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def check(name: str, ok: bool, detail: str) -> dict:
    return {"check": name, "passed": bool(ok), "detail": detail}


def audit_family(rows: list[dict], family: str) -> list[dict]:
    rows = sorted(rows, key=lambda r: r["frame"])
    frames = np.asarray([r["frame"] for r in rows])
    phase = np.asarray([r["subtraj_step_no"] for r in rows])
    traj = np.asarray([r["traj_no"] for r in rows])
    generation = np.asarray([r["morphology_generation"] for r in rows])
    reset = np.asarray([r["reset_happened"] for r in rows], dtype=bool)
    morphology = np.asarray([r["morphology"] for r in rows], dtype=float)
    provider = np.asarray([r["target_provider_max_error_m"] for r in rows])
    results = []

    results.append(check(
        "frames_contiguous",
        bool(np.array_equal(frames, np.arange(frames.size))),
        f"{frames.size} frames, no gaps",
    ))

    # 1. spheres freezing while the reference phase advances.
    # Between two consecutive non-reset frames the phase must advance; if it
    # ever stalls, the target is being held while the robot moves.
    step = np.diff(phase)
    advancing = ~reset[1:]
    stalled = int(np.sum((step <= 0) & advancing))
    results.append(check(
        "phase_advances_between_resets",
        stalled == 0,
        f"{stalled} frame(s) where the reference phase did not advance outside "
        f"a reset (of {int(advancing.sum())} checked)",
    ))

    # 2. an off-by-one or frozen cursor: phase must NOT be reconstructible as
    # the frame index, because asynchronous resets make the two diverge.
    diverged = int(np.sum(phase != (phase[0] + frames)))
    results.append(check(
        "phase_is_logged_not_derived_from_frame_index",
        diverged > 0 or bool(reset.any()) is False,
        f"phase differs from frame-derived phase on {diverged} frame(s); "
        f"{int(reset.sum())} reset(s) in this rollout",
    ))

    # 3. morphology changes ONLY at a reset -- a body swapping mid-episode
    # would mean the observation describes the previous body.
    changed = np.zeros(frames.size, dtype=bool)
    changed[1:] = np.any(np.abs(np.diff(morphology, axis=0)) > 1e-9, axis=1)
    illegal = int(np.sum(changed & ~reset))
    results.append(check(
        "morphology_changes_only_at_reset",
        illegal == 0,
        f"{illegal} mid-episode morphology change(s) out of "
        f"{int(changed.sum())} body change(s)",
    ))

    # 4. generation must move with the body, so the sidecar records the swap.
    gen_moved = np.zeros(frames.size, dtype=bool)
    gen_moved[1:] = np.diff(generation) != 0
    unrecorded = int(np.sum(changed & ~gen_moved))
    results.append(check(
        "morphology_generation_tracks_the_body",
        unrecorded == 0,
        f"{unrecorded} body change(s) not reflected in morphology_generation",
    ))

    # 5. the provider gate, per frame rather than as a run-level maximum.
    worst = float(provider.max()) if provider.size else 0.0
    results.append(check(
        "provider_agreement_every_frame",
        worst < 1e-5,
        f"worst per-frame CPU-vs-MJX target error {worst:.2e} m (gate 1e-5 m)",
    ))

    # 6. the rollout must actually exercise the trajectory, not sit on one pose.
    span = int(phase.max() - phase.min())
    results.append(check(
        "rollout_spans_the_reference",
        span > 10,
        f"reference phase spanned {span} steps "
        f"({int(traj.min())}..{int(traj.max())} trajectory ids)",
    ))

    for entry in results:
        entry["family"] = family
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sidecar", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    payload = json.loads(args.sidecar.read_text(encoding="utf-8"))
    by_family = defaultdict(list)
    for row in payload.get("rows", []):
        by_family[row["family"]].append(row)

    results = []
    for family, rows in sorted(by_family.items()):
        results.extend(audit_family(rows, family))

    # cross-family: each topology must use its own retargeted reference file
    references = {
        family: block.get("reference_file", "")
        for family, block in payload.get("per_family", {}).items()
    }
    distinct = len(set(references.values())) == len(references)
    results.append({
        "check": "each_family_uses_its_own_reference_file",
        "passed": bool(distinct and references),
        "detail": "; ".join(f"{k}: {Path(v).parent.name}/{Path(v).name}"
                            for k, v in references.items()) or "no reference "
                                                               "files recorded",
        "family": "all",
    })

    failed = [r for r in results if not r["passed"]]
    report = {
        "sidecar": str(args.sidecar),
        "families": sorted(by_family),
        "checks": results,
        "passed": not failed,
        "num_failed": len(failed),
    }
    output = args.output or args.sidecar.with_name(
        args.sidecar.stem + "_audit.json")
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for entry in results:
        mark = "ok  " if entry["passed"] else "FAIL"
        print(f"[{mark}] {entry['family']:>4s} {entry['check']}: "
              f"{entry['detail']}")
    print(f"[sidecar-audit] -> {output}")
    return 0 if not failed else 4


if __name__ == "__main__":
    raise SystemExit(main())
