"""Prove that catalog bodies really are physically different MJX models.

Descriptor uniqueness is not evidence of physical difference.  This audit takes
representative and boundary entries from a catalog and checks the model arrays
that ``_apply_morphology`` is supposed to change: segment offsets, body masses,
inertias and mimic-site positions.  It also asserts the invariants that make a
body simulable - positive masses and inertias, ordered joint ranges, and finite
reset/step outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

WORKSPACE = Path(__file__).resolve().parents[2]
SCRIPTS = WORKSPACE / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from scaling.embodiment_catalog import EmbodimentCatalog  # noqa: E402
from scaling.online_h1 import register_online_h1_env  # noqa: E402

NOMINAL_XML = WORKSPACE / "generated_variants" / "h1_morphology_nominal" / "h1.xml"

# Arrays the online morphology is expected to change, and the index groups that
# must actually differ for each coordinate.
CHANGED_ARRAYS = ("body_pos", "body_ipos", "body_mass", "body_inertia", "site_pos")


def audit_descriptor(env, descriptor: np.ndarray, nominal) -> dict:
    model = env._apply_morphology(env.sys, jnp.asarray(descriptor, dtype=jnp.float32))
    result = {"descriptor": [float(v) for v in descriptor]}
    for name in CHANGED_ARRAYS:
        changed = np.asarray(getattr(model, name))
        base = np.asarray(getattr(nominal, name))
        delta = np.abs(changed - base)
        result[name] = {
            "max_abs_difference_from_nominal": float(delta.max()),
            "num_entries_differing": int(np.sum(delta > 1e-9)),
        }
    body_mass = np.asarray(model.body_mass)
    body_inertia = np.asarray(model.body_inertia)
    joint_range = np.asarray(model.jnt_range)
    result["invariants"] = {
        "all_masses_positive": bool(np.all(body_mass[1:] > 0.0)),
        "all_inertias_positive": bool(np.all(body_inertia[1:] > 0.0)),
        "joint_ranges_ordered": bool(np.all(joint_range[:, 1] >= joint_range[:, 0])),
        "total_mass_kg": float(body_mass.sum()),
        "finite_arrays": bool(
            all(
                np.isfinite(np.asarray(getattr(model, n))).all() for n in CHANGED_ARRAYS
            )
        ),
    }
    return result


def leg_length(env, descriptor) -> float:
    """Vertical span of the knee+ankle offsets, i.e. the kinematic leg length."""
    model = env._apply_morphology(env.sys, jnp.asarray(descriptor, dtype=jnp.float32))
    body_pos = np.asarray(model.body_pos)
    return float(np.abs(body_pos[env._leg_position_ids][:, 2]).sum() / 2.0)


def main() -> None:
    args = parse_args()
    catalog = EmbodimentCatalog.load(args.catalog)
    env_cls = register_online_h1_env("MjxH1CatalogAudit", args.xml)
    env = env_cls()
    nominal = env._apply_morphology(env.sys, jnp.ones(4, dtype=jnp.float32))

    normalized = catalog.normalized()
    extreme_order = np.argsort(-np.max(np.abs(normalized), axis=1))
    selected = np.unique(
        np.concatenate(
            [
                np.linspace(
                    0, len(catalog) - 1, num=min(args.num_representative, len(catalog))
                )
                .round()
                .astype(int),
                extreme_order[: min(args.num_boundary, len(catalog))],
            ]
        )
    )

    audits = []
    for index in selected:
        audit = audit_descriptor(env, catalog.descriptors[index], nominal)
        audit["body_id"] = int(catalog.body_ids[index])
        audit["catalog_index"] = int(index)
        audit["kinematic_leg_length_m"] = leg_length(env, catalog.descriptors[index])
        audits.append(audit)

    # Distinctness across the audited bodies, not merely against nominal.
    masses = np.asarray(
        [
            np.asarray(
                env._apply_morphology(
                    env.sys, jnp.asarray(catalog.descriptors[i], dtype=jnp.float32)
                ).body_mass
            )
            for i in selected
        ]
    )
    positions = np.asarray(
        [
            np.asarray(
                env._apply_morphology(
                    env.sys, jnp.asarray(catalog.descriptors[i], dtype=jnp.float32)
                ).body_pos
            )
            for i in selected
        ]
    )
    unique_masses = len(np.unique(np.round(masses, 9), axis=0))
    unique_positions = len(np.unique(np.round(positions, 9), axis=0))
    # Two bodies are distinct if they differ in *any* audited array, not in every
    # one: shoulder width changes segment offsets without changing any mass, so
    # requiring per-array uniqueness would flag a perfectly valid catalog.
    combined = np.concatenate(
        [masses.reshape(len(selected), -1), positions.reshape(len(selected), -1)],
        axis=1,
    )
    unique_combined = len(np.unique(np.round(combined, 9), axis=0))

    # Reset/step finiteness on the audited bodies.
    slots = jnp.asarray(selected % catalog.num_bodies, dtype=jnp.int32)
    catalog_env = register_online_h1_env("MjxH1CatalogAuditSim", args.xml)(
        catalog_descriptors=catalog.descriptors,
        catalog_mode="fixed_balanced",
    )
    keys = jax.random.split(jax.random.PRNGKey(args.seed), len(selected))
    state = jax.jit(jax.vmap(catalog_env.mjx_reset_with_slot, in_axes=(0, 0)))(
        keys, slots
    )
    action = jnp.zeros((len(selected), catalog_env.info.action_space.shape[0]))
    next_state = jax.jit(jax.vmap(catalog_env.mjx_step, in_axes=(0, 0)))(state, action)

    report = {
        "experiment": "catalog_physical_audit",
        "catalog": str(args.catalog),
        "catalog_hash": catalog.content_hash,
        "catalog_split": catalog.split,
        "num_bodies": catalog.num_bodies,
        "num_audited": int(len(selected)),
        "mesh_limitation": (
            "collision and visual meshes are the shared nominal H1 meshes; this "
            "audit claims kinematic and inertial diversity only"
        ),
        "unique_body_mass_vectors_among_audited": int(unique_masses),
        "unique_body_pos_arrays_among_audited": int(unique_positions),
        "unique_combined_mass_and_pos_among_audited": int(unique_combined),
        "all_audited_bodies_distinct": bool(unique_combined == len(selected)),
        "distinctness_definition": (
            "bodies differ in at least one audited model array; per-array counts "
            "may be lower because e.g. shoulder width changes offsets but no mass"
        ),
        "kinematic_leg_length_min_m": float(
            min(a["kinematic_leg_length_m"] for a in audits)
        ),
        "kinematic_leg_length_max_m": float(
            max(a["kinematic_leg_length_m"] for a in audits)
        ),
        "total_mass_min_kg": float(
            min(a["invariants"]["total_mass_kg"] for a in audits)
        ),
        "total_mass_max_kg": float(
            max(a["invariants"]["total_mass_kg"] for a in audits)
        ),
        "all_invariants_hold": bool(
            all(
                a["invariants"]["all_masses_positive"]
                and a["invariants"]["all_inertias_positive"]
                and a["invariants"]["joint_ranges_ordered"]
                and a["invariants"]["finite_arrays"]
                for a in audits
            )
        ),
        "reset_observations_finite": bool(
            np.isfinite(np.asarray(state.observation)).all()
        ),
        "step_observations_finite": bool(
            np.isfinite(np.asarray(next_state.observation)).all()
        ),
        "step_rewards_finite": bool(np.isfinite(np.asarray(next_state.reward)).all()),
        "per_body": audits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    summary = {k: v for k, v in report.items() if k != "per_body"}
    print(json.dumps(summary, indent=2), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--xml", type=Path, default=NOMINAL_XML)
    parser.add_argument("--num-representative", type=int, default=24)
    parser.add_argument("--num-boundary", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
