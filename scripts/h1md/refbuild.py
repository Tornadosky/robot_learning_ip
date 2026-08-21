"""Canonical per-(body, arm) reference construction.

This exists because the same defect appeared twice: an evaluation script built
ONE reference (the `fk` one) and scored every arm's policy against it. A policy
evaluated against a reference it was not trained on looks broken — the
`shared_nominal` checkpoint reported 166 steps that way and 799 when scored
against its own reference. The bug is silent, plausible-looking, and easy to
reintroduce whenever a new script needs a reference.

Every evaluator, renderer and trainer should import `build_reference` from here
rather than re-deriving it.

Arms:
  fk              nominal joint angles on this body, root re-grounded per frame
  shared_nominal  the NOMINAL body's re-grounded reference, used unchanged
  ik_scaled       Gauss-Newton IK onto root-relative site targets scaled by the
                  carrying limb, then clamped to joint limits AND re-grounded
                  (the re-grounding is not optional — see Finding 32)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE / "scripts"))
sys.path.insert(0, str(WORKSPACE / "scripts" / "h1md"))

from c6_reward_discrimination import build_model, catalog, reground  # noqa: E402

ARMS = ("fk", "shared_nominal", "ik_scaled")


def build_reference(body_name: str, arm: str, qpos: np.ndarray, xml_root: Path) -> np.ndarray:
    """Return the qpos reference for (body, arm). Raises on an unknown arm."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")

    bodies = dict(catalog())
    model = build_model(body_name, bodies[body_name], xml_root)

    if arm == "fk":
        return reground(model, qpos)[0]

    if arm == "shared_nominal":
        nominal = build_model("body00_nominal", bodies["body00_nominal"], xml_root)
        return reground(nominal, qpos)[0]

    import jax.numpy as jnp

    from c3_reference_methods import IKKernel, LIMB_SCALE_INDEX

    kern = IKKernel(xml_root / "h1_morphology_c2_body00_nominal" / "h1.xml")
    morph = bodies[body_name]
    m = jnp.asarray(np.array([[morph["leg_length_scale"], morph["arm_length_scale"],
                               morph["shoulder_width_scale"]]], dtype=np.float32))
    Q = jnp.asarray(qpos.astype(np.float32))
    nom = np.asarray(kern.fk_bt(jnp.ones((1, 3), jnp.float32), Q))[0]
    nom_targets = jnp.asarray(nom[:, kern.TARGET_SITES, :])
    root_nom = jnp.asarray(qpos[:, 0:3].astype(np.float32))
    rel = nom_targets - root_nom[:, None, :]
    s = m[0][LIMB_SCALE_INDEX][None, :, None]
    root = root_nom.at[:, 2].multiply(m[0][0])
    xs, _ = kern.ik(m, Q, (root[:, None, :] + rel * s)[None])
    xs = np.asarray(xs)[0]

    q = qpos.copy()
    q[:, kern.ACT_QADR] = np.clip(xs[:, : kern.NU], kern.jnt_low, kern.jnt_high)
    q[:, 0:3] += xs[:, kern.NU: kern.NU + 3]
    return reground(model, q)[0]
