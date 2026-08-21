"""Contact-quality metrics + target-vs-achieved overlays for one checkpoint.

From one logged rollout per checkpoint (same machinery as eval_fk_tracking):
  - foot penetration: world foot-site z below the floor, on the SAMPLED body
  - foot slip: XY drift of a foot site while it is in contact (z < 3 cm)
  - jerk: mean |third difference| of actuated joint positions
  - command-switch is covered by the renderer's switch_metrics.json
  - overlay PNGs: reference vs achieved foot/hand site heights, env 0/family

All kinematics on CPU MuJoCo with the verified numpy morphology mutation.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
SCRIPTS = WORKSPACE / "scripts"
for p in (str(SCRIPTS), str(SCRIPTS / "h1md")):
    if p not in sys.path:
        sys.path.insert(0, p)

import mujoco
import numpy as np

from scaling.evaluate_cross_humanoid_policy import (
    _env_args,
    _find_manifest,
    _resolve_checkpoint,
)
from scaling.eval_fk_tracking import rollout_logged
from scaling.family_morphology import FAMILY_BODIES
from scaling.parallel_cross_humanoid_train import (
    build_cross_humanoid_env,
    trainer_for,
)

FOOT_SITES = ("left_foot_mimic", "right_foot_mimic")
HAND_SITES = ("left_hand_mimic", "right_hand_mimic")


def analyze_group(env, gi, robot, log, out_png: Path, tag: str):
    raw = env._raw_envs[gi]
    base_model = raw._model
    bodies = FAMILY_BODIES[robot]
    leg_ids = [
        mujoco.mj_name2id(base_model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in (*bodies.knee_bodies, *bodies.ankle_bodies)
    ]
    base_leg_z = base_model.body_pos[leg_ids, 2].copy()
    model = copy.deepcopy(base_model)
    data = mujoco.MjData(model)
    foot_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, s) for s in FOOT_SITES
    ]
    hand_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, s) for s in HAND_SITES
    ]
    traj = raw.th.traj.data
    nq_root = 7

    def world_sites(qpos):
        data.qpos[:] = qpos
        mujoco.mj_kinematics(model, data)
        return data.site_xpos[foot_ids].copy(), data.site_xpos[hand_ids].copy()

    steps, n_envs = log["qpos"].shape[0], log["qpos"].shape[1]
    pen, slip, jerks = [], [], []
    overlay = {"t": [], "foot_z_ref": [], "foot_z_act": [], "hand_z_ref": [], "hand_z_act": []}
    for e in range(n_envs):
        fell = log["absorbing"][:, e]
        alive_until = int(fell.argmax()) if fell.any() else steps
        morph = log["morph"][0, e]
        model.body_pos[leg_ids, 2] = base_leg_z * float(morph[0])
        prev_xy = None
        qser = log["qpos"][:alive_until, e, nq_root:]
        if qser.shape[0] > 3:
            jerks.append(float(np.mean(np.abs(np.diff(qser, n=3, axis=0)))))
        for t in range(0, alive_until):
            feet, hands = world_sites(log["qpos"][t, e])
            z = feet[:, 2]
            pen.append(float(np.maximum(0.0, -z).max()))
            in_contact = z < 0.03
            if prev_xy is not None and in_contact.any():
                d = np.linalg.norm(feet[in_contact, :2] - prev_xy[in_contact, :2], axis=-1)
                slip.append(float(d.max()))
            prev_xy = feet
            if e == 0:
                ref = traj.get(int(log["traj_no"][t, e]), int(log["step_no"][t, e]), np)
                rfeet, rhands = world_sites(np.asarray(ref.qpos))
                overlay["t"].append(t)
                overlay["foot_z_act"].append(float(z.min()))
                overlay["foot_z_ref"].append(float(rfeet[:, 2].min()))
                overlay["hand_z_act"].append(float(hands[:, 2].mean()))
                overlay["hand_z_ref"].append(float(rhands[:, 2].mean()))

    metrics = {
        "penetration_mean_m": float(np.mean(pen)) if pen else None,
        "penetration_p99_m": float(np.percentile(pen, 99)) if pen else None,
        "slip_mean_m_per_step": float(np.mean(slip)) if slip else None,
        "slip_p95_m_per_step": float(np.percentile(slip, 95)) if slip else None,
        "jerk_mean_rad_per_step3": float(np.mean(jerks)) if jerks else None,
        "n_pen_samples": len(pen),
    }

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    t = overlay["t"]
    for ax, key, label in (
        (axes[0], "foot_z", "min foot-site height (m)"),
        (axes[1], "hand_z", "mean hand-site height (m)"),
    ):
        ax.plot(t, overlay[f"{key}_ref"], color="#8a8f98", lw=2, label="reference (FK, sampled body)")
        ax.plot(t, overlay[f"{key}_act"], color="#2f6fdd", lw=2, label="achieved")
        ax.set_ylabel(label, fontsize=9)
        ax.grid(alpha=0.25, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].set_xlabel("control step", fontsize=9)
    fig.suptitle(f"{tag} · {robot} · target vs achieved (env 0)", fontsize=11)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--envs-per-robot", type=int, default=4)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--robots", nargs="*", default=None)
    parser.add_argument(
        "--reference-root", type=Path,
        default=WORKSPACE / "external_data" / "cross_humanoid",
    )
    parser.add_argument("--use-mjwarp", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--media-dir", type=Path,
        default=WORKSPACE / "experiments" / "urma_fk_targets_20260815" / "media",
    )
    args = parser.parse_args()

    checkpoint = _resolve_checkpoint(args.checkpoint)
    _, manifest = _find_manifest(checkpoint)
    trainer = trainer_for(str(manifest.get("backbone", "masked_mlp")))
    env_args = _env_args(args, manifest)
    agent_conf, agent_state = trainer.load_agent(checkpoint)
    env, _ = build_cross_humanoid_env(env_args)
    buffer = getattr(agent_conf, "actor_latent_buffer", None)
    logs = rollout_logged(env, agent_conf, agent_state, args.steps, args.seed, buffer)

    result = {"checkpoint": str(checkpoint), "tag": args.tag, "per_robot": {}}
    for gi, group in enumerate(env.groups):
        png = args.media_dir / f"overlay_{args.tag}_{group.name}.png"
        result["per_robot"][group.name] = analyze_group(
            env, gi, group.name, logs[gi], png, args.tag
        )
        print(f"[contact] {group.name}: {result['per_robot'][group.name]}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[contact] -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
