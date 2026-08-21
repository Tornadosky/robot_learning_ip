"""Crop + resample a loco_mujoco Trajectory npz between rates/windows (P0.1).

Both pipelines (loco-mujoco trainer and loco_mjx/urma2) read the same
Trajectory npz schema on the same MuJoCo models, so a two-way reference swap
is only a window/rate alignment:

  X1 direction: urma2's 40 Hz clip -> our 8 s window at 100 Hz
  X2 direction: our 100 Hz window  -> a 40 Hz clip for urma2's loader

qpos is linearly interpolated (root quaternion slerped), qvel is recomputed by
finite differences at the OUTPUT rate, and every derived field (xpos, xquat,
cvel, subtree_com, site_xpos, site_xmat) is rebuilt by forward kinematics --
the same discipline as make_control_references.py, whose helpers this reuses.

The --template file supplies Trajectory.info (joint names, frequency): pick a
file that already lives at the OUTPUT rate.
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

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from loco_mujoco.trajectory import Trajectory  # noqa: E402

from scaling.make_control_references import (  # noqa: E402
    _finite_difference_qvel,
    _free_joint_qpos_layout,
    _recompute,
    _slerp,
)
from scaling.parallel_cross_humanoid_train import HUMANOIDS  # noqa: E402


def resample(model, src_qpos: np.ndarray, src_freq: float,
             start_frame: float, out_frames: int, out_freq: float):
    """Sample src_qpos at out_freq starting at start_frame (source frames)."""
    n_src = src_qpos.shape[0]
    u = start_frame + np.arange(out_frames) * (src_freq / out_freq)
    if u[-1] > n_src - 1 + 1e-9:
        raise ValueError(
            f"window [{u[0]:.1f}, {u[-1]:.1f}] exceeds source length {n_src}")
    u = np.clip(u, 0.0, n_src - 1)
    i0 = np.floor(u).astype(int)
    i1 = np.minimum(i0 + 1, n_src - 1)
    t = u - i0
    qpos = (1.0 - t)[:, None] * src_qpos[i0] + t[:, None] * src_qpos[i1]
    pos_s, quat_s = _free_joint_qpos_layout(model)
    qpos[:, quat_s] = _slerp(src_qpos[i0][:, quat_s], src_qpos[i1][:, quat_s], t)
    qvel = _finite_difference_qvel(model, qpos, out_freq)
    return qpos, qvel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True,
                        help="Trajectory npz already at the output rate; "
                             "supplies info/schema")
    parser.add_argument("--robot", choices=list(HUMANOIDS), required=True)
    parser.add_argument("--src-start-frame", type=float, default=0.0)
    parser.add_argument("--out-frames", type=int, required=True)
    parser.add_argument("--out-frequency", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare-with", type=Path, default=None,
                        help="optional npz on the same grid: report qpos/site "
                             "diffs after alignment (W0.4)")
    args = parser.parse_args()

    src = Trajectory.load(str(args.source))
    template = Trajectory.load(str(args.template))
    src_freq = float(np.asarray(src.info.frequency))
    tmpl_freq = float(np.asarray(template.info.frequency))
    if abs(tmpl_freq - args.out_frequency) > 1e-6:
        raise SystemExit(
            f"template frequency {tmpl_freq} != requested {args.out_frequency}; "
            "pick a template at the output rate")

    model = mujoco.MjModel.from_xml_path(str(HUMANOIDS[args.robot].xml_path))
    src_qpos = np.asarray(src.data.qpos, dtype=np.float64)
    if src_qpos.shape[1] != model.nq:
        raise SystemExit(f"source nq {src_qpos.shape[1]} != model nq {model.nq}")

    qpos, qvel = resample(model, src_qpos, src_freq,
                          args.src_start_frame, args.out_frames,
                          args.out_frequency)
    traj = _recompute(model, qpos, qvel, template)
    # _recompute keeps the template's split_points; rewrite for the new length
    # (a single trajectory spanning the whole window).
    import jax.numpy as jnp
    traj = Trajectory(
        info=traj.info,
        data=traj.data.replace(
            split_points=jnp.asarray([0, args.out_frames], dtype=jnp.int32)),
        transitions=traj.transitions,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    traj.save(str(args.output))

    report = {
        "source": str(args.source), "template": str(args.template),
        "output": str(args.output), "robot": args.robot,
        "src_freq": src_freq, "out_freq": args.out_frequency,
        "src_start_frame": args.src_start_frame,
        "out_frames": args.out_frames,
        "root_z_mean": float(qpos[:, 2].mean()),
        "root_z_min": float(qpos[:, 2].min()),
        "qvel_p95": float(np.percentile(np.abs(qvel), 95)),
    }
    if args.compare_with is not None:
        other = Trajectory.load(str(args.compare_with))
        oq = np.asarray(other.data.qpos, dtype=np.float64)
        n = min(oq.shape[0], qpos.shape[0])
        dq = qpos[:n] - oq[:n]
        os_ = np.asarray(other.data.site_xpos)[:n]
        ns_ = np.asarray(traj.data.site_xpos)[:n]
        report["compare"] = {
            "frames_compared": int(n),
            "qpos_rms_per_dim_max": float(np.sqrt((dq ** 2).mean(0)).max()),
            "qpos_rms_overall": float(np.sqrt((dq ** 2).mean())),
            "root_pos_rms_m": float(np.sqrt((dq[:, :3] ** 2).mean())),
            "joint_rms_rad": float(np.sqrt((dq[:, 7:] ** 2).mean())),
            "site_rms_m": float(np.sqrt(((ns_ - os_) ** 2).sum(-1)).mean()),
            "site_max_m": float(np.sqrt(((ns_ - os_) ** 2).sum(-1)).max()),
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
