"""C1 -- freeze and hash the H1 dance2_subject4 motion window.

Phase 0 of H1_MORPHOLOGY_DEEPMIMIC_GOAL.md: confirm the clip identity, resolve
which frame indexing the historical 19482:20282 window actually refers to, and
emit a manifest with content hashes so every later arm can assert it used the
same reference.

There are TWO different frame rates for the same clip in this repository:

  * the raw LAFAN1 cache npz  (external_data/amass_converted/LAFAN1/UnitreeH1/)
  * ``env.th.traj``           (the TrajectoryHandler resamples to the env's
                               control frequency)

They are the same motion at different sampling rates, and the cache file is
keyed by (env family, clip) only -- not by frequency -- so a CPU env and an Mjx
env silently share one file. The frozen window is therefore defined in SECONDS
and the frame indices at each rate are derived from that.

Run under WSL dance_env (LOCOMUJOCO_VARIABLES paths are posix).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import yaml

import loco_mujoco
from loco_mujoco.trajectory import Trajectory

# Historical window as used by external_data/cross_humanoid/.../manifest_start19482_800f_*.json,
# expressed at the 100 Hz env.th.traj rate of the CPU UnitreeH1 env.
HIST_START_100HZ, HIST_FRAMES_100HZ = 19482, 800


def sha256_arrays(**arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for name in sorted(arrays):
        a = np.ascontiguousarray(np.asarray(arrays[name], dtype=np.float64))
        h.update(name.encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def cache_path(env_name: str, clip: str) -> Path:
    variables = yaml.safe_load(open(loco_mujoco.PATH_TO_VARIABLES))
    return (
        Path(variables["LOCOMUJOCO_CONVERTED_LAFAN1_PATH"])
        / env_name.replace("Mjx", "")
        / f"{clip}.npz"
    )


def describe(qpos: np.ndarray, qvel: np.ndarray) -> dict:
    speed = np.linalg.norm(qvel[:, 6:], axis=1)
    return {
        "n_frames": int(len(qpos)),
        "root_z_min": float(qpos[:, 2].min()),
        "root_z_max": float(qpos[:, 2].max()),
        "root_xy_travel_m": float(np.linalg.norm(qpos[-1, :2] - qpos[0, :2])),
        "joint_speed_mean": float(speed.mean()),
        "joint_speed_max": float(speed.max()),
        "nonfinite": int((~np.isfinite(qpos)).sum() + (~np.isfinite(qvel)).sum()),
        "loop_discontinuity_joint_l2": float(np.linalg.norm(qpos[-1, 7:] - qpos[0, 7:])),
        "hash_qpos_qvel": sha256_arrays(qpos=qpos, qvel=qvel),
    }


def energy_window_start(qvel: np.ndarray, window: int) -> int:
    energy = np.linalg.norm(qvel[:, 6:], axis=1)
    cum = np.concatenate(([0.0], np.cumsum(energy)))
    return int(np.argmax(cum[window:] - cum[:-window]))


def git_rev(path: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception as exc:  # provenance is best effort
        return f"unavailable: {exc}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clip", default="dance2_subject4")
    ap.add_argument("--cpu-env", default="UnitreeH1")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--skip-env", action="store_true",
                    help="Skip building the CPU env (raw-cache facts only).")
    args = ap.parse_args()

    import jax
    import mujoco

    src = cache_path(args.cpu_env, args.clip)
    raw = Trajectory.load(str(src))
    raw_freq = float(raw.info.frequency)
    raw_qpos, raw_qvel = np.asarray(raw.data.qpos), np.asarray(raw.data.qvel)

    manifest = {
        "component": "C1_motion_identity",
        "clip": args.clip,
        "raw_cache": {
            "path": str(src),
            "bytes": src.stat().st_size,
            "frequency_hz": raw_freq,
            "n_samples": int(raw.data.n_samples),
            "duration_s": int(raw.data.n_samples) / raw_freq,
            "qpos_dim": int(raw_qpos.shape[1]),
            "qvel_dim": int(raw_qvel.shape[1]),
            "n_joint_names": len(raw.info.joint_names),
            "hash_full": sha256_arrays(qpos=raw_qpos, qvel=raw_qvel),
            "note": (
                "cache key is (env family, clip) only -- CPU and Mjx envs share this file, "
                "so its frequency reflects whichever env wrote it last"
            ),
        },
    }

    if not args.skip_env:
        from loco_mujoco.task_factories import ImitationFactory, LAFAN1DatasetConf

        env = ImitationFactory.make(
            args.cpu_env, lafan1_dataset_conf=LAFAN1DatasetConf([args.clip])
        )
        th = env.th.traj
        env_freq = float(th.info.frequency)
        env_qpos, env_qvel = np.asarray(th.data.qpos), np.asarray(th.data.qvel)

        start = HIST_START_100HZ
        frames = HIST_FRAMES_100HZ
        in_range = start + frames <= len(env_qpos)
        window = {
            "definition": "canonical window, defined in seconds and derived per rate",
            "start_s": start / env_freq,
            "duration_s": frames / env_freq,
            "at_env_rate": {
                "frequency_hz": env_freq,
                "start_frame": start,
                "end_frame": start + frames,
                "n_frames": frames,
                "in_range": bool(in_range),
            },
            "at_raw_cache_rate": {
                "frequency_hz": raw_freq,
                "start_frame": int(round(start / env_freq * raw_freq)),
                "n_frames": int(round(frames / env_freq * raw_freq)),
                "in_range": bool(
                    round(start / env_freq * raw_freq) + round(frames / env_freq * raw_freq)
                    <= len(raw_qpos)
                ),
            },
        }
        manifest["env_traj"] = {
            "env": args.cpu_env,
            "frequency_hz": env_freq,
            "n_samples": int(th.data.n_samples),
            "duration_s": int(th.data.n_samples) / env_freq,
            "resample_ratio_vs_raw": env_freq / raw_freq,
            "hash_full": sha256_arrays(qpos=env_qpos, qvel=env_qvel),
        }
        manifest["window"] = window
        if in_range:
            manifest["window"]["stats_at_env_rate"] = describe(
                env_qpos[start:start + frames], env_qvel[start:start + frames]
            )
            manifest["window"]["hash_window"] = sha256_arrays(
                qpos=env_qpos[start:start + frames], qvel=env_qvel[start:start + frames]
            )
            auto = energy_window_start(env_qvel, frames)
            manifest["window"]["auto_energy_window_start"] = auto
            manifest["window"]["historical_equals_auto"] = bool(auto == start)

        rs = window["at_raw_cache_rate"]
        if rs["in_range"]:
            s, n = rs["start_frame"], rs["n_frames"]
            manifest["window"]["stats_at_raw_rate"] = describe(
                raw_qpos[s:s + n], raw_qvel[s:s + n]
            )

    manifest["env"] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "jax": jax.__version__,
        "jax_backend": jax.default_backend(),
        "mujoco": mujoco.__version__,
        "loco_mujoco_path": str(Path(loco_mujoco.__file__).parent),
        "loco_mujoco_rev": git_rev(Path(loco_mujoco.__file__).parents[1]),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
