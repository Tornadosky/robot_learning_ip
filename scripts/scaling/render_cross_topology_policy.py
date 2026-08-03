"""Render a trained cross-topology policy driving every robot it was trained on.

The existing renderers under ``scripts/`` target the static-XML morphology
pipeline and cannot load a padded cross-topology checkpoint.  This one rolls the
policy out inside the same ``ParallelMorphVecEnv`` the evaluator uses, records
each robot's ``qpos`` per step, and replays those trajectories through MuJoCo's
offscreen renderer.

Rendering is decoupled from simulation on purpose: MJX runs on the GPU and has
no renderer, so we capture state during the rollout and draw it afterwards on
the CPU model.  Run with ``MUJOCO_GL=osmesa``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

WORKSPACE = Path(__file__).resolve().parents[2]
SCRIPTS = WORKSPACE / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from scaling.evaluate_cross_humanoid_policy import (  # noqa: E402
    _env_args,
    _find_manifest,
    _resolve_checkpoint,
)
from scaling.parallel_cross_humanoid_train import (  # noqa: E402
    build_cross_humanoid_env,
    trainer_for,
)


def rollout(env, agent_conf, agent_state, steps: int, seed: int, zero_action: bool):
    """Step the padded env, returning {group_index: (steps, nq)} qpos tracks."""
    keys = jax.random.split(jax.random.PRNGKey(seed), env.num_envs)
    observation, state = jax.jit(env.reset)(keys)
    variables = {
        "params": agent_state.train_state.params,
        "run_stats": agent_state.train_state.run_stats,
    }
    step = jax.jit(env.step)

    def act(obs):
        if zero_action:
            return jnp.zeros((obs.shape[0], env.max_action_dim), dtype=obs.dtype)
        (policy, _), _ = agent_conf.network.apply(variables, obs, mutable=["run_stats"])
        return policy.mean()

    tracks = [[] for _ in env.groups]
    for _ in range(steps):
        for gi, group_state in enumerate(state.group_states):
            tracks[gi].append(np.asarray(group_state.env_state.data.qpos[0]))
        observation, _, _, _, _, state = step(state, act(observation))
    return [np.stack(t) for t in tracks]


def render_track(model, qpos, width, height, distance, elevation):
    """Replay one qpos trajectory offscreen with a camera tracking the root."""
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height, width)
    camera = mujoco.MjvCamera()
    camera.azimuth, camera.elevation, camera.distance = 135.0, elevation, distance
    frames = []
    for row in qpos:
        data.qpos[:] = row
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        camera.lookat[:] = [data.qpos[0], data.qpos[1], 0.75]
        renderer.update_scene(data, camera)
        frames.append(renderer.render().copy())
    renderer.close()
    return frames


def label(frame, text, sub):
    """Burn a caption strip into the top of a frame (no font deps: cv2 is present)."""
    import cv2

    out = frame.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (16, 22, 32), -1)
    cv2.putText(out, text, (9, 15), cv2.FONT_HERSHEY_DUPLEX, 0.42, (238, 242, 247), 1,
                cv2.LINE_AA)
    cv2.putText(out, sub, (9, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (150, 168, 188), 1,
                cv2.LINE_AA)
    return out


def main():
    args = parse_args()
    args.checkpoint = _resolve_checkpoint(args.checkpoint)
    _, manifest = _find_manifest(args.checkpoint)
    backbone = str(manifest.get("backbone", "masked_mlp"))
    env_args = _env_args(
        SimpleNamespace(
            robots=None,
            reference_root=args.reference_root,
            use_mjwarp=False,
            envs_per_robot=1,
        ),
        manifest,
    )
    agent_conf, agent_state = trainer_for(backbone).load_agent(args.checkpoint)
    env, _ = build_cross_humanoid_env(env_args)
    print(f"[render] {backbone} · {list(env.names)} · {args.steps} steps", flush=True)

    tracks = rollout(env, agent_conf, agent_state, args.steps, args.seed, args.zero_action)

    panels = []
    for gi, group in enumerate(env.groups):
        model = env.envs[gi]._model
        frames = render_track(model, tracks[gi], args.width, args.height,
                              args.distance, args.elevation)
        joints = int(env.group_action_dims[gi])
        tag = "zero action" if args.zero_action else f"{backbone} policy"
        panels.append([label(f, f"{group.name.upper()}  ·  {joints} joints", tag)
                       for f in frames])
        print(f"[render] {group.name}: {len(frames)} frames", flush=True)

    import cv2

    grid = [np.hstack([p[i] for p in panels]) for i in range(len(panels[0]))]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw = args.output.with_suffix(".raw.mp4")
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
                             (grid[0].shape[1], grid[0].shape[0]))
    for frame in grid:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"[render] wrote {raw}", flush=True)

    if args.ffmpeg:
        # A Windows ffmpeg.exe invoked from WSL cannot resolve /mnt/c paths.
        def host(path):
            text = str(path)
            if args.ffmpeg.endswith(".exe") and text.startswith("/mnt/"):
                return text[5].upper() + ":" + text[6:].replace("/", "\\")
            return text

        subprocess.run([args.ffmpeg, "-y", "-loglevel", "error", "-i", host(raw),
                        "-c:v", "libx264", "-crf", "32", "-preset", "slow", "-an",
                        "-movflags", "+faststart", host(args.output)], check=True)
        raw.unlink()
        print(f"[render] encoded {args.output} "
              f"({args.output.stat().st_size // 1024} KB)", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=240)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--height", type=int, default=400)
    p.add_argument("--distance", type=float, default=3.4)
    p.add_argument("--elevation", type=float, default=-12.0)
    p.add_argument("--seed", type=int, default=20260801)
    p.add_argument("--zero-action", action="store_true")
    p.add_argument("--ffmpeg", default=None, help="ffmpeg binary for web encoding")
    p.add_argument("--reference-root", type=Path,
                   default=WORKSPACE / "external_data" / "cross_humanoid")
    return p.parse_args()


if __name__ == "__main__":
    main()
