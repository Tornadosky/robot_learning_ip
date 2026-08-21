"""C35 -- one policy, several randomized bodies, side by side, in one clip.

The continuous-morphology policies live in the online-morphology MJX env, where
each of the 512 environments carries its *own* body as a set of dynamic model
arrays. That is why they never appeared in the earlier videos: the CPU renderer
only knows how to draw a body that exists as an XML.

This bridges the two. It rolls the policy out in the MJX env, reads each
environment's morphology descriptor out of the carry, rebuilds that exact body as
a CPU `MjModel` through the same four-scalar generator, and renders the panels
into one tiled clip.

The target spheres are `traj.data.site_xpos` — the shared reference the online
trainer actually scores every body against, so the same spheres appear in every
panel. That is not a simplification for display; it is the `shared_nominal`
construction the multi-body run trains on.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE / "scripts"))
sys.path.insert(0, str(WORKSPACE / "scripts" / "scaling"))
sys.path.insert(0, str(WORKSPACE / "scripts" / "h1md"))

from c9_shared_policy import MIMIC_SITES  # noqa: E402
from c15_render_targets import SITE_COLOURS, SPHERE_R  # noqa: E402
from h1_morphology_variants import H1MorphologyPreset, create_h1_variant_xml  # noqa: E402
from online_h1_train import build_online_env  # noqa: E402

DIMS = ("leg_length_scale", "arm_length_scale", "shoulder_width_scale", "torso_mass_scale")


def body_from_descriptor(desc: np.ndarray, tag: str, root: Path):
    preset = H1MorphologyPreset(
        name=f"c35_{tag}", label=tag,
        leg_length_scale=float(desc[0]), arm_length_scale=float(desc[1]),
        shoulder_width_scale=float(desc[2]), torso_mass_scale=float(desc[3]),
    )
    xml = create_h1_variant_xml(preset, output_root=root)
    # add the target spheres to this body's own XML (relative asset paths, so it
    # must be written alongside)
    spec = mujoco.MjSpec.from_file(str(xml))
    for name in MIMIC_SITES:
        b = spec.worldbody.add_body()
        b.name = f"target_{name}"
        b.mocap = True
        b.pos = [0.0, 0.0, -5.0]
        g = b.add_geom()
        g.name = f"targetgeom_{name}"
        g.type = mujoco.mjtGeom.mjGEOM_SPHERE
        g.size = [SPHERE_R, 0.0, 0.0]
        g.rgba = list(SITE_COLOURS[name])
        g.contype = 0
        g.conaffinity = 0
        g.group = 2
        g.mass = 0.0
    out = xml.parent / "h1_with_targets.xml"
    out.write_text(spec.to_xml(), encoding="utf-8")
    return mujoco.MjModel.from_xml_path(str(out))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--panels", type=int, default=6)
    ap.add_argument("--horizon", type=int, default=400)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--pw", type=int, default=300)
    ap.add_argument("--ph", type=int, default=250)
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    from loco_mujoco.algorithms import PPOJax
    from loco_mujoco.core.wrappers import LogWrapper, VecEnv
    from PIL import Image, ImageDraw

    manifest = json.loads((args.checkpoint.parent.parent / "manifest.json").read_text(encoding="utf-8"))
    freq = float(manifest["frequency_hz"])
    env_args = SimpleNamespace(
        clip=str(manifest["clip"]),
        duration=float(manifest["window_frames"]) / freq,
        start_frame=int(manifest["window_start_frame"]),
        run_tag="c35_render", use_mjwarp=False,
        backbone=str(manifest.get("backbone", "mlp")),
        resample_per_episode=False,
        morph_low=list(manifest["morphology_low"]),
        morph_high=list(manifest["morphology_high"]),
        catalog=None, catalog_mode=str(manifest.get("catalog_mode", "continuous")),
        catalog_stride=1, keep_morph_bounds=True,
        reward_weights=str(manifest.get("reward_weights_preset", "dance")),
        terminal_handler=manifest.get("terminal_handler"),
        goal_type=str(manifest.get("goal_type", "GoalTrajMimic")),
        max_root_deviation=None,      # measure freely; do not cut the rollout short
        # Pin every panel to frame 0 so all bodies dance the same 8 s in step.
        # With the default random RSI each panel starts at a different phase, the
        # clip ends at a different step, and that difference reads as some bodies
        # failing early when in fact none of them fell.
        th_params=dict(random_start=False, fixed_start_conf=(0, 0)),
    )
    env, _meta = build_online_env(env_args)
    traj = env.th.traj

    agent_conf, agent_state = PPOJax.load_agent(args.checkpoint)
    ts = agent_state.train_state
    if agent_conf.config.experiment.n_seeds > 1:
        ts = jax.tree.map(lambda x: x[0], ts)
    variables = {"params": ts.params, "run_stats": ts.run_stats}

    def act(obs):
        (pi, _), _ = agent_conf.network.apply(variables, obs, mutable=["run_stats"])
        return pi.mean()

    wrapped = VecEnv(LogWrapper(env))
    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.panels)
    obs, state = jax.jit(wrapped.reset)(keys)
    morphs = np.asarray(state.env_state.additional_carry.morphology)

    def step(carry, _):
        obs, state, alive = carry
        a = act(obs)
        obs2, _, _, done, _, state2 = wrapped.step(state, a)
        alive = alive & ~done.astype(jnp.bool_)
        return (obs2, state2, alive), (state2.env_state.data.qpos,
                                       state2.env_state.additional_carry.traj_state.subtraj_step_no,
                                       alive)

    init = (obs, state, jnp.ones((args.panels,), dtype=jnp.bool_))
    _, (qpos, phase, alive) = jax.jit(lambda c: jax.lax.scan(step, c, None, args.horizon))(init)
    qpos, phase, alive = np.asarray(qpos), np.asarray(phase), np.asarray(alive)
    print(f"rolled out {args.panels} bodies x {args.horizon} steps")
    for i, m in enumerate(morphs):
        print(f"  body {i}: " + ", ".join(f"{d.split('_')[0]} {v:.2f}" for d, v in zip(DIMS, m))
              + f"  | survived {int(alive[:, i].sum())}")

    traj_qpos = np.asarray(traj.data.qpos)

    # Targets must be computed PER BODY, exactly as MorphMimicReward does at
    # training time. Reading traj.data.site_xpos would draw the nominal body's
    # targets in every panel -- which is what this renderer did before, and it
    # made the retargeting invisible even though training was using it.
    from mujoco import mjx as _mjx

    _d0 = _mjx.make_data(env.sys)
    _site_ids_mjx = np.array([mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_SITE, n)
                              for n in MIMIC_SITES])

    _Q = jnp.asarray(traj_qpos)

    @jax.jit
    def _targets_for(morph):
        """All frames for one body, vmapped -- 800 separate FK calls per body is
        minutes; one vmapped call is instant."""
        m = env._apply_morphology(env.sys, morph)
        fk = jax.vmap(lambda q: _mjx.kinematics(m, _d0.replace(qpos=q)).site_xpos)
        return fk(_Q)[:, _site_ids_mjx]

    # (panel, frame, site, 3) -- one target set per body per frame
    per_body_targets = np.stack([
        np.asarray(_targets_for(jnp.asarray(morphs[i], dtype=jnp.float32)))
        for i in range(args.panels)
    ])
    spread = np.linalg.norm(per_body_targets[:, :, 1] - per_body_targets[:, :, 3], axis=-1).mean(axis=1)
    print("per-body hand-to-hand target span (m): " + ", ".join(f"{v:.3f}" for v in spread))
    print(f"  spread across the six panels: {(spread.max() - spread.min()) * 100:.1f} cm")
    xml_root = WORKSPACE / "experiments" / "h1_morphology_deepmimic_20260808" / "body_catalog" / "xml_multibody"

    models, site_ids = [], None
    for i, m in enumerate(morphs):
        mdl = body_from_descriptor(m, f"b{i}", xml_root)
        models.append(mdl)
        if site_ids is None:
            site_ids = np.array([mujoco.mj_name2id(mdl, mujoco.mjtObj.mjOBJ_SITE, n) for n in MIMIC_SITES])

    rows = int(np.ceil(args.panels / args.cols))
    frames = []
    datas = [mujoco.MjData(m) for m in models]
    mocaps = [[m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"target_{n}")]
               for n in MIMIC_SITES] for m in models]
    renderers = [mujoco.Renderer(m, height=args.ph, width=args.pw) for m in models]

    try:
        for t in range(0, args.horizon, args.stride):
            tiles = []
            for i in range(args.panels):
                mdl, d, r = models[i], datas[i], renderers[i]
                d.qpos[:] = qpos[t, i]
                ph = int(np.clip(phase[t, i], 0, per_body_targets.shape[1] - 1))
                for j, mid in enumerate(mocaps[i]):
                    d.mocap_pos[mid] = per_body_targets[i, ph, j]
                mujoco.mj_forward(mdl, d)
                cam = mujoco.MjvCamera()
                mujoco.mjv_defaultCamera(cam)
                cam.azimuth, cam.elevation = 135, -14
                rxy = np.array([d.qpos[0], d.qpos[1]])
                txy = per_body_targets[i, ph][:, :2].mean(axis=0)
                mid_xy = 0.5 * (rxy + txy)
                cam.lookat[:] = [mid_xy[0], mid_xy[1], 0.85]
                cam.distance = 3.2 + 1.3 * float(np.linalg.norm(rxy - txy))
                r.update_scene(d, camera=cam)
                img = Image.fromarray(r.render())
                draw = ImageDraw.Draw(img)
                label = (f"legs {morphs[i][0]:.2f}  arms {morphs[i][1]:.2f}  "
                         f"mass {morphs[i][3]:.2f}  | hand tgt {spread[i]:.2f} m")
                draw.rectangle([0, 0, args.pw, 16], fill=(20, 24, 21))
                draw.text((5, 3), label, fill=(226, 232, 226))
                if not alive[t, i]:
                    # `done` here fires on reaching the end of the reference clip
                    # as well as on falling, so it must not be labelled a fall.
                    draw.text((args.pw - 62, 3), "episode end", fill=(224, 136, 128))
                tiles.append(img)
            sheet = Image.new("RGB", (args.cols * args.pw, rows * args.ph), (20, 24, 21))
            for i, img in enumerate(tiles):
                sheet.paste(img, ((i % args.cols) * args.pw, (i // args.cols) * args.ph))
            frames.append(sheet)
    finally:
        for r in renderers:
            r.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(args.out, save_all=True, append_images=frames[1:],
                   duration=int(round(1000 * args.stride / freq)), loop=0, quality=40, method=4)
    err = np.linalg.norm(qpos[:, :, :2] - traj_qpos[np.clip(phase, 0, len(traj_qpos) - 1)][:, :, :2],
                         axis=-1)
    print(f"\nwrote {args.out} ({args.out.stat().st_size // 1024} KB, {len(frames)} frames)")
    print(f"mean root error across panels: {float(err[alive].mean()):.2f} m")


if __name__ == "__main__":
    main()
