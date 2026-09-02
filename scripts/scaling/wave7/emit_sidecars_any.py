"""Emit `_win` (raw 11x4 window) and `_zq` (FSQ code from the tokenizer ENCODER
only) sidecars for any loco_mjx robot, with joint names in the MODEL's actuator
convention (what tracking_clip.py maps by name) and the loader's alias tables
applied to the clip. No decoder / descriptor needed, so it covers robots the
tokenizer never saw (BoosterT1, Atlas, Apollo).

  python scripts/scaling/wave7/emit_sidecars_any.py --tokenizer experiments/fsq_khaendler/tokenizer_m20 \
     --clip-dir experiments/fsq_khaendler/clips_5r --robots booster_t1 atlas apptronik_apollo --clip super20.npz walk1_subject5.npz
(runs in WSL ~/jaxgpu with JAX_PLATFORMS=cpu, or the Windows .venv)
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(R / "scripts/scaling"))
sys.path.insert(0, str(R / "loco_mjx"))
from khaendler_fsq_clip import build_windows, load_clip_feet  # noqa: E402

ROBOT_DIR = {"unitree_h1": "UnitreeH1", "unitree_g1": "UnitreeG1", "booster_t1": "BoosterT1",
             "atlas": "Atlas", "apptronik_apollo": "Apollo", "fourier_gr1t2": "FourierGR1T2", "talos": "Talos"}
TABLES = R / "loco_mjx/loco_mjx/environments/locomotion/urma2/mjx/clip_tables"


def actuator_joint_names(robot):
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(R / "loco_mjx/loco_mjx/environments/robots" / robot / "data/plane.xml"))
    names = []
    for a in range(m.nu):
        if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT:
            names.append(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(m.actuator_trnid[a, 0])))
    return names


def load_joints_aliased(npz, model_joint_names, aliases, signs):
    d = np.load(npz, allow_pickle=True)
    cj = [str(n) for n in d["joint_names"]]
    if cj[0] not in ("root", "reference", "floating_base_joint", "floating_base"):
        raise ValueError(f"{npz}: expected a free joint first, got {cj[0]!r}")
    clip_joints = [aliases.get(n, n) for n in cj[1:]]
    missing = [n for n in model_joint_names if n not in clip_joints]
    if missing:
        raise ValueError(f"{npz}: actuator joints without clip column: {missing}")
    perm = np.array([clip_joints.index(n) for n in model_joint_names])
    sg = np.array([signs.get(n, 1.0) for n in model_joint_names], dtype=np.float32)
    qpos = np.asarray(d["qpos"], dtype=np.float32)[:, 7:][:, perm] * sg
    qvel = np.asarray(d["qvel"], dtype=np.float32)[:, 6:][:, perm] * sg
    return qpos, qvel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--clip-dir", required=True)
    ap.add_argument("--robots", nargs="+", required=True, help="loco_mjx robot names (unitree_h1, booster_t1, atlas, apptronik_apollo)")
    ap.add_argument("--clip", nargs="+", required=True)
    ap.add_argument("--no-zq", action="store_true")
    a = ap.parse_args()
    tok = Path(a.tokenizer)
    cfg = json.loads((tok / "config.json").read_text())
    lookahead, foot, latent, levels = int(cfg["lookahead"]), bool(cfg.get("foot_channels", False)), int(cfg["latent_dim"]), int(cfg["fsq_levels"])
    C = 4 if foot else 2

    import jax, jax.numpy as jnp
    from loco_mjx.algorithms.urma2.mjx.fsq_cotrain import CoTrainEncoder, FSQ, load_tokenizer_encoder_params
    enc = CoTrainEncoder(latent_dim=latent, network_width_multiplier=float(cfg.get("width", 1.0)))
    fsq = FSQ(levels=tuple([levels] * latent))
    dummy = jnp.zeros((1, lookahead + 1, 1, C), jnp.float32)
    p0 = enc.init(jax.random.PRNGKey(0), dummy)["params"]
    params = load_tokenizer_encoder_params(str(tok / "params.msgpack"), p0)

    @jax.jit
    def encode(w):  # (B, N, J, C) -> (B, J, latent)
        return fsq.apply({}, enc.apply({"params": params}, w))

    for robot in a.robots:
        sub = ROBOT_DIR[robot]
        tab = json.load(open(TABLES / f"{sub}.json")) if (TABLES / f"{sub}.json").exists() else {}
        aliases, signs = tab.get("joint_aliases", {}), tab.get("signs", {})
        jn = actuator_joint_names(robot)
        for clip in a.clip:
            src = Path(a.clip_dir) / sub / clip
            if not src.exists():
                print(f"{robot}/{clip}: missing, skipped"); continue
            qpos, qvel = load_joints_aliased(src, jn, aliases, signs)
            feet = load_clip_feet(src) if foot else None
            w = build_windows(qpos, qvel, lookahead, feet)  # (T, N, J, C)
            T, N, J, Cc = w.shape
            flat = np.transpose(w, (0, 2, 1, 3)).reshape(T, J, N * Cc).astype(np.float32)
            np.savez(src.with_name(src.stem + "_win.npz"), z_q=flat, joint_names=np.array(jn), rows=N, channels=Cc, lookahead=lookahead, foot_channels=foot)
            msg = f"{robot}/{clip}: win {flat.shape}"
            if not a.no_zq:
                zq = np.concatenate([np.asarray(encode(jnp.asarray(w[i:i + 2048]))) for i in range(0, T, 2048)], axis=0).astype(np.float32)
                np.savez(src.with_name(src.stem + "_zq.npz"), z_q=zq, joint_names=np.array(jn))
                msg += f" zq {zq.shape} unique-codes {len(np.unique(zq.reshape(-1, latent), axis=0))}"
            print(msg, flush=True)


if __name__ == "__main__":
    main()
