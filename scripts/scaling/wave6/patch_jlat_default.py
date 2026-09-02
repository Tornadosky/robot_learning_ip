"""Wave 6: make the LEGACY shared-Dense token routing the explicit default
(joint_latent_channels 0) and add the JLAT_CH knob to the trainer sbatch and the
CE_SIDECAR knob to crosseval_token3.sbatch. Idempotent, atomic, byte-preserving."""
import os
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[3]


def rw(path, pairs, marker, enc="latin-1"):
    if not path.exists():
        print(f"{path}: absent, skipped"); return
    s = path.read_bytes().decode(enc)
    if marker in s:
        print(f"{path.name}: already patched"); return
    for old, new in pairs:
        assert s.count(old) == 1, (path.name, s.count(old), old[:80])
        s = s.replace(old, new)
    tmp = path.with_suffix(path.suffix + ".tmp_patch")
    tmp.write_bytes(s.encode(enc)); os.replace(tmp, path)
    print(f"{path.name}: patched")


rw(R / "loco_mjx/loco_mjx/algorithms/urma2/mjx/default_config.py", [(
    "    config.joint_latent_channels = -1\n",
    "    # WAVE 6 (2026-09-02): default 0 = the LEGACY shared-Dense routing every\n"
    "    # token arm before 2026-09-02 actually trained with (the -1 resolution was\n"
    "    # silently returning 0). Pass -1 explicitly for the real separate projection.\n"
    "    config.joint_latent_channels = 0\n")], "LEGACY shared-Dense routing every", enc="utf-8")

for sb in (R / "experiments/urma2_h1g1/viper_train.sbatch", R / "viper_train.sbatch"):
    rw(sb, [('  --algorithm.joint_latent_encoder_dim="${JLAT_ENC_DIM:-4}" \\\n',
             '  --algorithm.joint_latent_encoder_dim="${JLAT_ENC_DIM:-4}" \\\n  --algorithm.joint_latent_channels="${JLAT_CH:-0}" \\\n'),
            ("LEGW AUX_COEFF", "JLAT_CH LEGW AUX_COEFF")], "JLAT_CH")

for ce in (R / "scripts/scaling/crosseval_token3.sbatch", R / "crosseval_token3.sbatch"):
    rw(ce, [('CE_FLAGS+=(--latent_divisor "${CE_DIVISOR:-1.0}")\n',
             'CE_FLAGS+=(--latent_divisor "${CE_DIVISOR:-1.0}")\nCE_FLAGS+=(--latent_sidecar "${CE_SIDECAR:-_zq}")\n')], "CE_SIDECAR")

for cm in (R / "experiments/fsq_khaendler/crosseval_motion.py", R / "crosseval_motion.py"):
    if not cm.exists():
        print(f"{cm}: absent, skipped"); continue
    s = cm.read_bytes().decode("utf-8")
    if "latent_sidecar" in s:
        print(f"{cm}: already patched"); continue
    old = '    p.add_argument("--reference_hold", type=int, default=1,\n'
    assert s.count(old) == 1
    s = s.replace(old, '    p.add_argument("--latent_sidecar", default="_zq",\n'
                       '                   help="tracking_clip_latent_sidecar_suffix, as trained (_win for co-training arms)")\n' + old)
    old2 = '        config.environment.command.tracking_clip_latent_obs_divisor = float(args.latent_divisor)\n'
    assert s.count(old2) == 1
    s = s.replace(old2, old2 + '        config.environment.command.tracking_clip_latent_sidecar_suffix = str(args.latent_sidecar)\n')
    old3 = '        "latent_replaces": args.latent_replaces,\n'
    assert s.count(old3) == 1
    s = s.replace(old3, old3 + '        "latent_sidecar": str(args.latent_sidecar),\n')
    tmp = cm.with_suffix(".py.tmp_patch"); tmp.write_bytes(s.encode("utf-8")); os.replace(tmp, cm)
    print(f"{cm}: patched")
print("JLAT/CE PATCH COMPLETE")
