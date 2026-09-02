"""Wave 6: SONIC-style co-training patch (encoder-in-the-loop FSQ token).

Env publishes the raw reference window per joint (sidecar `<clip>_win.npz`,
loaded through the existing latent path via tracking_clip_latent_sidecar_suffix);
the policy encodes it online (CoTrainEncoder -> FSQ straight-through), the
code replaces the precomputed token in the tk4 routing, and a per-joint recon
head regularises the code. Anchored, idempotent, atomic writes.
"""
import os
from pathlib import Path

R = Path(__file__).resolve().parents[3]


def patch(path, pairs, marker):
    s = path.read_text(encoding="utf-8")
    if marker in s:
        print(f"{path.name}: already patched")
        return
    norm = lambda t: "\n".join(l.rstrip() if l.strip() == "" else l for l in t.split("\n"))
    s = norm(s)
    for old, new, cnt in pairs:
        old = norm(old); new = norm(new)
        assert s.count(old) == cnt, (path.name, s.count(old), old[:100])
        s = s.replace(old, new)
    data = s.encode("utf-8")
    tmp = path.with_suffix(path.suffix + ".tmp_patch")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    print(f"{path.name}: patched")


# ------------------------------------------------ env config + sidecar suffix
patch(R / "loco_mjx/loco_mjx/environments/locomotion/urma2/mjx/default_config.py", [(
    '            "tracking_clip_latent_obs_divisor": 1.0,\n',
    '            "tracking_clip_latent_obs_divisor": 1.0,\n'
    '            # Which sidecar the latent block is read from: "_zq" = the\n'
    '            # precomputed FSQ code (default); "_win" = the raw reference window\n'
    '            # for SONIC-style co-training (wave 6), encoded online in the policy.\n'
    '            "tracking_clip_latent_sidecar_suffix": "_zq",\n', 1)],
    "tracking_clip_latent_sidecar_suffix")

patch(R / "loco_mjx/loco_mjx/environments/locomotion/urma2/mjx/command_functions/tracking_clip.py", [(
    '            zq_path = resolve_clip_path(env, command_config).replace(".npz", "_zq.npz")\n',
    '            _sfx = str(command_config.get("tracking_clip_latent_sidecar_suffix", "_zq"))\n'
    '            zq_path = resolve_clip_path(env, command_config).replace(".npz", _sfx + ".npz")\n', 1)],
    "tracking_clip_latent_sidecar_suffix")

# ------------------------------------------------ algorithm config
patch(R / "loco_mjx/loco_mjx/algorithms/urma2/mjx/default_config.py", [(
    "    config.aux_token_learning_rate = 1.5e-3\n",
    "    config.aux_token_learning_rate = 1.5e-3\n"
    "    # SONIC-STYLE CO-TRAINING (wave 6). cotrain_window_rows > 0 turns it on:\n"
    "    # the env's per-joint latent block is then the RAW reference window\n"
    "    # (rows x channels, sidecar _win), encoded online by a per-joint FSQ\n"
    "    # encoder inside the policy (straight-through, PPO gradients reach it)\n"
    "    # plus a reconstruction head. The code (cotrain_latent_dim wide, divided\n"
    "    # by cotrain_code_divisor) takes the precomputed token's place in the tk4\n"
    "    # routing; the critic drops the window block. 0 = off = bit-identical.\n"
    "    config.cotrain_window_rows = 0\n"
    "    config.cotrain_window_channels = 4\n"
    "    config.cotrain_latent_dim = 32\n"
    "    config.cotrain_fsq_levels = 8\n"
    "    config.cotrain_recon_coeff = 1.0\n"
    "    config.cotrain_code_divisor = 10.0\n"
    "    config.cotrain_freeze_encoder = False\n"
    "    config.cotrain_init_encoder = \"\"  # tokenizer params.msgpack; empty = from scratch\n", 1)],
    "cotrain_window_rows")

# ------------------------------------------------ policy.py
P = []
P.append(("from loco_mjx.algorithms.urma2.network_width import scaled_width\n",
          "from loco_mjx.algorithms.urma2.network_width import scaled_width\n"
          "from loco_mjx.algorithms.urma2.mjx.fsq_cotrain import CoTrainEncoder, FSQ, CoTrainReconHead\n", 1))
P.append(("            config.algorithm.joint_latent_channels, config.algorithm.joint_latent_encoder_dim\n        ), \n",
          "            config.algorithm.joint_latent_channels, config.algorithm.joint_latent_encoder_dim,\n"
          "            int(config.algorithm.cotrain_window_rows), int(config.algorithm.cotrain_window_channels),\n"
          "            int(config.algorithm.cotrain_latent_dim), int(config.algorithm.cotrain_fsq_levels),\n"
          "            bool(config.algorithm.cotrain_freeze_encoder), float(config.algorithm.cotrain_code_divisor),\n"
          "        ), \n", 1))
P.append(("    joint_latent_channels: int = 0\n    joint_latent_encoder_dim: int = 0\n\n    @nn.compact\n    def __call__(self, joint_description, joint_state, general_state):\n",
          "    joint_latent_channels: int = 0\n    joint_latent_encoder_dim: int = 0\n"
          "    cotrain_rows: int = 0\n    cotrain_chans: int = 0\n    cotrain_latent_dim: int = 32\n"
          "    cotrain_levels: int = 8\n    cotrain_freeze: bool = False\n    cotrain_code_divisor: float = 10.0\n\n"
          "    @nn.compact\n    def __call__(self, joint_description, joint_state, general_state):\n", 1))
COTRAIN_BLOCK = '''        joints_present = joint_description[..., [-1]]
        joint_description = joint_description[..., :-1]
        if self.cotrain_rows > 0:
            # SONIC-STYLE CO-TRAINING (wave 6): the trailing W channels of the
            # joint state are the RAW reference window (rows x channels, the
            # tokenizer's own input layout). Encode per joint online, quantise
            # with straight-through FSQ, and let the code take the token's place.
            _W = self.cotrain_rows * self.cotrain_chans
            _win = joint_state[..., -_W:]
            _base = joint_state[..., :-_W]
            _lead = _win.shape[:-2]
            _J = _win.shape[-2]
            _x = _win.reshape((-1, _J, self.cotrain_rows, self.cotrain_chans))
            _x = jnp.transpose(_x, (0, 2, 1, 3))  # (B, rows, J, C)
            _z = CoTrainEncoder(latent_dim=self.cotrain_latent_dim, network_width_multiplier=1.0, name="encoder_cotrain")(_x)
            _zq = FSQ(levels=tuple([self.cotrain_levels] * self.cotrain_latent_dim), name="encoder_cotrain_fsq")(_z)
            if self.cotrain_freeze:
                _zq = jax.lax.stop_gradient(_zq)
            _zq = _zq.reshape(_lead + (_J, self.cotrain_latent_dim))
            _recon = CoTrainReconHead(out_dim=_W, name="encoder_cotrain_recon")(_zq, joint_description)
            _recon_err = jnp.mean(jnp.square(_recon - jax.lax.stop_gradient(_win)), axis=-1, keepdims=True) * joints_present
            _recon_loss = jnp.sum(_recon_err) / jnp.maximum(jnp.sum(joints_present), 1.0)
            self.sow("intermediates", "cotrain_recon_loss", _recon_loss)
            self.sow("intermediates", "cotrain_code", _zq)
            joint_state = jnp.concatenate([_base, _zq / self.cotrain_code_divisor], axis=-1)
'''
P.append(("        joints_present = joint_description[..., [-1]]\n        joint_description = joint_description[..., :-1]\n",
          COTRAIN_BLOCK, 1))
patch(R / "loco_mjx/loco_mjx/algorithms/urma2/mjx/policy.py", P, "cotrain_rows")

# ------------------------------------------------ critic.py
C = []
C.append(("                  config.algorithm.joint_latent_channels, config.algorithm.joint_latent_encoder_dim)\n",
          "                  (0 if int(config.algorithm.cotrain_window_rows) > 0 else config.algorithm.joint_latent_channels), config.algorithm.joint_latent_encoder_dim,\n"
          "                  int(config.algorithm.cotrain_window_rows) * int(config.algorithm.cotrain_window_channels) if int(config.algorithm.cotrain_window_rows) > 0 else 0)\n", 1))
C.append(("    joint_latent_channels: int = 0\n    joint_latent_encoder_dim: int = 0\n",
          "    joint_latent_channels: int = 0\n    joint_latent_encoder_dim: int = 0\n    cotrain_drop: int = 0\n", 1))
C.append(("    def __call__(self, joint_description, joint_state, foot_description, foot_state, general_state):\n",
          "    def __call__(self, joint_description, joint_state, foot_description, foot_state, general_state):\n"
          "        if self.cotrain_drop > 0:\n"
          "            # co-training: the raw window block is the policy encoder's input,\n"
          "            # not a critic feature (raw radians/rad/s would enter ~500x too loud).\n"
          "            joint_state = joint_state[..., :-self.cotrain_drop]\n", 1))
patch(R / "loco_mjx/loco_mjx/algorithms/urma2/mjx/critic.py", C, "cotrain_drop")

# ------------------------------------------------ urma2.py
U = []
U.append(('''        _env_latent_channels = int(getattr(self.train_env, "tracking_latent_joint_dim", 0))
        if int(self.config.algorithm.joint_latent_channels) < 0:
            self.config.algorithm.joint_latent_channels = _env_latent_channels
        elif int(self.config.algorithm.joint_latent_channels) != _env_latent_channels:
''', '''        # WAVE 6 FIX (2026-09-02): the MultiEnvironment wrapper has no
        # `tracking_latent_joint_dim` attribute, so the old getattr(..., 0)
        # ALWAYS resolved 0 -- every "tk4" arm (Viper and local) trained with
        # the token through the shared Dense(8) (param counts of tk1/tk2/tk4
        # are identical: 1486130). Read the per-robot envs instead. A stored
        # value of 0 with token channels present is that LEGACY layout and is
        # honoured (old checkpoints keep loading); -1 resolves to the real
        # split; any other mismatch is an error.
        _per_robot = [int(x) for x in self.train_env.call("tracking_latent_joint_dim")]
        if len(set(_per_robot)) != 1:
            raise ValueError(f"train robots disagree on tracking_latent_joint_dim: {_per_robot}")
        _env_latent_channels = _per_robot[0]
        # SONIC-STYLE CO-TRAINING (wave 6): the env block is the raw window
        # (rows*channels wide); the policy-side token width is the code width.
        self.cotrain_on = int(self.config.algorithm.cotrain_window_rows) > 0
        if self.cotrain_on:
            _win_w = int(self.config.algorithm.cotrain_window_rows) * int(self.config.algorithm.cotrain_window_channels)
            if _env_latent_channels != _win_w:
                raise ValueError(f"cotrain: environment publishes {_env_latent_channels} latent channels but the window is {_win_w} (rows*channels); use the _win sidecar with tracking_clip_latent_dim={_win_w}")
            self.config.algorithm.joint_latent_channels = int(self.config.algorithm.cotrain_latent_dim)
        elif int(self.config.algorithm.joint_latent_channels) < 0:
            self.config.algorithm.joint_latent_channels = _env_latent_channels
        elif int(self.config.algorithm.joint_latent_channels) == 0 and _env_latent_channels > 0:
            rlx_logger.warning(f"joint_latent_channels=0 with {_env_latent_channels} token channels in the observation: LEGACY shared-Dense routing (pre-2026-09-02 checkpoints)")
        elif int(self.config.algorithm.joint_latent_channels) != _env_latent_channels:
''', 1))

U.append(('''        policy_params = self.policy.init(policy_key, dummy_policy_joint_descriptions, dummy_policy_joint_observations, dummy_policy_general_state)
''', '''        policy_params = self.policy.init(policy_key, dummy_policy_joint_descriptions, dummy_policy_joint_observations, dummy_policy_general_state)
        if self.cotrain_on:
            self.cotrain_recon_coeff = float(self.config.algorithm.cotrain_recon_coeff)
            self.cotrain_code_divisor = float(self.config.algorithm.cotrain_code_divisor)
            _init_path = str(self.config.algorithm.cotrain_init_encoder or "")
            if _init_path and not os.path.exists(_init_path):
                # A checkpoint stores the path it was trained with (e.g. Viper's);
                # at evaluation the encoder weights come from the checkpoint
                # itself, so a missing file is only fatal for training.
                if str(getattr(self.config.runner, "mode", "train")) == "train" and not getattr(self.config.runner, "load_model", ""):
                    raise FileNotFoundError(f"cotrain_init_encoder not found: {_init_path}")
                rlx_logger.warning(f"cotrain: init file {_init_path} absent; relying on the checkpoint's encoder weights")
                _init_path = ""
            if _init_path:
                from loco_mjx.algorithms.urma2.mjx.fsq_cotrain import load_tokenizer_encoder_params
                import flax.core
                _pp = flax.core.unfreeze(policy_params)
                _pp["params"]["encoder_cotrain"] = load_tokenizer_encoder_params(_init_path, _pp["params"]["encoder_cotrain"])
                policy_params = _pp
                rlx_logger.info(f"cotrain: encoder initialised from {_init_path}")
            rlx_logger.info(f"cotrain ON: rows={self.config.algorithm.cotrain_window_rows} chans={self.config.algorithm.cotrain_window_channels} code={self.config.algorithm.cotrain_latent_dim} recon_coeff={self.cotrain_recon_coeff} freeze={self.config.algorithm.cotrain_freeze_encoder}")
''', 1))

OLD_PI = '''                            if self.aux_on:
                                (action_mean, action_logstd), _inter = self.policy.apply(policy_params, policy_joint_descriptions, policy_joint_observations, policy_general_state, mutable=["intermediates"])
                                _lat = _inter["intermediates"]["perjoint_latent"][0]
                                if self.aux_token_detach_trunk:
                                    _lat = jax.lax.stop_gradient(_lat)
                                _pred = self.aux_head.apply(aux_params, _lat)
                                _tgt_joint_obs = self._decode_train_obs(aux_target_states[i], i)[1]
                                _tgt = jax.lax.stop_gradient(_tgt_joint_obs[..., -self.joint_latent_channels:])
                                _aux_err = jnp.sum(jnp.mean(jnp.square(_pred - _tgt), axis=-1) * joints_present) / jnp.maximum(joints_present.sum(), 1.0)
                                aux_loss = _aux_err * aux_valid[i]
                            else:
                                action_mean, action_logstd = self.policy.apply(policy_params, policy_joint_descriptions, policy_joint_observations, policy_general_state)
                                aux_loss = None
'''
NEW_PI = '''                            if self.aux_on or self.cotrain_on:
                                (action_mean, action_logstd), _inter = self.policy.apply(policy_params, policy_joint_descriptions, policy_joint_observations, policy_general_state, mutable=["intermediates"])
                            else:
                                action_mean, action_logstd = self.policy.apply(policy_params, policy_joint_descriptions, policy_joint_observations, policy_general_state)
                                _inter = None
                            recon_loss = _inter["intermediates"]["cotrain_recon_loss"][0] if self.cotrain_on else None
                            if self.aux_on:
                                _lat = _inter["intermediates"]["perjoint_latent"][0]
                                if self.aux_token_detach_trunk:
                                    _lat = jax.lax.stop_gradient(_lat)
                                _pred = self.aux_head.apply(aux_params, _lat)
                                _tgt_dec = self._decode_train_obs(aux_target_states[i], i)
                                if self.cotrain_on:
                                    # target = the CODE the (current) encoder emits for the future window
                                    _, _ti = self.policy.apply(policy_params, _tgt_dec[0], _tgt_dec[1], _tgt_dec[2], mutable=["intermediates"])
                                    _tgt = jax.lax.stop_gradient(_ti["intermediates"]["cotrain_code"][0] / self.cotrain_code_divisor)
                                else:
                                    _tgt = jax.lax.stop_gradient(_tgt_dec[1][..., -self.joint_latent_channels:])
                                _aux_err = jnp.sum(jnp.mean(jnp.square(_pred - _tgt), axis=-1) * joints_present) / jnp.maximum(joints_present.sum(), 1.0)
                                aux_loss = _aux_err * aux_valid[i]
                            else:
                                aux_loss = None
'''
U.append((OLD_PI, NEW_PI, 1))

U.append(('''                            if self.aux_on:
                                loss = loss + self.aux_token_coeff * aux_loss
''', '''                            if self.aux_on:
                                loss = loss + self.aux_token_coeff * aux_loss
                            if self.cotrain_on:
                                loss = loss + self.cotrain_recon_coeff * recon_loss
''', 1))
U.append(('''                            if self.aux_on:
                                metric["loss/aux_token_loss"] = aux_loss
''', '''                            if self.aux_on:
                                metric["loss/aux_token_loss"] = aux_loss
                            if self.cotrain_on:
                                metric["loss/cotrain_recon_loss"] = recon_loss
''', 1))
patch(R / "loco_mjx/loco_mjx/algorithms/urma2/mjx/urma2.py", U, "SONIC-STYLE CO-TRAINING (wave 6)")

# ------------------------------------------------ launch scripts
L = R / "scripts/scaling/wave6/local_train.sh"
s = L.read_text(encoding="utf-8") if L.exists() else "COTRAIN_ROWS"  # local-only launcher
if "COTRAIN_ROWS" not in s:
    s = s.replace('LEGW=${LEGW:-1.0}; SEED=${SEED:-1}; EXTRA=${EXTRA:-}\n',
                  'LEGW=${LEGW:-1.0}; SEED=${SEED:-1}; EXTRA=${EXTRA:-}\n'
                  'LATENT_DIM=${LATENT_DIM:-32}; LATENT_DIVISOR=${LATENT_DIVISOR:-10.0}; SIDECAR=${SIDECAR:-_zq}\n'
                  'COTRAIN_ROWS=${COTRAIN_ROWS:-0}; COTRAIN_CH=${COTRAIN_CH:-4}; COTRAIN_RECON=${COTRAIN_RECON:-1.0}\n'
                  'COTRAIN_FREEZE=${COTRAIN_FREEZE:-False}; COTRAIN_INIT=${COTRAIN_INIT:-}\n')
    s = s.replace('  --environment.command.tracking_clip_latent_dim=32 \\\n', '  --environment.command.tracking_clip_latent_dim="$LATENT_DIM" \\\n')
    s = s.replace('  --environment.command.tracking_clip_latent_obs_divisor=10.0 \\\n',
                  '  --environment.command.tracking_clip_latent_obs_divisor="$LATENT_DIVISOR" \\\n'
                  '  --environment.command.tracking_clip_latent_sidecar_suffix="$SIDECAR" \\\n')
    s = s.replace('  --algorithm.aux_token_detach_trunk="$AUX_DETACH" \\\n',
                  '  --algorithm.aux_token_detach_trunk="$AUX_DETACH" \\\n'
                  '  --algorithm.cotrain_window_rows="$COTRAIN_ROWS" \\\n'
                  '  --algorithm.cotrain_window_channels="$COTRAIN_CH" \\\n'
                  '  --algorithm.cotrain_recon_coeff="$COTRAIN_RECON" \\\n'
                  '  --algorithm.cotrain_freeze_encoder="$COTRAIN_FREEZE" \\\n'
                  '  --algorithm.cotrain_init_encoder="$COTRAIN_INIT" \\\n')
    s = s.replace('legw=$LEGW seed=$SEED', 'legw=$LEGW cotrain=$COTRAIN_ROWS/$COTRAIN_CH/$COTRAIN_RECON/$COTRAIN_FREEZE/[$COTRAIN_INIT] sidecar=$SIDECAR dim=$LATENT_DIM div=$LATENT_DIVISOR seed=$SEED')
    L.write_bytes(s.encode("utf-8")); print("local_train.sh: patched")

SB = R / "experiments/urma2_h1g1/viper_train.sbatch"
if not SB.exists():
    SB = R / "viper_train.sbatch"  # Viper layout
s = SB.read_bytes().decode("latin-1") if SB.exists() else "COTRAIN_ROWS"  # byte-preserving: cp1252 dashes
if "COTRAIN_ROWS" not in s:
    s = s.replace('         --environment.command.tracking_clip_latent_obs_divisor="${LATENT_DIVISOR:-1.0}"\n',
                  '         --environment.command.tracking_clip_latent_obs_divisor="${LATENT_DIVISOR:-1.0}"\n'
                  '         --environment.command.tracking_clip_latent_sidecar_suffix="${SIDECAR:-_zq}"\n')
    s = s.replace('  --algorithm.aux_token_detach_trunk="${AUX_DETACH:-False}" \\\n',
                  '  --algorithm.aux_token_detach_trunk="${AUX_DETACH:-False}" \\\n'
                  '  --algorithm.cotrain_window_rows="${COTRAIN_ROWS:-0}" \\\n'
                  '  --algorithm.cotrain_window_channels="${COTRAIN_CH:-4}" \\\n'
                  '  --algorithm.cotrain_recon_coeff="${COTRAIN_RECON:-1.0}" \\\n'
                  '  --algorithm.cotrain_freeze_encoder="${COTRAIN_FREEZE:-False}" \\\n'
                  '  --algorithm.cotrain_init_encoder="${COTRAIN_INIT:-}" \\\n')
    s = s.replace("LEGW AUX_COEFF AUX_HORIZON AUX_DETACH", "LEGW AUX_COEFF AUX_HORIZON AUX_DETACH SIDECAR COTRAIN_ROWS COTRAIN_CH COTRAIN_RECON COTRAIN_FREEZE COTRAIN_INIT", 1)
    assert "COTRAIN_ROWS" in s and "SIDECAR:-_zq" in s
    SB.write_bytes(s.encode("latin-1")); print("sbatch: patched")

# ------------------------------------------------ crosseval
CE = R / "experiments/fsq_khaendler/crosseval_motion.py"
if not CE.exists():
    CE = R / "crosseval_motion.py"  # Viper layout
s = CE.read_text(encoding="utf-8") if CE.exists() else "latent_sidecar"
if "latent_sidecar" not in s:
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
    CE.write_bytes(s.encode("utf-8")); print("crosseval_motion.py: patched")
print("COTRAIN PATCH COMPLETE")
