"""Wave 6: apply the AUX TOKEN HEAD patch to urma2.py / policy.py / viper_train.sbatch.

Anchored string replacements with occurrence asserts; idempotent (skips a file
that already carries the patch). Run from anywhere:  python patch_aux_head.py
"""
from pathlib import Path

R = Path(__file__).resolve().parents[3]


def patch(path, pairs, marker):
    s = path.read_text(encoding="utf-8")
    if marker in s:
        print(f"{path.name}: already patched")
        return
    # whitespace-only lines in the source are normalised so anchors match
    norm = lambda t: "\n".join(l.rstrip() if l.strip() == "" else l for l in t.split("\n"))
    s = norm(s)
    # Viper's tree (jax 0.7.1) still uses the multi-line jax.device_put_replicated
    # form; normalise it to the local flax_replicate form so the anchors match.
    _old_rep = ("        policy_state = jax.device_put_replicated(\n            self.policy_state, jax.local_devices()\n        )\n"
                "        critic_state = jax.device_put_replicated(\n            self.critic_state, jax.local_devices()\n        )\n")
    if _old_rep in s and "flax_replicate" not in s:
        s = s.replace(_old_rep, "        # jax.device_put_replicated was removed in JAX 0.10; Flax's helper is the pmap drop-in.\n"
                                "        policy_state = flax_replicate(self.policy_state, jax.local_devices())\n"
                                "        critic_state = flax_replicate(self.critic_state, jax.local_devices())\n")
        s = s.replace("import jax.numpy as jnp\n", "import jax.numpy as jnp\nfrom flax.jax_utils import replicate as flax_replicate\n", 1)
        print(f"{path.name}: replicate form normalised")
    for old, new, cnt in pairs:
        old = norm(old); new = norm(new)
        assert s.count(old) == cnt, (path.name, s.count(old), old[:90])
        s = s.replace(old, new)
    # atomic: encode fully first, then write a sibling and rename over the
    # original (a failed in-place write_text truncated urma2.py once).
    data = s.encode("utf-8")
    tmp = path.with_suffix(path.suffix + ".tmp_patch")
    tmp.write_bytes(data)
    import os
    os.replace(tmp, path)
    print(f"{path.name}: patched")


# ---------------------------------------------------------------- policy.py
HEAD = '''class AuxTokenHead(nn.Module):
    """Wave-6 auxiliary next-token head (docs/notes/AUX_TOKEN_HEAD_SPEC.md).

    Reads the per-joint trunk latent sown by Policy and predicts the FSQ token
    channels of the observation `aux_token_horizon` steps ahead. Lives in its
    own TrainState in urma2.py, so policy checkpoints are unchanged.
    """
    hidden: int
    out_dim: int

    @nn.compact
    def __call__(self, perjoint_latent):
        x = nn.Dense(self.hidden, name="aux_dense_0")(perjoint_latent)
        x = nn.gelu(x)
        return nn.Dense(self.out_dim, name="aux_dense_1")(x)


def get_processed_action_function():'''
patch(R / "loco_mjx/loco_mjx/algorithms/urma2/mjx/policy.py",
      [("def get_processed_action_function():", HEAD, 1)], "class AuxTokenHead")
SOW_ANCHOR = "        joint_description_latent = nn.Dense(scaled_width(256, self.network_width_multiplier), name=\"encoder_dense_1\")(joint_description_latent)\n"
patch(R / "loco_mjx/loco_mjx/algorithms/urma2/mjx/policy.py",
      [(SOW_ANCHOR, SOW_ANCHOR
        + "        # AUX TOKEN HEAD (wave 6): expose the per-joint trunk latent for the\n"
        + "        # self-supervised next-token head in urma2.py. sow() adds NO parameters,\n"
        + "        # so checkpoints stay byte-compatible in both directions.\n"
        + "        self.sow(\"intermediates\", \"perjoint_latent\", joint_description_latent)\n", 1)],
      "\"perjoint_latent\"")

# ---------------------------------------------------------------- algorithm default_config.py
patch(R / "loco_mjx/loco_mjx/algorithms/urma2/mjx/default_config.py", [(
    "    config.joint_latent_encoder_dim = 4\n",
    "    config.joint_latent_encoder_dim = 4\n"
    "    # AUX TOKEN HEAD (wave 6, docs/notes/AUX_TOKEN_HEAD_SPEC.md). A second head\n"
    "    # on the per-joint trunk latent predicts the FSQ token `aux_token_horizon`\n"
    "    # control steps ahead (target = the token channels of that later\n"
    "    # observation, already at obs scale). coeff 0.0 = off = bit-identical.\n"
    "    # detach_trunk=True stops the gradient at the trunk (head-only probe: does\n"
    "    # the trunk already encode next-token information?).\n"
    "    config.aux_token_coeff = 0.0\n"
    "    config.aux_token_horizon = 1\n"
    "    config.aux_token_hidden = 64\n"
    "    config.aux_token_detach_trunk = False\n"
    "    config.aux_token_learning_rate = 1.5e-3\n", 1)], "aux_token_coeff")

# ---------------------------------------------------------------- urma2.py
U = []
U.append(("from loco_mjx.algorithms.urma2.mjx.policy import get_policy",
          "from loco_mjx.algorithms.urma2.mjx.policy import get_policy, AuxTokenHead", 1))

DEL = "        del dummy_policy_joint_descriptions, dummy_policy_joint_observations, dummy_critic_joint_descriptions, dummy_critic_joint_observations, dummy_critic_feet_descriptions, dummy_critic_feet_observations, dummy_policy_general_state, dummy_critic_general_state\n"
AUX_INIT = '''        # AUX TOKEN HEAD (wave 6). Separate module + TrainState: nothing about
        # the policy/critic param trees or checkpoints changes. coeff 0.0 = off
        # = the exact pre-existing code path (Python-level branches below).
        self.aux_token_coeff = float(self.config.algorithm.aux_token_coeff)
        self.aux_on = self.aux_token_coeff > 0.0
        self.aux_token_horizon = int(self.config.algorithm.aux_token_horizon)
        self.aux_token_detach_trunk = bool(self.config.algorithm.aux_token_detach_trunk)
        if self.aux_on:
            if self.joint_latent_channels <= 0:
                raise ValueError("algorithm.aux_token_coeff > 0 requires the FSQ token in the observation (joint_latent_channels > 0)")
            if self.aux_token_horizon < 1 or self.aux_token_horizon >= self.nr_steps:
                raise ValueError(f"algorithm.aux_token_horizon={self.aux_token_horizon} must be in [1, nr_steps-1={self.nr_steps - 1}]")
            self.aux_head = AuxTokenHead(hidden=int(self.config.algorithm.aux_token_hidden), out_dim=self.joint_latent_channels)
            _, _aux_inter = self.policy.apply(policy_params, dummy_policy_joint_descriptions, dummy_policy_joint_observations, dummy_policy_general_state, mutable=["intermediates"])
            _aux_latent = _aux_inter["intermediates"]["perjoint_latent"][0]
            aux_params = self.aux_head.init(jax.random.fold_in(policy_key, 7), _aux_latent)
            self.aux_state = TrainState.create(
                apply_fn=self.aux_head.apply,
                params=aux_params,
                tx=optax.chain(optax.clip_by_global_norm(self.max_grad_norm), optax.adam(float(self.config.algorithm.aux_token_learning_rate)))
            )
            rlx_logger.info(f"Aux token head ON: coeff={self.aux_token_coeff} horizon={self.aux_token_horizon} detach_trunk={self.aux_token_detach_trunk} params={sum(p.size for p in tree.flatten(aux_params))}")
        else:
            self.aux_head = None
            self.aux_state = TrainState.create(apply_fn=None, params={}, tx=optax.identity())

'''
U.append((DEL, AUX_INIT + DEL, 1))

U.append(("        def jitable_train_function(policy_state, critic_state, multi_env_state, key, device_id):",
          "        def jitable_train_function(policy_state, critic_state, aux_state, multi_env_state, key, device_id):", 1))
U.append(("policy_state, prev_policy_state, critic_state, multi_env_state, key = multi_learning_and_eval_save_iteration_carry",
          "policy_state, prev_policy_state, critic_state, aux_state, multi_env_state, key = multi_learning_and_eval_save_iteration_carry", 1))
U.append(("policy_state, prev_policy_state, critic_state, multi_env_state, key = learning_iteration_carry",
          "policy_state, prev_policy_state, critic_state, aux_state, multi_env_state, key = learning_iteration_carry", 2))
U.append(("                    return (new_policy_state, prev_policy_state, new_critic_state, multi_env_state, key), None",
          "                    return (new_policy_state, prev_policy_state, new_critic_state, new_aux_state, multi_env_state, key), None", 1))
U.append(("jax.lax.scan(learning_iteration, (policy_state, prev_policy_state, critic_state, multi_env_state, subkey), jnp.arange(self.nr_updates_per_multi_learning_iteration))",
          "jax.lax.scan(learning_iteration, (policy_state, prev_policy_state, critic_state, aux_state, multi_env_state, subkey), jnp.arange(self.nr_updates_per_multi_learning_iteration))", 1))
U.append(("                return (policy_state, prev_policy_state, critic_state, multi_env_state, key), None\n\n            jax.lax.scan(multi_learning_and_eval_save_iteration, (policy_state, prev_policy_state, critic_state, multi_env_state, key), jnp.arange(start_chunk, self.nr_multi_learning_and_eval_save_iterations))",
          "                return (policy_state, prev_policy_state, critic_state, aux_state, multi_env_state, key), None\n\n            jax.lax.scan(multi_learning_and_eval_save_iteration, (policy_state, prev_policy_state, critic_state, aux_state, multi_env_state, key), jnp.arange(start_chunk, self.nr_multi_learning_and_eval_save_iterations))", 1))
U.append(('        train_function = jax.pmap(jitable_train_function, axis_name="i", donate_argnums=(0, 1, 2, 3))',
          '        train_function = jax.pmap(jitable_train_function, axis_name="i", donate_argnums=(0, 1, 2, 3, 4))', 1))
U.append(("        critic_state = flax_replicate(self.critic_state, jax.local_devices())\n",
          "        critic_state = flax_replicate(self.critic_state, jax.local_devices())\n        aux_state = flax_replicate(self.aux_state, jax.local_devices())\n", 1))
U.append(("        train_function(policy_state, critic_state, multi_env_state, train_key, jnp.arange(self.nr_devices))",
          "        train_function(policy_state, critic_state, aux_state, multi_env_state, train_key, jnp.arange(self.nr_devices))", 1))

BATCH = "                    states, next_states, actions, rewards, values, terminations, log_probs, infos = batch\n\n"
TARGETS = '''                    # AUX TOKEN HEAD targets: the observation H steps ahead (token
                    # channels are sliced out inside loss_fn), valid only when no
                    # episode boundary lies inside the window.
                    if self.aux_on:
                        _H = self.aux_token_horizon
                        if _H == 1:
                            aux_target_states = next_states
                            aux_valid = 1.0 - terminations.astype(jnp.float32)
                        else:
                            _pad = jnp.repeat(states[-1:], _H, axis=0)
                            aux_target_states = jnp.concatenate([states, _pad], axis=0)[_H:_H + self.nr_steps]
                            _term = terminations.astype(jnp.float32)
                            _c = jnp.concatenate([jnp.zeros_like(_term[:1]), jnp.cumsum(_term, axis=0)], axis=0)
                            _c = jnp.concatenate([_c, jnp.repeat(_c[-1:], _H, axis=0)], axis=0)
                            _in_window = _c[_H:_H + self.nr_steps] - _c[:self.nr_steps]
                            _t_ok = (jnp.arange(self.nr_steps) + _H <= self.nr_steps - 1).astype(jnp.float32)[:, None]
                            aux_valid = (_in_window == 0).astype(jnp.float32) * _t_ok
                    else:
                        aux_target_states = None
                        aux_valid = None

'''
U.append((BATCH, BATCH + TARGETS, 1))

U.append(("                    def loss_fn(policy_params, critic_params, states, actions, log_probs, returns, advantages):",
          "                    def loss_fn(policy_params, critic_params, aux_params, states, actions, log_probs, returns, advantages, aux_target_states, aux_valid):", 1))

OLD_PI = '''                            # Policy loss
                            action_mean, action_logstd = self.policy.apply(policy_params, policy_joint_descriptions, policy_joint_observations, policy_general_state)
'''
NEW_PI = '''                            # Policy loss
                            if self.aux_on:
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
U.append((OLD_PI, NEW_PI, 1))

OLD_COMB = '''                            # Combine losses
                            loss = pg_loss - self.entropy_coef * entropy_loss + self.critic_coef * critic_loss
'''
NEW_COMB = OLD_COMB + '''                            if self.aux_on:
                                loss = loss + self.aux_token_coeff * aux_loss
'''
U.append((OLD_COMB, NEW_COMB, 1))

OLD_MET = '''                                "policy/std_dev": action_std_mean
                            }
'''
NEW_MET = OLD_MET + '''                            if self.aux_on:
                                metric["loss/aux_token_loss"] = aux_loss
'''
U.append((OLD_MET, NEW_MET, 1))

OLD_VMAP = '''                    vmap_loss_fn = jax.vmap(loss_fn, in_axes=(None, None, 0, 0, 0, 0, 0), out_axes=0)
                    safe_mean = lambda x: jnp.mean(x) if x is not None else x
                    mean_vmapped_loss_fn = lambda *a, **k: jax.tree.map(safe_mean, vmap_loss_fn(*a, **k))
                    grad_loss_fn = jax.value_and_grad(mean_vmapped_loss_fn, argnums=(0, 1), has_aux=True)
'''
NEW_VMAP = '''                    if self.aux_on:
                        batch_aux_targets = aux_target_states.reshape((self.nr_steps, self.nr_train_robots, self.nr_envs_per_train_robot, *self.os_shape)).transpose((0, 2, 1, 3)).reshape((self.nr_steps * self.nr_envs_per_train_robot, self.nr_train_robots, *self.os_shape))
                        batch_aux_valid = aux_valid.reshape((self.nr_steps, self.nr_train_robots, self.nr_envs_per_train_robot)).transpose((0, 2, 1)).reshape((self.nr_steps * self.nr_envs_per_train_robot, self.nr_train_robots))
                        _aux_axis = 0
                    else:
                        batch_aux_targets = None
                        batch_aux_valid = None
                        _aux_axis = None

                    vmap_loss_fn = jax.vmap(loss_fn, in_axes=(None, None, None, 0, 0, 0, 0, 0, _aux_axis, _aux_axis), out_axes=0)
                    safe_mean = lambda x: jnp.mean(x) if x is not None else x
                    mean_vmapped_loss_fn = lambda *a, **k: jax.tree.map(safe_mean, vmap_loss_fn(*a, **k))
                    grad_loss_fn = jax.value_and_grad(mean_vmapped_loss_fn, argnums=((0, 1, 2) if self.aux_on else (0, 1)), has_aux=True)
'''
U.append((OLD_VMAP, NEW_VMAP, 1))

U.append(('''                    def minibatch_update(carry, minibatch_indices):
                        policy_state, critic_state = carry
''', '''                    def minibatch_update(carry, minibatch_indices):
                        policy_state, critic_state, aux_state = carry
''', 1))

OLD_GRAD = '''                        (loss, (metrics)), (policy_gradients, critic_gradients) = grad_loss_fn(
                            policy_state.params,
                            critic_state.params,
                            batch_states[minibatch_indices],
                            batch_actions[minibatch_indices],
                            batch_log_probs[minibatch_indices],
                            batch_returns[minibatch_indices],
                            minibatch_advantages
                        )

                        policy_gradients = jax.lax.pmean(policy_gradients, axis_name='i')
                        critic_gradients = jax.lax.pmean(critic_gradients, axis_name='i')
                        policy_state = policy_state.apply_gradients(grads=policy_gradients)
                        critic_state = critic_state.apply_gradients(grads=critic_gradients)
'''
NEW_GRAD = '''                        _mb_aux_targets = batch_aux_targets[minibatch_indices] if self.aux_on else None
                        _mb_aux_valid = batch_aux_valid[minibatch_indices] if self.aux_on else None
                        (loss, (metrics)), _grads = grad_loss_fn(
                            policy_state.params,
                            critic_state.params,
                            aux_state.params if self.aux_on else None,
                            batch_states[minibatch_indices],
                            batch_actions[minibatch_indices],
                            batch_log_probs[minibatch_indices],
                            batch_returns[minibatch_indices],
                            minibatch_advantages,
                            _mb_aux_targets,
                            _mb_aux_valid,
                        )
                        if self.aux_on:
                            policy_gradients, critic_gradients, aux_gradients = _grads
                            aux_gradients = jax.lax.pmean(aux_gradients, axis_name='i')
                            aux_state = aux_state.apply_gradients(grads=aux_gradients)
                        else:
                            policy_gradients, critic_gradients = _grads

                        policy_gradients = jax.lax.pmean(policy_gradients, axis_name='i')
                        critic_gradients = jax.lax.pmean(critic_gradients, axis_name='i')
                        policy_state = policy_state.apply_gradients(grads=policy_gradients)
                        critic_state = critic_state.apply_gradients(grads=critic_gradients)
'''
U.append((OLD_GRAD, NEW_GRAD, 1))

U.append(('''                        carry = (policy_state, critic_state)

                        return carry, (metrics)

                    init_carry = (policy_state, critic_state)
                    carry, (optimization_metrics) = jax.lax.scan(minibatch_update, init_carry, batch_indices)
                    new_policy_state, new_critic_state = carry
''', '''                        carry = (policy_state, critic_state, aux_state)

                        return carry, (metrics)

                    init_carry = (policy_state, critic_state, aux_state)
                    carry, (optimization_metrics) = jax.lax.scan(minibatch_update, init_carry, batch_indices)
                    new_policy_state, new_critic_state, new_aux_state = carry
''', 1))

patch(R / "loco_mjx/loco_mjx/algorithms/urma2/mjx/urma2.py", U, "AUX TOKEN HEAD (wave 6)")

# ---------------------------------------------------------------- sbatch
SB = R / "experiments/urma2_h1g1/viper_train.sbatch"
if not SB.exists():
    SB = R / "viper_train.sbatch"  # Viper layout
if not SB.exists():
    print("sbatch: absent, skipped"); raise SystemExit(0)
s = SB.read_bytes().decode("latin-1")
if "aux_token_coeff" not in s:
    old = '  --algorithm.joint_latent_encoder_dim="${JLAT_ENC_DIM:-4}" \\\n'
    assert s.count(old) == 1, s.count(old)
    s = s.replace(old, old + '  --algorithm.aux_token_coeff="${AUX_COEFF:-0.0}" \\\n  --algorithm.aux_token_horizon="${AUX_HORIZON:-1}" \\\n  --algorithm.aux_token_detach_trunk="${AUX_DETACH:-False}" \\\n')
if "LEGW AUX_COEFF" not in s:
    s = s.replace("LATENT_HOLD LATENT_OBS LATENT_DIM LATENT_REPLACES", "LEGW AUX_COEFF AUX_HORIZON AUX_DETACH LATENT_HOLD LATENT_OBS LATENT_DIM LATENT_REPLACES", 1)
SB.write_bytes(s.encode("latin-1"))
print("sbatch: ok")
