"""Apply the 2026-08-29 token-routing edits to a loco_mjx tree, in place.

Written after a wholesale file upload REGRESSED Viper. The two trees have
diverged in BOTH directions: Viper's ``tracking_clip.py`` carries
``tracking_clip_reference_lead``, which the local checkout lacks, while the
local ``algorithms/urma2`` carries ``update_guard.py``, which Viper lacks.
Copying whole files across therefore silently deletes work in one direction or
the other. This patches by anchored replacement instead, so each tree keeps
everything else it has.

Idempotent: a file that already carries the patch marker is reported and left
alone rather than edited twice. Every anchor must match exactly once or the
script aborts without writing, so a drifted file fails loudly instead of being
half-patched.

    python apply_token_patch.py <path to the loco_mjx PACKAGE dir>
    # e.g. /ptmp/akalenik/urma/loco_mjx/loco_mjx
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path


def patch(path: Path, edits, marker: str) -> bool:
    """Apply (old, new) edits to `path`. `marker` proves it is already done."""
    if not path.exists():
        print("  %s: ABSENT, skipped" % path.name)
        return False
    src = path.read_text(encoding="utf-8")
    if marker in src:
        print("  %s: already patched" % path.name)
        return False
    for old, new in edits:
        n = src.count(old)
        if n != 1:
            raise SystemExit(
                "ABORT %s: anchor matched %d times, expected 1:\n%s" % (path, n, old[:200]))
        src = src.replace(old, new)
    ast.parse(src)  # never write a file that will not import
    path.write_text(src, encoding="utf-8")
    print("  %s: PATCHED" % path.name)
    return True


ENV_CFG_OLD = '            "tracking_clip_latent_hold": 1,'

ENV_CFG_NEW = '''            "tracking_clip_latent_hold": 1,
            # WHERE the token enters the observation.
            #   "per_joint" -- latent_dim extra channels on EVERY joint block,
            #                  the pre-2026-08-29 behaviour (bit-identical).
            #   "global"    -- ONE pooled motion embedding in the general
            #                  observation; per-joint blocks keep their width.
            #   "both"      -- per-joint AND pooled, for an ablation.
            # urma2's joint-state path is a single Dense(8). Under "per_joint"
            # the token turns that 5-channel input into 37 channels squeezed
            # through the SAME 8 units that must carry position, velocity and
            # the reference. "global" routes the code into the 512-wide core,
            # where nothing is bottlenecked.
            "tracking_clip_latent_scope": "per_joint",
            # Per-channel divisor for the latent block, mirroring the divisors
            # every other observation channel already has in environment.py's
            # set_norm table (position 6.5, velocity 180, previous action 10).
            # The latent block had NO entry -- because no index list existed to
            # address it -- so FSQ codes entered raw at std ~0.51 beside
            # channels normalized to std ~0.01-0.10, i.e. ~559x the combined
            # input energy of all five real channels. 1.0 reproduces that
            # exactly; it is the default only so no past arm is invalidated.
            "tracking_clip_latent_obs_divisor": 1.0,'''

ENV_EDITS = [
    ('''        self.tracking_latent_dim = (
            int(self.env_config["command"]["tracking_clip_latent_dim"])
            if self.tracking_latent_active else 0
        )''',
     '''        self.tracking_latent_dim = (
            int(self.env_config["command"]["tracking_clip_latent_dim"])
            if self.tracking_latent_active else 0
        )
        # WHERE the token enters. "per_joint" reproduces the old layout exactly;
        # "global" pools it into ONE motion embedding in the general
        # observation, off the joint-state bottleneck entirely.
        _latent_scope = str(self.env_config["command"].get("tracking_clip_latent_scope", "per_joint"))
        if _latent_scope not in ("per_joint", "global", "both"):
            raise ValueError(
                f"tracking_clip_latent_scope={_latent_scope!r}; expected "
                "'per_joint', 'global' or 'both'")
        self.tracking_latent_scope = _latent_scope
        self.tracking_latent_joint_dim = (
            self.tracking_latent_dim if _latent_scope in ("per_joint", "both") else 0
        )
        self.tracking_latent_global_dim = (
            self.tracking_latent_dim if _latent_scope in ("global", "both") else 0
        )'''),
    ('''            + int(self.tracking_reference_velocity_active) + self.tracking_latent_dim
        )''',
     '''            + int(self.tracking_reference_velocity_active) + self.tracking_latent_joint_dim
        )'''),
    ('''        self.policy_joint_reference_obs_idx = []
        self.policy_joint_reference_velocity_obs_idx = []
        for _ in range(self.nr_actuators):''',
     '''        self.policy_joint_reference_obs_idx = []
        self.policy_joint_reference_velocity_obs_idx = []
        # The latent block never had an index list, which is why it is the ONE
        # channel group absent from the set_norm table below, and the reason a
        # token-ablation control had nothing to address.
        self.policy_joint_latent_obs_idx = []
        for _ in range(self.nr_actuators):'''),
    ('''                self.policy_joint_reference_velocity_obs_idx.append(current_observation_idx)
                current_observation_idx += 1
            current_observation_idx += self.tracking_latent_dim''',
     '''                self.policy_joint_reference_velocity_obs_idx.append(current_observation_idx)
                current_observation_idx += 1
            self.policy_joint_latent_obs_idx.extend(
                [current_observation_idx + i for i in range(self.tracking_latent_joint_dim)])
            current_observation_idx += self.tracking_latent_joint_dim'''),
    ('''        self.policy_joint_reference_velocity_obs_idx = jnp.array(
            self.policy_joint_reference_velocity_obs_idx, dtype=jnp.int32
        )''',
     '''        self.policy_joint_reference_velocity_obs_idx = jnp.array(
            self.policy_joint_reference_velocity_obs_idx, dtype=jnp.int32
        )
        self.policy_joint_latent_obs_idx = jnp.array(
            self.policy_joint_latent_obs_idx, dtype=jnp.int32
        )'''),
    ('''        self.critic_joint_reference_obs_idx = []
        self.critic_joint_reference_velocity_obs_idx = []
        for _ in range(self.nr_actuators):''',
     '''        self.critic_joint_reference_obs_idx = []
        self.critic_joint_reference_velocity_obs_idx = []
        self.critic_joint_latent_obs_idx = []
        for _ in range(self.nr_actuators):'''),
    ('''                self.critic_joint_reference_velocity_obs_idx.append(current_observation_idx)
                current_observation_idx += 1
            current_observation_idx += self.tracking_latent_dim''',
     '''                self.critic_joint_reference_velocity_obs_idx.append(current_observation_idx)
                current_observation_idx += 1
            self.critic_joint_latent_obs_idx.extend(
                [current_observation_idx + i for i in range(self.tracking_latent_joint_dim)])
            current_observation_idx += self.tracking_latent_joint_dim'''),
    ('''        self.critic_joint_reference_velocity_obs_idx = jnp.array(
            self.critic_joint_reference_velocity_obs_idx, dtype=jnp.int32
        )''',
     '''        self.critic_joint_reference_velocity_obs_idx = jnp.array(
            self.critic_joint_reference_velocity_obs_idx, dtype=jnp.int32
        )
        self.critic_joint_latent_obs_idx = jnp.array(
            self.critic_joint_latent_obs_idx, dtype=jnp.int32
        )'''),
    ('''        self.critic_root_heading_obs_idx = jnp.array(
            [current_observation_idx + i for i in range(nr_heading_obs)], dtype=jnp.int32)
        current_observation_idx += nr_heading_obs''',
     '''        self.critic_root_heading_obs_idx = jnp.array(
            [current_observation_idx + i for i in range(nr_heading_obs)], dtype=jnp.int32)
        current_observation_idx += nr_heading_obs

        # The pooled FSQ motion embedding, when the token is routed globally.
        # It sits in the GENERAL block, so it reaches the 512-wide core intact
        # instead of being squeezed through the joint-state bottleneck.
        self.policy_motion_latent_obs_idx = jnp.array(
            [current_observation_idx + i for i in range(self.tracking_latent_global_dim)],
            dtype=jnp.int32)
        current_observation_idx += self.tracking_latent_global_dim

        self.critic_motion_latent_obs_idx = jnp.array(
            [current_observation_idx + i for i in range(self.tracking_latent_global_dim)],
            dtype=jnp.int32)
        current_observation_idx += self.tracking_latent_global_dim'''),
    ('''            self.policy_root_heading_obs_idx,
            self.policy_tracked_frame_obs_idx,''',
     '''            self.policy_root_heading_obs_idx,
            self.policy_motion_latent_obs_idx,
            self.policy_tracked_frame_obs_idx,'''),
    ('''            self.critic_root_heading_obs_idx,
            self.critic_tracked_frame_obs_idx,''',
     '''            self.critic_root_heading_obs_idx,
            self.critic_motion_latent_obs_idx,
            self.critic_tracked_frame_obs_idx,'''),
    ('''        set_norm(self.policy_joint_previous_actions_obs_idx, 10.0, 0.0, False)
        set_norm(self.critic_joint_previous_actions_obs_idx, 10.0, 0.0, False)''',
     '''        set_norm(self.policy_joint_previous_actions_obs_idx, 10.0, 0.0, False)
        set_norm(self.critic_joint_previous_actions_obs_idx, 10.0, 0.0, False)
        # The FSQ latent block. Every other channel group above is divided onto
        # a comparable scale; this one had no entry at all, because it had no
        # index list. Measured on the H1 dance2_subject4 sidecar the codes have
        # std 0.508 against post-norm neighbours at std 0.012-0.100 -- so at
        # divisor 1.0 the 32-channel block carries ~559x the input energy of the
        # five real channels combined, into the same Dense(8). Default stays 1.0
        # so no previously trained arm changes.
        _latent_divisor = float(self.env_config["command"].get("tracking_clip_latent_obs_divisor", 1.0))
        set_norm(self.policy_joint_latent_obs_idx, _latent_divisor, 0.0, False)
        set_norm(self.critic_joint_latent_obs_idx, _latent_divisor, 0.0, False)
        set_norm(self.policy_motion_latent_obs_idx, _latent_divisor, 0.0, False)
        set_norm(self.critic_motion_latent_obs_idx, _latent_divisor, 0.0, False)'''),
    ('''            if self.tracking_latent_active:
                reference_latent = internal_state["reference_joint_latent"]''',
     '''            if self.tracking_latent_joint_dim > 0:
                reference_latent = internal_state["reference_joint_latent"]'''),
    ('''        critic_root_heading = policy_root_heading
        policy_tracked_frame =''',
     '''        critic_root_heading = policy_root_heading
        # Pooled motion embedding: one code per timestep for the whole body.
        # Averaged over MAPPED actuators upstream (unmapped ones observe zeros
        # and would drag the mean toward the origin), so it is topology
        # invariant -- the same 32 numbers whether the robot has 19 or 23
        # joints, which is what makes it usable as a shared motion channel
        # across families.
        if self.tracking_latent_global_dim > 0:
            policy_motion_latent = internal_state["reference_motion_latent"]
        else:
            policy_motion_latent = jnp.zeros(0)
        critic_motion_latent = policy_motion_latent
        policy_tracked_frame ='''),
    ('''            policy_root_heading,
            critic_root_heading,
            policy_tracked_frame,''',
     '''            policy_root_heading,
            critic_root_heading,
            policy_motion_latent,
            critic_motion_latent,
            policy_tracked_frame,'''),
]

CLIP_EDITS = [
    ('        self.latent_hold = max(int(command_config.get("tracking_clip_latent_hold", 1)), 1)',
     '''        self.latent_hold = max(int(command_config.get("tracking_clip_latent_hold", 1)), 1)
        self.latent_scope = str(command_config.get("tracking_clip_latent_scope", "per_joint"))
        self.latent_global = self.latent_scope in ("global", "both")'''),
    ('''            latents = _np.zeros((clip.nr_frames, env.nr_actuators, self.latent_dim), dtype=_np.float32)
            mapped = 0''',
     '''            latents = _np.zeros((clip.nr_frames, env.nr_actuators, self.latent_dim), dtype=_np.float32)
            mapped = 0
            mapped_ids = []'''),
    ('''                if name in zq_names:
                    latents[:, actuator_id, :] = z_q[:, zq_names.index(name), :]
                    mapped += 1''',
     '''                if name in zq_names:
                    latents[:, actuator_id, :] = z_q[:, zq_names.index(name), :]
                    mapped += 1
                    mapped_ids.append(actuator_id)'''),
    ('''            if mapped == 0:
                raise ValueError(f"{zq_path}: no latent joints mapped onto this robot")
            self.clip_latents = jnp.array(latents)''',
     '''            if mapped == 0:
                raise ValueError(f"{zq_path}: no latent joints mapped onto this robot")
            self.clip_latents = jnp.array(latents)
            # Pooled motion embedding for the "global" scope. Averaged over the
            # MAPPED actuators only: unmapped rows are zeros by construction
            # above, and including them would shrink the embedding toward the
            # origin by a factor that depends on the robot's actuator count --
            # it would encode the topology, not the motion.
            self.clip_latents_global = jnp.array(
                latents[:, mapped_ids, :].mean(axis=1)) if self.latent_global else None'''),
    ('''            internal_state["reference_joint_latent"] = jnp.zeros(
                (self.env.nr_actuators, self.latent_dim), dtype=jnp.float32
            )''',
     '''            internal_state["reference_joint_latent"] = jnp.zeros(
                (self.env.nr_actuators, self.latent_dim), dtype=jnp.float32
            )
            if self.latent_global:
                internal_state["reference_motion_latent"] = jnp.zeros(
                    (self.latent_dim,), dtype=jnp.float32
                )'''),
    ('            internal_state["reference_joint_latent"] = self.clip_latents[nearest]',
     '''            internal_state["reference_joint_latent"] = self.clip_latents[nearest]
            if self.latent_global:
                # Same frame index as the per-joint stream, so the two scopes
                # can never disagree about which instant is being described.
                internal_state["reference_motion_latent"] = self.clip_latents_global[nearest]'''),
]

ENC_FIELDS = (
    "    network_width_multiplier: float",
    """    network_width_multiplier: float
    joint_latent_channels: int = 0
    joint_latent_encoder_dim: int = 0""")

POLICY_CTOR = (
    "            config.algorithm.network_width_multiplier\n        ), ",
    "            config.algorithm.network_width_multiplier,\n"
    "            config.algorithm.joint_latent_channels, config.algorithm.joint_latent_encoder_dim\n"
    "        ), ")

CRITIC_CTOR = (
    ", config.algorithm.stability_epsilon, config.algorithm.network_width_multiplier)",
    ", config.algorithm.stability_epsilon, config.algorithm.network_width_multiplier,\n"
    "                  config.algorithm.joint_latent_channels, config.algorithm.joint_latent_encoder_dim)")

ALG_CFG = (
    "    config.softmax_temperature = 1.0",
    '''    # FSQ token routing. urma2's joint-state path is a single Dense(8) over
    # the joint's channels; with the token appended per joint that Dense carries
    # 5 real channels AND 32 code channels through the same 8 units.
    # joint_latent_channels > 0 splits it: the base channels keep their Dense(8)
    # and the trailing token channels get their own projection, concatenated.
    # -1 resolves the width from the environment (0 when the token is off or
    # routed globally); any other value is asserted against the environment
    # rather than trusted. joint_latent_encoder_dim = 0 restores the single
    # shared Dense(8) even when the token is present.
    config.joint_latent_channels = -1
    config.joint_latent_encoder_dim = 4
    config.softmax_temperature = 1.0''')

TRAINER = (
    "        self.joint_observation_size = self.train_env.joint_observation_size",
    '''        self.joint_observation_size = self.train_env.joint_observation_size
        # How many TRAILING channels of the per-joint block are the FSQ token.
        # Resolved from the environment so the network split can never disagree
        # with the observation layout; -1 means "resolve", any other value is
        # asserted against the environment rather than trusted.
        _env_latent_channels = int(getattr(self.train_env, "tracking_latent_joint_dim", 0))
        if int(self.config.algorithm.joint_latent_channels) < 0:
            self.config.algorithm.joint_latent_channels = _env_latent_channels
        elif int(self.config.algorithm.joint_latent_channels) != _env_latent_channels:
            raise ValueError(
                f"algorithm.joint_latent_channels="
                f"{int(self.config.algorithm.joint_latent_channels)} but the environment "
                f"publishes {_env_latent_channels} per-joint latent channels")
        self.joint_latent_channels = int(self.config.algorithm.joint_latent_channels)''')


def encoder_edit(dense_name: str):
    old = '        latent_joint_state = nn.Dense(8, name="%s")(joint_state)' % dense_name
    new = '''        # urma2's joint-state path is an 8-unit bottleneck. With the FSQ token
        # appended per joint, `joint_state` grows 5 -> 37 channels and those 8
        # units must carry position, velocity, previous action and the reference
        # ALONGSIDE 32 code channels. joint_latent_channels > 0 gives the token
        # its own projection instead of making it compete for the bottleneck.
        # (The token also reaches the wider mask encoder above, which is not a
        # bottleneck and is deliberately left alone.)
        if self.joint_latent_channels > 0 and self.joint_latent_encoder_dim > 0:
            _L = self.joint_latent_channels
            latent_joint_state = jnp.concatenate([
                nn.Dense(8, name="%s")(joint_state[..., :-_L]),
                nn.Dense(self.joint_latent_encoder_dim, name="%s_token")(joint_state[..., -_L:]),
            ], axis=-1)
        else:
            latent_joint_state = nn.Dense(8, name="%s")(joint_state)''' % (
        dense_name, dense_name, dense_name)
    return (old, new)


def main(root: Path) -> None:
    env = root / "environments" / "locomotion" / "urma2" / "mjx"

    print("environment config:")
    patch(env / "default_config.py", [(ENV_CFG_OLD, ENV_CFG_NEW)],
          marker="tracking_clip_latent_scope")
    print("environment:")
    patch(env / "environment.py", ENV_EDITS, marker="policy_motion_latent_obs_idx")
    print("command function:")
    patch(env / "command_functions" / "tracking_clip.py", CLIP_EDITS,
          marker="clip_latents_global")

    for label, sub, trainer in (("algorithm (mjx)", "mjx", "urma2.py"),
                                ("algorithm (mjx_split)", "mjx_split", "urma2_split.py")):
        d = root / "algorithms" / "urma2" / sub
        if not d.exists():
            print("%s: directory absent, skipped" % label)
            continue
        print("%s:" % label)
        patch(d / "policy.py",
              [encoder_edit("encoder_latent_state"), ENC_FIELDS, POLICY_CTOR],
              marker="encoder_latent_state_token")
        patch(d / "critic.py",
              [encoder_edit("encoder_joint_latent_state"), ENC_FIELDS, CRITIC_CTOR],
              marker="encoder_joint_latent_state_token")
        patch(d / "default_config.py", [ALG_CFG], marker="joint_latent_encoder_dim")
        patch(d / trainer, [TRAINER], marker="_env_latent_channels")

    print("\nTOKEN PATCH COMPLETE")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]).resolve())
