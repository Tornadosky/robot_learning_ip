#!/usr/bin/env python3
"""Anchored patch: add the night5 SWING MATCH reward term to Viper's tree.

Same edits as the local 2026-08-31 change, applied by unique anchors so a
diverged Viper file aborts loudly instead of being clobbered. Backups:
*.bak_night5. Runs ON VIPER.
"""
import shutil
import sys

TRACK = "/ptmp/akalenik/urma/loco_mjx/loco_mjx/environments/locomotion/urma2/mjx/reward_functions/tracking.py"
CONF = "/ptmp/akalenik/urma/loco_mjx/loco_mjx/environments/locomotion/urma2/mjx/default_config.py"

EDITS_TRACK = [
    (
        '        self.foot_height_coeff = (\n'
        '            self.foot_height_weight_ratio * reward_config["joint_tracking_coeff"] * env.dt\n'
        '        )\n',
        '        self.foot_height_coeff = (\n'
        '            self.foot_height_weight_ratio * reward_config["joint_tracking_coeff"] * env.dt\n'
        '        )\n'
        '        # SWING MATCH (night5): per-foot LINEAR hinge on lift deficit,\n'
        '        # gated on the REFERENCE foot being airborne. FOOTH/air_time have\n'
        '        # no usable gradient for a planted foot; this does.\n'
        '        self.swing_match_weight_ratio = float(\n'
        '            reward_config.get("deepmimic_swing_match_weight_ratio", 0.0))\n'
        '        self.swing_match_ref_threshold_m = float(\n'
        '            reward_config.get("deepmimic_swing_match_ref_threshold_m", 0.02))\n'
        '        self.swing_match_active = (\n'
        '            self.swing_match_weight_ratio > 0.0 and env.nr_feet > 0 and env.has_free_base\n'
        '        )\n'
        '        self.swing_match_coeff = (\n'
        '            self.swing_match_weight_ratio * reward_config["joint_tracking_coeff"] * env.dt\n'
        '        )\n',
    ),
    (
        '                 or self.foot_height_active)\n        )\n',
        '                 or self.foot_height_active or self.swing_match_active)\n        )\n',
    ),
    (
        '        foot_height_reward = 0.0\n',
        '        foot_height_reward = 0.0\n        swing_match_reward = 0.0\n',
    ),
    (
        '            if self.log_info:\n'
        '                short_name = self.env.robot_config["short_name"]\n'
        '                info[f"env_info/foot_height_error/{short_name}"] = foot_height_error\n'
        '                info[f"env_info/foot_height_reward/{short_name}"] = foot_height_reward\n',
        '            ref_airborne_gate = (\n'
        '                reference_feet_over_ground > self.swing_match_ref_threshold_m\n'
        '            ).astype(jnp.float32)\n'
        '            swing_deficit = jnp.maximum(\n'
        '                reference_feet_over_ground - robot_feet_over_ground, 0.0\n'
        '            ) / body_length_scale\n'
        '            swing_match_error = jnp.mean(ref_airborne_gate * swing_deficit)\n'
        '            if self.swing_match_active:\n'
        '                swing_match_reward = -(\n'
        '                    tracking_gate * self.swing_match_coeff * swing_match_error\n'
        '                )\n'
        '            if self.log_info:\n'
        '                short_name = self.env.robot_config["short_name"]\n'
        '                info[f"env_info/foot_height_error/{short_name}"] = foot_height_error\n'
        '                info[f"env_info/foot_height_reward/{short_name}"] = foot_height_reward\n'
        '                info[f"env_info/swing_match_error/{short_name}"] = swing_match_error\n'
        '                info[f"env_info/swing_match_reward/{short_name}"] = swing_match_reward\n'
        '                info[f"env_info/ref_airborne_frac/{short_name}"] = jnp.mean(ref_airborne_gate)\n',
    ),
    (
        '                  + root_heading_reward + foot_height_reward\n',
        '                  + root_heading_reward + foot_height_reward + swing_match_reward\n',
    ),
]
EDITS_CONF = [
    (
        '            "deepmimic_foot_height_weight_ratio": 0.0,\n',
        '            "deepmimic_foot_height_weight_ratio": 0.0,\n'
        '            "deepmimic_swing_match_weight_ratio": 0.0,\n'
        '            "deepmimic_swing_match_ref_threshold_m": 0.02,\n',
    ),
]


def apply(path, edits):
    src = open(path).read()
    if "swing_match" in src:
        print(f"{path}: ALREADY PATCHED")
        return
    for old, new in edits:
        n = src.count(old)
        if n != 1:
            print(f"ABORT {path}: anchor x{n}: {old[:70]!r}")
            sys.exit(1)
    shutil.copy2(path, path + ".bak_night5")
    for old, new in edits:
        src = src.replace(old, new)
    compile(src, path, "exec")
    open(path, "w").write(src)
    print(f"{path}: PATCHED")


apply(TRACK, EDITS_TRACK)
apply(CONF, EDITS_CONF)
print("OK")
