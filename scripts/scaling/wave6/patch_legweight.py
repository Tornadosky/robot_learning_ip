"""Wave 6: per-chain (leg-weighted) tracking kernel patch -- tracking.py + env
default_config.py. Anchored, idempotent, atomic. Weight 1.0 = bit-identical."""
import os
from pathlib import Path

R = Path(__file__).resolve().parents[3]


def patch(path, pairs, marker):
    s = path.read_bytes().decode("utf-8")
    if marker in s:
        print(f"{path.name}: already patched")
        return
    for old, new, cnt in pairs:
        assert s.count(old) == cnt, (path.name, s.count(old), old[:100])
        s = s.replace(old, new)
    tmp = path.with_suffix(path.suffix + ".tmp_patch")
    tmp.write_bytes(s.encode("utf-8"))
    os.replace(tmp, path)
    print(f"{path.name}: patched")


patch(R / "loco_mjx/loco_mjx/environments/locomotion/urma2/mjx/default_config.py", [(
    '            "joint_tracking_temperature": 0.25,\n',
    '            "joint_tracking_temperature": 0.25,\n'
    '            # PER-CHAIN KERNEL (wave 6, roadmap F4). Actuators whose joint name\n'
    '            # matches joint_tracking_leg_regex get weight joint_tracking_leg_weight\n'
    '            # in the mean-squared joint error (weighted mean, so the error scale\n'
    '            # is unchanged at weight 1.0). 1.0 = off = bit-identical. Rationale:\n'
    '            # arms are ~2x cheaper to track than legs, so a shared kernel buys\n'
    '            # arm reward and parks the legs (legs sit 10-15% above the\n'
    '            # zero-action floor on every policy measured 2026-09-02).\n'
    '            "joint_tracking_leg_weight": 1.0,\n'
    '            "joint_tracking_leg_regex": "hip|knee|ankle",\n', 1)],
    "joint_tracking_leg_weight")

T = R / "loco_mjx/loco_mjx/environments/locomotion/urma2/mjx/reward_functions/tracking.py"
patch(T, [(
    '        self.joint_tracking_temperature = reward_config["joint_tracking_temperature"]\n',
    '        self.joint_tracking_temperature = reward_config["joint_tracking_temperature"]\n'
    '        # PER-CHAIN KERNEL (wave 6, F4): per-actuator weights on the joint error.\n'
    '        # Built once from joint names; weight 1.0 everywhere is bit-identical.\n'
    '        import re as _re\n'
    '        _leg_w = float(reward_config.get("joint_tracking_leg_weight", 1.0))\n'
    '        _leg_re = _re.compile(str(reward_config.get("joint_tracking_leg_regex", "hip|knee|ankle")), _re.IGNORECASE)\n'
    '        _names = list(getattr(env, "actuator_joint_names", []))\n'
    '        _w = np.ones(env.nr_actuators, dtype=np.float32)\n'
    '        if _leg_w != 1.0 and _names:\n'
    '            for _i, _n in enumerate(_names):\n'
    '                if _n is not None and _leg_re.search(str(_n)):\n'
    '                    _w[_i] = _leg_w\n'
    '        self.joint_tracking_weights = jnp.array(_w)\n'
    '        self.joint_tracking_weighted = bool(_leg_w != 1.0)\n', 1), (
    '        joint_errors = joint_errors * self.env.actuator_is_joint_transmission\n'
    '        trackable_count = jnp.maximum(jnp.sum(self.env.actuator_is_joint_transmission), 1.0)\n'
    '        mean_squared_joint_error = jnp.sum(jnp.square(joint_errors)) / trackable_count\n',
    '        joint_errors = joint_errors * self.env.actuator_is_joint_transmission\n'
    '        # trackable_count is also used by the qvel term below: keep it unconditional\n'
    '        trackable_count = jnp.maximum(jnp.sum(self.env.actuator_is_joint_transmission), 1.0)\n'
    '        if self.joint_tracking_weighted:\n'
    '            _wmask = self.joint_tracking_weights * self.env.actuator_is_joint_transmission\n'
    '            mean_squared_joint_error = jnp.sum(_wmask * jnp.square(joint_errors)) / jnp.maximum(jnp.sum(_wmask), 1.0)\n'
    '        else:\n'
    '            mean_squared_joint_error = jnp.sum(jnp.square(joint_errors)) / trackable_count\n', 1)],
    "joint_tracking_weighted")
# fix-up for trees patched by the first version (trackable_count only in the else branch)
patch(T, [(
    '        if self.joint_tracking_weighted:\n'
    '            _wmask = self.joint_tracking_weights * self.env.actuator_is_joint_transmission\n'
    '            mean_squared_joint_error = jnp.sum(_wmask * jnp.square(joint_errors)) / jnp.maximum(jnp.sum(_wmask), 1.0)\n'
    '        else:\n'
    '            trackable_count = jnp.maximum(jnp.sum(self.env.actuator_is_joint_transmission), 1.0)\n'
    '            mean_squared_joint_error = jnp.sum(jnp.square(joint_errors)) / trackable_count\n',
    '        # trackable_count is also used by the qvel term below: keep it unconditional\n'
    '        trackable_count = jnp.maximum(jnp.sum(self.env.actuator_is_joint_transmission), 1.0)\n'
    '        if self.joint_tracking_weighted:\n'
    '            _wmask = self.joint_tracking_weights * self.env.actuator_is_joint_transmission\n'
    '            mean_squared_joint_error = jnp.sum(_wmask * jnp.square(joint_errors)) / jnp.maximum(jnp.sum(_wmask), 1.0)\n'
    '        else:\n'
    '            mean_squared_joint_error = jnp.sum(jnp.square(joint_errors)) / trackable_count\n', 1)],
    "keep it unconditional")
print("LEGWEIGHT PATCH COMPLETE")
