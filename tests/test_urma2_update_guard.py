from pathlib import Path
import importlib.util
import jax.numpy as jnp
import numpy as np

PATH = Path(__file__).resolve().parents[1] / "loco_mjx/loco_mjx/algorithms/urma2/mjx/update_guard.py"
spec = importlib.util.spec_from_file_location("urma2_update_guard", PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
select_policy_state = module.select_policy_state


def value(tree):
    return float(np.asarray(tree["w"]))


def test_accepted_update_selects_candidate():
    selected, rejected = select_policy_state({"w": jnp.array(1.)}, {"w": jnp.array(2.)},
                                             jnp.array([.01, .02]), .15)
    assert value(selected) == 2
    assert float(np.asarray(rejected)) == 0


def test_high_or_nonfinite_kl_keeps_current():
    current, candidate = {"w": jnp.array(1.)}, {"w": jnp.array(2.)}
    for kl in (jnp.array([.2]), jnp.array([jnp.nan])):
        selected, rejected = select_policy_state(current, candidate, kl, .15)
        assert value(selected) == 1
        assert float(np.asarray(rejected)) == 1


def test_zero_kl_is_not_rejected():
    selected, rejected = select_policy_state({"w": jnp.array(1.)}, {"w": jnp.array(3.)},
                                             jnp.zeros(8), .15)
    assert value(selected) == 3
    assert float(np.asarray(rejected)) == 0
