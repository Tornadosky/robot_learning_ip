"""The config guard must catch the bugs it was built for.

A guard that silently stops detecting is worse than no guard, because it is
reported as a clean bill of health. It already happened once: restricting the
flag parser to lines STARTING with `--` made it miss the sbatch's
`EXTRA_ARGS_ARR=--a=1 --b=2` line, which is exactly where the design-B arms pass
their latent flags -- 19 arms were flagged as broken when the fault was the tool.

Each test reproduces one historical bug verbatim from a real training log and a
real crosseval artifact, so the checks are pinned against the failures they exist
for, not against synthetic fixtures.
"""
import copy
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FSQ = ROOT / "experiments/fsq_khaendler"
sys.path.insert(0, str(FSQ))

LOG = FSQ / "_guard_logs/e1_ctl_s1.out"
EVAL = FSQ / "wave4_controls/e1_ctl_s1__s0.json"
pytestmark = pytest.mark.skipif(
    not (LOG.exists() and EVAL.exists()),
    reason="needs a fetched training log + crosseval (scripts/scaling/fetch_guard_logs.sh)")


def _run(cfg, series, evals):
    from pipeline_guard import check_dead_terms, check_train_eval, load_defaults
    buf = io.StringIO()
    with redirect_stdout(buf):
        n = check_dead_terms(cfg, series)
        n += check_train_eval({**load_defaults(), **cfg}, evals)
    return n, buf.getvalue()


def _log(tmp_path, replace=None):
    from pipeline_guard import parse_log
    src = LOG.read_text(encoding="utf-8", errors="replace")
    if replace:
        assert replace[0] in src, "anchor missing -- the fixture log changed"
        src = src.replace(*replace)
    p = tmp_path / "log.out"
    p.write_text(src, encoding="utf-8")
    return parse_log(p)


def _eval(tmp_path, **overrides):
    d = json.loads(EVAL.read_text())
    d = copy.deepcopy(d)
    d["eval_condition"].update(overrides)
    p = tmp_path / "eval.json"
    p.write_text(json.dumps(d))
    return [str(p)]


def test_real_arm_is_clean(tmp_path):
    """The negative control. A guard that flags everything is useless."""
    cfg, series = _log(tmp_path)
    n, out = _run(cfg, series, [str(EVAL)])
    assert n == 0, out


def test_catches_a_dead_reward_term(tmp_path):
    """Bug 2: root_heading_tracking_reward was exactly 0 for a whole campaign."""
    cfg, series = _log(tmp_path, (
        "--environment.reward.root_heading_tracking_weight_ratio=0.0",
        "--environment.reward.root_heading_tracking_weight_ratio=0.20"))
    n, out = _run(cfg, series, [])
    assert n >= 1 and "IDENTICALLY ZERO" in out, out


def test_catches_refbias_train_eval_mismatch(tmp_path):
    """Bug 7: CE_REFBIAS defaulted to 1.0 while the arm trained at 0.0."""
    cfg, series = _log(tmp_path)
    n, out = _run(cfg, series, _eval(tmp_path, refbias=1.0))
    assert n >= 1 and "refbias" in out, out


def test_catches_observation_width_mismatch_against_the_default(tmp_path):
    """Bug 8: the eval asks for heading obs the arm never trained with.

    The training log does not mention the flag at all, so this only works if the
    guard resolves absent flags to their default_config value rather than
    skipping them.
    """
    cfg, series = _log(tmp_path)
    n, out = _run(cfg, series, _eval(tmp_path, root_heading_obs="True"))
    assert n >= 1 and "root_heading_obs" in out, out


def test_parser_reads_flags_from_the_extra_args_line(tmp_path):
    """The regression that made the guard cry wolf on 19 design-B arms."""
    from pipeline_guard import parse_log
    p = tmp_path / "x.out"
    p.write_text("EXTRA_ARGS_ARR=--environment.command.tracking_clip_latent_obs=True "
                 "--environment.command.tracking_clip_latent_dim=4\n", encoding="utf-8")
    cfg, _ = parse_log(p)
    assert cfg.get("tracking_clip_latent_obs") == "True"
    assert cfg.get("tracking_clip_latent_dim") == "4"
