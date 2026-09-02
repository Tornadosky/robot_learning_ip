"""The three contact-quality terms must be APPLIED ONCE, not twice.

`tracking_post_contact_penalties=True` stashes foot_slip + ground_penetration +
feet_orientation for TrackingReward to re-apply past its clip, so that contact
quality can compete with the imitation bonuses instead of being clipped away
before they are added. Until 2026-08-29 the same three were ALSO left inside
`reward_penalty`, so the flag charged them twice. Measured cost of the extra
copy on H1: +17.2 % tracking error and +62.7 % ankle error, bought against
-60 % penetration (REPORT Sec 12.2) -- and the whole of that cost is cancelled by
re-dosing the contact weights (Sec 13.3), which is what a double count looks like.

This is a STRUCTURAL test, deliberately. The numeric path needs a compiled MJX
environment with a clip, which is a training run, not a unit test; what a unit
test can do is pin the arithmetic so the count cannot silently go back to two.
The behavioural confirmation is a wave-7 arm at a deliberate weight.
"""
import ast
from pathlib import Path

REWARDS = (Path(__file__).resolve().parents[1]
           / "loco_mjx/loco_mjx/environments/locomotion/urma2/mjx/reward_functions")
CONTACT_TERMS = ("foot_slip_reward", "ground_penetration_reward",
                 "feet_orientation_reward")


def _reward_body(path, class_name):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "reward_and_info":
                    return item
    raise AssertionError(f"{class_name}.reward_and_info not found in {path.name}")


def _assignments(fn, target):
    """Every RHS assigned to `target`, as source-ish text."""
    out = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == target:
                    out.append(ast.dump(node.value))
    return out


def test_contact_terms_are_not_inside_reward_penalty():
    """The unconditional penalty sum must no longer name the three terms."""
    fn = _reward_body(REWARDS / "default.py", "DefaultReward")
    assigns = _assignments(fn, "reward_penalty")
    assert assigns, "reward_penalty is not assigned -- the reward was restructured"
    base = assigns[0]
    for term in CONTACT_TERMS:
        assert term not in base, (
            f"{term} is back inside the unconditional reward_penalty sum; with "
            "tracking_post_contact_penalties=True it would be charged twice")


def test_contact_terms_are_added_back_when_the_move_is_off():
    """With the move off, the terms must still be charged -- exactly once."""
    fn = _reward_body(REWARDS / "default.py", "DefaultReward")
    guarded = [
        node for node in ast.walk(fn)
        if isinstance(node, ast.If)
        and "move_contact_quality_past_clip" in ast.dump(node.test)
        and isinstance(node.test, ast.UnaryOp)          # `if not ...`
    ]
    assert len(guarded) == 1, (
        "expected exactly one `if not move_contact_quality_past_clip:` branch "
        "restoring the contact terms to reward_penalty")
    assert "contact_quality_reward" in ast.dump(guarded[0])


def test_tracking_applies_the_stash_only_under_its_flag():
    """TrackingReward is the second application site and must stay conditional."""
    fn = _reward_body(REWARDS / "tracking.py", "TrackingReward")
    conditional = [
        node for node in ast.walk(fn)
        if isinstance(node, ast.If)
        and "post_contact_penalties" in ast.dump(node.test)
        and "tracking_post_contact_penalty" in ast.dump(node)
    ]
    assert len(conditional) == 1, (
        "the post-contact stash must be applied inside exactly one "
        "`if self.post_contact_penalties:` branch")


def test_the_stash_holds_exactly_the_three_terms():
    fn = _reward_body(REWARDS / "default.py", "DefaultReward")
    dumped = [d for d in _assignments(fn, "contact_quality_reward")]
    assert len(dumped) == 1, "contact_quality_reward must be assigned exactly once"
    for term in CONTACT_TERMS:
        assert term in dumped[0], f"{term} missing from the post-contact stash"
