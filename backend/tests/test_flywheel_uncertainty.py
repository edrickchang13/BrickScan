"""Unit tests for the flywheel review-queue trigger (should_flag_for_review).

The margin signal decides which scans get surfaced for human review. The two
properties that matter operationally:
  * a confident, well-separated scan is NOT flagged (no review-queue spam);
  * the margin is measured to the next genuinely DIFFERENT part — mold/print
    variants of the same brick (3001 vs 3001a vs 3001pr0001) are collapsed first
    (FLYWHEEL.md §1), so a confident brick whose runner-up is just a variant of
    itself is not needlessly flagged.
"""
from __future__ import annotations

from app.local_inventory.flywheel_routes import should_flag_for_review


def _preds(*pairs):
    return [{"part_num": p, "confidence": c} for p, c in pairs]


def test_confident_and_separated_not_flagged():
    v = should_flag_for_review(_preds(("3001", 0.90), ("3002", 0.40)))
    assert v["flag"] is False
    assert v["reason"] is None


def test_low_margin_flagged():
    v = should_flag_for_review(_preds(("3001", 0.82), ("3002", 0.80)))
    assert v["flag"] is True
    assert v["reason"] == "low_margin"
    assert abs(v["margin"] - 0.02) < 1e-6


def test_low_absolute_confidence_flagged_even_when_separated():
    v = should_flag_for_review(_preds(("3001", 0.40), ("3002", 0.10)))
    assert v["flag"] is True
    assert v["reason"] == "low_confidence"


def test_mold_variant_runner_up_not_flagged():
    # 3001 vs 3001a are the same base mold → margin is measured to 3002 (0.60),
    # so a confidently-correct brick is NOT flagged just because a variant of
    # itself ranks second.
    v = should_flag_for_review(
        _preds(("3001", 0.90), ("3001a", 0.89), ("3002", 0.30))
    )
    assert v["flag"] is False
    assert abs(v["margin"] - 0.60) < 1e-6


def test_print_variant_runner_up_not_flagged():
    v = should_flag_for_review(
        _preds(("3001", 0.90), ("3001pr0001", 0.88), ("9999", 0.20))
    )
    assert v["flag"] is False


def test_only_variants_of_one_part_has_no_margin():
    # No DIFFERENT base part anywhere → margin is None; confident → not flagged.
    v = should_flag_for_review(_preds(("3001", 0.95), ("3001a", 0.90)))
    assert v["flag"] is False
    assert v["margin"] is None


def test_empty_predictions_not_flagged():
    v = should_flag_for_review([])
    assert v["flag"] is False
    assert v["reason"] == "no_predictions"
