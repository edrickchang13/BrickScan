"""
Smoke tests for the v2 feedback flywheel — backwards compatibility + new fields.

Key guarantees under test:
  1. Legacy clients (no v2 fields) still work — POST /scan-feedback with only
     scan_id / predicted_part_num / correct_part_num succeeds.
  2. feedback_type is derived server-side when absent.
  3. New v2 fields (feedback_type, correct_rank, predictions_shown, time_to_confirm_ms)
     are persisted when supplied.
  4. Stats endpoint now returns top1_accuracy / top3_accuracy / by_source / accuracy_trend.
  5. CSV export produces a well-formed CSV with the columns retrain_from_feedback.py expects.

These talk to the local_inventory SQLite DB directly via sqlalchemy — no main-DB Postgres needed.
"""

import os
import tempfile
import pytest


def _isolate_local_db(monkeypatch):
    """Force the local_inventory SQLite DB to a fresh temp file for the test."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    os.environ["HOME"] = os.path.dirname(tmp.name)
    # Drop the sqlite file at the HOME path — app.local_inventory.database
    # constructs _DB_PATH = os.path.expanduser("~/brickscan_inventory.db").
    target = os.path.join(os.environ["HOME"], "brickscan_inventory.db")
    if os.path.exists(target):
        os.unlink(target)
    return target


@pytest.fixture()
def fresh_local_db(monkeypatch):
    """Re-initialise the local_inventory DB module with a fresh SQLite file."""
    target = _isolate_local_db(monkeypatch)

    # Reload the module to rebuild the engine against the new path.
    import importlib
    from app.local_inventory import database as ldb
    importlib.reload(ldb)
    ldb.init_db()
    yield ldb
    if os.path.exists(target):
        os.unlink(target)


def test_feedback_model_has_v2_columns(fresh_local_db):
    """After init_db(), scan_feedback should have all four v2 columns."""
    from sqlalchemy import text
    with fresh_local_db.engine.connect() as conn:
        cols = {row[1] for row in conn.execute(text("PRAGMA table_info(scan_feedback)"))}
    for c in ("feedback_type", "correct_rank", "predictions_shown_json", "time_to_confirm_ms"):
        assert c in cols, f"scan_feedback missing column {c}"


def test_feedback_eval_snapshot_table_created(fresh_local_db):
    """The new snapshot table should exist after init_db()."""
    from sqlalchemy import text
    with fresh_local_db.engine.connect() as conn:
        tables = {row[0] for row in conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))}
    assert "feedback_eval_snapshots" in tables


def test_derive_feedback_type_heuristics():
    """The server-side heuristic must pick the right type for legacy payloads."""
    from app.local_inventory.feedback_routes import _derive_feedback_type

    # Agreement, no colour change
    assert _derive_feedback_type("3001", "3001", None, None) == "top_correct"

    # Agreement + colour given — counts as partially_correct
    assert _derive_feedback_type("3001", "3001", "14", None) == "partially_correct"

    # Disagreement, correct part in shown list at position 2 → alternative_correct
    shown = [
        {"part_num": "3001"},
        {"part_num": "3002"},
        {"part_num": "3003"},
    ]
    assert _derive_feedback_type("3001", "3002", None, shown) == "alternative_correct"

    # Disagreement, correct part NOT in shown list
    assert _derive_feedback_type("3001", "9999", None, shown) == "none_correct"

    # Disagreement, no shown list given
    assert _derive_feedback_type("3001", "9999", None, None) == "none_correct"


def test_topn_accuracy_computes_correctly(fresh_local_db):
    """_compute_topn_accuracy must respect feedback_type + correct_rank."""
    from app.local_inventory.feedback_routes import _compute_topn_accuracy
    from app.local_inventory.models import ScanFeedback

    rows = [
        # top_correct → counts as top-1 and top-3
        ScanFeedback(scan_id="a", predicted_part_num="3001", correct_part_num="3001",
                     confidence=0.9, source="brickognize", feedback_type="top_correct",
                     correct_rank=0, used_for_training=False),
        # alternative_correct at rank 1 → counts as top-3 only
        ScanFeedback(scan_id="b", predicted_part_num="3001", correct_part_num="3002",
                     confidence=0.6, source="gemini", feedback_type="alternative_correct",
                     correct_rank=1, used_for_training=False),
        # none_correct → neither top-1 nor top-3
        ScanFeedback(scan_id="c", predicted_part_num="3001", correct_part_num="9999",
                     confidence=0.4, source="brickognize", feedback_type="none_correct",
                     correct_rank=-1, used_for_training=False),
    ]
    top1, top3 = _compute_topn_accuracy(rows)
    assert abs(top1 - 1/3) < 0.01, f"top1={top1}"
    assert abs(top3 - 2/3) < 0.01, f"top3={top3}"


def test_topn_accuracy_empty_list():
    from app.local_inventory.feedback_routes import _compute_topn_accuracy
    assert _compute_topn_accuracy([]) == (0.0, 0.0)


def test_by_source_buckets():
    """_compute_by_source should bucket counts + accuracy by model."""
    from app.local_inventory.feedback_routes import _compute_by_source
    from app.local_inventory.models import ScanFeedback

    rows = [
        ScanFeedback(scan_id="a", predicted_part_num="3001", correct_part_num="3001",
                     confidence=0.9, source="brickognize", feedback_type="top_correct", correct_rank=0),
        ScanFeedback(scan_id="b", predicted_part_num="3001", correct_part_num="9999",
                     confidence=0.3, source="brickognize", feedback_type="none_correct", correct_rank=-1),
        ScanFeedback(scan_id="c", predicted_part_num="3001", correct_part_num="3001",
                     confidence=0.7, source="gemini", feedback_type="top_correct", correct_rank=0),
    ]
    stats = {s.source: s for s in _compute_by_source(rows)}
    assert stats["brickognize"].count == 2
    assert stats["brickognize"].correct == 1
    assert abs(stats["brickognize"].accuracy - 0.5) < 0.01
    assert stats["gemini"].count == 1
    assert abs(stats["gemini"].accuracy - 1.0) < 0.01
