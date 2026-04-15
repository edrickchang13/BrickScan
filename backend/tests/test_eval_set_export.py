"""
Regression tests for GET /api/local-inventory/feedback/eval-set.json.

Guarantees:
  1. Endpoint exists and returns JSON (not CSV or HTML).
  2. Only rows with feedback_type IN ('top_correct','alternative_correct','none_correct')
     AND image_path IS NOT NULL are returned.
  3. partially_correct rows are excluded (part ID was right, only colour wrong).
  4. Schema fields are stable (image_path, correct_part_num, feedback_type, correct_rank, etc.).
  5. `limit` query param caps the result count.
  6. Newest rows come first (ORDER BY timestamp DESC).

Uses the same isolated-SQLite-DB pattern as test_feedback_v2.py.
"""

import json
import os
import tempfile
from datetime import datetime, timedelta, timezone

import pytest


@pytest.fixture()
def fresh_local_db():
    """
    Ensure each test starts from an empty scan_feedback table. The
    local_inventory SQLite engine is a module-level singleton — reloading
    the database module only updates database.engine, not references
    already imported into models/feedback_routes/etc — so we clear rows
    rather than try to swap the file.
    """
    from app.local_inventory import database as ldb
    from app.local_inventory.models import ScanFeedback, FeedbackEvalSnapshot
    ldb.init_db()
    session = ldb.SessionLocal()
    try:
        session.query(ScanFeedback).delete()
        session.query(FeedbackEvalSnapshot).delete()
        session.commit()
    finally:
        session.close()
    yield ldb
    # Clean up after so later tests start clean too
    session = ldb.SessionLocal()
    try:
        session.query(ScanFeedback).delete()
        session.query(FeedbackEvalSnapshot).delete()
        session.commit()
    finally:
        session.close()


def _seed_feedback(ldb, rows):
    """Insert ScanFeedback rows into the fresh local DB."""
    from app.local_inventory.models import ScanFeedback
    session = ldb.SessionLocal()
    try:
        for r in rows:
            session.add(ScanFeedback(**r))
        session.commit()
    finally:
        session.close()


def _call_endpoint(limit=500):
    """Import and call the endpoint handler directly — avoids needing httpx."""
    import asyncio
    from app.local_inventory.feedback_routes import export_eval_set
    from app.local_inventory import database as ldb
    session = ldb.SessionLocal()
    try:
        resp = asyncio.run(export_eval_set(limit=limit, include_used=False, db=session))
    finally:
        session.close()
    # resp is a starlette Response; parse its body
    return json.loads(resp.body.decode())


def test_endpoint_returns_json_array(fresh_local_db):
    """Empty DB → empty JSON array (not 500)."""
    result = _call_endpoint()
    assert result == []


def test_only_valid_feedback_types_returned(fresh_local_db):
    """partially_correct + legacy (feedback_type=None) rows must be excluded."""
    now = datetime.now(timezone.utc)
    _seed_feedback(fresh_local_db, [
        # Included
        dict(scan_id="a", predicted_part_num="3001", correct_part_num="3001",
             image_path="/tmp/a.jpg", confidence=0.9, source="brickognize",
             feedback_type="top_correct", correct_rank=0, timestamp=now),
        dict(scan_id="b", predicted_part_num="3001", correct_part_num="3002",
             image_path="/tmp/b.jpg", confidence=0.6, source="gemini",
             feedback_type="alternative_correct", correct_rank=1, timestamp=now),
        dict(scan_id="c", predicted_part_num="3001", correct_part_num="9999",
             image_path="/tmp/c.jpg", confidence=0.4, source="brickognize",
             feedback_type="none_correct", correct_rank=-1, timestamp=now),
        # Excluded: partially_correct
        dict(scan_id="d", predicted_part_num="3001", correct_part_num="3001",
             image_path="/tmp/d.jpg", confidence=0.9, source="brickognize",
             feedback_type="partially_correct", correct_rank=0, timestamp=now),
        # Excluded: no image_path
        dict(scan_id="e", predicted_part_num="3001", correct_part_num="3001",
             image_path=None, confidence=0.9, source="brickognize",
             feedback_type="top_correct", correct_rank=0, timestamp=now),
        # Excluded: legacy feedback_type=None
        dict(scan_id="f", predicted_part_num="3001", correct_part_num="3001",
             image_path="/tmp/f.jpg", confidence=0.9, source="brickognize",
             feedback_type=None, correct_rank=None, timestamp=now),
    ])
    result = _call_endpoint()
    scan_ids = {r["scan_id"] for r in result}
    assert scan_ids == {"a", "b", "c"}, f"expected only a,b,c — got {scan_ids}"


def test_schema_stable(fresh_local_db):
    """Every row should have the documented fields."""
    now = datetime.now(timezone.utc)
    _seed_feedback(fresh_local_db, [
        dict(scan_id="a", predicted_part_num="3001", correct_part_num="3001",
             correct_color_id="4", image_path="/tmp/a.jpg", confidence=0.9,
             source="brickognize", feedback_type="top_correct", correct_rank=0,
             timestamp=now),
    ])
    result = _call_endpoint()
    assert len(result) == 1
    row = result[0]
    for field in ("image_path", "correct_part_num", "correct_color_id",
                  "original_prediction", "source", "confidence", "timestamp",
                  "scan_id", "feedback_type", "correct_rank", "predictions_shown"):
        assert field in row, f"missing field: {field}"
    assert row["feedback_type"] == "top_correct"
    assert row["correct_rank"] == 0
    assert row["correct_color_id"] == "4"


def test_limit_param_caps_results(fresh_local_db):
    now = datetime.now(timezone.utc)
    _seed_feedback(fresh_local_db, [
        dict(scan_id=f"s{i}", predicted_part_num="3001", correct_part_num="3001",
             image_path=f"/tmp/{i}.jpg", confidence=0.9, source="brickognize",
             feedback_type="top_correct", correct_rank=0,
             timestamp=now - timedelta(minutes=i))
        for i in range(10)
    ])
    result = _call_endpoint(limit=3)
    assert len(result) == 3


def test_ordering_newest_first(fresh_local_db):
    now = datetime.now(timezone.utc)
    _seed_feedback(fresh_local_db, [
        dict(scan_id="old", predicted_part_num="3001", correct_part_num="3001",
             image_path="/tmp/old.jpg", confidence=0.9, source="brickognize",
             feedback_type="top_correct", correct_rank=0,
             timestamp=now - timedelta(days=2)),
        dict(scan_id="new", predicted_part_num="3001", correct_part_num="3001",
             image_path="/tmp/new.jpg", confidence=0.9, source="brickognize",
             feedback_type="top_correct", correct_rank=0,
             timestamp=now),
    ])
    result = _call_endpoint()
    assert result[0]["scan_id"] == "new"
    assert result[1]["scan_id"] == "old"
