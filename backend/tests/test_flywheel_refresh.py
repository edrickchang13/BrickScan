"""Unit tests for the operational flywheel: refresh loop + monitoring metrics.

These exercise the PURE logic and control flow of app/ml/flywheel_refresh.py and
app/ml/flywheel_metrics.py without needing the encoder, sklearn, or a live DB —
the heavy collaborators (ingest_confirmed, gallery_status) are monkeypatched and
the DB is a light fake that mimics the SQLAlchemy query chain the code uses.

The properties under test are the ones that make the loop safe to schedule:
  * idempotency / resumability — used_for_training is the durable cursor
  * graceful degradation — a missing encoder leaves rows un-ingested for retry
  * the crop is NOT re-journaled on replay (journal_crops=False)
  * metrics: confusion-pair ranking, part-number normalisation, time-window volume
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent


# ── light fakes ────────────────────────────────────────────────────────────────
class _Col:
    """Stand-in for a SQLAlchemy Column used at class level (.isnot/.asc/==)."""
    def isnot(self, _x):
        return ("isnot", _x)

    def asc(self):
        return ("asc",)

    def __eq__(self, _x):
        return ("eq", _x)


class FakeFeedback:
    image_path = _Col()
    used_for_training = _Col()
    timestamp = _Col()

    def __init__(self, id, pred, corr, img, used, cid=None, ts=0):
        self.id = id
        self.predicted_part_num = pred
        self.correct_part_num = corr
        self.image_path = img
        self.used_for_training = used
        self.correct_color_id = cid
        self.timestamp = ts


class _Query:
    def __init__(self, rows):
        self._rows = rows
        self._lim = 9999

    def filter(self, *a, **k):
        return self

    def order_by(self, *a, **k):
        return self

    def limit(self, n):
        self._lim = n
        return self

    def all(self):
        eligible = [
            r for r in self._rows
            if r.image_path is not None and r.used_for_training is False
        ]
        eligible.sort(key=lambda r: r.timestamp)
        return eligible[: self._lim]

    def count(self):
        return len([
            r for r in self._rows
            if r.image_path is not None and r.used_for_training is False
        ])


class FakeDB:
    def __init__(self, rows):
        self._rows = rows
        self.committed = 0
        self.rolled_back = 0

    def query(self, _model):
        return _Query(self._rows)

    def commit(self):
        self.committed += 1

    def rollback(self):
        self.rolled_back += 1


def _install_fake_app(monkeypatch, *, ingest_calls, sizes, fail_part="NOENC"):
    """Register fake app.* modules so the function-local imports resolve."""
    for name in ("app", "app.ml", "app.local_inventory"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    models = types.ModuleType("app.local_inventory.models")
    models.ScanFeedback = FakeFeedback
    monkeypatch.setitem(sys.modules, "app.local_inventory.models", models)

    ingest = types.ModuleType("app.ml.flywheel_ingest")

    class _Ingest:
        def __init__(self, embedded, vs, el, notes=None):
            self.embedded = embedded
            self.visual_search_added = vs
            self.embedding_library_added = el
            self.notes = notes or []

    def ingest_confirmed(crop_bytes, part_num, color_id=None, journal_crops=True):
        ingest_calls.append((part_num, color_id, len(crop_bytes), journal_crops))
        if part_num == fail_part:
            return _Ingest(False, False, False,
                           ["encoder not deployed — part gallery append skipped"])
        sizes["v"] += 1
        sizes["e"] += 1
        return _Ingest(True, True, True)

    ingest.ingest_confirmed = ingest_confirmed
    ingest.gallery_status = lambda: {
        "visual_search_size": sizes["v"], "embedding_library_size": sizes["e"],
    }
    monkeypatch.setitem(sys.modules, "app.ml.flywheel_ingest", ingest)


def _load(mod_name, rel_path, monkeypatch, tmp):
    spec = importlib.util.spec_from_file_location(mod_name, str(_BACKEND / rel_path))
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, mod_name, mod)
    spec.loader.exec_module(mod)
    return mod


# ── refresh loop ─────────────────────────────────────────────────────────────
@pytest.fixture
def fake_crops():
    tmp = Path(tempfile.mkdtemp())

    def crop(name):
        p = tmp / name
        p.write_bytes(b"jpeg-" + name.encode())
        return str(p)

    return tmp, crop


def _rows(crop):
    good1, good2, noenc = crop("a.jpg"), crop("b.jpg"), crop("c.jpg")
    missing = str(Path(crop("d.jpg")).parent / "gone.jpg")
    Path(missing).unlink(missing_ok=True)  # ensure it's absent
    return [
        FakeFeedback("r1", "3002", "3001", good1, False, "4", 1),   # correction → appends
        FakeFeedback("r2", "3001", "3001", good2, False, "5", 2),   # agreement → appends
        FakeFeedback("r3", "x", "NOENC", noenc, False, None, 3),    # appends nothing (encoder)
        FakeFeedback("r4", "y", "unknown", good1, False, None, 4),  # skip: no part
        FakeFeedback("r5", "z", "3009", missing, False, None, 5),   # skip: no image file
        FakeFeedback("r6", "q", "3010", None, False, None, 6),      # not eligible: no path
        FakeFeedback("r7", "w", "3011", good2, True, None, 7),      # already used
    ]


def test_refresh_dry_run_appends_nothing(monkeypatch, fake_crops):
    tmp, crop = fake_crops
    calls, sizes = [], {"v": 10, "e": 5}
    _install_fake_app(monkeypatch, ingest_calls=calls, sizes=sizes)
    fr = _load("app.ml.flywheel_refresh", "app/ml/flywheel_refresh.py", monkeypatch, tmp)
    monkeypatch.setattr(fr, "REFRESH_STATE_DIR", tmp / "st")
    monkeypatch.setattr(fr, "REFRESH_STATE_FILE", tmp / "st" / "last.json")

    res = fr.refresh_galleries(FakeDB(_rows(crop)), dry_run=True)
    assert res.candidates == 5
    assert res.ingested == 0
    assert calls == []


def test_refresh_is_idempotent_and_resumable(monkeypatch, fake_crops):
    tmp, crop = fake_crops
    calls, sizes = [], {"v": 10, "e": 5}
    _install_fake_app(monkeypatch, ingest_calls=calls, sizes=sizes)
    fr = _load("app.ml.flywheel_refresh", "app/ml/flywheel_refresh.py", monkeypatch, tmp)
    monkeypatch.setattr(fr, "REFRESH_STATE_DIR", tmp / "st")
    monkeypatch.setattr(fr, "REFRESH_STATE_FILE", tmp / "st" / "last.json")

    rows = _rows(crop)
    db = FakeDB(rows)
    size_before = sizes["v"]

    res = fr.refresh_galleries(db)
    assert (res.candidates, res.ingested, res.embedded) == (5, 2, 2)
    assert res.marked_used == 2
    assert res.skipped_no_part == 1
    assert res.skipped_no_image == 1
    assert db.committed == 1
    assert res.visual_search_size_before == size_before
    assert res.visual_search_size_after == size_before + 2
    # crop already on disk — replay must NOT re-journal it
    assert all(c[3] is False for c in calls)
    # the appended rows are marked; the encoder-less row is left for retry
    assert rows[0].used_for_training is True
    assert rows[1].used_for_training is True
    assert rows[2].used_for_training is False
    assert any("encoder not deployed" in n for n in res.notes)

    # Re-run: the two used rows are skipped; nothing new ingested or committed.
    calls.clear()
    res2 = fr.refresh_galleries(db)
    assert (res2.candidates, res2.ingested, res2.marked_used) == (3, 0, 0)
    assert db.committed == 1  # no second commit


def test_refresh_only_corrections_skips_agreements(monkeypatch, fake_crops):
    tmp, crop = fake_crops
    calls, sizes = [], {"v": 0, "e": 0}
    _install_fake_app(monkeypatch, ingest_calls=calls, sizes=sizes)
    fr = _load("app.ml.flywheel_refresh", "app/ml/flywheel_refresh.py", monkeypatch, tmp)
    monkeypatch.setattr(fr, "REFRESH_STATE_DIR", tmp / "st")
    monkeypatch.setattr(fr, "REFRESH_STATE_FILE", tmp / "st" / "last.json")

    fr.refresh_galleries(FakeDB(_rows(crop)), only_corrections=True)
    parts = [c[0] for c in calls]
    # r1 (3002→3001) is a correction; r2 (3001→3001) is an agreement and must be skipped.
    assert parts.count("3001") == 1
    assert "NOENC" in parts  # r3 (x→NOENC) is a correction too


def test_heavy_refresh_status_graceful_when_absent(monkeypatch, fake_crops):
    tmp, _ = fake_crops
    calls, sizes = [], {"v": 0, "e": 0}
    _install_fake_app(monkeypatch, ingest_calls=calls, sizes=sizes)
    fr = _load("app.ml.flywheel_refresh", "app/ml/flywheel_refresh.py", monkeypatch, tmp)
    monkeypatch.setattr(fr, "HEAVY_REFRESH_SCRIPT", tmp / "does_not_exist.sh")
    status = fr.heavy_refresh_status()
    assert status["available"] is False
    assert "no-op" in status["note"]


# ── metrics ──────────────────────────────────────────────────────────────────
class _MRow:
    def __init__(self, pred, corr, cid=None, ts=None):
        self.predicted_part_num = pred
        self.correct_part_num = corr
        self.correct_color_id = cid
        self.timestamp = ts


def _load_metrics(monkeypatch):
    tmp = Path(tempfile.mkdtemp())
    return _load("app.ml.flywheel_metrics", "app/ml/flywheel_metrics.py", monkeypatch, tmp)


def test_metrics_confusion_pairs_and_coverage(monkeypatch):
    fm = _load_metrics(monkeypatch)
    now = datetime.now(timezone.utc)
    rows = [
        _MRow("3002", "3001", "4", now),
        _MRow("3002", "3001", "4", now),       # same confusion twice
        _MRow("3001", "3001", "5", now),        # agreement, new colour
        _MRow("99", "3005", None, now),         # confusion, no colour
        _MRow("0003001", "3001", None, now),    # leading zeros normalise to 3001
    ]
    pairs, cov = fm._confusion_pairs_and_coverage(rows)
    assert pairs[0] == {"predicted_part_num": "3002", "correct_part_num": "3001", "count": 2}
    assert cov == {"parts_with_feedback": 2, "colors_with_feedback": 2}  # {3001,3005}, {4,5}


def test_metrics_recent_volume_windows(monkeypatch):
    fm = _load_metrics(monkeypatch)
    now = datetime.now(timezone.utc)
    rows = [
        _MRow("a", "a", None, now),                       # 24h
        _MRow("a", "a", None, now - timedelta(hours=2)),  # 24h
        _MRow("a", "a", None, now - timedelta(days=3)),   # 7d
        _MRow("a", "a", None, now - timedelta(days=10)),  # 30d
        _MRow("a", "a", None, now - timedelta(days=40)),  # outside all windows
        _MRow("a", "a", None, None),                       # missing ts → total only
    ]
    vol = fm._recent_volume(rows, now=now)
    assert vol == {"last_24h": 2, "last_7d": 3, "last_30d": 4, "total": 6}


# ── DB-backed integration: scheduled refresh against the real SQLite schema ──────
#
# Proves the operational loop works end-to-end against the actual ScanFeedback /
# FeedbackEvalSnapshot tables (the same schema /flywheel/confirm writes): insert
# confirmed rows with crops on disk, monkeypatch ONLY the frozen-encoder gallery
# hot path (so no ONNX is needed), run refresh_galleries, and assert the rows are
# folded in, the used_for_training cursor flips, and a re-run is a no-op.
@pytest.fixture()
def fresh_local_db(monkeypatch, tmp_path):
    """Re-init the local_inventory SQLite DB against a fresh temp file."""
    import os
    monkeypatch.setenv("HOME", str(tmp_path))
    target = tmp_path / "brickscan_inventory.db"
    if target.exists():
        target.unlink()
    import importlib
    from app.local_inventory import database as ldb
    importlib.reload(ldb)
    ldb.init_db()
    yield ldb


def test_refresh_against_real_db_schema(monkeypatch, fresh_local_db, tmp_path):
    from app.local_inventory.models import ScanFeedback

    # A real crop on disk that the refresh reads back.
    crop = tmp_path / "crop1.jpg"
    crop.write_bytes(b"\xff\xd8\xff fake jpeg bytes")

    db = fresh_local_db.SessionLocal()
    try:
        db.add(ScanFeedback(
            scan_id="scan_it_1", predicted_part_num="3002",
            correct_part_num="3001", correct_color_id="4",
            image_path=str(crop), confidence=0.5, source="flywheel_confirm",
            used_for_training=False, feedback_type="none_correct",
        ))
        # A row with no crop on disk — must be skipped (skip_no_image), cursor unchanged.
        db.add(ScanFeedback(
            scan_id="scan_it_2", predicted_part_num="x",
            correct_part_num="3009", correct_color_id=None,
            image_path=str(tmp_path / "missing.jpg"), confidence=0.5,
            source="scan-feedback", used_for_training=False,
        ))
        db.commit()

        # Monkeypatch the encoder hot path so no ONNX model is required.
        import app.ml.flywheel_ingest as fi
        appended = []

        def fake_ingest(crop_bytes, part_num, color_id=None, journal_crops=True):
            appended.append((part_num, color_id, journal_crops))
            return fi.IngestResult(
                part_num=part_num, color_id=color_id, embedded=True,
                visual_search_added=True, embedding_library_added=True,
            )

        import app.ml.flywheel_refresh as fr
        monkeypatch.setattr(fr, "ingest_confirmed", fake_ingest, raising=True)
        monkeypatch.setattr(
            "app.ml.flywheel_ingest.gallery_status",
            lambda: {"visual_search_size": 0, "embedding_library_size": 0},
        )
        monkeypatch.setattr(fr, "REFRESH_STATE_DIR", tmp_path / "state")
        monkeypatch.setattr(fr, "REFRESH_STATE_FILE", tmp_path / "state" / "last.json")

        res = fr.refresh_galleries(db, limit=100)
        assert res.candidates == 2
        assert res.ingested == 1            # only the row with a real crop
        assert res.skipped_no_image == 1
        assert res.marked_used == 1
        assert appended == [("3001", "4", False)]  # crop already on disk → no re-journal

        # Cursor flipped in the DB → re-run finds nothing to do.
        res2 = fr.refresh_galleries(db, limit=100)
        assert res2.candidates == 1         # only the missing-crop row remains eligible
        assert res2.ingested == 0
    finally:
        db.close()
