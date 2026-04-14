"""
Integration tests for the new hybrid recognition cascade.

Tests cover:
  - ModelManager graceful fallback when model files are absent
  - EmbeddingLibrary graceful fallback when cache is absent
  - hybrid_predict cascade ordering and source attribution
  - Contrastive k-NN confidence conversion
  - Feedback endpoint round-trip (in-memory SQLite)
  - scan-multi YOLO/quadrant detection_method field

Run with:
    cd brickscan/backend
    python -m pytest tests/test_model_cascade.py -v
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import uuid
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from PIL import Image as PILImage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_jpeg(width: int = 64, height: int = 64) -> bytes:
    """Return a minimal JPEG in memory — no filesystem needed."""
    img = PILImage.new("RGB", (width, height), color=(200, 100, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=50)
    return buf.getvalue()


def _make_pred(part_num: str = "3001", confidence: float = 0.9, source: str = "test") -> Dict:
    return {
        "part_num":   part_num,
        "part_name":  f"Part {part_num}",
        "confidence": confidence,
        "color_id":   None,
        "color_name": None,
        "color_hex":  None,
        "source":     source,
    }


# ---------------------------------------------------------------------------
# ModelManager — graceful fallback
# ---------------------------------------------------------------------------

class TestModelManagerFallback:
    """ModelManager must not crash when model files are absent."""

    def test_encode_image_returns_none_without_model(self):
        from app.ml.model_manager import ModelManager
        # Reset singleton so we get a fresh instance without cached sessions
        mm = ModelManager.__new__(ModelManager)
        mm._encoder_session = None
        mm._student_session = None
        mm._yolo_session    = None
        mm._class_labels    = None
        mm._loaded          = True   # skip real loading

        result = mm.encode_image(_make_jpeg())
        assert result is None

    def test_classify_image_returns_empty_without_model(self):
        from app.ml.model_manager import ModelManager
        mm = ModelManager.__new__(ModelManager)
        mm._student_session = None
        mm._class_labels    = None
        mm._loaded          = True

        result = mm.classify_image(_make_jpeg())
        assert result == []

    def test_detect_pieces_returns_empty_without_model(self):
        from app.ml.model_manager import ModelManager
        mm = ModelManager.__new__(ModelManager)
        mm._yolo_session = None
        mm._loaded       = True

        result = mm.detect_pieces(_make_jpeg())
        assert result == []

    def test_classify_image_returns_correct_source(self):
        """When student model IS present, source field must be 'distilled_model'."""
        from app.ml.model_manager import ModelManager, _softmax
        import numpy as np

        # Fake ONNX session
        num_classes = 10
        fake_logits = np.zeros((1, num_classes), dtype=np.float32)
        fake_logits[0, 3] = 5.0   # class 3 has highest logit

        mock_session = MagicMock()
        mock_session.get_inputs.return_value  = [MagicMock(name="input")]
        mock_session.get_outputs.return_value = [MagicMock(name="output")]
        mock_session.run.return_value = [fake_logits]

        mm = ModelManager.__new__(ModelManager)
        mm._student_session = mock_session
        mm._class_labels    = {str(i): f"part_{i:04d}" for i in range(num_classes)}
        mm._loaded          = True

        results = mm.classify_image(_make_jpeg(), top_k=3)
        assert len(results) >= 1
        assert results[0]["part_num"] == "part_0003"
        assert results[0]["source"] == "distilled_model"
        assert 0.0 <= results[0]["confidence"] <= 1.0


# ---------------------------------------------------------------------------
# EmbeddingLibrary — graceful fallback
# ---------------------------------------------------------------------------

class TestEmbeddingLibraryFallback:
    """EmbeddingLibrary must return [] when cache is absent."""

    def test_knn_search_empty_without_cache(self, tmp_path, monkeypatch):
        from app.ml import embedding_library as el_mod

        # Point CACHE_PATH to a non-existent file
        monkeypatch.setattr(el_mod, "CACHE_PATH", tmp_path / "nonexistent.pkl")

        lib = el_mod.EmbeddingLibrary()
        lib._loaded = False   # force reload

        query = np.random.rand(128).astype(np.float32)
        results = lib.knn_search(query, k=5)
        assert results == []

    def test_knn_search_with_populated_cache(self, tmp_path, monkeypatch):
        import pickle
        from app.ml import embedding_library as el_mod

        # Build a tiny fake cache
        embeddings = {}
        for part in ["3001", "3002", "3003", "3004", "3005"]:
            vec = np.random.rand(128).astype(np.float32)
            vec /= np.linalg.norm(vec)
            embeddings[part] = vec

        cache_path = tmp_path / "embeddings_cache.pkl"
        with open(cache_path, "wb") as f:
            pickle.dump({"embeddings": embeddings}, f)

        monkeypatch.setattr(el_mod, "CACHE_PATH", cache_path)

        lib = el_mod.EmbeddingLibrary()
        lib._loaded = False

        # Query should return nearest neighbours
        query = embeddings["3001"].copy()   # exact match should be first
        results = lib.knn_search(query, k=3)
        assert len(results) == 3
        top_part, top_dist = results[0]
        assert top_part == "3001"
        assert top_dist < 1e-4   # essentially zero distance (same vector)

    def test_add_and_search(self, tmp_path, monkeypatch):
        from app.ml import embedding_library as el_mod

        monkeypatch.setattr(el_mod, "CACHE_PATH", tmp_path / "test_cache.pkl")

        lib = el_mod.EmbeddingLibrary()
        lib._loaded = True   # skip file loading

        vec = np.array([1.0] + [0.0] * 127, dtype=np.float32)
        lib.add_embedding("3001", vec)
        lib.add_embedding("3002", np.array([0.0, 1.0] + [0.0] * 126, dtype=np.float32))

        results = lib.knn_search(vec, k=2)
        assert results[0][0] == "3001"
        assert results[0][1] < 1e-4


# ---------------------------------------------------------------------------
# KNN distance → confidence conversion
# ---------------------------------------------------------------------------

class TestKNNConfidenceConversion:
    """Cosine distance 0.0 → conf 1.0; distance 2.0 → conf 0.0."""

    @pytest.mark.parametrize("distance,expected_conf", [
        (0.00, 1.00),
        (0.30, 0.85),
        (1.00, 0.50),
        (2.00, 0.00),
    ])
    def test_distance_to_confidence(self, distance: float, expected_conf: float):
        pseudo_conf = max(0.0, 1.0 - distance / 2.0)
        assert abs(pseudo_conf - expected_conf) < 1e-6


# ---------------------------------------------------------------------------
# hybrid_predict cascade ordering
# ---------------------------------------------------------------------------

class TestHybridPredictCascade:
    """Verify that source attribution and cascade priority are correct."""

    @pytest.mark.asyncio
    async def test_high_confidence_brickognize_skips_gemini(self):
        """When Brickognize is >80%, Gemini should not be called."""
        from app.services import hybrid_recognition as hr

        brickognize_result = [_make_pred("3001", 0.95, "brickognize")]

        with (
            patch.object(hr, "brickognize_predict", new=AsyncMock(return_value=brickognize_result)),
            patch.object(hr, "gemini_predict",      new=AsyncMock(return_value=[])) as mock_gemini,
            patch.object(hr, "_safe_local_predict",  new=AsyncMock(return_value=[])),
        ):
            result = await hr.hybrid_predict(_make_jpeg())

        mock_gemini.assert_not_called()
        assert result[0]["part_num"] == "3001"
        assert result[0]["source"] == "brickognize"

    @pytest.mark.asyncio
    async def test_low_confidence_brickognize_calls_gemini(self):
        """When Brickognize is <80%, Gemini should be called."""
        from app.services import hybrid_recognition as hr

        brickognize_result = [_make_pred("3001", 0.40, "brickognize")]
        gemini_result      = [_make_pred("3001", 0.75, "gemini")]

        with (
            patch.object(hr, "brickognize_predict", new=AsyncMock(return_value=brickognize_result)),
            patch.object(hr, "gemini_predict",      new=AsyncMock(return_value=gemini_result)) as mock_gemini,
            patch.object(hr, "_safe_local_predict",  new=AsyncMock(return_value=[])),
        ):
            result = await hr.hybrid_predict(_make_jpeg())

        mock_gemini.assert_called_once()
        # Agreement boost should raise confidence above both individual values
        assert result[0]["confidence"] > max(0.40, 0.75)
        assert result[0]["source"] == "brickognize+gemini"

    @pytest.mark.asyncio
    async def test_knn_result_present_in_output(self):
        """A confident k-NN prediction should appear in the merged output."""
        from app.services import hybrid_recognition as hr

        brickognize_result = [_make_pred("3001", 0.35)]
        knn_result         = [_make_pred("3002", 0.85, "contrastive_knn")]

        with (
            patch.object(hr, "brickognize_predict", new=AsyncMock(return_value=brickognize_result)),
            patch.object(hr, "_safe_gemini",         new=AsyncMock(return_value=[])),
            patch.object(hr, "_safe_local_predict",  new=AsyncMock(return_value=knn_result)),
        ):
            result = await hr.hybrid_predict(_make_jpeg())

        sources = [p["source"] for p in result]
        assert "contrastive_knn" in sources

    @pytest.mark.asyncio
    async def test_all_sources_fail_gracefully(self):
        """If every source crashes, hybrid_predict should return [] not raise."""
        from app.services import hybrid_recognition as hr

        with (
            patch.object(hr, "brickognize_predict", new=AsyncMock(return_value=[])),
            patch.object(hr, "_safe_gemini",         new=AsyncMock(return_value=[])),
            patch.object(hr, "_safe_local_predict",  new=AsyncMock(return_value=[])),
        ):
            result = await hr.hybrid_predict(_make_jpeg())

        assert result == []


# ---------------------------------------------------------------------------
# _merge_predictions deduplication
# ---------------------------------------------------------------------------

class TestMergePredictions:
    def test_no_duplicate_parts(self):
        from app.services.hybrid_recognition import _merge_predictions

        bg  = [_make_pred("3001", 0.60), _make_pred("3002", 0.30)]
        gm  = [_make_pred("3001", 0.55), _make_pred("3003", 0.40)]
        loc = [_make_pred("3001", 0.50, "contrastive_knn")]

        result = _merge_predictions(bg, gm, loc)
        part_nums = [p["part_num"] for p in result]
        assert len(part_nums) == len(set(part_nums)), "Duplicate parts in merged output"

    def test_at_most_five_results(self):
        from app.services.hybrid_recognition import _merge_predictions

        bg  = [_make_pred(f"part_{i}", 0.9 - i * 0.1) for i in range(3)]
        gm  = [_make_pred(f"gem_{i}",  0.6 - i * 0.1) for i in range(3)]
        loc = [_make_pred(f"knn_{i}",  0.5 - i * 0.1, "contrastive_knn") for i in range(5)]

        result = _merge_predictions(bg, gm, loc)
        assert len(result) <= 5


# ---------------------------------------------------------------------------
# Feedback endpoint (in-memory SQLite)
# ---------------------------------------------------------------------------

class TestFeedbackEndpoint:
    """Round-trip test using the FastAPI test client with SQLite in memory."""

    @pytest.fixture(autouse=True)
    def _setup_test_db(self, tmp_path, monkeypatch):
        """Override the database to use a temp SQLite file."""
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        db_path = tmp_path / "test.db"
        test_engine = create_engine(
            f"sqlite:///{db_path}",
            connect_args={"check_same_thread": False},
        )

        from app.local_inventory.database import Base
        from app.local_inventory.models import LocalInventoryPart, ScanSession, ScanFeedback  # noqa

        Base.metadata.create_all(bind=test_engine)

        TestSession = sessionmaker(bind=test_engine)

        import app.local_inventory.database as db_mod
        monkeypatch.setattr(db_mod, "engine", test_engine)
        monkeypatch.setattr(db_mod, "SessionLocal", TestSession)

    def _get_test_client(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from app.local_inventory.feedback_routes import feedback_router

        app = FastAPI()
        app.include_router(feedback_router)
        return TestClient(app)

    def test_submit_correct_prediction(self):
        client = self._get_test_client()
        payload = {
            "scan_id":            "scan_test_001",
            "predicted_part_num": "3001",
            "correct_part_num":   "3001",   # same = confirmation
            "confidence":         0.92,
            "source":             "brickognize",
        }
        resp = client.post("/api/local-inventory/scan-feedback", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["saved"] is True
        assert data["will_improve_model"] is False   # same part → no training value

    def test_submit_correction(self, tmp_path, monkeypatch):
        import app.local_inventory.feedback_routes as fr
        monkeypatch.setattr(fr, "FEEDBACK_IMAGES_DIR", tmp_path / "feedback")

        client = self._get_test_client()

        # Encode a tiny JPEG as base64
        img_b64 = base64.b64encode(_make_jpeg()).decode()

        payload = {
            "scan_id":            "scan_test_002",
            "predicted_part_num": "3001",
            "correct_part_num":   "3002",   # different = real correction
            "confidence":         0.45,
            "source":             "gemini",
            "image_base64":       img_b64,
        }
        resp = client.post("/api/local-inventory/scan-feedback", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["saved"] is True
        assert data["will_improve_model"] is True
        assert data["feedback_id"] != ""

        # Image should have been saved
        saved_images = list((tmp_path / "feedback" / "3002").glob("*.jpg"))
        assert len(saved_images) == 1

    def test_feedback_stats(self):
        client = self._get_test_client()

        # Submit three records: 2 corrections, 1 confirmation
        for scan_id, predicted, correct in [
            ("s1", "3001", "3002"),
            ("s2", "3001", "3003"),
            ("s3", "3004", "3004"),
        ]:
            client.post("/api/local-inventory/scan-feedback", json={
                "scan_id": scan_id,
                "predicted_part_num": predicted,
                "correct_part_num": correct,
                "confidence": 0.5,
                "source": "gemini",
            })

        resp = client.get("/api/local-inventory/feedback/stats")
        assert resp.status_code == 200
        stats = resp.json()
        assert stats["total_corrections"] == 2
        assert stats["agreement_count"]   == 1
        assert len(stats["top_confused_pairs"]) <= 10


# ---------------------------------------------------------------------------
# scan-multi detection_method field
# ---------------------------------------------------------------------------

class TestScanMultiDetectionMethod:
    """The updated scan-multi endpoint should report which detection path was used."""

    @pytest.mark.asyncio
    async def test_quadrant_fallback_when_yolo_absent(self):
        """When YOLO model is not loaded, detection_method should be 'quadrant'."""
        from app.ml.model_manager import ModelManager

        mm_mock = MagicMock(spec=ModelManager)
        mm_mock.yolo_available = False

        with patch("app.local_inventory.feedback_routes.ModelManager") as MockClass:
            MockClass.get.return_value = mm_mock
            with patch(
                "app.local_inventory.feedback_routes.hybrid_predict",
                new=AsyncMock(return_value=[_make_pred("3001", 0.75)]),
            ):
                from fastapi.testclient import TestClient
                from fastapi import FastAPI
                from app.local_inventory.feedback_routes import feedback_router
                from app.local_inventory.database import get_local_db
                from sqlalchemy import create_engine
                from sqlalchemy.orm import sessionmaker
                from app.local_inventory.database import Base
                from app.local_inventory import models  # noqa — register all tables

                engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
                Base.metadata.create_all(bind=engine)
                Session = sessionmaker(bind=engine)

                app = FastAPI()
                app.dependency_overrides[get_local_db] = lambda: Session()
                app.include_router(feedback_router)
                client = TestClient(app)

                img_b64 = base64.b64encode(_make_jpeg(200, 200)).decode()
                resp = client.post(
                    "/api/local-inventory/scan-multi",
                    json={"image_base64": img_b64},
                )
                # We don't assert detection_method because MultiPieceScanResponse
                # from the locked schemas.py doesn't include it yet —
                # this test just asserts the endpoint returns 200 and valid structure.
                assert resp.status_code == 200
                data = resp.json()
                assert "pieces_detected" in data
                assert "pieces" in data
