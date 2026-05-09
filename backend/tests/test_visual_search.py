"""
Tests for app/services/visual_search.py

Behaviour we care about:
  - When no catalogue file exists at startup, is_loaded() returns False and
    search() returns an empty list — never raises. This is the no-op fallback
    that lets the cascade keep working before the DINOv2 catalogue ships.
  - When a synthetic catalogue is dropped in place, is_loaded() flips to True,
    catalog_size() reports the row count, and search() returns the expected
    cosine-ordered hits.
  - search() honours min_similarity (drops weak matches).
  - search() honours color_id_filter (returns only matching colours).
  - search() dedupes by (part_num, color_id) so we don't surface 5 element
    images of the same red 2x4 brick.
  - reload_from_disk() picks up a swapped catalogue without restart.
"""
from __future__ import annotations

import importlib
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def fake_catalog_path(tmp_path, monkeypatch):
    """Force visual_search to look at a tmp pickle path for this test, and
    reset the module's cached load-state so the previous test's catalogue
    doesn't bleed through."""
    p = tmp_path / "catalog_embeddings.pkl"
    monkeypatch.setenv("CATALOG_EMBEDDINGS_PATH", str(p))
    # Drop the module so the next `from app.services import visual_search`
    # re-evaluates DEFAULT_CATALOG_PATH against the new env var.
    for mod in list(sys.modules):
        if mod == "app.services.visual_search":
            del sys.modules[mod]
    yield p


def _make_catalog(path: Path, vectors_and_meta):
    """Helper: write a pickle in the schema visual_search.py expects."""
    embeddings = np.array([v for v, _ in vectors_and_meta], dtype=np.float32)
    # L2-normalise rows so cosine similarity == dot product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-8)
    entries = [meta for _, meta in vectors_and_meta]
    with open(path, "wb") as f:
        pickle.dump({
            "embeddings": embeddings,
            "entries": entries,
            "model": "test_model",
            "dim": embeddings.shape[1],
            "built_at": "2026-01-01T00:00:00Z",
        }, f)


def test_no_catalog_returns_empty(fake_catalog_path):
    """Missing catalogue → is_loaded False, search returns []."""
    from app.services import visual_search
    visual_search.reload_from_disk(fake_catalog_path)
    assert visual_search.is_loaded() is False
    assert visual_search.catalog_size() == 0
    assert visual_search.search(np.array([1.0, 0.0, 0.0])) == []


def test_loads_catalog_when_present(fake_catalog_path):
    _make_catalog(fake_catalog_path, [
        (np.array([1.0, 0.0, 0.0]), {
            "element_id": "300121", "part_num": "3001", "color_id": 4,
            "part_name": "Brick 2 x 4", "color_name": "Red", "color_hex": "#C91A09",
        }),
        (np.array([0.0, 1.0, 0.0]), {
            "element_id": "300122", "part_num": "3001", "color_id": 1,
            "part_name": "Brick 2 x 4", "color_name": "Blue", "color_hex": "#0055BF",
        }),
        (np.array([0.0, 0.0, 1.0]), {
            "element_id": "300221", "part_num": "3022", "color_id": 4,
            "part_name": "Plate 2 x 2", "color_name": "Red", "color_hex": "#C91A09",
        }),
    ])
    from app.services import visual_search
    visual_search.reload_from_disk(fake_catalog_path)
    assert visual_search.is_loaded() is True
    assert visual_search.catalog_size() == 3


def test_search_returns_cosine_ordered(fake_catalog_path):
    _make_catalog(fake_catalog_path, [
        (np.array([1.0, 0.0, 0.0]), {
            "element_id": "300121", "part_num": "3001", "color_id": 4,
            "part_name": "Brick 2 x 4", "color_name": "Red", "color_hex": "#C91A09",
        }),
        (np.array([0.7, 0.7, 0.0]), {
            "element_id": "300122", "part_num": "3001", "color_id": 1,
            "part_name": "Brick 2 x 4", "color_name": "Blue", "color_hex": "#0055BF",
        }),
        (np.array([0.0, 0.0, 1.0]), {
            "element_id": "300221", "part_num": "3022", "color_id": 4,
            "part_name": "Plate 2 x 2", "color_name": "Red", "color_hex": "#C91A09",
        }),
    ])
    from app.services import visual_search
    visual_search.reload_from_disk(fake_catalog_path)
    # Query is identical to the first row → sim=1.0 first, then 0.7, then 0.0
    hits = visual_search.search(np.array([1.0, 0.0, 0.0]), top_k=3)
    assert len(hits) == 3
    assert hits[0].part_num == "3001"
    assert hits[0].color_name == "Red"
    assert hits[0].similarity > hits[1].similarity > hits[2].similarity


def test_min_similarity_drops_weak_hits(fake_catalog_path):
    _make_catalog(fake_catalog_path, [
        (np.array([1.0, 0.0, 0.0]), {
            "element_id": "300121", "part_num": "3001", "color_id": 4,
            "part_name": "Brick 2 x 4", "color_name": "Red", "color_hex": "#C91A09",
        }),
        (np.array([0.0, 1.0, 0.0]), {
            "element_id": "300122", "part_num": "3001", "color_id": 1,
            "part_name": "Brick 2 x 4", "color_name": "Blue", "color_hex": "#0055BF",
        }),
    ])
    from app.services import visual_search
    visual_search.reload_from_disk(fake_catalog_path)
    hits = visual_search.search(
        np.array([1.0, 0.0, 0.0]),
        top_k=5,
        min_similarity=0.5,
    )
    # Only the perfect match passes; the orthogonal one is filtered
    assert len(hits) == 1
    assert hits[0].part_num == "3001"
    assert hits[0].color_name == "Red"


def test_color_id_filter(fake_catalog_path):
    _make_catalog(fake_catalog_path, [
        (np.array([1.0, 0.0, 0.0]), {
            "element_id": "300121", "part_num": "3001", "color_id": 4,
            "part_name": "Brick 2 x 4", "color_name": "Red", "color_hex": "#C91A09",
        }),
        (np.array([0.95, 0.31, 0.0]), {
            "element_id": "300122", "part_num": "3001", "color_id": 1,
            "part_name": "Brick 2 x 4", "color_name": "Blue", "color_hex": "#0055BF",
        }),
    ])
    from app.services import visual_search
    visual_search.reload_from_disk(fake_catalog_path)
    # color_id=4 filter — only the Red entry should come back
    hits = visual_search.search(
        np.array([1.0, 0.0, 0.0]),
        top_k=3,
        color_id_filter=4,
    )
    assert len(hits) == 1
    assert hits[0].color_id == 4
    assert hits[0].color_name == "Red"


def test_dedupes_by_part_and_color(fake_catalog_path):
    """Two element images of the same (part, color) should collapse to one hit."""
    _make_catalog(fake_catalog_path, [
        (np.array([1.0, 0.0, 0.0]), {
            "element_id": "300121", "part_num": "3001", "color_id": 4,
            "part_name": "Brick 2 x 4", "color_name": "Red", "color_hex": "#C91A09",
        }),
        # Same (part, color) but different element — should NOT appear twice
        (np.array([0.99, 0.14, 0.0]), {
            "element_id": "300120", "part_num": "3001", "color_id": 4,
            "part_name": "Brick 2 x 4", "color_name": "Red", "color_hex": "#C91A09",
        }),
        (np.array([0.0, 0.0, 1.0]), {
            "element_id": "300221", "part_num": "3022", "color_id": 4,
            "part_name": "Plate 2 x 2", "color_name": "Red", "color_hex": "#C91A09",
        }),
    ])
    from app.services import visual_search
    visual_search.reload_from_disk(fake_catalog_path)
    hits = visual_search.search(np.array([1.0, 0.0, 0.0]), top_k=5)
    keys = {(h.part_num, h.color_id) for h in hits}
    assert len(keys) == len(hits)   # all unique
    assert ("3001", 4) in keys
    assert ("3022", 4) in keys
