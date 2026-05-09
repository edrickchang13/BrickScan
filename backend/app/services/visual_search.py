"""
Visual-search service — k-NN retrieval over a precomputed catalogue of
LEGO part embeddings, powered by the DINOv2 + LoRA contrastive encoder.

Why this exists:
  The classifier head has a fixed output space (currently 1000 classes).
  To recognise a brick that's not in those 1000 classes, we either retrain
  (slow) or fall through to retrieval. This module is the retrieval path.

Pipeline at query time:
  1. ModelManager.encode_image(image_bytes) → 128/384-d float vector
  2. Cosine top-K search over catalogue embeddings
  3. Optional re-rank by Brickognize / classifier confidence when present

Catalogue:
  Indexed by element_id (one row per Rebrickable canonical image). Each
  row has the embedding vector + the Rebrickable part_num, color_id,
  part_name, color_name, color_hex. Built offline by
  ml/scripts/precompute_catalog_embeddings.py.

Search backend:
  - sklearn NearestNeighbors when catalogue size <50K (fits in memory,
    cosine-distance brute force is fine — 30ms per query at 50K × 384d)
  - FAISS IndexFlatIP when ≥50K (still brute force, but vectorised C++)
  - FAISS HNSW when ≥500K (approximate, sub-ms; rebuild offline)

We pick automatically based on catalogue size at load time.

This module is import-safe even when no catalogue exists yet — calls to
search() return [] until a catalogue file is dropped at the configured
path. No GPU dependency at query time.
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Default catalogue path — built by ml/scripts/precompute_catalog_embeddings.py
# and shipped to backend/data/catalog_embeddings.pkl. Override via
# CATALOG_EMBEDDINGS_PATH env var.
DEFAULT_CATALOG_PATH = Path(
    os.environ.get(
        "CATALOG_EMBEDDINGS_PATH",
        Path(__file__).resolve().parent.parent.parent
        / "data" / "catalog_embeddings.pkl",
    )
)


@dataclass
class CatalogEntry:
    """One row of the precomputed catalogue."""
    element_id: str
    part_num: str
    color_id: Optional[int]
    part_name: str
    color_name: str
    color_hex: str
    # The embedding itself lives in a separate big numpy array — we store
    # only the row index here to keep this dataclass cheap.
    row_index: int


@dataclass
class SearchHit:
    part_num: str
    part_name: str
    color_id: Optional[int]
    color_name: str
    color_hex: str
    element_id: str
    similarity: float          # cosine similarity in [-1, 1]; higher = closer


_LOCK = threading.Lock()
_LOADED = False
_ENTRIES: List[CatalogEntry] = []
_EMBEDDINGS: Optional[np.ndarray] = None     # shape (N, D), L2-normalised
_INDEX = None                                # backend-specific index handle


def _load_catalog(path: Path = DEFAULT_CATALOG_PATH) -> None:
    """Idempotent loader. Reads a pickle written by precompute_catalog_embeddings.py:

        {
          "embeddings": np.ndarray (N, D) float32 L2-normalised,
          "entries":    List[dict] with keys element_id, part_num, color_id,
                                       part_name, color_name, color_hex,
          "model":      str (e.g. "dinov2_lora_v1"),
          "dim":        int,
          "built_at":   ISO timestamp,
        }
    """
    global _LOADED, _ENTRIES, _EMBEDDINGS, _INDEX
    with _LOCK:
        if _LOADED:
            return
        if not path.exists():
            logger.info(
                "visual_search: catalogue %s not found — search disabled until built", path,
            )
            _LOADED = True
            return
        t0 = time.time()
        try:
            with open(path, "rb") as f:
                blob = pickle.load(f)
            embeddings = blob["embeddings"]
            entries_raw = blob["entries"]
            assert embeddings.ndim == 2 and embeddings.shape[0] == len(entries_raw)
            _EMBEDDINGS = embeddings.astype(np.float32, copy=False)
            _ENTRIES = [
                CatalogEntry(
                    element_id=e.get("element_id", ""),
                    part_num=e.get("part_num", ""),
                    color_id=e.get("color_id"),
                    part_name=e.get("part_name", ""),
                    color_name=e.get("color_name", ""),
                    color_hex=e.get("color_hex", ""),
                    row_index=i,
                )
                for i, e in enumerate(entries_raw)
            ]
        except Exception as e:
            logger.error("visual_search: failed to load %s: %s", path, e)
            _LOADED = True
            return

        _INDEX = _build_index(_EMBEDDINGS)
        _LOADED = True
        logger.info(
            "visual_search: loaded %d entries (%dD), %s, %.1fs",
            _EMBEDDINGS.shape[0], _EMBEDDINGS.shape[1],
            type(_INDEX).__name__, time.time() - t0,
        )


class _NumpyBruteForceIndex:
    """Pure-numpy brute-force inner-product search.

    Used as the bottom-of-the-barrel fallback when neither FAISS nor sklearn
    is installed. Plenty fast at <50K rows × ~384d on CPU (~5-15ms / query).
    Catalogue rows MUST be L2-normalised already (visual_search guarantees that
    in _load_catalog so cosine == dot product).
    """
    def __init__(self, embeddings: np.ndarray):
        self._emb = embeddings  # (N, D), L2-normalised

    def search(self, query: np.ndarray, k: int):
        # query is (1, D); compute (1, N) similarity, then top-k
        sims = query @ self._emb.T          # inner product == cosine here
        idx = np.argsort(-sims, axis=1)[:, :k]
        ordered = np.take_along_axis(sims, idx, axis=1)
        return ordered, idx


def _build_index(emb: np.ndarray):
    """Pick the right ANN backend for the catalogue size."""
    n = emb.shape[0]
    # Try FAISS first when it's a reasonable scale; brute-force above 500K
    # would still be O(N) per query.
    if n >= 50_000:
        try:
            import faiss
            d = emb.shape[1]
            if n >= 500_000:
                # Approximate HNSW for very large catalogues
                index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
                index.hnsw.efConstruction = 200
                index.hnsw.efSearch = 64
                index.add(np.ascontiguousarray(emb))
            else:
                # Exact IP brute force on FAISS — vectorised C++, plenty fast
                index = faiss.IndexFlatIP(d)
                index.add(np.ascontiguousarray(emb))
            return index
        except ImportError:
            logger.info("visual_search: faiss not installed; trying sklearn")
    # Try sklearn NearestNeighbors with cosine metric.
    try:
        from sklearn.neighbors import NearestNeighbors  # type: ignore
        nn = NearestNeighbors(metric="cosine", algorithm="brute")
        nn.fit(emb)
        return nn
    except ImportError:
        logger.info("visual_search: sklearn not installed; using numpy brute force")
    # Final fallback: pure numpy. Always works because numpy is mandatory.
    return _NumpyBruteForceIndex(emb)


def _query(emb_query: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (similarity, row_indices) of top_k matches.
    Similarity is cosine in [-1, 1] (higher better)."""
    assert _EMBEDDINGS is not None
    q = emb_query.astype(np.float32, copy=False).reshape(1, -1)
    n = _EMBEDDINGS.shape[0]
    k = min(top_k, n)

    # Normalise query (catalogue is already L2-normalised by the precompute step)
    norm = np.linalg.norm(q)
    if norm > 1e-8:
        q = q / norm

    if hasattr(_INDEX, "search"):
        # FAISS or _NumpyBruteForceIndex — both return (similarity, index)
        # with the same (1, k) shape. IP search on already-normalised vectors
        # gives cosine.
        sims, idx = _INDEX.search(q, k)
        return sims[0], idx[0]
    # sklearn path — returns cosine *distance*, convert to similarity
    distances, indices = _INDEX.kneighbors(q, n_neighbors=k)
    sims = 1.0 - distances[0]
    return sims, indices[0]


# ── Public API ──────────────────────────────────────────────────────────────

def is_loaded() -> bool:
    """True iff a catalogue was successfully loaded (i.e. the catalogue
    pickle existed at startup and parsed cleanly)."""
    _load_catalog()
    return _EMBEDDINGS is not None


def catalog_size() -> int:
    _load_catalog()
    return int(_EMBEDDINGS.shape[0]) if _EMBEDDINGS is not None else 0


def search(
    embedding: Optional[np.ndarray],
    top_k: int = 5,
    *,
    min_similarity: float = 0.0,
    color_id_filter: Optional[int] = None,
) -> List[SearchHit]:
    """
    Cosine-similarity k-NN over the catalogue.

    Args:
        embedding:   the query vector — typically from
                     ModelManager.encode_image(image_bytes). Pass None and
                     this returns []; lets callers no-op cleanly when the
                     encoder model isn't loaded.
        top_k:       number of hits to return.
        min_similarity: drop hits below this threshold. Useful when you'd
                     rather show "no match" than show a cosmetically high
                     match that's actually noise.
        color_id_filter: when set, only return hits whose stored color_id
                     matches. Used by the cascade when an upstream colour
                     classifier has already nailed the colour and we want
                     retrieval to stay in-colour.

    Returns: list of SearchHit, highest similarity first. Empty when no
    catalogue is loaded or the embedding is None.
    """
    if embedding is None:
        return []
    _load_catalog()
    if _EMBEDDINGS is None:
        return []

    # Pull more than top_k when filtering by colour, so we don't end up
    # with fewer results than requested after the filter.
    over_fetch = top_k * 6 if color_id_filter is not None else top_k
    sims, idx = _query(embedding, over_fetch)

    out: List[SearchHit] = []
    seen_keys: set = set()  # dedupe by (part_num, color_id) — multiple
                            # element images per part shouldn't all surface.
    for sim, row in zip(sims, idx):
        if sim < min_similarity:
            continue
        if 0 <= row < len(_ENTRIES):
            entry = _ENTRIES[int(row)]
            if color_id_filter is not None and entry.color_id != color_id_filter:
                continue
            key = (entry.part_num, entry.color_id)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            out.append(SearchHit(
                part_num=entry.part_num,
                part_name=entry.part_name,
                color_id=entry.color_id,
                color_name=entry.color_name,
                color_hex=entry.color_hex,
                element_id=entry.element_id,
                similarity=float(sim),
            ))
            if len(out) >= top_k:
                break
    return out


def reload_from_disk(path: Optional[Path] = None) -> None:
    """Force a re-read of the catalogue. Call after dropping in a freshly
    built pickle without restarting the server.

    Pass `path` to override the default location (mostly useful in tests).
    """
    global _LOADED, _ENTRIES, _EMBEDDINGS, _INDEX
    with _LOCK:
        _LOADED = False
        _ENTRIES = []
        _EMBEDDINGS = None
        _INDEX = None
    _load_catalog(path or DEFAULT_CATALOG_PATH)
