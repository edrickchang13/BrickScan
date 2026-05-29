"""
EmbeddingLibrary: k-NN search over a cached dict of contrastive embeddings.

Flow
----
1. At first use, look for  backend/data/embeddings_cache.pkl.
2. If found, load {part_num -> float32[128]} and build a sklearn NearestNeighbors
   index (cosine, brute-force — fast for up to ~100k parts on CPU).
3. knn_search(query, k=5) returns (part_num, cosine_distance) pairs, ascending.
   Cosine distance 0.0 = identical, 2.0 = maximally different.
   Threshold 0.30 is a reliable "confident match" boundary.
4. Embeddings can be added at runtime; call save_cache() to persist them.
5. build_from_images(dir, encoder) builds the cache from part image folders.

The cache is normally built offline on the DGX Spark, then copied here.
Everything fails gracefully — knn_search returns [] when the index is empty.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
CACHE_PATH = _BACKEND_DIR / "data" / "embeddings_cache.pkl"

# Cosine distance at or below this value = confident k-NN match
KNN_CONFIDENCE_THRESHOLD = 0.30


class EmbeddingLibrary:
    """Singleton k-NN index over contrastive embeddings of known LEGO parts."""

    _instance: Optional["EmbeddingLibrary"] = None

    def __init__(self) -> None:
        self._embeddings: Dict[str, np.ndarray] = {}
        self._part_nums:  List[str] = []
        # part_num -> how many confirmed exemplars have been merged into its
        # prototype (used by add_exemplar's running-mean merge). Persisted in
        # the cache so the running mean is correct across restarts.
        self._exemplar_counts: Dict[str, int] = {}
        self._knn = None
        self._loaded: bool = False
        self._dirty: bool = False

    @classmethod
    def get(cls) -> "EmbeddingLibrary":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── lazy init ─────────────────────────────────────────────────────────────
    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        self._load_cache()
        if self._embeddings:
            self._rebuild_index()

    def _load_cache(self) -> None:
        if not CACHE_PATH.exists():
            logger.info(
                "No embeddings cache at %s — k-NN disabled until cache is built", CACHE_PATH
            )
            return
        try:
            with open(CACHE_PATH, "rb") as f:
                data = pickle.load(f)
            self._embeddings = data.get("embeddings", {})
            # Restore per-part exemplar counts (default 1 for legacy caches that
            # predate the flywheel merge, so the next merge uses a sane weight).
            self._exemplar_counts = data.get("exemplar_counts", {}) or {}
            for p in self._embeddings:
                self._exemplar_counts.setdefault(p, 1)
            logger.info("Loaded %d embeddings from cache", len(self._embeddings))
        except Exception as e:
            logger.error("Failed to load embeddings cache: %s", e)

    def _rebuild_index(self) -> None:
        if not self._embeddings:
            self._knn = None
            return
        try:
            from sklearn.neighbors import NearestNeighbors
            self._part_nums = list(self._embeddings.keys())
            matrix = np.stack([self._embeddings[p].astype(np.float32) for p in self._part_nums])
            self._knn = NearestNeighbors(
                n_neighbors=min(10, len(self._part_nums)),
                metric="cosine",
                algorithm="brute",
                n_jobs=-1,
            )
            self._knn.fit(matrix)
            self._dirty = False
            logger.info("k-NN index built: %d parts", len(self._part_nums))
        except ImportError:
            logger.warning("scikit-learn not installed — k-NN disabled. pip install scikit-learn")
        except Exception as e:
            logger.error("Failed to build k-NN index: %s", e)

    # ── public API ─────────────────────────────────────────────────────────────
    @property
    def size(self) -> int:
        self._ensure_loaded(); return len(self._embeddings)

    def knn_search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """
        Find k nearest parts.

        Returns:
            List of (part_num, cosine_distance) sorted ascending.
            Empty list when index is empty or unavailable.
        """
        self._ensure_loaded()
        if self._dirty:
            self._rebuild_index()
        if self._knn is None or not self._part_nums:
            return []
        try:
            q = query_embedding.reshape(1, -1).astype(np.float32)
            k_actual = min(k, len(self._part_nums))
            dists, idxs = self._knn.kneighbors(q, n_neighbors=k_actual)
            return [(self._part_nums[int(i)], float(d)) for d, i in zip(dists[0], idxs[0])]
        except Exception as e:
            logger.error("knn_search: %s", e)
            return []

    def add_embedding(self, part_num: str, embedding: np.ndarray) -> None:
        """Add/update one embedding. Index is rebuilt lazily on next search."""
        self._ensure_loaded()
        self._embeddings[part_num] = embedding.astype(np.float32)
        self._dirty = True

    def add_exemplar(
        self, part_num: str, embedding: np.ndarray, *,
        persist: bool = False, momentum: Optional[float] = None,
    ) -> None:
        """Fold a CONFIRMED exemplar into a part's prototype — flywheel ingest.

        Unlike add_embedding (which overwrites), this MERGES the new exemplar
        into the running per-part vector and re-normalises, so repeated
        confirmations of the same part sharpen its prototype instead of being
        lost. The index keeps exactly one vector per part_num (k-NN over part
        prototypes), so the index never bloats no matter how many corrections
        arrive.

        merge rule:
          - first exemplar for a part:      store it (normalised)
          - subsequent exemplars:           v <- normalise((1-m)*v_old + m*v_new)
            with m defaulting to a simple running mean (1/count) when `momentum`
            is None, or a fixed EMA weight when given (favours recent crops).

        The index is rebuilt lazily on the next search (sets _dirty). With
        persist=True the cache is written so the exemplar survives a restart.
        """
        self._ensure_loaded()
        new = embedding.astype(np.float32)
        nn = np.linalg.norm(new)
        if nn > 1e-8:
            new = new / nn

        old = self._embeddings.get(part_num)
        if old is None:
            merged = new
            self._exemplar_counts[part_num] = 1
        else:
            cnt = self._exemplar_counts.get(part_num, 1)
            m = (1.0 / (cnt + 1)) if momentum is None else float(momentum)
            merged = (1.0 - m) * old + m * new
            mn = np.linalg.norm(merged)
            merged = merged / mn if mn > 1e-8 else merged
            self._exemplar_counts[part_num] = cnt + 1

        self._embeddings[part_num] = merged.astype(np.float32)
        self._dirty = True
        if persist:
            self.save_cache()

    def save_cache(self) -> None:
        """Write embeddings dict to disk."""
        self._ensure_loaded()
        try:
            CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(CACHE_PATH, "wb") as f:
                pickle.dump(
                    {"embeddings": self._embeddings,
                     "exemplar_counts": self._exemplar_counts},
                    f, protocol=4,
                )
            logger.info("Saved %d embeddings → %s", len(self._embeddings), CACHE_PATH)
        except Exception as e:
            logger.error("save_cache: %s", e)

    def build_from_images(self, image_dir: str, encoder, force_rebuild: bool = False) -> int:
        """
        Encode part images and add to cache.

        image_dir structure:
            <image_dir>/3001/img1.jpg
            <image_dir>/3001/img2.jpg
            ...
        Averages embeddings across multiple images per part and re-normalises.
        Returns number of new embeddings added.
        """
        self._ensure_loaded()
        base = Path(image_dir)
        if not base.exists():
            logger.warning("Image dir not found: %s", image_dir)
            return 0

        added = 0
        for part_dir in sorted(base.iterdir()):
            if not part_dir.is_dir():
                continue
            part_num = part_dir.name
            if not force_rebuild and part_num in self._embeddings:
                continue
            imgs = list(part_dir.glob("*.jpg")) + list(part_dir.glob("*.png"))
            if not imgs:
                continue
            embs: List[np.ndarray] = []
            for p in imgs[:5]:
                try:
                    emb = encoder.encode_image(p.read_bytes())
                    if emb is not None:
                        embs.append(emb)
                except Exception:
                    pass
            if embs:
                mean = np.mean(embs, axis=0).astype(np.float32)
                norm = np.linalg.norm(mean)
                self._embeddings[part_num] = mean / norm if norm > 1e-8 else mean
                added += 1

        if added:
            self._dirty = True
            self.save_cache()
            logger.info("Built %d new embeddings, saved cache", added)
        return added
