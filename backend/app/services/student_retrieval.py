"""
Student-retrieval service — server-side k-NN retrieval over the on-device
gallery index, powered by the distilled FastViT-SA24 *student* encoder.

Why this exists:
  The student was trained by embedding distillation from the DINOv2 teacher
  and validated at 90.1% top-1 (single-frame) / 95.6% recall@3 on the 439-class
  real-photo val set. The mobile app already ships it as the on-device engine;
  this module gives the *backend* the same validated retrieval path so the
  server's hybrid cascade no longer depends solely on third-party APIs
  (Brickognize / Gemini).

Pipeline at query time:
  1. _embed(image_bytes) → 768-d L2-normalised float32 vector
       (student bakes ImageNet normalisation internally → feed RGB[0,1] 224)
  2. int8 cosine k-NN over the gallery index (dequantised via the global scale)
  3. group neighbours by part_num → ranked top-k Rebrickable part predictions

Artifacts (configurable via env vars):
  STUDENT_ONNX_PATH      student.onnx (768-d output, L2-norm baked? no — we
                         L2-normalise here; the index rows ARE normalised).
                         Default: mobile/assets/models/student.onnx
  STUDENT_GALLERY_PATH   gallery_index.json (rn-retrieval partIndex format:
                         {version, dim, count, scale, vectors:b64 int8, partNums}).
                         Default: backend/data/gallery_index.json (a server copy
                         of the validated 12.5k-exemplar / 439-class index).

Search backend:
  Pure numpy brute force. The gallery is ~12.5k × 768 int8; a single query is
  one (1, N) matmul → ~5-10 ms on CPU. No FAISS / sklearn / usearch dependency
  so the module imports cleanly anywhere numpy is present.

Graceful degradation:
  Every public entry point returns [] / False when onnxruntime is missing or
  the model / index files are absent, so the cascade keeps working exactly as
  before the student was wired in. Nothing here ever raises into the caller.
"""
from __future__ import annotations

import base64
import io
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# backend/app/services/student_retrieval.py → parent x3 = backend/
_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
# Server copy of the validated gallery index lives under backend/data/.
_DEFAULT_GALLERY_PATH = _BACKEND_DIR / "data" / "gallery_index.json"
# The 86 MB student ONNX is shipped with the mobile bundle; the server loads it
# from there by default rather than duplicating it. Override via env var to
# point at a backend-local copy in production.
_DEFAULT_STUDENT_PATH = (
    _BACKEND_DIR.parent / "mobile" / "assets" / "models" / "student.onnx"
)

STUDENT_ONNX_PATH = Path(os.environ.get("STUDENT_ONNX_PATH", str(_DEFAULT_STUDENT_PATH)))
STUDENT_GALLERY_PATH = Path(
    os.environ.get("STUDENT_GALLERY_PATH", str(_DEFAULT_GALLERY_PATH))
)

# Student input geometry — MUST match ml/scripts/student_index_eval.py exactly,
# or query embeddings won't align with the gallery rows.
#   224x224 BILINEAR, RGB scaled to [0,1], CHW. ImageNet normalisation is BAKED
#   INTO the ONNX graph, so we deliberately do NOT subtract mean / divide std
#   here (unlike app/ml/model_manager._preprocess, which feeds a different,
#   externally-normalised model).
_STUDENT_INPUT_SIZE = 224

# Cosine-similarity floor for a neighbour to count toward a part's score.
# The index rows are L2-normalised, so dot product == cosine in [-1, 1].
# Below this we treat the hit as noise. Conservative default — the cascade's
# own thresholds / calibration do the final gating.
DEFAULT_MIN_SIMILARITY = 0.20


@dataclass
class StudentHit:
    """One ranked part prediction from the student retrieval engine."""
    part_num: str
    confidence: float          # aggregated cosine similarity in [0, 1]
    neighbor_count: int        # how many of the top neighbours voted for it
    best_similarity: float     # the single closest neighbour's cosine sim


class _StudentRetriever:
    """Singleton owning the student ONNX session + the int8 gallery matrix.

    Lazily initialised on first use. All failures (missing onnxruntime, missing
    files, malformed index) degrade to a disabled-but-safe state where
    ``available`` is False and ``query`` returns [].
    """

    _instance: Optional["_StudentRetriever"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._session: Optional[Any] = None
        self._in_name: Optional[str] = None
        self._out_name: Optional[str] = None
        # Gallery: int8 matrix (N, D), the global dequant scale, and the
        # parallel part_num list. _gallery_f32 holds the dequantised float32
        # rows (already L2-normalised) for the matmul hot path.
        self._gallery_i8: Optional[np.ndarray] = None
        self._gallery_f32: Optional[np.ndarray] = None
        self._scale: float = 1.0 / 127.0
        self._part_nums: List[str] = []
        self._dim: int = 0
        self._loaded: bool = False

    @classmethod
    def get(cls) -> "_StudentRetriever":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ── lazy init ──────────────────────────────────────────────────────────────
    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            self._loaded = True       # set first so a failure doesn't re-try every call
            self._load_session()
            self._load_gallery()

    def _ort_providers(self) -> List[str]:
        try:
            import onnxruntime as ort
            if "CUDAExecutionProvider" in ort.get_available_providers():
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        except Exception:
            pass
        return ["CPUExecutionProvider"]

    def _load_session(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError:
            logger.info(
                "student_retrieval: onnxruntime not installed — student tier disabled"
            )
            return
        if not STUDENT_ONNX_PATH.exists():
            logger.info(
                "student_retrieval: student ONNX %s not found — student tier disabled",
                STUDENT_ONNX_PATH,
            )
            return
        try:
            t0 = time.time()
            sess = ort.InferenceSession(
                str(STUDENT_ONNX_PATH), providers=self._ort_providers()
            )
            self._session = sess
            self._in_name = sess.get_inputs()[0].name
            self._out_name = sess.get_outputs()[0].name
            logger.info(
                "student_retrieval: loaded %s (%s, in=%s out=%s) in %.2fs",
                STUDENT_ONNX_PATH.name, sess.get_providers()[0],
                self._in_name, self._out_name, time.time() - t0,
            )
        except Exception as e:
            logger.error("student_retrieval: failed to load student ONNX: %s", e)
            self._session = None

    def _load_gallery(self, path: Path = STUDENT_GALLERY_PATH) -> None:
        """Parse the rn-retrieval partIndex JSON and dequantise the int8 rows.

        Expected schema (flat, as written by ml/scripts/student_index_eval.py):
            {
              "version": 1,
              "dim":      768,
              "count":    12500,
              "scale":    0.007874...,        # global int8 → float scale (1/127)
              "vectors":  "<base64 int8, count*dim row-major>",
              "partNums": ["10197", "10197", ...]   # length == count
            }
        """
        if not path.exists():
            logger.info(
                "student_retrieval: gallery index %s not found — student tier disabled",
                path,
            )
            return
        try:
            import json
            t0 = time.time()
            with open(path, "r") as f:
                blob = json.load(f)

            dim = int(blob["dim"])
            count = int(blob["count"])
            scale = float(blob.get("scale", 1.0 / 127.0))
            part_nums = [str(p) for p in blob["partNums"]]

            raw = np.frombuffer(base64.b64decode(blob["vectors"]), dtype=np.int8)
            if raw.size != dim * count:
                logger.error(
                    "student_retrieval: gallery vector size mismatch "
                    "(decoded=%d, expected dim*count=%d) — disabling",
                    raw.size, dim * count,
                )
                return
            if len(part_nums) != count:
                logger.error(
                    "student_retrieval: partNums length %d != count %d — disabling",
                    len(part_nums), count,
                )
                return

            gallery_i8 = raw.reshape(count, dim)
            # Dequantise once up front. Rows are already (approximately)
            # L2-normalised by the offline builder, so dot product ≈ cosine.
            # We re-normalise defensively so int8 rounding can't skew the
            # similarity scale across rows.
            gallery_f32 = gallery_i8.astype(np.float32) * scale
            norms = np.linalg.norm(gallery_f32, axis=1, keepdims=True)
            np.divide(gallery_f32, norms, out=gallery_f32, where=norms > 1e-8)

            self._gallery_i8 = gallery_i8
            self._gallery_f32 = np.ascontiguousarray(gallery_f32)
            self._scale = scale
            self._part_nums = part_nums
            self._dim = dim
            logger.info(
                "student_retrieval: loaded gallery %d exemplars × %dD, "
                "%d unique parts, %.2fs",
                count, dim, len(set(part_nums)), time.time() - t0,
            )
        except Exception as e:
            logger.error("student_retrieval: failed to load gallery %s: %s", path, e)
            self._gallery_f32 = None

    # ── public surface ───────────────────────────────────────────────────────────
    @property
    def available(self) -> bool:
        """True iff both the student session and a non-empty gallery loaded."""
        self._ensure_loaded()
        return (
            self._session is not None
            and self._gallery_f32 is not None
            and self._gallery_f32.shape[0] > 0
        )

    @property
    def gallery_size(self) -> int:
        self._ensure_loaded()
        return int(self._gallery_f32.shape[0]) if self._gallery_f32 is not None else 0

    def _embed(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """Decode → student input tensor → L2-normalised 768-d float32, or None.

        ImageNet normalisation is baked into the ONNX graph, so the input is
        RGB scaled to [0,1] (NOT externally mean/std-normalised). 224×224
        BILINEAR matches the offline embed in student_index_eval.py.
        """
        if self._session is None:
            return None
        try:
            from PIL import Image
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img = img.resize(
                (_STUDENT_INPUT_SIZE, _STUDENT_INPUT_SIZE), Image.Resampling.BILINEAR
            )
            arr = np.asarray(img, dtype=np.float32) / 255.0      # HWC [0,1]
            arr = np.transpose(arr, (2, 0, 1))                   # CHW
            tensor = np.expand_dims(arr, 0)                      # (1, 3, 224, 224)
            out = self._session.run([self._out_name], {self._in_name: tensor})[0]
            emb = np.asarray(out, dtype=np.float32).reshape(-1)
            norm = float(np.linalg.norm(emb))
            return emb / norm if norm > 1e-8 else emb
        except Exception as e:
            logger.error("student_retrieval._embed: %s", e)
            return None

    def query(
        self,
        image_bytes: bytes,
        top_k: int = 5,
        *,
        knn: int = 20,
        min_similarity: float = DEFAULT_MIN_SIMILARITY,
    ) -> List[StudentHit]:
        """Embed the crop and return up to `top_k` ranked part predictions.

        We take the `knn` nearest gallery exemplars (cosine) and aggregate them
        by part_num: a part's score is the max cosine similarity of its voting
        exemplars (matches the single-frame retrieval protocol the student was
        evaluated under — nearest-exemplar wins). neighbor_count is surfaced so
        the cascade can prefer parts with multiple supporting exemplars.

        Returns [] when the engine is unavailable or the embedding fails — never
        raises.
        """
        self._ensure_loaded()
        if not self.available:
            return []
        emb = self._embed(image_bytes)
        if emb is None:
            return []
        try:
            gallery = self._gallery_f32
            assert gallery is not None
            # Cosine similarity to every exemplar; rows are L2-normalised and we
            # normalised `emb`, so the dot product is cosine in [-1, 1].
            sims = gallery @ emb                       # (N,)
            n = sims.shape[0]
            k = int(min(max(knn, top_k), n))
            # argpartition for the top-k, then sort just those k descending.
            part_idx = np.argpartition(-sims, k - 1)[:k]
            part_idx = part_idx[np.argsort(-sims[part_idx])]

            # Aggregate by part_num: nearest-exemplar score + vote count.
            best: dict = {}                            # part_num -> [best_sim, count]
            order: List[str] = []                      # first-seen order (== sim order)
            for i in part_idx:
                sim = float(sims[int(i)])
                if sim < min_similarity:
                    continue
                pn = self._part_nums[int(i)]
                if pn in best:
                    best[pn][1] += 1
                    if sim > best[pn][0]:
                        best[pn][0] = sim
                else:
                    best[pn] = [sim, 1]
                    order.append(pn)

            hits = [
                StudentHit(
                    part_num=pn,
                    confidence=max(0.0, min(1.0, best[pn][0])),
                    neighbor_count=best[pn][1],
                    best_similarity=best[pn][0],
                )
                for pn in order
            ]
            # Already in descending-similarity order via `order`, but sort
            # defensively in case a later neighbour raised an earlier part's max.
            hits.sort(key=lambda h: h.best_similarity, reverse=True)
            return hits[:top_k]
        except Exception as e:
            logger.error("student_retrieval.query: %s", e)
            return []


# ── module-level convenience API ─────────────────────────────────────────────────

def is_available() -> bool:
    """True iff the student ONNX + gallery index are both loaded and usable."""
    return _StudentRetriever.get().available


def gallery_size() -> int:
    return _StudentRetriever.get().gallery_size


def predict(
    image_bytes: bytes,
    top_k: int = 5,
    *,
    knn: int = 20,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
) -> List[StudentHit]:
    """Top-k Rebrickable part predictions for a query crop via student k-NN.

    Thin wrapper over the singleton so callers don't touch the class. Returns []
    when the engine is unavailable (graceful degradation)."""
    return _StudentRetriever.get().query(
        image_bytes, top_k=top_k, knn=knn, min_similarity=min_similarity
    )


def reload() -> None:
    """Drop the cached singleton so the next call re-reads the model + index.

    Useful in tests, or after dropping in a freshly built gallery index without
    restarting the server.
    """
    with _StudentRetriever._lock:
        _StudentRetriever._instance = None
