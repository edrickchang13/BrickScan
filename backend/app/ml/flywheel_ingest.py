"""
Active-learning FLYWHEEL ingest — append a CONFIRMED exemplar to the galleries
so the very next scan benefits, with NO retraining.

This is the server-side counterpart to ml/scripts/gallery_index.py (on-device)
and ml/scripts/color_gallery_append.py (color). The recognition spine is a
frozen embedder (RGB 224 -> L2-normalised float vector) + cosine k-NN over a
gallery. New parts/colors are learned by INSERTING exemplars — never by
retraining a network.

What `ingest_confirmed` does, given a confirmed (crop, part_num, color_id):
  1. Embed the crop ONCE with the frozen encoder (ModelManager.encode_image).
  2. APPEND that embedding to the part galleries the scan cascade already reads:
       - visual_search catalogue (element-level; part + colour), via a new
         runtime add_entry() so the (part, colour) hit surfaces immediately.
       - EmbeddingLibrary (part-level k-NN), which already supports runtime add.
     Both are persisted to disk so the exemplar survives a restart.
  3. Record the colour exemplar for the colour gallery. The portable colour
     model (models/color_v1/color_model.npz) is appended to OFF-LINE on the
     Spark by ml/scripts/color_gallery_append.py — there's no LDA at query time
     in the backend — so here we journal the confirmed colour crop to
     data/flywheel/color_exemplars/<color_id>/ for the next gallery rebuild and
     return the path. (Mirrors how feedback_images/ feeds the part galleries.)

Everything degrades gracefully: if the encoder model isn't deployed yet, the
part-gallery append is skipped (reported in the result) and the colour exemplar
is still journaled. No call here raises on a missing model.
"""
from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
# Confirmed colour crops land here, one dir per Rebrickable colour id, ready for
# ml/scripts/color_gallery_append.py append-dir to fold into color_model.npz.
COLOR_EXEMPLAR_DIR = _BACKEND_DIR / "data" / "flywheel" / "color_exemplars"
# Confirmed part crops (mirrors feedback_images/, but specifically the ones we
# embedded + appended, so an offline rebuild of the catalogue can reuse them).
PART_EXEMPLAR_DIR = _BACKEND_DIR / "data" / "flywheel" / "part_exemplars"


@dataclass
class IngestResult:
    """Outcome of folding one confirmed scan into the galleries."""
    part_num: str
    color_id: Optional[str]
    embedded: bool = False               # did the frozen encoder produce a vector?
    visual_search_added: bool = False    # appended to the element-level catalogue
    embedding_library_added: bool = False  # appended to the part-level k-NN
    color_exemplar_path: Optional[str] = None  # journaled colour crop, if saved
    part_exemplar_path: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "part_num": self.part_num,
            "color_id": self.color_id,
            "embedded": self.embedded,
            "visual_search_added": self.visual_search_added,
            "embedding_library_added": self.embedding_library_added,
            "color_exemplar_path": self.color_exemplar_path,
            "part_exemplar_path": self.part_exemplar_path,
            "gallery_updated": self.visual_search_added or self.embedding_library_added,
            "notes": self.notes,
        }


def _save_crop(crop_bytes: bytes, root: Path, key: str) -> Optional[str]:
    """Persist a crop under root/<key>/<sha>.jpg (re-encoded JPEG). Returns path."""
    import hashlib
    try:
        from PIL import Image as PILImage
        d = root / str(key)
        d.mkdir(parents=True, exist_ok=True)
        sha = hashlib.sha1(crop_bytes).hexdigest()[:12]
        out = d / f"{sha}.jpg"
        if not out.exists():
            PILImage.open(io.BytesIO(crop_bytes)).convert("RGB").save(
                str(out), format="JPEG", quality=90)
        return str(out)
    except Exception as e:
        logger.warning("flywheel: failed to save crop under %s: %s", root, e)
        return None


def ingest_confirmed(
    crop_bytes: bytes,
    part_num: str,
    color_id: Optional[str] = None,
    *,
    part_name: str = "",
    color_name: str = "",
    color_hex: str = "",
    element_id: str = "",
    journal_crops: bool = True,
) -> IngestResult:
    """Append one CONFIRMED (crop, part_num, color_id) to the galleries.

    Returns an IngestResult describing exactly what was updated. Never raises on
    a missing model — the recognition cascade keeps working either way; this just
    makes the next scan of the same brick land it faster.
    """
    part_num = str(part_num).strip()
    cid = str(color_id).strip() if color_id is not None and str(color_id).strip() else None
    result = IngestResult(part_num=part_num, color_id=cid)

    if not part_num or part_num.lower() in ("", "unknown"):
        result.notes.append("empty/unknown part_num — skipped gallery append")
        return result

    # ── 1. Embed once with the frozen encoder ────────────────────────────────
    embedding: Optional[np.ndarray] = None
    try:
        from app.ml.model_manager import ModelManager
        mm = ModelManager.get()
        if mm.encoder_available:
            embedding = mm.encode_image(crop_bytes)   # L2-normalised float32[D] or None
        else:
            result.notes.append("encoder not deployed — part gallery append skipped")
    except Exception as e:
        logger.warning("flywheel: encode failed: %s", e)
        result.notes.append(f"encode error: {e}")

    if embedding is not None:
        result.embedded = True

        # ── 2a. Append to the element-level visual-search catalogue ──────────
        try:
            from app.services import visual_search
            added = visual_search.add_entry(
                embedding=embedding,
                part_num=part_num,
                color_id=int(cid) if (cid and cid.lstrip("-").isdigit()) else None,
                part_name=part_name,
                color_name=color_name,
                color_hex=color_hex,
                element_id=element_id,
                persist=True,
            )
            result.visual_search_added = bool(added)
            if added:
                logger.info("flywheel: appended %s/%s to visual_search catalogue",
                            part_num, cid)
        except Exception as e:
            logger.warning("flywheel: visual_search append failed: %s", e)
            result.notes.append(f"visual_search error: {e}")

        # ── 2b. Append to the part-level EmbeddingLibrary k-NN ───────────────
        # NOTE: EmbeddingLibrary keys by part_num and AVERAGES new exemplars into
        # the running per-part vector (re-normalising), so repeated confirmations
        # of the same part sharpen its prototype rather than bloating the index.
        try:
            from app.ml.embedding_library import EmbeddingLibrary
            lib = EmbeddingLibrary.get()
            lib.add_exemplar(part_num, embedding, persist=True)
            result.embedding_library_added = True
            logger.info("flywheel: folded %s into EmbeddingLibrary (size=%d)",
                        part_num, lib.size)
        except AttributeError:
            # Older EmbeddingLibrary without add_exemplar: fall back to add_embedding
            try:
                from app.ml.embedding_library import EmbeddingLibrary
                lib = EmbeddingLibrary.get()
                lib.add_embedding(part_num, embedding)
                lib.save_cache()
                result.embedding_library_added = True
            except Exception as e:
                logger.warning("flywheel: EmbeddingLibrary append failed: %s", e)
                result.notes.append(f"embedding_library error: {e}")
        except Exception as e:
            logger.warning("flywheel: EmbeddingLibrary append failed: %s", e)
            result.notes.append(f"embedding_library error: {e}")

    # ── 3. Journal the crops for offline gallery rebuilds (part + colour) ────
    if journal_crops:
        result.part_exemplar_path = _save_crop(crop_bytes, PART_EXEMPLAR_DIR, part_num)
        if cid is not None:
            result.color_exemplar_path = _save_crop(crop_bytes, COLOR_EXEMPLAR_DIR, cid)
            if result.color_exemplar_path:
                logger.info("flywheel: journaled colour exemplar for color %s -> %s",
                            cid, result.color_exemplar_path)

    return result


def gallery_status() -> dict:
    """Cheap snapshot of the live galleries — used by the flywheel stats route."""
    status = {
        "encoder_available": False,
        "visual_search_loaded": False,
        "visual_search_size": 0,
        "embedding_library_size": 0,
        "color_exemplars_pending": 0,
        "part_exemplars_journaled": 0,
    }
    try:
        from app.ml.model_manager import ModelManager
        status["encoder_available"] = ModelManager.get().encoder_available
    except Exception:
        pass
    try:
        from app.services import visual_search
        status["visual_search_loaded"] = visual_search.is_loaded()
        status["visual_search_size"] = visual_search.catalog_size()
    except Exception:
        pass
    try:
        from app.ml.embedding_library import EmbeddingLibrary
        status["embedding_library_size"] = EmbeddingLibrary.get().size
    except Exception:
        pass
    try:
        if COLOR_EXEMPLAR_DIR.exists():
            status["color_exemplars_pending"] = sum(
                1 for _ in COLOR_EXEMPLAR_DIR.rglob("*.jpg"))
        if PART_EXEMPLAR_DIR.exists():
            status["part_exemplars_journaled"] = sum(
                1 for _ in PART_EXEMPLAR_DIR.rglob("*.jpg"))
    except Exception:
        pass
    status["updated_at"] = time.time()
    return status
