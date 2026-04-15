"""
Hybrid LEGO brick recognition service — updated cascade.

New pipeline (replaces local ONNX EfficientNet-B4 at the tail):
  1. Brickognize API      — primary, fast ConvNeXt-T
  2. Gemini 2.5 Flash     — fallback when Brickognize is uncertain
  3. Contrastive k-NN     — 128-dim DINOv2 embedding similarity search
                             (source="contrastive_knn")
  4. Distilled MobileNetV3 — knowledge-distilled from DINOv2 teacher
                             (source="distilled_model")

Steps 3-4 use the new ModelManager / EmbeddingLibrary singletons which
fail gracefully (returning [] / None) when the Spark-trained model files
are not yet present, so the existing behaviour is fully preserved during
development.

The source field is propagated to the mobile app for display.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Awaitable, Callable, Dict, List, Optional

from app.services.brickognize_client import identify_part as brickognize_predict
from app.services.gemini_service import identify_piece as gemini_predict
from app.services.ml_inference import predict as onnx_predict   # legacy EfficientNet fallback
from app.services.ml_inference import predict_with_tta

logger = logging.getLogger(__name__)

# Confidence thresholds
BRICKOGNIZE_HIGH_CONFIDENCE = 0.80
BRICKOGNIZE_LOW_CONFIDENCE  = 0.40
GEMINI_MIN_CONFIDENCE       = 0.50
AGREEMENT_BOOST             = 0.10

# k-NN cosine distance below which we trust the result (0 = identical, 2 = max)
KNN_DISTANCE_THRESHOLD = 0.30

# Gates for the optional cascade enhancements. All disabled by default —
# Brickognize alone (with Gemini fallback for low-confidence scans) is the
# baseline that produced the best perceived accuracy. Re-enable per-feature
# via env vars once the FeedbackStatsScreen shows enough data to A/B them.
#
#   SCAN_TTA_ENABLED=true        # 4-rotation TTA over local ONNX (legacy model)
#   SCAN_ALWAYS_RUN_GEMINI=true  # run Gemini even when Brickognize is confident
TTA_ENABLED            = os.environ.get("SCAN_TTA_ENABLED", "false").lower() == "true"
ALWAYS_RUN_GEMINI      = os.environ.get("SCAN_ALWAYS_RUN_GEMINI", "false").lower() == "true"
TTA_TRIGGER_CONFIDENCE = 0.70  # only invoke TTA when top prediction is below this

ProgressCb = Optional[Callable[[str, int, str, Optional[Dict[str, Any]]], Awaitable[None]]]


async def _emit(cb: ProgressCb, stage: str, percent: int, message: str = "",
                partial: Optional[Dict[str, Any]] = None) -> None:
    if cb is None:
        return
    try:
        await cb(stage, percent, message, partial)
    except Exception as e:
        logger.debug("Progress callback failed (non-fatal): %s", e)


def _normalize_part_num(part_num: str) -> str:
    return part_num.strip().lower().lstrip("0") or "0"


def _sources_agree(pred_a: Dict, pred_b: Dict) -> bool:
    a = _normalize_part_num(pred_a.get("part_num", ""))
    b = _normalize_part_num(pred_b.get("part_num", ""))
    return a == b and a not in ("", "unknown")


def _merge_predictions(
    brickognize_preds: List[Dict],
    gemini_preds: List[Dict],
    local_preds: List[Dict],
) -> List[Dict]:
    """
    Merge predictions from all sources into a ranked list of up to 5.

    Priority:
      1. Brickognize high-confidence  (> 80%) → use it (boost if Gemini agrees)
      2. Brickognize + Gemini agree   → boost and use Brickognize details
      3. Disagreement                 → keep both, higher confidence first
      4. Fill remaining slots from local_preds (k-NN or distilled)
    """
    results: List[Dict] = []
    seen: set = set()

    bg_top = brickognize_preds[0] if brickognize_preds else None
    gm_top = gemini_preds[0]      if gemini_preds      else None

    if bg_top and bg_top.get("confidence", 0) >= BRICKOGNIZE_HIGH_CONFIDENCE:
        entry = {**bg_top, "source": "brickognize"}
        if gm_top and _sources_agree(bg_top, gm_top):
            entry["confidence"] = min(1.0, entry["confidence"] + AGREEMENT_BOOST)
            entry["source"] = "brickognize+gemini"
        results.append(entry)
        seen.add(_normalize_part_num(entry["part_num"]))

    elif bg_top and gm_top:
        if _sources_agree(bg_top, gm_top):
            entry = {**bg_top, "source": "brickognize+gemini"}
            entry["confidence"] = min(
                1.0,
                max(bg_top["confidence"], gm_top.get("confidence", 0)) + AGREEMENT_BOOST,
            )
            results.append(entry)
            seen.add(_normalize_part_num(entry["part_num"]))
        else:
            bg_conf = bg_top.get("confidence", 0)
            gm_conf = gm_top.get("confidence", 0)
            first, second = (bg_top, gm_top) if bg_conf >= gm_conf else (gm_top, bg_top)
            first_src  = "brickognize" if first  is bg_top else "gemini"
            second_src = "brickognize" if second is bg_top else "gemini"
            results.append({**first,  "source": first_src})
            seen.add(_normalize_part_num(first["part_num"]))
            norm2 = _normalize_part_num(second["part_num"])
            if norm2 not in seen:
                results.append({**second, "source": second_src})
                seen.add(norm2)

    elif bg_top:
        results.append({**bg_top, "source": "brickognize"})
        seen.add(_normalize_part_num(bg_top["part_num"]))
    elif gm_top:
        results.append({**gm_top, "source": "gemini"})
        seen.add(_normalize_part_num(gm_top["part_num"]))

    # Fill with remaining predictions from all sources
    remaining: List[Dict] = []
    for p in brickognize_preds[1:]:
        remaining.append({**p, "source": "brickognize"})
    for p in (gemini_preds[1:] if gemini_preds else []):
        remaining.append({**p, "source": "gemini"})
    for p in local_preds:
        remaining.append(p)  # source already set by model_manager / onnx_predict

    remaining.sort(key=lambda x: x.get("confidence", 0), reverse=True)
    for p in remaining:
        if len(results) >= 5:
            break
        norm = _normalize_part_num(p.get("part_num", ""))
        if norm not in seen and norm != "unknown":
            results.append(p)
            seen.add(norm)

    return results


async def hybrid_predict(
    image_bytes: bytes,
    progress_cb: ProgressCb = None,
    enable_tta: bool = True,
) -> List[Dict[str, Any]]:
    """
    Run the full hybrid cascade and return ranked predictions.

    Always runs Brickognize + Gemini + local models in parallel (the old
    "skip Gemini if Brickognize is confident" gate caused obscure-part misses
    when Brickognize was *confidently wrong*). The merge logic still rewards
    cross-source agreement.

    If `enable_tta` and the top result is below TTA_TRIGGER_CONFIDENCE,
    re-runs the local ONNX classifier under 4-rotation TTA to stabilise.

    Each prediction dict contains:
      part_num, part_name, confidence, color_name, color_hex, color_id, source

    Args:
        image_bytes:  raw JPEG/PNG bytes of the scan
        progress_cb:  optional async fn(stage, percent, msg, partial) for SSE streaming
        enable_tta:   if False, skip the TTA stabilisation pass even when conditions match
    """
    logger.info("Starting hybrid recognition pipeline")
    await _emit(progress_cb, "brickognize_start", 10, "Querying Brickognize…")

    # Always run Brickognize + local models in parallel (cheap).
    # Gemini is only joined in if Brickognize is uncertain OR ALWAYS_RUN_GEMINI
    # is forced on. This protects accuracy: when Brickognize is confidently
    # right, Gemini's hallucinations would otherwise pollute slots 2-3.
    bg_task    = asyncio.create_task(_safe_brickognize(image_bytes))
    local_task = asyncio.create_task(_safe_local_predict(image_bytes))

    brickognize_preds = await bg_task
    bg_top_conf = brickognize_preds[0].get("confidence", 0) if brickognize_preds else 0
    await _emit(
        progress_cb, "brickognize_done", 35,
        f"Brickognize: {bg_top_conf * 100:.0f}% confidence",
        partial={"predictions": brickognize_preds[:3]},
    )

    gemini_preds: List[Dict] = []
    should_run_gemini = ALWAYS_RUN_GEMINI or bg_top_conf < BRICKOGNIZE_HIGH_CONFIDENCE
    if should_run_gemini:
        await _emit(progress_cb, "gemini_start", 45, "Asking Gemini for a second opinion…")
        gemini_preds = await _safe_gemini(image_bytes)
        gm_top_conf = gemini_preds[0].get("confidence", 0) if gemini_preds else 0
        await _emit(
            progress_cb, "gemini_done", 65,
            f"Gemini: {gm_top_conf * 100:.0f}% confidence",
            partial={"predictions": gemini_preds[:3]},
        )
    else:
        logger.info(
            "Brickognize high-confidence (%.0f%%) — skipping Gemini",
            bg_top_conf * 100,
        )
        await _emit(
            progress_cb, "gemini_skipped", 65,
            f"Brickognize confident ({bg_top_conf * 100:.0f}%) — skipping Gemini",
        )

    await _emit(progress_cb, "local_models", 75, "Running local models…")
    local_preds = await local_task

    await _emit(progress_cb, "merge", 85, "Merging cross-source results…")
    merged = _merge_predictions(brickognize_preds, gemini_preds, local_preds)

    # TTA stabilisation pass — only when no source is confident
    top_conf = merged[0].get("confidence", 0) if merged else 0
    if enable_tta and TTA_ENABLED and top_conf < TTA_TRIGGER_CONFIDENCE:
        await _emit(progress_cb, "tta", 90, "Stabilising with rotation augmentation…")
        try:
            tta_preds = await predict_with_tta(image_bytes, rotations=4)
            tta_with_source = [{**p, "source": "tta_local"} for p in tta_preds]
            merged = _merge_predictions(brickognize_preds, gemini_preds, tta_with_source + local_preds)
        except Exception as e:
            logger.warning("TTA stabilisation failed (non-fatal): %s", e)

    if merged:
        top = merged[0]
        logger.info(
            "Hybrid result: %s (%s) — %.1f%% [%s]",
            top.get("part_num"), top.get("part_name"),
            top.get("confidence", 0) * 100, top.get("source"),
        )
    else:
        logger.warning("Hybrid recognition returned no predictions")

    return merged


async def _safe_brickognize(image_bytes: bytes) -> List[Dict]:
    try:
        return await brickognize_predict(image_bytes) or []
    except Exception as e:
        logger.warning("Brickognize failed: %s", e)
        return []


async def _safe_gemini(image_bytes: bytes) -> List[Dict]:
    try:
        return await gemini_predict(image_bytes) or []
    except Exception as e:
        logger.warning("Gemini failed: %s", e)
        return []


async def _safe_local_predict(image_bytes: bytes) -> List[Dict]:
    """
    Run the local model sub-cascade:
      1. Contrastive k-NN (fast, requires embeddings cache)
      2. Distilled MobileNetV3 (if k-NN isn't confident enough)
      3. Legacy EfficientNet ONNX (final fallback)

    Returns a merged list with the best local prediction first.
    """
    try:
        from app.ml.model_manager import ModelManager
        from app.ml.embedding_library import EmbeddingLibrary, KNN_CONFIDENCE_THRESHOLD

        mm  = ModelManager.get()
        lib = EmbeddingLibrary.get()

        results: List[Dict] = []

        # ── Step 1: Contrastive k-NN ─────────────────────────────────────────
        knn_top: Optional[Dict] = None
        if mm.encoder_available and lib.size > 0:
            embedding = mm.encode_image(image_bytes)
            if embedding is not None:
                neighbours = lib.knn_search(embedding, k=5)
                if neighbours:
                    for part_num, distance in neighbours:
                        # Convert cosine distance → pseudo-confidence (1 - dist/2)
                        pseudo_conf = max(0.0, 1.0 - distance / 2.0)
                        entry: Dict = {
                            "part_num":   part_num,
                            "part_name":  "",
                            "confidence": pseudo_conf,
                            "color_id":   None,
                            "color_name": None,
                            "color_hex":  None,
                            "source":     "contrastive_knn",
                        }
                        results.append(entry)

                    top_dist = neighbours[0][1]
                    if top_dist <= KNN_CONFIDENCE_THRESHOLD:
                        knn_top = results[0]
                        logger.info(
                            "k-NN match: %s (dist=%.3f, conf=%.0f%%)",
                            knn_top["part_num"], top_dist, knn_top["confidence"] * 100,
                        )

        # ── Step 2: Distilled student (if k-NN uncertain or unavailable) ─────
        if knn_top is None and mm.student_available:
            student_preds = mm.classify_image(image_bytes, top_k=5)
            if student_preds:
                logger.info(
                    "Distilled model: %s (%.0f%%)",
                    student_preds[0]["part_num"], student_preds[0]["confidence"] * 100,
                )
                # Merge: deduplicate with k-NN results, student first when k-NN uncertain
                seen_parts = {p["part_num"] for p in results}
                merged_local: List[Dict] = student_preds.copy()
                for p in results:
                    if p["part_num"] not in {x["part_num"] for x in merged_local}:
                        merged_local.append(p)
                merged_local.sort(key=lambda x: x.get("confidence", 0), reverse=True)
                return merged_local[:5]

        # k-NN was confident — return its results (they're already sorted)
        if results:
            return results[:5]

    except Exception as e:
        logger.warning("New local models failed: %s — trying legacy ONNX", e)

    # ── Step 3: Legacy EfficientNet ONNX (final fallback) ────────────────────
    try:
        legacy = await onnx_predict(image_bytes) or []
        return [{**p, "source": "local_model"} for p in legacy]
    except Exception as e:
        logger.warning("Legacy ONNX fallback failed: %s", e)
        return []
