"""
Active-learning FLYWHEEL routes — uncertainty sampling + confirm/ingest.

This EXTENDS the feedback pipeline in feedback_routes.py (it does not modify it).
Register after the existing routers in main.py:

    from app.local_inventory.flywheel_routes import flywheel_router
    app.include_router(flywheel_router)

Endpoints
---------
  POST /api/local-inventory/flywheel/confirm
        A CONFIRMED (crop, part_num, color_id) — embed the crop and APPEND it to
        the galleries (no retraining), then log the confirmation as feedback so
        accuracy stats still see it. The next scan of the same brick benefits
        immediately.

  POST /api/local-inventory/flywheel/check-uncertainty
        Given the top-k predictions of a scan, return whether it should be
        flagged for human review (k-NN top-1→top-2 margin < tau, OR top-1 below
        an absolute confidence floor). Pure function over the request — no I/O.

  GET  /api/local-inventory/flywheel/status
        Live gallery sizes + how many confirmed exemplars are journaled for the
        next offline colour-gallery rebuild.

The uncertainty trigger
-----------------------
The recognition spine ranks candidates by cosine similarity in embedding space.
The most informative thing to ask a user about is a scan where the top two
DIFFERENT parts are nearly tied — the model literally can't separate them. So
the primary signal is the MARGIN:

    margin = sim(top-1) - sim(next prediction with a different part_num)

`FLYWHEEL_MARGIN_TAU` (default 0.05, cosine-similarity units) is the gap below
which we flag for review. We also keep an absolute floor (`conf < tau_abs`,
default 0.55, matching visual_search's confident-match floor) so a uniformly
low-confidence scan is flagged even when nothing is close behind it. A user
correction is, of course, always a trigger — that path is the existing
/scan-feedback endpoint, which the confirm endpoint below also writes to.
"""
from __future__ import annotations

import base64
import logging
import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.local_inventory.database import get_local_db
from app.local_inventory.models import ScanFeedback as ScanFeedbackModel

logger = logging.getLogger(__name__)

# Margin (top-1 minus next-different-part), in cosine-similarity units, below
# which a scan is "uncertain" and worth a review prompt. Tunable via env so it
# can be retuned from the FeedbackStatsScreen data without a code change.
FLYWHEEL_MARGIN_TAU = float(os.environ.get("FLYWHEEL_MARGIN_TAU", "0.05"))
# Absolute top-1 confidence floor: flag a scan whose best guess is below this
# even if nothing is close behind it. 0.55 matches visual_search.min_similarity.
FLYWHEEL_CONF_FLOOR = float(os.environ.get("FLYWHEEL_CONF_FLOOR", "0.55"))

flywheel_router = APIRouter(
    prefix="/api/local-inventory",
    tags=["flywheel"],
)


# ── uncertainty signal (pure function, also importable by the scan path) ──────

def _norm_part(pn: Optional[str]) -> str:
    return (pn or "").strip().lower().lstrip("0") or "0"


def should_flag_for_review(
    predictions: List[Dict[str, Any]],
    tau: float = FLYWHEEL_MARGIN_TAU,
    conf_floor: float = FLYWHEEL_CONF_FLOOR,
) -> Dict[str, Any]:
    """Decide whether a scan's predictions warrant a human-review prompt.

    `predictions` is the ranked list the cascade returns (dicts with at least
    `part_num` and `confidence`). Returns:
        {flag: bool, reason: str|None, margin: float|None, top1_confidence: float}

    Triggers (either is sufficient):
      - low_margin: gap from top-1 to the next DIFFERENT part_num is < tau
      - low_confidence: top-1 confidence < conf_floor
    A scan that is confident AND well-separated is not flagged.
    """
    if not predictions:
        return {"flag": False, "reason": "no_predictions", "margin": None,
                "top1_confidence": 0.0}

    top = predictions[0]
    top_conf = float(top.get("confidence", 0.0) or 0.0)
    top_part = _norm_part(top.get("part_num"))

    # Margin to the next prediction that is a DIFFERENT part (skip mold/colour
    # variants of the same part — those aren't a confusion worth asking about).
    margin: Optional[float] = None
    for p in predictions[1:]:
        if _norm_part(p.get("part_num")) != top_part:
            margin = top_conf - float(p.get("confidence", 0.0) or 0.0)
            break

    if top_conf < conf_floor:
        return {"flag": True, "reason": "low_confidence", "margin": margin,
                "top1_confidence": top_conf}
    if margin is not None and margin < tau:
        return {"flag": True, "reason": "low_margin", "margin": margin,
                "top1_confidence": top_conf}
    return {"flag": False, "reason": None, "margin": margin,
            "top1_confidence": top_conf}


# ── request/response schemas ──────────────────────────────────────────────────

class PredictionIn(BaseModel):
    part_num: str
    confidence: float = 0.0


class CheckUncertaintyRequest(BaseModel):
    predictions: List[PredictionIn] = Field(..., description="Ranked scan predictions")
    tau: Optional[float] = Field(None, description="Override the margin threshold")
    conf_floor: Optional[float] = Field(None, description="Override the confidence floor")


class CheckUncertaintyResponse(BaseModel):
    flag: bool
    reason: Optional[str]
    margin: Optional[float]
    top1_confidence: float
    tau: float


class ConfirmRequest(BaseModel):
    """A confirmed (crop, part_num, color_id) to fold into the galleries."""
    scan_id: str = Field(..., description="Client-generated stable scan event id")
    part_num: str = Field(..., description="The CONFIRMED part number")
    color_id: Optional[str] = Field(None, description="Confirmed Rebrickable colour id")
    image_base64: str = Field(..., description="Base64 JPEG of the confirmed crop")
    # Optional richer labels to store on the appended catalogue row.
    part_name: str = ""
    color_name: str = ""
    color_hex: str = ""
    element_id: str = ""
    # Provenance for the feedback log so accuracy stats stay correct.
    predicted_part_num: Optional[str] = Field(
        None, description="What the model had predicted (defaults to part_num = a confirmation)")
    confidence: float = 0.0
    source: str = "flywheel_confirm"
    # If this confirmation corrected a wrong top-1, set feedback_type explicitly;
    # otherwise it's logged as top_correct.
    feedback_type: Optional[str] = None
    correct_rank: Optional[int] = None


class ConfirmResponse(BaseModel):
    ingested: bool
    gallery_updated: bool
    embedded: bool
    visual_search_added: bool
    embedding_library_added: bool
    color_exemplar_path: Optional[str]
    feedback_id: Optional[str]
    notes: List[str]


# ── POST /flywheel/check-uncertainty ──────────────────────────────────────────

@flywheel_router.post("/flywheel/check-uncertainty", response_model=CheckUncertaintyResponse)
async def check_uncertainty(req: CheckUncertaintyRequest) -> CheckUncertaintyResponse:
    """Return whether these predictions should be flagged for review. Pure."""
    tau = req.tau if req.tau is not None else FLYWHEEL_MARGIN_TAU
    floor = req.conf_floor if req.conf_floor is not None else FLYWHEEL_CONF_FLOOR
    verdict = should_flag_for_review(
        [p.model_dump() for p in req.predictions], tau=tau, conf_floor=floor)
    return CheckUncertaintyResponse(tau=tau, **verdict)


# ── POST /flywheel/confirm ─────────────────────────────────────────────────────

@flywheel_router.post("/flywheel/confirm", response_model=ConfirmResponse)
async def confirm_and_ingest(
    req: ConfirmRequest,
    db: Session = Depends(get_local_db),
) -> ConfirmResponse:
    """Fold a CONFIRMED scan into the galleries (no retrain) and log feedback.

    Order of operations:
      1. Decode the crop.
      2. flywheel_ingest.ingest_confirmed → embed once, append to part galleries,
         journal the colour exemplar for the next colour-gallery rebuild.
      3. Persist a ScanFeedback row so /feedback/stats keeps an accurate picture
         (feedback_type defaults to top_correct when the confirmation matched the
         model, or whatever the client passes when it was a correction).
    """
    # 1. decode
    raw = req.image_base64
    if "," in raw:
        raw = raw.split(",", 1)[1]
    try:
        crop_bytes = base64.b64decode(raw)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"bad image_base64: {e}")

    # 2. ingest (never raises on a missing model)
    from app.ml.flywheel_ingest import ingest_confirmed
    result = ingest_confirmed(
        crop_bytes=crop_bytes,
        part_num=req.part_num,
        color_id=req.color_id,
        part_name=req.part_name,
        color_name=req.color_name,
        color_hex=req.color_hex,
        element_id=req.element_id,
    )

    # 3. log feedback (reuses the same table feedback_routes.py writes)
    feedback_id: Optional[str] = None
    try:
        predicted = req.predicted_part_num or req.part_num
        is_correction = _norm_part(predicted) != _norm_part(req.part_num)
        ftype = req.feedback_type or ("none_correct" if is_correction else "top_correct")
        record = ScanFeedbackModel(
            scan_id=req.scan_id,
            predicted_part_num=predicted,
            correct_part_num=req.part_num,
            correct_color_id=req.color_id,
            image_path=result.part_exemplar_path,   # the crop we just journaled
            confidence=req.confidence,
            source=req.source,
            used_for_training=False,
            feedback_type=ftype,
            correct_rank=req.correct_rank,
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        feedback_id = record.id
    except Exception as e:
        logger.warning("flywheel/confirm: feedback log failed (non-fatal): %s", e)
        db.rollback()

    payload = result.as_dict()
    logger.info("flywheel/confirm: scan=%s part=%s color=%s -> %s",
                req.scan_id, req.part_num, req.color_id, payload)
    return ConfirmResponse(
        ingested=True,
        gallery_updated=payload["gallery_updated"],
        embedded=payload["embedded"],
        visual_search_added=payload["visual_search_added"],
        embedding_library_added=payload["embedding_library_added"],
        color_exemplar_path=payload["color_exemplar_path"],
        feedback_id=feedback_id,
        notes=payload["notes"],
    )


# ── GET /flywheel/status ───────────────────────────────────────────────────────

@flywheel_router.get("/flywheel/status")
async def flywheel_status() -> Dict[str, Any]:
    """Live gallery sizes + pending colour exemplars + the active thresholds."""
    from app.ml.flywheel_ingest import gallery_status
    s = gallery_status()
    s["margin_tau"] = FLYWHEEL_MARGIN_TAU
    s["conf_floor"] = FLYWHEEL_CONF_FLOOR
    return s
