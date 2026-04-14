"""
Pydantic schemas for Workstream B — Active Learning Pipeline.

These extend the existing schemas in schemas.py without modifying it.

REGISTRATION: In main.py, after the existing local_inventory router is included,
add:
    from app.local_inventory.feedback_routes import feedback_router
    app.include_router(feedback_router)

(Add it BEFORE the existing local_inventory router if you want the
 updated /scan-multi endpoint with YOLO to take precedence.)
"""

from pydantic import BaseModel, Field
from typing import Optional, List


class ScanFeedback(BaseModel):
    """
    User correction submitted after a scan.

    `scan_id` is generated on the mobile client as:
        `scan_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
    It doesn't match any server record — it's just a stable dedup key.
    """

    scan_id: str = Field(..., description="Client-generated stable scan event ID")
    predicted_part_num: str = Field(..., description="What the model predicted")
    correct_part_num: str = Field(..., description="What the user says it actually is")
    correct_color_id: Optional[str] = Field(None, description="Correct Rebrickable color ID")
    image_base64: Optional[str] = Field(
        None,
        description=(
            "Original scan image as base64 JPEG. "
            "Saved to disk only when correct_part_num != predicted_part_num."
        ),
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence at scan time")
    source: str = Field(
        ...,
        description=(
            "Which model made the prediction: "
            "'brickognize', 'gemini', 'contrastive_knn', 'distilled_model', 'local_model'"
        ),
    )


class ScanFeedbackResponse(BaseModel):
    saved: bool
    will_improve_model: bool  # True when correct_part_num != predicted_part_num
    feedback_id: str


class FeedbackConfusionPair(BaseModel):
    predicted_part_num: str
    correct_part_num: str
    count: int


class FeedbackStats(BaseModel):
    """
    Aggregate view of the active-learning feedback collection.
    Returned by GET /api/local-inventory/feedback/stats.
    """

    total_corrections: int = Field(
        ..., description="Corrections where correct_part_num != predicted_part_num"
    )
    agreement_count: int = Field(
        ..., description="Times user confirmed prediction was correct (same part)"
    )
    top_confused_pairs: List[FeedbackConfusionPair] = Field(
        ..., description="Top 10 predicted→correct confusion pairs"
    )
    parts_with_feedback: int = Field(
        ..., description="Unique correct_part_num values that have saved images"
    )
    images_saved: int = Field(
        ..., description="Total labelled JPEG images saved to feedback_images/"
    )
    pending_training: int = Field(
        ..., description="Corrections not yet exported for a training run"
    )
