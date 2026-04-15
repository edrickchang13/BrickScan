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
from typing import Any, Dict, List, Literal, Optional


FeedbackType = Literal[
    "top_correct",          # user tapped "Yes, top pick is right"       (rank 0)
    "alternative_correct",  # user tapped a non-top prediction from top-5 (rank 1..N)
    "none_correct",         # user searched for correct part (wasn't shown) (rank -1)
    "partially_correct",    # right brick, wrong colour
]


class PredictionShown(BaseModel):
    """One entry in the top-5 that was shown on ScanResultScreen at feedback time."""
    part_num: str
    part_name: Optional[str] = None
    confidence: float
    source: Optional[str] = None
    color_id: Optional[str] = None
    color_hex: Optional[str] = None


class ScanFeedback(BaseModel):
    """
    User correction submitted after a scan.

    `scan_id` is generated on the mobile client as:
        `scan_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
    It doesn't match any server record — it's just a stable dedup key.

    The `feedback_type`, `correct_rank`, `predictions_shown`, and
    `time_to_confirm_ms` fields are optional — older mobile clients omit
    them and the backend will derive feedback_type heuristically from the
    part_num comparison.
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

    # v2 fields — all optional for backwards compat
    feedback_type: Optional[FeedbackType] = Field(
        None,
        description="Explicit three-way feedback signal. Derived server-side when absent.",
    )
    correct_rank: Optional[int] = Field(
        None,
        description="Where the right answer sat in the shown top-5 (0..N), or -1 if not shown.",
    )
    predictions_shown: Optional[List[PredictionShown]] = Field(
        None,
        description="The top-5 shown to the user at feedback time (for confusion analysis).",
    )
    time_to_confirm_ms: Optional[int] = Field(
        None, ge=0, description="How long the user deliberated, in milliseconds."
    )


class ScanFeedbackResponse(BaseModel):
    saved: bool
    will_improve_model: bool  # True when correct_part_num != predicted_part_num
    feedback_id: str


class FeedbackConfusionPair(BaseModel):
    predicted_part_num: str
    correct_part_num: str
    count: int


class SourceStats(BaseModel):
    """Per-model accuracy breakdown (e.g. Brickognize vs. Gemini vs. k-NN)."""
    source: str
    count: int                   # total feedback rows attributable to this source
    correct: int                 # top_correct + partially_correct (brick was right)
    accuracy: float              # correct / count, 0.0 when count == 0


class AccuracyTrendPoint(BaseModel):
    """One point on the weekly accuracy chart."""
    week_ending: str             # ISO date
    top1_accuracy: float
    top3_accuracy: float
    sample_size: int


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

    # v2 fields
    top1_accuracy: float = Field(
        0.0, description="Rolling 30-day top-1 accuracy (top pick was right)."
    )
    top3_accuracy: float = Field(
        0.0, description="Rolling 30-day top-3 accuracy (correct answer was in top-3)."
    )
    by_source: List[SourceStats] = Field(
        default_factory=list, description="Per-model accuracy breakdown."
    )
    accuracy_trend: List[AccuracyTrendPoint] = Field(
        default_factory=list, description="Weekly snapshots over the last 8 weeks."
    )


class FeedbackSnapshotResponse(BaseModel):
    """Returned by POST /feedback/snapshot."""
    snapshot_date: str
    top1_accuracy: float
    top3_accuracy: float
    sample_size: int
    by_source: Dict[str, Dict[str, Any]]
