"""
SQLAlchemy models for local inventory system.

LocalInventoryPart: Represents a part in the user's physical inventory
ScanSession:        Groups scans into named sessions
ScanFeedback:       Active-learning corrections — user says "model was wrong, it's actually X"
"""

from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    Boolean,
    DateTime,
    Text,
    UniqueConstraint,
)
from datetime import datetime, timezone
import uuid
from app.local_inventory.database import Base


class LocalInventoryPart(Base):
    """
    Represents a single LEGO part in the user's local inventory.
    """

    __tablename__ = "local_inventory_parts"
    __table_args__ = (
        UniqueConstraint(
            "part_num",
            "color_id",
            name="uq_local_part_color",
        ),
    )

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    part_num = Column(String(50), nullable=False, index=True)
    color_id = Column(Integer, nullable=True, index=True)
    color_name = Column(String(100), nullable=True)
    color_hex = Column(String(10), nullable=True)
    quantity = Column(Integer, nullable=False, default=1)
    confidence = Column(Float, nullable=False, default=0.0)
    user_confirmed = Column(Boolean, nullable=False, default=False)
    image_path = Column(Text, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,
    )


class ScanSession(Base):
    """Groups multiple scans into a named session for organisation."""

    __tablename__ = "scan_sessions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    set_name = Column(String(255), nullable=False, index=True)
    completed = Column(Boolean, nullable=False, default=False, index=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


class ScanFeedback(Base):
    """
    Active-learning correction submitted by the user after a scan.

    When the model gets a prediction wrong, the user can say "it's actually 3001"
    (for example).  These corrections are:
      - Stored here for training data collection
      - Saved as labelled images in data/feedback_images/<correct_part_num>/
      - Aggregated via GET /feedback/stats to show confusion pairs

    `used_for_training` is set to True after the feedback record has been
    exported / incorporated into a training run.
    """

    __tablename__ = "scan_feedback"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))

    # Client-generated ID (e.g. `scan_<timestamp>_<random>`) ties the feedback
    # back to the original scan event without us storing every scan.
    scan_id = Column(String(100), nullable=False, index=True)

    predicted_part_num = Column(String(50), nullable=False, index=True)
    correct_part_num   = Column(String(50), nullable=False, index=True)
    correct_color_id   = Column(String(20), nullable=True)

    # Local path to the saved JPEG (set only when image_base64 was provided
    # and the correction differs from the prediction)
    image_path = Column(Text, nullable=True)

    confidence = Column(Float, nullable=False, default=0.0)
    source     = Column(String(50), nullable=False, default="")

    # False until this record is exported into a training dataset
    used_for_training = Column(Boolean, nullable=False, default=False, index=True)

    # Three-way feedback typing. NULL = legacy row written before v2 schema.
    #   top_correct         — user tapped "yes, top pick is right"  (rank 0)
    #   alternative_correct — user tapped a non-top prediction      (rank 1..N)
    #   none_correct        — user typed a correct_part_num that wasn't shown (rank -1)
    #   partially_correct   — right brick, wrong colour             (rank 0, color diff)
    feedback_type = Column(String(30), nullable=True, index=True)

    # Position of the correct answer in the top-5 that was shown.
    # 0 = top pick, 1..N = alternative, -1 = wasn't in the list.
    correct_rank = Column(Integer, nullable=True, index=True)

    # Full top-5 serialised at feedback time so we can compute top-N accuracy
    # and per-source accuracy later without re-running inference.
    predictions_shown_json = Column(Text, nullable=True)

    # UX telemetry: how long the user deliberated before tapping a choice.
    time_to_confirm_ms = Column(Integer, nullable=True)

    timestamp = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,
    )


class FeedbackEvalSnapshot(Base):
    """
    Weekly frozen accuracy snapshot. One row per run of POST /feedback/snapshot.

    Powers the "model has improved from 62% → 78%" trend chart on the
    FeedbackStatsScreen. Computed from the rolling window of ScanFeedback
    rows at snapshot time.
    """

    __tablename__ = "feedback_eval_snapshots"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    snapshot_date = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,
    )

    # Top-1 = top pick was correct; Top-3 = correct answer was in shown top-3.
    top1_accuracy = Column(Float, nullable=False)
    top3_accuracy = Column(Float, nullable=False)

    # JSON: {"brickognize": {"count": 42, "accuracy": 0.78}, "gemini": {...}, ...}
    by_source_json = Column(Text, nullable=True)

    # How many feedback rows fed this snapshot (for confidence intervals)
    sample_size = Column(Integer, nullable=False, default=0)

    # The window end — snapshots look back N days from this.
    window_days = Column(Integer, nullable=False, default=30)
