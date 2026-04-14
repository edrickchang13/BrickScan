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

    timestamp = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,
    )
