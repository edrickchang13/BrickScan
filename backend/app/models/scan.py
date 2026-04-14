"""
Scan model — records each camera scan attempt by a user.
Tracks the model's prediction, confidence, user confirmation, and timing.
"""

from sqlalchemy import Column, String, Integer, Float, ForeignKey, DateTime
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime, timezone
import uuid
from app.core.database import Base


class Scan(Base):
    __tablename__ = "scans"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    image_s3_key = Column(String, nullable=True)
    prediction = Column(String, nullable=True, index=True)   # predicted part_num
    confidence = Column(Float, nullable=True)
    confirmed_part_num = Column(String, nullable=True, index=True)
    set_num = Column(String, nullable=True, index=True)      # set context if provided
    processing_time_ms = Column(Integer, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,
    )
