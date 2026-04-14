from sqlalchemy import Column, String, Integer, Float, ForeignKey, DateTime, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
import uuid
from app.core.database import Base


class InventoryItem(Base):
    __tablename__ = "inventory_items"
    __table_args__ = (UniqueConstraint("user_id", "part_id", "color_id", name="uq_user_part_color"),)

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    part_id = Column(UUID(as_uuid=True), ForeignKey("parts.id"), nullable=False)
    color_id = Column(UUID(as_uuid=True), ForeignKey("colors.id"), nullable=False)
    quantity = Column(Integer, nullable=False, default=1)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


class ScanLog(Base):
    __tablename__ = "scan_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    image_s3_key = Column(String, nullable=True)
    predicted_part_num = Column(String, nullable=True, index=True)
    confidence = Column(Float, nullable=True)
    confirmed_part_num = Column(String, nullable=True, index=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,
    )
