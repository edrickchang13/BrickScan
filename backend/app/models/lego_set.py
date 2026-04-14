from sqlalchemy import Column, String, Integer, Boolean, ForeignKey, DateTime, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
import uuid
from app.core.database import Base


class Theme(Base):
    __tablename__ = "themes"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, unique=True, nullable=False, index=True)
    parent_id = Column(UUID(as_uuid=True), ForeignKey("themes.id"), nullable=True)

    sets = relationship("LegoSet", back_populates="theme")


class LegoSet(Base):
    __tablename__ = "lego_sets"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    set_num = Column(String, unique=True, nullable=False, index=True)
    name = Column(String, nullable=False)
    year = Column(Integer, nullable=True)
    theme_id = Column(UUID(as_uuid=True), ForeignKey("themes.id"), nullable=True)
    num_parts = Column(Integer, nullable=True)
    img_url = Column(String, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    theme = relationship("Theme", back_populates="sets")
    parts = relationship("SetPart", back_populates="lego_set", cascade="all, delete-orphan")


class SetPart(Base):
    __tablename__ = "set_parts"
    __table_args__ = (UniqueConstraint("set_id", "part_id", "color_id", name="uq_set_part_color"),)

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    set_id = Column(UUID(as_uuid=True), ForeignKey("lego_sets.id"), nullable=False)
    part_id = Column(UUID(as_uuid=True), ForeignKey("parts.id"), nullable=False)
    color_id = Column(UUID(as_uuid=True), ForeignKey("colors.id"), nullable=False)
    quantity = Column(Integer, nullable=False)
    is_spare = Column(Boolean, default=False, nullable=False)

    lego_set = relationship("LegoSet", back_populates="parts")
