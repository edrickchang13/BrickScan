from sqlalchemy import Column, String, Integer, ForeignKey, Boolean, DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
import uuid
from app.core.database import Base


class Color(Base):
    __tablename__ = "colors"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    rebrickable_id = Column(Integer, unique=True, nullable=False, index=True)
    name = Column(String, nullable=False)
    hex_code = Column(String, nullable=True)
    is_transparent = Column(Boolean, default=False, nullable=False)

    parts = relationship("Part", back_populates="color")


class PartCategory(Base):
    __tablename__ = "part_categories"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, unique=True, nullable=False, index=True)

    parts = relationship("Part", back_populates="category")


class Part(Base):
    __tablename__ = "parts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    part_num = Column(String, unique=True, nullable=False, index=True)
    name = Column(String, nullable=False)
    part_cat_id = Column(UUID(as_uuid=True), ForeignKey("part_categories.id"), nullable=True)
    year_from = Column(Integer, nullable=True)
    year_to = Column(Integer, nullable=True)
    image_url = Column(String, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    category = relationship("PartCategory", back_populates="parts")
