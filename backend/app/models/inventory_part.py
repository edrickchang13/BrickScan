"""
InventoryPart model — denormalized Rebrickable inventory_parts data.
Maps set_num + part_num + color_id → quantity for fast set completion lookups.
Populated by the data pipeline from Rebrickable CSV dumps.
"""

from sqlalchemy import Column, String, Integer, Boolean, ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from app.core.database import Base
import uuid


class InventoryPart(Base):
    __tablename__ = "inventory_parts"
    __table_args__ = (
        UniqueConstraint("set_num", "part_num", "color_id", "is_spare", name="uq_inventory_part"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    set_num = Column(String, nullable=False, index=True)
    part_num = Column(String, nullable=False, index=True)
    color_id = Column(Integer, nullable=False)  # Rebrickable color id (int)
    quantity = Column(Integer, nullable=False, default=1)
    is_spare = Column(Boolean, default=False, nullable=False)
