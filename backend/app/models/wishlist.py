"""
SQLAlchemy model for user wishlists.
Tracks which LEGO sets users want to build.
"""

from sqlalchemy import Column, String, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from app.core.database import Base


class Wishlist(Base):
    """
    Represents a user's wishlist of LEGO sets they want to build.

    Attributes:
    - id: Unique identifier (UUID)
    - user_id: Foreign key to User
    - set_num: Foreign key to LegoSet (LEGO set number)
    - added_at: When the set was added to wishlist
    """

    __tablename__ = "wishlist"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    set_num = Column(
        String(50), ForeignKey("lego_sets.set_num", ondelete="CASCADE"), nullable=False
    )
    added_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships (no back_populates — User/LegoSet don't declare wishlists)
    user = relationship("User")
    lego_set = relationship("LegoSet")

    # Unique constraint: each user can only wishlist a set once
    __table_args__ = (UniqueConstraint("user_id", "set_num", name="uq_wishlist_user_set"),)

    def __repr__(self):
        return f"<Wishlist(user_id={self.user_id}, set_num={self.set_num})>"
