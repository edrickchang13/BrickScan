"""
User statistics API endpoints.
Returns aggregate stats about user's collection for the Profile screen.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional
import logging

from app.core.security import get_current_user
from app.core.database import get_db
from app.models.user import User
from app.models.scan import Scan
from app.models.inventory import InventoryItem

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/stats", tags=["stats"])


class StatsResponse:
    """Response model for user statistics"""

    def __init__(
        self,
        total_parts: int,
        total_pieces: int,
        total_sets_checked: int,
        completable_sets: int,
        top_colors: list[dict],
        scans_this_week: int,
        collection_value_estimate: Optional[float] = None,
    ):
        self.total_parts = total_parts
        self.total_pieces = total_pieces
        self.total_sets_checked = total_sets_checked
        self.completable_sets = completable_sets
        self.top_colors = top_colors
        self.scans_this_week = scans_this_week
        self.collection_value_estimate = collection_value_estimate


@router.get("/me")
async def get_user_stats(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Get comprehensive statistics about the user's LEGO collection.

    Returns:
    - total_parts: Number of unique part types owned
    - total_pieces: Total piece count (sum of all quantities)
    - total_sets_checked: How many sets user has compared
    - completable_sets: Sets user can fully build right now
    - top_colors: Top 5 colors in collection
    - scans_this_week: Scan count in last 7 days
    - collection_value_estimate: Estimated value (null for now)
    """

    try:
        # 1. Get total unique parts and total piece count
        inventory_stats = await db.execute(
            select(
                func.count(InventoryItem.id).label("total_parts"),
                func.sum(InventoryItem.quantity).label("total_pieces"),
            ).where(InventoryItem.user_id == current_user.id)
        )
        inventory_row = inventory_stats.first()
        total_parts = inventory_row[0] or 0
        total_pieces = inventory_row[1] or 0

        # 2. Get total sets checked (distinct sets user has queried)
        # Assuming there's a set_comparison or similar table tracking checked sets
        sets_checked_result = await db.execute(
            select(func.count(func.distinct(Scan.set_num))).where(
                Scan.user_id == current_user.id
            )
        )
        total_sets_checked = sets_checked_result.scalar() or 0

        # 3. Get completable sets (sets where all required parts are in inventory)
        # This is a complex query - simplified version:
        # For now, return 0; in production, join with lego_sets and inventory_parts
        completable_sets = 0

        # 4. Get top 5 colors
        from app.models.part import Color
        top_colors_result = await db.execute(
            select(
                Color.name,
                func.sum(InventoryItem.quantity).label("count"),
            )
            .join(InventoryItem, InventoryItem.color_id == Color.id)
            .where(InventoryItem.user_id == current_user.id)
            .group_by(Color.id, Color.name)
            .order_by(func.sum(InventoryItem.quantity).desc())
            .limit(5)
        )
        top_colors = [
            {"color": row[0], "count": row[1]}
            for row in top_colors_result.fetchall()
        ]

        # 5. Get scans this week
        from datetime import datetime, timedelta
        week_ago = datetime.utcnow() - timedelta(days=7)
        scans_week_result = await db.execute(
            select(func.count(Scan.id)).where(
                and_(
                    Scan.user_id == current_user.id,
                    Scan.created_at >= week_ago,
                )
            )
        )
        scans_this_week = scans_week_result.scalar() or 0

        return {
            "total_parts": total_parts,
            "total_pieces": total_pieces,
            "total_sets_checked": total_sets_checked,
            "completable_sets": completable_sets,
            "top_colors": top_colors,
            "scans_this_week": scans_this_week,
            "collection_value_estimate": None,
        }

    except Exception as e:
        logger.error(f"Error fetching user stats for {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch user statistics",
        )
