"""
Wishlist API endpoints.
Manage user's wishlisted LEGO sets and track build completion percentage.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select, and_, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from datetime import datetime, timezone
import logging

from app.core.security import get_current_user
from app.core.database import get_db
from app.models.user import User
from app.models.wishlist import Wishlist
from app.models.lego_set import LegoSet
from app.models.inventory import InventoryItem
from app.models.inventory_part import InventoryPart

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/wishlist", tags=["wishlist"])


class WishlistSetResponse:
    """Response for a wishlisted set with completion percentage"""

    def __init__(
        self,
        set_num: str,
        name: str,
        theme: str,
        year: int,
        num_parts: int,
        img_url: Optional[str],
        completion_percentage: float,
        added_at: datetime,
    ):
        self.set_num = set_num
        self.name = name
        self.theme = theme
        self.year = year
        self.num_parts = num_parts
        self.img_url = img_url
        self.completion_percentage = completion_percentage
        self.added_at = added_at


async def calculate_set_completion(
    user_id: str, set_num: str, db: AsyncSession
) -> float:
    """
    Calculate what percentage of a set the user can currently build.

    Returns completion % (0-100).
    """
    # Get all parts needed for this set from inventory_parts
    needed_parts_result = await db.execute(
        select(
            InventoryPart.part_num,
            InventoryPart.color_id,
            InventoryPart.quantity,
        ).where(InventoryPart.set_num == set_num)
    )
    needed_parts = needed_parts_result.fetchall()

    if not needed_parts:
        return 0.0

    total_needed = 0
    total_have = 0

    for part_num, color_id, quantity_needed in needed_parts:
        total_needed += quantity_needed

        # Check how many of this part+color user has
        user_has_result = await db.execute(
            select(func.sum(InventoryItem.quantity)).where(
                and_(
                    InventoryItem.user_id == user_id,
                    InventoryItem.part_num == part_num,
                    InventoryItem.color_id == color_id,
                )
            )
        )
        quantity_have = user_has_result.scalar() or 0
        total_have += min(quantity_have, quantity_needed)

    if total_needed == 0:
        return 0.0

    return (total_have / total_needed) * 100


@router.get("", response_model=list[dict])
async def get_wishlist(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[dict]:
    """
    Get user's wishlisted sets with completion percentages.

    Returns list of sets with:
    - set info (set_num, name, theme, year, num_parts, img_url)
    - completion_percentage: % of parts user currently owns
    - added_at: when added to wishlist
    """
    try:
        # Get all wishlisted sets for user
        wishlist_result = await db.execute(
            select(Wishlist)
            .where(Wishlist.user_id == current_user.id)
            .options(selectinload(Wishlist.lego_set))
            .order_by(Wishlist.added_at.desc())
        )
        wishlist_items = wishlist_result.scalars().unique().all()

        response = []
        for item in wishlist_items:
            set_info = item.lego_set
            completion = await calculate_set_completion(
                current_user.id, set_info.set_num, db
            )

            response.append(
                {
                    "set_num": set_info.set_num,
                    "name": set_info.name,
                    "theme": set_info.theme,
                    "year": set_info.year,
                    "num_parts": set_info.num_parts,
                    "img_url": set_info.img_url,
                    "completion_percentage": round(completion, 1),
                    "added_at": item.added_at.isoformat(),
                }
            )

        return response

    except Exception as e:
        logger.error(f"Error fetching wishlist for {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch wishlist",
        )


@router.post("/{set_num}")
async def add_to_wishlist(
    set_num: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Add a set to user's wishlist.

    Args:
    - set_num: LEGO set number (e.g., "10307")

    Returns the wishlisted set with completion percentage.
    """
    try:
        # Check if set exists
        set_result = await db.execute(
            select(LegoSet).where(LegoSet.set_num == set_num)
        )
        lego_set = set_result.scalar_one_or_none()
        if not lego_set:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Set {set_num} not found",
            )

        # Check if already wishlisted
        existing_result = await db.execute(
            select(Wishlist).where(
                and_(
                    Wishlist.user_id == current_user.id,
                    Wishlist.set_num == set_num,
                )
            )
        )
        if existing_result.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Set already in wishlist",
            )

        # Add to wishlist
        wishlist_item = Wishlist(
            user_id=current_user.id,
            set_num=set_num,
            added_at=datetime.now(timezone.utc),
        )
        db.add(wishlist_item)
        await db.commit()

        completion = await calculate_set_completion(
            current_user.id, set_num, db
        )

        return {
            "set_num": lego_set.set_num,
            "name": lego_set.name,
            "theme": lego_set.theme,
            "year": lego_set.year,
            "num_parts": lego_set.num_parts,
            "img_url": lego_set.img_url,
            "completion_percentage": round(completion, 1),
            "added_at": wishlist_item.added_at.isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(
            f"Error adding {set_num} to wishlist for {current_user.id}: {e}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to add to wishlist",
        )


@router.delete("/{set_num}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_from_wishlist(
    set_num: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Remove a set from user's wishlist.

    Args:
    - set_num: LEGO set number (e.g., "10307")
    """
    try:
        # Find and delete wishlist item
        result = await db.execute(
            select(Wishlist).where(
                and_(
                    Wishlist.user_id == current_user.id,
                    Wishlist.set_num == set_num,
                )
            )
        )
        wishlist_item = result.scalar_one_or_none()

        if not wishlist_item:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Wishlist item not found",
            )

        await db.delete(wishlist_item)
        await db.commit()

    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(
            f"Error removing {set_num} from wishlist for {current_user.id}: {e}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to remove from wishlist",
        )


@router.get("/buildable")
async def get_buildable_sets(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[dict]:
    """
    Get only wishlisted sets that user can fully build right now (100% completion).

    Returns list of buildable wishlisted sets.
    """
    try:
        # Get all wishlisted sets
        wishlist_result = await db.execute(
            select(Wishlist)
            .where(Wishlist.user_id == current_user.id)
            .options(selectinload(Wishlist.lego_set))
            .order_by(Wishlist.added_at.desc())
        )
        wishlist_items = wishlist_result.scalars().unique().all()

        buildable = []
        for item in wishlist_items:
            set_info = item.lego_set
            completion = await calculate_set_completion(
                current_user.id, set_info.set_num, db
            )

            # Only include sets with 100% completion
            if completion >= 99.5:  # Allow small floating point variance
                buildable.append(
                    {
                        "set_num": set_info.set_num,
                        "name": set_info.name,
                        "theme": set_info.theme,
                        "year": set_info.year,
                        "num_parts": set_info.num_parts,
                        "img_url": set_info.img_url,
                        "completion_percentage": 100.0,
                        "added_at": item.added_at.isoformat(),
                    }
                )

        return buildable

    except Exception as e:
        logger.error(f"Error fetching buildable sets for {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch buildable sets",
        )


from typing import Optional
