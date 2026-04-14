"""
Parts API with search, filtering, and color variants.
Returns part information with all known color variants from inventory_parts.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy import select, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional
import logging

from app.core.database import get_db
from app.core.cache import get_cache
from app.models.part import Part
from app.models.part import Color
from app.models.inventory_part import InventoryPart
from app.models.scan import Scan
from app.models.user import User
from app.core.security import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/parts", tags=["parts"])


@router.get("")
async def search_parts(
    search: Optional[str] = Query(None, min_length=1),
    category: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Search for LEGO parts with filtering and pagination.

    Args:
    - search: Text to search in part name/number (optional)
    - category: Filter by part category (optional)
    - limit: Max results (1-100, default 20)
    - offset: Pagination offset (default 0)

    Returns:
    - total: Total matching parts
    - limit: Limit used
    - offset: Offset used
    - parts: List of matching parts with basic info
    """
    try:
        # Build query for counting
        count_query = select(func.count(Part.id))

        # Search filter
        if search:
            search_term = f"%{search}%"
            count_query = count_query.where(
                or_(
                    Part.part_num.ilike(search_term),
                    Part.name.ilike(search_term),
                )
            )

        # Category filter
        if category:
            count_query = count_query.where(Part.category == category)

        # Get total count
        count_result = await db.execute(count_query)
        total = count_result.scalar() or 0

        # Build query for results
        query = select(Part)

        if search:
            search_term = f"%{search}%"
            query = query.where(
                or_(
                    Part.part_num.ilike(search_term),
                    Part.name.ilike(search_term),
                )
            )

        if category:
            query = query.where(Part.category == category)

        # Get paginated results
        result = await db.execute(
            query.order_by(Part.name).offset(offset).limit(limit)
        )
        parts = result.scalars().all()

        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "parts": [
                {
                    "part_num": p.part_num,
                    "name": p.name,
                    "category": p.category,
                }
                for p in parts
            ],
        }

    except Exception as e:
        logger.error(f"Error searching parts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to search parts",
        )


@router.get("/{part_num}")
async def get_part_details(
    part_num: str,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Get detailed information about a specific part.

    Includes all known color variants from inventory_parts.

    Args:
    - part_num: Part number (e.g., "3003")

    Returns:
    - part_num: Part number
    - name: Part name
    - category: Part category
    - color_variants: List of colors this part exists in
    """
    try:
        # Fetch part
        result = await db.execute(
            select(Part).where(Part.part_num == part_num)
        )
        part = result.scalar_one_or_none()

        if not part:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Part {part_num} not found",
            )

        # Get all color variants for this part
        colors_result = await db.execute(
            select(Color.id, Color.name, Color.rgb)
            .distinct()
            .join(InventoryPart, InventoryPart.color_id == Color.id)
            .where(InventoryPart.part_num == part_num)
            .order_by(Color.name)
        )
        color_variants = [
            {
                "color_id": row[0],
                "color_name": row[1],
                "rgb": row[2],
            }
            for row in colors_result.fetchall()
        ]

        return {
            "part_num": part.part_num,
            "name": part.name,
            "category": part.category,
            "color_variants": color_variants,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching part details for {part_num}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch part details",
        )


@router.get("/{part_num}/colors")
async def get_part_colors(
    part_num: str,
    db: AsyncSession = Depends(get_db),
) -> list[dict]:
    """
    Get all known colors for a specific part.

    Args:
    - part_num: Part number

    Returns:
    - List of color objects with id, name, and rgb
    """
    try:
        # Verify part exists
        part_result = await db.execute(
            select(Part).where(Part.part_num == part_num)
        )
        if not part_result.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Part {part_num} not found",
            )

        # Get all colors for this part
        colors_result = await db.execute(
            select(Color.id, Color.name, Color.rgb)
            .distinct()
            .join(InventoryPart, InventoryPart.color_id == Color.id)
            .where(InventoryPart.part_num == part_num)
            .order_by(Color.name)
        )

        return [
            {
                "color_id": row[0],
                "color_name": row[1],
                "rgb": row[2],
            }
            for row in colors_result.fetchall()
        ]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching colors for part {part_num}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch part colors",
        )


@router.get("/{part_num}/substitutes")
async def get_part_substitutes(
    part_num: str,
    db: AsyncSession = Depends(get_db),
) -> list[dict]:
    """
    Get similar/substitute parts for a given part number.

    Uses Rebrickable API to find parts in the same category and scores them
    by dimensional similarity. Returns top 5 substitutes with similarity scores.

    Args:
    - part_num: Part number (e.g., "3003")

    Returns:
    - List of substitute parts with: part_num, name, similarity (0-1), reason, image_url
    """
    try:
        import aiohttp
        import asyncio
        from app.core.cache import get_cache

        cache = get_cache()
        cache_key = f"substitutes:{part_num}"

        # Check cache first (7 day TTL)
        cached = await cache.get(cache_key)
        if cached:
            return cached

        # Fetch part details from Rebrickable
        async with aiohttp.ClientSession() as session:
            # Get part info from Rebrickable API
            rebrickable_url = f"https://rebrickable.com/api/v3/lego/parts/{part_num}/"
            async with session.get(rebrickable_url) as resp:
                if resp.status != 200:
                    logger.warning(f"Rebrickable part not found: {part_num}")
                    return []

                part_data = await resp.json()
                part_category_id = part_data.get("part_cat_id")
                x_size = part_data.get("x_size")
                y_size = part_data.get("y_size")
                z_size = part_data.get("z_size")

                if not part_category_id:
                    return []

                # Get parts in same category
                category_url = f"https://rebrickable.com/api/v3/lego/parts/?part_cat_id={part_category_id}&page_size=50"
                async with session.get(category_url) as resp:
                    if resp.status != 200:
                        return []

                    category_data = await resp.json()
                    candidates = category_data.get("results", [])

                # Score candidates by dimensional similarity
                scored_candidates = []
                for candidate in candidates:
                    if candidate["part_num"] == part_num:
                        continue  # Skip self

                    cand_x = candidate.get("x_size")
                    cand_y = candidate.get("y_size")
                    cand_z = candidate.get("z_size")

                    # Scoring logic
                    if (x_size is not None and y_size is not None and
                        cand_x == x_size and cand_y == y_size):
                        score = 1.0
                        reason = "Same category & exact dimensions"
                    elif (x_size is not None and y_size is not None and
                          cand_x is not None and cand_y is not None):
                        score = 0.5
                        reason = "Same category, similar size"
                    else:
                        score = 0.0
                        reason = "Same category"

                    # Check for mold/print relationships in external_ids
                    external_ids = candidate.get("external_ids", {})
                    if external_ids and part_data.get("external_ids"):
                        part_ext = part_data.get("external_ids", {})
                        # Look for matching base IDs (same mold, different print)
                        if any(part_ext.get(k) == external_ids.get(k)
                               for k in set(list(part_ext.keys()) + list(external_ids.keys()))):
                            score = 0.95
                            reason = "Same mold, different print"

                    if score > 0:
                        scored_candidates.append({
                            "part_num": candidate["part_num"],
                            "name": candidate.get("name", candidate["part_num"]),
                            "similarity": score,
                            "reason": reason,
                            "image_url": candidate.get("part_img_url") or
                                         f"https://img.bricklink.com/ItemImage/PN/11/{candidate['part_num']}.png",
                        })

                # Sort by similarity (descending) and take top 5
                scored_candidates.sort(key=lambda x: x["similarity"], reverse=True)
                result = scored_candidates[:5]

                # Cache for 7 days
                await cache.set(cache_key, result, ttl_seconds=604800)

                return result

    except Exception as e:
        logger.error(f"Error fetching substitutes for {part_num}: {e}")
        return []


@router.get("/categories")
async def get_part_categories(
    db: AsyncSession = Depends(get_db),
) -> list[str]:
    """
    Get list of all part categories for filtering.

    Returns:
    - List of category names (alphabetically sorted)
    """
    try:
        cache = get_cache()
        cache_key = "parts:categories"

        # Try cache first
        cached = await cache.get(cache_key)
        if cached:
            return cached

        # Query database
        result = await db.execute(
            select(func.distinct(Part.category))
            .where(Part.category.isnot(None))
            .order_by(Part.category)
        )
        categories = [row[0] for row in result.fetchall()]

        # Cache for 1 hour
        await cache.set(cache_key, categories, ttl_seconds=3600)

        return categories

    except Exception as e:
        logger.error(f"Error fetching categories: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch categories",
        )


@router.get("/recent")
async def get_recent_parts(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    limit: int = Query(10, ge=1, le=50),
) -> list[dict]:
    """
    Get recently scanned parts by the user (for suggestions).

    Args:
    - limit: Max parts to return (default 10)

    Returns:
    - List of recently scanned parts (most recent first)
    """
    try:
        # Get recent scans for this user
        result = await db.execute(
            select(
                Scan.prediction,
                func.max(Scan.created_at).label("last_scanned"),
                func.count(Scan.id).label("scan_count"),
            )
            .where(Scan.user_id == current_user.id)
            .group_by(Scan.prediction)
            .order_by(func.max(Scan.created_at).desc())
            .limit(limit)
        )

        recent_scans = result.fetchall()

        # Was N+1 (one Part query per scan row); now batch-loads all Parts in
        # a single WHERE part_num IN (...) query. InventoryItem/SetPart have no
        # ORM relationship to Part (only FK columns), and Scan.prediction is a
        # raw part_num string — so batching by IN is the right fit here rather
        # than selectinload().
        part_nums = [row[0] for row in recent_scans if row[0] is not None]
        parts_by_num: dict[str, Part] = {}
        if part_nums:
            parts_result = await db.execute(
                select(Part).where(Part.part_num.in_(part_nums))
            )
            parts_by_num = {p.part_num: p for p in parts_result.scalars().all()}

        # Enrich with part details (preserves original response order/shape)
        parts = []
        for part_num, last_scanned, scan_count in recent_scans:
            part = parts_by_num.get(part_num)
            # Defensive: skip orphaned predictions whose Part no longer exists
            if part:
                parts.append(
                    {
                        "part_num": part.part_num,
                        "name": part.name,
                        "category": part.category,
                        "last_scanned": last_scanned.isoformat(),
                        "scan_count": scan_count,
                    }
                )

        return parts

    except Exception as e:
        logger.error(f"Error fetching recent parts for {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch recent parts",
        )
