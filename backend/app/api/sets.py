from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload
from sqlalchemy import or_
import redis.asyncio as redis
import json
from typing import Optional
from datetime import datetime, timezone

from app.core.database import get_db
from app.core.config import settings
from app.core.security import get_current_user
from app.models.lego_set import LegoSet, Theme, SetPart
from app.models.part import Part, Color
from app.models.inventory import InventoryItem
from app.schemas.lego_set import (
    SetSummarySchema, SetDetailSchema, SetPartSchema,
    SetCompletionSchema, MissingPartSchema
)

router = APIRouter(prefix="/sets", tags=["sets"])

redis_client = None


async def get_redis():
    global redis_client
    if redis_client is None:
        redis_client = await redis.from_url(settings.REDIS_URL)
    return redis_client


@router.get("", response_model=list[SetSummarySchema])
async def search_sets(
    search: Optional[str] = None,
    theme: Optional[str] = None,
    year: Optional[int] = None,
    limit: int = 20,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    query = select(LegoSet)

    if search:
        query = query.where(
            or_(
                LegoSet.set_num.ilike(f"%{search}%"),
                LegoSet.name.ilike(f"%{search}%"),
            )
        )

    if year:
        query = query.where(LegoSet.year == year)

    if theme:
        theme_result = await db.execute(
            select(Theme).where(Theme.name.ilike(f"%{theme}%"))
        )
        theme_obj = theme_result.scalars().first()
        if theme_obj:
            query = query.where(LegoSet.theme_id == theme_obj.id)

    query = query.offset(offset).limit(limit)
    result = await db.execute(query)
    sets = result.scalars().all()

    return [
        SetSummarySchema(
            id=str(s.id),
            set_num=s.set_num,
            name=s.name,
            year=s.year,
            theme_id=str(s.theme_id) if s.theme_id else None,
            num_parts=s.num_parts,
            img_url=s.img_url,
        )
        for s in sets
    ]


@router.get("/{set_num}", response_model=SetDetailSchema)
async def get_set(
    set_num: str,
    db: AsyncSession = Depends(get_db),
):
    cache = await get_redis()
    cache_key = f"set:{set_num}"

    cached = await cache.get(cache_key)
    if cached:
        return json.loads(cached)

    result = await db.execute(
        select(LegoSet)
        .where(LegoSet.set_num == set_num)
        .options(selectinload(LegoSet.parts))
    )
    lego_set = result.unique().scalars().first()

    if not lego_set:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Set not found",
        )

    theme_name = None
    if lego_set.theme_id:
        theme_result = await db.execute(
            select(Theme).where(Theme.id == lego_set.theme_id)
        )
        theme = theme_result.scalars().first()
        if theme:
            theme_name = theme.name

    parts_data = []
    for set_part in lego_set.parts:
        part_result = await db.execute(
            select(Part).where(Part.id == set_part.part_id)
        )
        part = part_result.scalars().first()

        color_result = await db.execute(
            select(Color).where(Color.id == set_part.color_id)
        )
        color = color_result.scalars().first()

        if part and color:
            parts_data.append(
                SetPartSchema(
                    part_num=part.part_num,
                    part_name=part.name,
                    color_name=color.name,
                    color_hex=color.hex_code,
                    quantity=set_part.quantity,
                    is_spare=set_part.is_spare,
                )
            )

    response = SetDetailSchema(
        id=str(lego_set.id),
        set_num=lego_set.set_num,
        name=lego_set.name,
        year=lego_set.year,
        theme_id=str(lego_set.theme_id) if lego_set.theme_id else None,
        theme_name=theme_name,
        num_parts=lego_set.num_parts,
        img_url=lego_set.img_url,
        created_at=lego_set.created_at,
        parts=parts_data,
    )

    await cache.setex(cache_key, 86400, json.dumps(response.model_dump(), default=str))

    return response


@router.get("/{set_num}/parts", response_model=list[SetPartSchema])
async def get_set_parts(
    set_num: str,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(LegoSet)
        .where(LegoSet.set_num == set_num)
        .options(selectinload(LegoSet.parts))
    )
    lego_set = result.unique().scalars().first()

    if not lego_set:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Set not found",
        )

    parts_data = []
    for set_part in lego_set.parts:
        part_result = await db.execute(
            select(Part).where(Part.id == set_part.part_id)
        )
        part = part_result.scalars().first()

        color_result = await db.execute(
            select(Color).where(Color.id == set_part.color_id)
        )
        color = color_result.scalars().first()

        if part and color:
            parts_data.append(
                SetPartSchema(
                    part_num=part.part_num,
                    part_name=part.name,
                    color_name=color.name,
                    color_hex=color.hex_code,
                    quantity=set_part.quantity,
                    is_spare=set_part.is_spare,
                )
            )

    return parts_data


@router.get("/completion/scan", response_model=list[SetCompletionSchema])
async def scan_inventory_for_sets(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    limit: int = 50,
):
    """
    Scan all sets and return those with >30% completion.
    Sorted by completion % descending. Limit 50 results.
    """
    user_id = current_user.get("sub")
    cache = await get_redis()
    cache_key = f"completion-scan:{user_id}"

    # Check cache first
    cached = await cache.get(cache_key)
    if cached:
        return json.loads(cached)

    # Get user inventory
    inv_result = await db.execute(
        select(InventoryItem).where(InventoryItem.user_id == user_id)
    )
    inventory_items = inv_result.scalars().all()

    # Build inventory map
    inventory_map = {}
    for inv_item in inventory_items:
        part_result = await db.execute(
            select(Part).where(Part.id == inv_item.part_id)
        )
        part = part_result.scalars().first()
        if part:
            inventory_map[(part.part_num, inv_item.color_id)] = inv_item.quantity

    # Get all sets
    sets_result = await db.execute(
        select(LegoSet).options(selectinload(LegoSet.parts))
    )
    all_sets = sets_result.unique().scalars().all()

    results = []

    for lego_set in all_sets:
        total_parts_needed = 0
        unique_parts_needed = set()
        have_parts = 0
        have_unique = set()

        for set_part in lego_set.parts:
            if set_part.is_spare:
                continue

            part_result = await db.execute(
                select(Part).where(Part.id == set_part.part_id)
            )
            part = part_result.scalars().first()

            if not part:
                continue

            needed = set_part.quantity
            have = inventory_map.get((part.part_num, set_part.color_id), 0)
            have = min(have, needed)

            total_parts_needed += needed
            unique_parts_needed.add(part.part_num)
            have_parts += have

            if have > 0:
                have_unique.add(part.part_num)

        if total_parts_needed == 0:
            continue

        completion_pct = have_parts / total_parts_needed * 100

        if completion_pct >= 30:
            results.append(
                {
                    "set_num": lego_set.set_num,
                    "set_name": lego_set.name,
                    "total_parts": total_parts_needed,
                    "unique_parts": len(unique_parts_needed),
                    "have_parts": have_parts,
                    "have_unique": len(have_unique),
                    "completion_pct": round(completion_pct, 1),
                    "missing": [],
                    "cached_at": datetime.now(timezone.utc),
                }
            )

    # Sort by completion % descending
    results.sort(key=lambda x: x["completion_pct"], reverse=True)
    results = results[:limit]

    # Convert to schema
    response = [SetCompletionSchema(**r) for r in results]

    # Cache for 1 hour
    await cache.setex(
        cache_key, 3600, json.dumps([r.model_dump() for r in response], default=str)
    )

    return response


@router.get("/{set_num}/completion", response_model=SetCompletionSchema)
async def get_set_completion(
    set_num: str,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get completion status of a set against user's inventory.
    Compares set's BOM with user's scanned parts.
    """
    user_id = current_user.get("sub")
    cache = await get_redis()
    cache_key = f"completion:{set_num}:{user_id}"

    # Check cache first
    cached = await cache.get(cache_key)
    if cached:
        return json.loads(cached)

    # Get set from DB
    result = await db.execute(
        select(LegoSet)
        .where(LegoSet.set_num == set_num)
        .options(selectinload(LegoSet.parts))
    )
    lego_set = result.unique().scalars().first()

    if not lego_set:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Set not found",
        )

    # Get user inventory
    inv_result = await db.execute(
        select(InventoryItem).where(InventoryItem.user_id == user_id)
    )
    inventory_items = inv_result.scalars().all()

    # Build a map: (part_num, color_id) -> quantity
    inventory_map = {}
    for inv_item in inventory_items:
        part_result = await db.execute(
            select(Part).where(Part.id == inv_item.part_id)
        )
        part = part_result.scalars().first()
        if part:
            inventory_map[(part.part_num, inv_item.color_id)] = inv_item.quantity

    # Calculate completion
    total_parts_needed = 0
    unique_parts_needed = set()
    have_parts = 0
    have_unique = set()
    missing = []

    for set_part in lego_set.parts:
        if set_part.is_spare:
            continue

        part_result = await db.execute(
            select(Part).where(Part.id == set_part.part_id)
        )
        part = part_result.scalars().first()

        color_result = await db.execute(
            select(Color).where(Color.id == set_part.color_id)
        )
        color = color_result.scalars().first()

        if not part or not color:
            continue

        needed = set_part.quantity
        have = inventory_map.get((part.part_num, set_part.color_id), 0)
        have = min(have, needed)

        total_parts_needed += needed
        unique_parts_needed.add(part.part_num)
        have_parts += have

        if have > 0:
            have_unique.add(part.part_num)

        short = needed - have
        if short > 0:
            missing.append(
                MissingPartSchema(
                    part_num=part.part_num,
                    part_name=part.name,
                    color_id=str(set_part.color_id),
                    color_name=color.name,
                    color_hex=color.hex_code,
                    need=needed,
                    have=have,
                    short=short,
                )
            )

    # Calculate percentage
    completion_pct = (
        (have_parts / total_parts_needed * 100) if total_parts_needed > 0 else 0
    )

    response = SetCompletionSchema(
        set_num=lego_set.set_num,
        set_name=lego_set.name,
        total_parts=total_parts_needed,
        unique_parts=len(unique_parts_needed),
        have_parts=have_parts,
        have_unique=len(have_unique),
        completion_pct=round(completion_pct, 1),
        missing=sorted(missing, key=lambda x: x.short, reverse=True),
        cached_at=datetime.now(timezone.utc),
    )

    # Cache for 1 hour
    await cache.setex(cache_key, 3600, json.dumps(response.model_dump(), default=str))

    return response
