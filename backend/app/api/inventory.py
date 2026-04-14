from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import PlainTextResponse, Response
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload
from uuid import UUID
import csv
import io
from typing import Optional

from app.core.database import get_db
from app.core.security import get_current_user
from app.core.cache import get_cache
from app.models.inventory import InventoryItem
from app.models.part import Part, Color
from app.schemas.inventory import (
    InventoryItemSchema,
    AddInventoryRequest,
    UpdateInventoryRequest,
)
from pydantic import BaseModel
from datetime import datetime
import httpx
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/inventory", tags=["inventory"])


# ─── Duplicate detection endpoint ────────────────────────────────────────────

@router.get("/check")
async def check_inventory_duplicate(
    part_num: str,
    color_id: str,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Check if a part+color combination already exists in the user's inventory.

    Query params:
        part_num: LEGO part number (e.g., "3004")
        color_id: Color ID as string (e.g., "1" for white)

    Returns:
        {
            "exists": bool,
            "quantity": int (current quantity if exists),
            "inventory_part_id": str | null
        }
    """
    user_id = current_user.get("sub")

    # Find the part
    part_result = await db.execute(
        select(Part).where(Part.part_num == part_num)
    )
    part = part_result.scalars().first()

    if not part:
        return {
            "exists": False,
            "quantity": 0,
            "inventory_part_id": None,
        }

    # Convert color_id string to int for lookup
    try:
        color_id_int = int(color_id)
    except (ValueError, TypeError):
        return {
            "exists": False,
            "quantity": 0,
            "inventory_part_id": None,
        }

    # Find the color
    color_result = await db.execute(
        select(Color).where(Color.id == color_id_int)
    )
    color = color_result.scalars().first()

    if not color:
        return {
            "exists": False,
            "quantity": 0,
            "inventory_part_id": None,
        }

    # Check if user has this part+color combination
    existing = await db.execute(
        select(InventoryItem).where(
            (InventoryItem.user_id == user_id)
            & (InventoryItem.part_id == part.id)
            & (InventoryItem.color_id == color_id_int)
        )
    )
    existing_item = existing.scalars().first()

    if existing_item:
        return {
            "exists": True,
            "quantity": existing_item.quantity,
            "inventory_part_id": str(existing_item.id),
        }

    return {
        "exists": False,
        "quantity": 0,
        "inventory_part_id": None,
    }


# Schemas for inventory value response
class ValueBreakdownColor(BaseModel):
    color_name: str
    value_usd: float
    count: int


class TopValuablePart(BaseModel):
    part_num: str
    part_name: str
    color_name: str
    unit_price: float
    qty: int
    total: float


class InventoryValueResult(BaseModel):
    total_value_usd: float
    total_parts: int
    unique_parts: int
    breakdown_by_color: list[ValueBreakdownColor]
    breakdown_by_theme: list
    top_valuable_parts: list[TopValuablePart]
    cached_at: str


# Rebrickable to BrickLink color ID mapping
# Covers the 20 most common colors
REBRICKABLE_TO_BRICKLINK_COLOR = {
    1: 1,        # White
    0: 11,       # Black
    4: 5,        # Red
    2: 23,       # Blue (Rebrickable 2 = Dark Blue, BrickLink 23 = Dark Blue)
    3: 3,        # Yellow
    6: 2,        # Green (Rebrickable 6 = Green, BrickLink 2 = Green)
    71: 9,       # Light Bluish Gray
    72: 10,      # Dark Bluish Gray
    19: 2,       # Tan (BrickLink 2 = Dark Tan)
    154: 59,     # Dark Red
    25: 4,       # Orange
    102: 42,     # Medium Blue
    27: 34,      # Lime
    11: 80,      # Dark Green
    8: 8,        # Brown (actual color ID, not the confusing task description)
    308: 120,    # Dark Brown
    12: 12,      # Trans-Clear
    17: 17,      # Trans-Red
    15: 15,      # Trans-Blue
    14: 14,      # Trans-Yellow
}


@router.get("", response_model=list[InventoryItemSchema])
async def get_inventory(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    user_id = current_user.get("sub")

    result = await db.execute(
        select(InventoryItem)
        .where(InventoryItem.user_id == user_id)
        .options(
            selectinload(InventoryItem.user_id),
        )
    )
    items = result.scalars().all()

    response = []
    for item in items:
        part_result = await db.execute(
            select(Part).where(Part.id == item.part_id)
        )
        part = part_result.scalars().first()

        color_result = await db.execute(
            select(Color).where(Color.id == item.color_id)
        )
        color = color_result.scalars().first()

        if part and color:
            response.append(
                InventoryItemSchema(
                    id=str(item.id),
                    user_id=str(item.user_id),
                    part_id=str(item.part_id),
                    part_num=part.part_num,
                    part_name=part.name,
                    color_id=str(item.color_id),
                    color_name=color.name,
                    color_hex=color.hex_code,
                    quantity=item.quantity,
                    created_at=item.created_at,
                    updated_at=item.updated_at,
                )
            )

    return response


@router.post("", response_model=InventoryItemSchema)
async def add_inventory(
    request: AddInventoryRequest,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    user_id = current_user.get("sub")

    part_result = await db.execute(
        select(Part).where(Part.part_num == request.part_num)
    )
    part = part_result.scalars().first()

    if not part:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Part not found",
        )

    color_result = await db.execute(
        select(Color).where(Color.id == request.color_id)
    )
    color = color_result.scalars().first()

    if not color:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Color not found",
        )

    existing = await db.execute(
        select(InventoryItem).where(
            (InventoryItem.user_id == user_id)
            & (InventoryItem.part_id == part.id)
            & (InventoryItem.color_id == color.id)
        )
    )
    existing_item = existing.scalars().first()

    if existing_item:
        existing_item.quantity += request.quantity
        await db.commit()
        await db.refresh(existing_item)
        item = existing_item
    else:
        item = InventoryItem(
            user_id=user_id,
            part_id=part.id,
            color_id=color.id,
            quantity=request.quantity,
        )
        db.add(item)
        await db.commit()
        await db.refresh(item)

    return InventoryItemSchema(
        id=str(item.id),
        user_id=str(item.user_id),
        part_id=str(item.part_id),
        part_num=part.part_num,
        part_name=part.name,
        color_id=str(item.color_id),
        color_name=color.name,
        color_hex=color.hex_code,
        quantity=item.quantity,
        created_at=item.created_at,
        updated_at=item.updated_at,
    )


@router.put("/{item_id}", response_model=InventoryItemSchema)
async def update_inventory(
    item_id: str,
    request: UpdateInventoryRequest,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    user_id = current_user.get("sub")

    result = await db.execute(
        select(InventoryItem).where(
            (InventoryItem.id == item_id) & (InventoryItem.user_id == user_id)
        )
    )
    item = result.scalars().first()

    if not item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Inventory item not found",
        )

    part_result = await db.execute(
        select(Part).where(Part.id == item.part_id)
    )
    part = part_result.scalars().first()

    color_result = await db.execute(
        select(Color).where(Color.id == item.color_id)
    )
    color = color_result.scalars().first()

    item.quantity = request.quantity
    await db.commit()
    await db.refresh(item)

    return InventoryItemSchema(
        id=str(item.id),
        user_id=str(item.user_id),
        part_id=str(item.part_id),
        part_num=part.part_num,
        part_name=part.name,
        color_id=str(item.color_id),
        color_name=color.name,
        color_hex=color.hex_code,
        quantity=item.quantity,
        created_at=item.created_at,
        updated_at=item.updated_at,
    )


@router.delete("/{item_id}")
async def delete_inventory(
    item_id: str,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    user_id = current_user.get("sub")

    result = await db.execute(
        select(InventoryItem).where(
            (InventoryItem.id == item_id) & (InventoryItem.user_id == user_id)
        )
    )
    item = result.scalars().first()

    if not item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Inventory item not found",
        )

    await db.delete(item)
    await db.commit()

    return {"message": "Item deleted"}


@router.get("/export", response_class=PlainTextResponse)
async def export_inventory(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    user_id = current_user.get("sub")

    result = await db.execute(
        select(InventoryItem).where(InventoryItem.user_id == user_id)
    )
    items = result.scalars().all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Part Number", "Part Name", "Color", "Hex Code", "Quantity"])

    for item in items:
        part_result = await db.execute(
            select(Part).where(Part.id == item.part_id)
        )
        part = part_result.scalars().first()

        color_result = await db.execute(
            select(Color).where(Color.id == item.color_id)
        )
        color = color_result.scalars().first()

        if part and color:
            writer.writerow(
                [
                    part.part_num,
                    part.name,
                    color.name,
                    color.hex_code or "",
                    item.quantity,
                ]
            )

    return output.getvalue()


@router.get("/export/bricklink-xml")
async def export_bricklink_xml(
    set_num: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Export inventory as BrickLink Wanted List XML format.
    If set_num is provided, export missing parts for that set.
    Otherwise, export entire inventory as a "have list".
    """
    user_id = current_user.get("sub")

    xml_items = []

    if set_num:
        # TODO: Load missing parts from set completion endpoint
        # For now, just return empty inventory for set exports
        pass
    else:
        # Export entire inventory as a "have list"
        result = await db.execute(
            select(InventoryItem).where(InventoryItem.user_id == user_id)
        )
        items = result.scalars().all()

        for item in items:
            part_result = await db.execute(
                select(Part).where(Part.id == item.part_id)
            )
            part = part_result.scalars().first()

            color_result = await db.execute(
                select(Color).where(Color.id == item.color_id)
            )
            color = color_result.scalars().first()

            if part and color:
                # Map Rebrickable color ID to BrickLink color ID
                rebrickable_color_id = color.rebrickable_id
                bricklink_color_id = REBRICKABLE_TO_BRICKLINK_COLOR.get(
                    rebrickable_color_id, 16  # Default to gray if not mapped
                )

                xml_item = f"""    <ITEM>
      <ITEMTYPE>P</ITEMTYPE>
      <ITEMID>{part.part_num}</ITEMID>
      <COLOR>{bricklink_color_id}</COLOR>
      <MINQTY>{item.quantity}</MINQTY>
      <CONDITION>X</CONDITION>
      <NOTIFY>N</NOTIFY>
    </ITEM>
"""
                xml_items.append(xml_item)

    items_xml = "\n".join(xml_items)

    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<INVENTORY>
{items_xml}
</INVENTORY>
"""

    return Response(
        content=xml,
        media_type="application/xml",
        headers={
            "Content-Disposition": 'attachment; filename="brickscan_wanted.xml"'
        },
    )


async def get_bricklink_price(part_num: str, color_id: int, cache) -> Optional[float]:
    """
    Fetch price for a part from BrickLink API.
    Returns average price in USD, or None if unavailable.
    """
    from app.core.config import settings

    cache_key = f"bricklink_price:{part_num}:{color_id}"

    # Try cache first
    cached_price = await cache.get(cache_key)
    if cached_price is not None:
        return float(cached_price) if cached_price else None

    try:
        # BrickLink API endpoint for price guide
        url = f"https://api.bricklink.com/api/v1/items/part/{part_num}/price"
        params = {
            "colorId": color_id,
            "currencyCode": "USD",
            "guide_type": "sold",
        }

        headers = {
            "Authorization": f"Bearer {settings.BRICKLINK_TOKEN}",
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, params=params, headers=headers)

            if response.status_code == 200:
                data = response.json()
                avg_price = data.get("data", {}).get("avg_price")

                # Cache for 24 hours
                await cache.set(cache_key, avg_price, ttl_seconds=86400)
                return float(avg_price) if avg_price else None
            else:
                # Cache None for 24 hours to avoid repeated failures
                await cache.set(cache_key, None, ttl_seconds=86400)
                return None

    except Exception as e:
        logger.error(f"Failed to fetch BrickLink price for {part_num}: {e}")
        return None


@router.get("/value", response_model=InventoryValueResult)
async def get_inventory_value(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get total collection value based on BrickLink prices.
    Caches individual part prices for 24 hours.
    """
    user_id = current_user.get("sub")
    cache = get_cache()

    result = await db.execute(
        select(InventoryItem).where(InventoryItem.user_id == user_id)
    )
    items = result.scalars().all()

    total_value = 0.0
    total_parts = 0
    unique_parts_set = set()

    color_breakdown = {}  # color_name -> {value, count}
    parts_by_value = []   # {part_num, part_name, color_name, unit_price, qty, total}

    for item in items:
        part_result = await db.execute(
            select(Part).where(Part.id == item.part_id)
        )
        part = part_result.scalars().first()

        color_result = await db.execute(
            select(Color).where(Color.id == item.color_id)
        )
        color = color_result.scalars().first()

        if not part or not color:
            continue

        unique_parts_set.add((part.part_num, color.name))
        total_parts += item.quantity

        # Map to BrickLink color ID
        bricklink_color_id = REBRICKABLE_TO_BRICKLINK_COLOR.get(
            color.rebrickable_id, 16
        )

        # Fetch price from BrickLink
        unit_price = await get_bricklink_price(part.part_num, bricklink_color_id, cache)
        if unit_price is None:
            unit_price = 0.0

        part_total = unit_price * item.quantity
        total_value += part_total

        # Track by color
        if color.name not in color_breakdown:
            color_breakdown[color.name] = {"value": 0.0, "count": 0}
        color_breakdown[color.name]["value"] += part_total
        color_breakdown[color.name]["count"] += item.quantity

        # Track valuable parts
        parts_by_value.append({
            "part_num": part.part_num,
            "part_name": part.name,
            "color_name": color.name,
            "unit_price": unit_price,
            "qty": item.quantity,
            "total": part_total,
        })

    # Sort by total value (descending) and take top 10
    parts_by_value.sort(key=lambda x: x["total"], reverse=True)
    top_parts = [
        TopValuablePart(
            part_num=p["part_num"],
            part_name=p["part_name"],
            color_name=p["color_name"],
            unit_price=p["unit_price"],
            qty=p["qty"],
            total=p["total"],
        )
        for p in parts_by_value[:10]
    ]

    # Build color breakdown response
    color_list = [
        ValueBreakdownColor(
            color_name=color_name,
            value_usd=stats["value"],
            count=stats["count"],
        )
        for color_name, stats in color_breakdown.items()
    ]

    return InventoryValueResult(
        total_value_usd=round(total_value, 2),
        total_parts=total_parts,
        unique_parts=len(unique_parts_set),
        breakdown_by_color=color_list,
        breakdown_by_theme=[],  # TODO: implement theme breakdown
        top_valuable_parts=top_parts,
        cached_at=datetime.utcnow().isoformat() + "Z",
    )
