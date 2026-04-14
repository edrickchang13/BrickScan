from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload
from uuid import UUID

from app.models.lego_set import LegoSet, SetPart
from app.models.inventory import InventoryItem
from app.models.part import Part, Color
from app.schemas.inventory import BuildCheckResult, MissingPart


async def compare_inventory_to_set(
    user_id: str,
    set_num: str,
    db: AsyncSession,
) -> BuildCheckResult:
    result = await db.execute(
        select(LegoSet)
        .where(LegoSet.set_num == set_num)
        .options(selectinload(LegoSet.parts))
    )
    lego_set = result.unique().scalars().first()

    if not lego_set:
        return BuildCheckResult(
            total_needed=0,
            total_have=0,
            percent_complete=0.0,
            have=[],
            missing=[],
        )

    user_inventory_result = await db.execute(
        select(InventoryItem).where(InventoryItem.user_id == user_id)
    )
    user_items = user_inventory_result.scalars().all()

    user_inventory = {}
    for item in user_items:
        key = (item.part_id, item.color_id)
        user_inventory[key] = item.quantity

    have_list = []
    missing_list = []
    total_needed = 0
    total_have = 0

    for set_part in lego_set.parts:
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

        key = (set_part.part_id, set_part.color_id)
        quantity_needed = set_part.quantity
        quantity_have = user_inventory.get(key, 0)

        total_needed += quantity_needed

        missing_part_obj = MissingPart(
            part_num=part.part_num,
            part_name=part.name,
            color_name=color.name,
            color_hex=color.hex_code,
            quantity_needed=quantity_needed,
            quantity_have=quantity_have,
        )

        if quantity_have > 0:
            total_have += quantity_have
            have_list.append(missing_part_obj)

        if quantity_have < quantity_needed:
            missing_list.append(missing_part_obj)

    percent_complete = 0.0
    if total_needed > 0:
        percent_complete = (total_have / total_needed) * 100.0

    return BuildCheckResult(
        total_needed=total_needed,
        total_have=total_have,
        percent_complete=percent_complete,
        have=have_list,
        missing=missing_list,
    )
