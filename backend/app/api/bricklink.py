from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import PlainTextResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import get_current_user
from app.services.build_check import compare_inventory_to_set
from app.services.bricklink_service import generate_wanted_list_xml, bricklink_color_map
from app.schemas.inventory import BuildCheckResult

router = APIRouter(prefix="/bricklink", tags=["bricklink"])


@router.post("/wanted-list/{set_num}", response_class=PlainTextResponse)
async def generate_wanted_list(
    set_num: str,
    condition: str = "N",
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    user_id = current_user.get("sub")

    build_check: BuildCheckResult = await compare_inventory_to_set(user_id, set_num, db)

    xml = generate_wanted_list_xml(build_check.missing, condition)

    return xml


@router.get("/colors", response_model=dict)
async def get_bricklink_colors():
    return {"color_mapping": bricklink_color_map}
