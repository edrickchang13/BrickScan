from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


class InventoryItemSchema(BaseModel):
    id: str
    user_id: str
    part_id: str
    part_num: str
    part_name: str
    color_id: str
    color_name: str
    color_hex: Optional[str] = None
    quantity: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class AddInventoryRequest(BaseModel):
    part_num: str
    color_id: str
    quantity: int = 1


class UpdateInventoryRequest(BaseModel):
    quantity: int


class MissingPart(BaseModel):
    part_num: str
    part_name: str
    color_name: str
    color_hex: Optional[str] = None
    quantity_needed: int
    quantity_have: int


class BuildCheckResult(BaseModel):
    total_needed: int
    total_have: int
    percent_complete: float
    have: List[MissingPart]
    missing: List[MissingPart]
