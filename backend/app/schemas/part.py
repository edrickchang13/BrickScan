from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class ColorSchema(BaseModel):
    id: str
    rebrickable_id: int
    name: str
    hex_code: Optional[str] = None
    is_transparent: bool

    class Config:
        from_attributes = True


class PartSchema(BaseModel):
    id: str
    part_num: str
    name: str
    year_from: Optional[int] = None
    year_to: Optional[int] = None
    image_url: Optional[str] = None

    class Config:
        from_attributes = True


class PartDetailSchema(PartSchema):
    category_id: Optional[str] = None
    category_name: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True
