from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


class SetSummarySchema(BaseModel):
    id: str
    set_num: str
    name: str
    year: Optional[int] = None
    theme_id: Optional[str] = None
    num_parts: Optional[int] = None
    img_url: Optional[str] = None

    class Config:
        from_attributes = True


class SetPartSchema(BaseModel):
    part_num: str
    part_name: str
    color_name: str
    color_hex: Optional[str] = None
    quantity: int
    is_spare: bool

    class Config:
        from_attributes = True


class SetDetailSchema(SetSummarySchema):
    theme_name: Optional[str] = None
    created_at: datetime
    parts: Optional[List[SetPartSchema]] = None

    class Config:
        from_attributes = True


class MissingPartSchema(BaseModel):
    part_num: str
    part_name: str
    color_id: str
    color_name: str
    color_hex: Optional[str] = None
    need: int
    have: int
    short: int

    class Config:
        from_attributes = True


class SetCompletionSchema(BaseModel):
    set_num: str
    set_name: str
    total_parts: int
    unique_parts: int
    have_parts: int
    have_unique: int
    completion_pct: float
    missing: List[MissingPartSchema]
    cached_at: datetime

    class Config:
        from_attributes = True
