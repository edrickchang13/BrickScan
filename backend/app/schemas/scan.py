from pydantic import BaseModel
from typing import List, Optional


class ScanRequest(BaseModel):
    image_base64: str


class ScanPrediction(BaseModel):
    part_num: str
    part_name: str
    color_name: Optional[str] = None
    color_hex: Optional[str] = None
    confidence: float
    image_url: Optional[str] = None


class StudGrid(BaseModel):
    cols: int
    rows: int
    confidence: float


class ScanResponse(BaseModel):
    predictions: List[ScanPrediction]
    stud_grid: Optional[StudGrid] = None
    scan_id: Optional[str] = None
    thumbnail_url: Optional[str] = None


class ConfirmScanRequest(BaseModel):
    scan_log_id: str
    confirmed_part_num: str
