from pydantic import BaseModel
from typing import Any, Dict, List, Optional


class ScanRequest(BaseModel):
    image_base64: str


class ScanPrediction(BaseModel):
    part_num: str
    part_name: str
    color_name: Optional[str] = None
    color_hex: Optional[str] = None
    confidence: float
    image_url: Optional[str] = None
    source: Optional[str] = None


class StudGrid(BaseModel):
    cols: int
    rows: int
    confidence: float


class PieceDetection(BaseModel):
    """One detected brick in a multi-piece frame, with its own predictions."""
    bbox: Dict[str, float]            # {x1, y1, x2, y2} in normalised [0, 1]
    detector: str                     # "yolo" | "opencv" | "single"
    predictions: List[ScanPrediction]


class ScanResponse(BaseModel):
    predictions: List[ScanPrediction]
    detections: List[PieceDetection] = []
    stud_grid: Optional[StudGrid] = None
    scan_id: Optional[str] = None
    thumbnail_url: Optional[str] = None
    # URL of a Grad-CAM / occlusion-sensitivity heatmap overlay explaining the
    # top prediction (populated when the backend can introspect the model).
    heatmap_url: Optional[str] = None


class ConfirmScanRequest(BaseModel):
    scan_log_id: str
    confirmed_part_num: str


class ScanStartResponse(BaseModel):
    scan_id: str


class ScanStreamEvent(BaseModel):
    """One event emitted by the worker over SSE."""
    stage: str                        # e.g. "brickognize", "gemini", "merge", "done"
    percent: int                      # 0-100, or -1 for heartbeat
    message: str = ""
    partial: Optional[Dict[str, Any]] = None  # progressively-improving prediction set
