"""
Pydantic schemas for local inventory API requests and responses.

Used for request validation and response serialization.
"""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List


class ScanRequest(BaseModel):
    """
    Request to scan a LEGO brick image.

    Attributes:
        image_base64: Base64-encoded JPEG/PNG from mobile camera
        session_id: Optional session to group scans
    """

    image_base64: str = Field(
        ..., description="Base64-encoded image (JPEG or PNG)"
    )
    session_id: Optional[str] = Field(
        None, description="Optional scan session ID for grouping"
    )


class ScanPrediction(BaseModel):
    """Single prediction result from recognition pipeline."""

    part_num: str
    part_name: str = ""
    confidence: float = Field(..., ge=0.0, le=1.0)
    color_id: Optional[int] = None
    color_name: Optional[str] = None
    color_hex: Optional[str] = None
    source: Optional[str] = Field(
        None,
        description="Recognition source: 'brickognize', 'gemini', 'brickognize+gemini', 'local_model'",
    )
    image_url: Optional[str] = Field(
        None, description="Reference image URL from Brickognize"
    )


class ScanResponse(BaseModel):
    """
    Response from scanning a brick image.

    Attributes:
        status: "known" (confidence > 80%) or "uncertain" (< 80%)
        predictions: List of top-3 candidate predictions
        primary_prediction: The highest-confidence single prediction
        save_image: Whether the image was saved for retraining
    """

    status: str = Field(..., description="'known' or 'uncertain'")
    predictions: List[ScanPrediction] = Field(
        ..., description="Top-3 predictions from model"
    )
    primary_prediction: ScanPrediction = Field(
        ..., description="Highest-confidence prediction"
    )
    save_image: bool = Field(
        True, description="Whether image saved for potential retraining"
    )


class ConfirmPredictionRequest(BaseModel):
    """User confirmation of a scan prediction."""

    part_num: str = Field(..., description="Confirmed LEGO part number")
    color_id: Optional[int] = Field(None, description="Confirmed Rebrickable color ID")
    color_name: Optional[str] = Field(None, description="Confirmed color name")
    color_hex: Optional[str] = Field(None, description="Confirmed color hex (without #)")
    quantity: int = Field(1, ge=1, description="Number of this part")


class LocalInventoryPartSchema(BaseModel):
    """Represents a part in local inventory."""

    id: str
    part_num: str
    color_id: Optional[int]
    color_name: Optional[str]
    color_hex: Optional[str]
    quantity: int
    confidence: float
    user_confirmed: bool
    image_path: Optional[str]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class UpdateInventoryQuantityRequest(BaseModel):
    """Update quantity of an inventory part."""

    quantity: int = Field(..., ge=0, description="New quantity (0 to remove)")


class CorrectPredictionRequest(BaseModel):
    """
    User correction of a mispredicted part.

    Used when the model's prediction was wrong and user wants to fix it.
    """

    inventory_id: str = Field(..., description="LocalInventoryPart ID to correct")
    correct_part_num: str = Field(..., description="Correct LEGO part number")
    correct_color_id: Optional[int] = Field(
        None, description="Correct Rebrickable color ID"
    )
    correct_color_name: Optional[str] = Field(None, description="Correct color name")


class VideoScanRequest(BaseModel):
    """
    Request to scan a LEGO brick from multiple angles (video mode).

    Attributes:
        frames: List of base64-encoded images captured at different angles
        session_id: Optional session to group scans
    """

    frames: List[str] = Field(
        ...,
        min_length=2,
        max_length=10,
        description="List of base64-encoded frames from video capture",
    )
    session_id: Optional[str] = Field(
        None, description="Optional scan session ID"
    )


class VideoScanResponse(BaseModel):
    """
    Response from video scanning — aggregated across multiple frames.

    Attributes:
        status: "known" or "uncertain"
        predictions: Consensus predictions voted across all frames
        primary_prediction: Best consensus prediction
        frames_analyzed: How many frames were successfully analyzed
        agreement_score: How consistent predictions were across frames (0-1)
    """

    status: str = Field(..., description="'known' or 'uncertain'")
    predictions: List[ScanPrediction]
    primary_prediction: ScanPrediction
    frames_analyzed: int = Field(..., description="Number of frames successfully analyzed")
    agreement_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Cross-frame agreement (1.0 = all frames agree)",
    )
    save_image: bool = False


class MultiPieceScanRequest(BaseModel):
    """
    Request to scan multiple LEGO pieces in a single image.

    Attributes:
        image_base64: Base64-encoded image of multiple pieces spread out
        session_id: Optional session to group scans
    """

    image_base64: str = Field(
        ..., description="Base64-encoded image with multiple pieces"
    )
    session_id: Optional[str] = Field(
        None, description="Optional scan session ID"
    )


class DetectedPiece(BaseModel):
    """A single detected piece from multi-piece scan."""

    piece_index: int = Field(..., description="Index of this piece in the detection")
    predictions: List[ScanPrediction] = Field(
        ..., description="Top predictions for this piece"
    )
    primary_prediction: ScanPrediction
    bbox: Optional[List[float]] = Field(
        None, description="Bounding box [x1, y1, x2, y2] normalized 0-1"
    )


class MultiPieceScanResponse(BaseModel):
    """Response from multi-piece scanning."""

    pieces_detected: int = Field(..., description="Number of pieces found")
    pieces: List[DetectedPiece] = Field(
        ..., description="Per-piece predictions"
    )
    status: str = Field(..., description="'success' or 'partial'")


class ScanSessionRequest(BaseModel):
    """Start a new scan session."""

    set_name: str = Field(..., description="Name/description of this scan session")


class ScanSessionSchema(BaseModel):
    """Represents a scan session."""

    id: str
    set_name: str
    completed: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class InventoryExportRow(BaseModel):
    """Single row in CSV export."""

    part_num: str
    part_name: Optional[str]
    color_name: Optional[str]
    color_hex: Optional[str]
    quantity: int
    confidence: float
    user_confirmed: bool
