"""
Utility functions for local inventory system.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional
from app.local_inventory.constants import (
    CONFIDENCE_THRESHOLD_KNOWN,
    CONFIDENCE_THRESHOLD_UNCERTAIN,
    STATUS_KNOWN,
    STATUS_UNCERTAIN,
)

logger = logging.getLogger(__name__)


def determine_confidence_status(confidence: float) -> str:
    """
    Classify prediction confidence as "known" or "uncertain".

    Args:
        confidence: Model confidence score (0.0 to 1.0)

    Returns:
        STATUS_KNOWN (>= 80%) or STATUS_UNCERTAIN (< 80%)
    """
    if confidence >= CONFIDENCE_THRESHOLD_KNOWN:
        return STATUS_KNOWN
    return STATUS_UNCERTAIN


def format_confidence_percent(confidence: float) -> str:
    """Format confidence as percentage with 1 decimal place."""
    return f"{confidence * 100:.1f}%"


def normalize_part_num(part_num: str) -> str:
    """
    Normalize LEGO part number (basic sanitization).

    Strips whitespace, converts to uppercase.

    Args:
        part_num: Raw part number string

    Returns:
        Normalized part number
    """
    return part_num.strip().upper()


def is_valid_part_num(part_num: str) -> bool:
    """
    Basic validation for LEGO part numbers.

    Valid if non-empty after normalization.

    Args:
        part_num: Part number to validate

    Returns:
        True if valid format
    """
    return bool(normalize_part_num(part_num))


def format_datetime_iso(dt: datetime) -> str:
    """Format datetime as ISO 8601 string."""
    if dt:
        return dt.isoformat()
    return ""


def summarize_inventory(
    total_parts: int,
    total_quantity: int,
    confirmed_count: int,
    uncertain_count: int,
) -> Dict[str, Any]:
    """
    Create human-readable summary of inventory state.

    Args:
        total_parts: Number of unique parts
        total_quantity: Total physical pieces
        confirmed_count: Count of user-confirmed parts
        uncertain_count: Count of unconfirmed predictions

    Returns:
        Dictionary with summary text and stats
    """
    confirmed_pct = (
        (confirmed_count / total_parts * 100)
        if total_parts > 0
        else 0
    )

    return {
        "total_unique_parts": total_parts,
        "total_pieces": total_quantity,
        "confirmed_parts": confirmed_count,
        "uncertain_parts": uncertain_count,
        "confirmed_percentage": round(confirmed_pct, 1),
        "summary_text": (
            f"{total_parts} unique parts ({total_quantity} total pieces), "
            f"{confirmed_count} confirmed ({confirmed_pct:.1f}%)"
        ),
    }


class ConfidenceAnalysis:
    """
    Helper for analyzing confidence scores across multiple predictions.

    Usage:
        analysis = ConfidenceAnalysis(predictions)
        print(analysis.average_confidence)
        print(analysis.status)
    """

    def __init__(self, predictions: list):
        """
        Initialize with list of prediction dicts.

        Each prediction should have 'confidence' key.

        Args:
            predictions: List of prediction dictionaries
        """
        self.predictions = predictions
        self._analyze()

    def _analyze(self) -> None:
        """Internal analysis of prediction confidence."""
        if not self.predictions:
            self.average_confidence = 0.0
            self.max_confidence = 0.0
            self.min_confidence = 0.0
            self.status = STATUS_UNCERTAIN
            return

        confidences = [
            p.get("confidence", 0.0)
            for p in self.predictions
        ]

        self.average_confidence = sum(confidences) / len(confidences)
        self.max_confidence = max(confidences)
        self.min_confidence = min(confidences)
        self.status = determine_confidence_status(self.max_confidence)

    def __repr__(self) -> str:
        return (
            f"ConfidenceAnalysis("
            f"avg={format_confidence_percent(self.average_confidence)}, "
            f"max={format_confidence_percent(self.max_confidence)}, "
            f"status={self.status})"
        )
