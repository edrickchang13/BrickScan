"""
Brickognize API client for LEGO brick recognition.

Brickognize is a free, purpose-built API using ConvNeXt-T (~28M params)
trained specifically on LEGO parts, sets, and minifigures.

API docs: https://api.brickognize.com/
No API key required — completely free.

Returns predictions with Rebrickable-compatible part numbers.
"""

from __future__ import annotations

import base64
import logging
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

BRICKOGNIZE_API_URL = "https://api.brickognize.com/predict/"


async def identify_part(image_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Send an image to Brickognize and return predictions.

    The API accepts multipart/form-data with an image file.
    Returns a list of prediction dicts with keys:
      - part_num (str): Rebrickable part number
      - part_name (str): Human-readable name
      - confidence (float): 0.0–1.0
      - color_name (str|None)
      - color_hex (str|None)
      - color_id (int|None)
      - source (str): "brickognize"
      - image_url (str|None): Reference image URL
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Brickognize expects multipart with "query_image" field
            files = {"query_image": ("scan.jpg", image_bytes, "image/jpeg")}
            response = await client.post(BRICKOGNIZE_API_URL, files=files)
            response.raise_for_status()

            data = response.json()
            logger.debug("Brickognize raw response: %s", data)

            items = data.get("items", [])
            if not items:
                logger.info("Brickognize returned no predictions")
                return []

            predictions = []
            for item in items[:5]:  # Top 5
                # Brickognize returns id, name, img_url, score, type
                pred = {
                    "part_num": item.get("id", "unknown"),
                    "part_name": item.get("name", ""),
                    "confidence": float(item.get("score", 0.0)),
                    "color_name": item.get("color_name"),
                    "color_hex": item.get("color_hex"),
                    "color_id": item.get("color_id"),
                    "source": "brickognize",
                    "image_url": item.get("img_url"),
                    "item_type": item.get("type", "part"),
                }
                predictions.append(pred)

            if predictions:
                top = predictions[0]
                logger.info(
                    "Brickognize top: %s (%s) - %.1f%%",
                    top["part_num"],
                    top["part_name"],
                    top["confidence"] * 100,
                )

            return predictions

    except httpx.TimeoutException:
        logger.warning("Brickognize API timed out")
        return []
    except httpx.HTTPStatusError as e:
        logger.warning("Brickognize API HTTP error: %s", e.response.status_code)
        return []
    except Exception as e:
        logger.error("Brickognize API error: %s", e)
        return []
