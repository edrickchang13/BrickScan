import httpx
import json
import base64
from typing import List, Dict, Any, Optional
from app.core.config import settings

SYSTEM_PROMPT = """You are a LEGO expert. Identify this LEGO piece from the image provided.

Return a JSON array with up to 3 predictions. Each prediction must be a valid JSON object with these exact fields:
- part_num: The BrickLink or Rebrickable part number (string)
- part_name: The full official LEGO part name (string)
- color_name: The official LEGO color name (string)
- color_hex: The hex color code if known (string or null)
- confidence: A confidence score between 0.0 and 1.0 (number)

Example response:
[
  {
    "part_num": "3001",
    "part_name": "Brick 2x4",
    "color_name": "Red",
    "color_hex": "#C91A09",
    "confidence": 0.95
  }
]

Return ONLY valid JSON. No markdown, no explanations, no code blocks."""


async def identify_piece(image_bytes: bytes) -> List[Dict[str, Any]]:
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

    headers = {
        "Content-Type": "application/json",
    }

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": SYSTEM_PROMPT,
                    },
                    {
                        "inlineData": {
                            "mimeType": "image/jpeg",
                            "data": image_base64,
                        }
                    },
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0.3,
            "topK": 10,
            "topP": 0.95,
            "maxOutputTokens": 500,
        },
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                url,
                headers=headers,
                json=payload,
                params={"key": settings.GEMINI_API_KEY},
                timeout=30.0,
            )
            response.raise_for_status()

            data = response.json()

            if "candidates" not in data or len(data["candidates"]) == 0:
                return []

            content = data["candidates"][0].get("content", {})
            parts = content.get("parts", [])

            if not parts:
                return []

            text_response = parts[0].get("text", "")

            text_response = text_response.strip()
            if text_response.startswith("```json"):
                text_response = text_response[7:]
            if text_response.startswith("```"):
                text_response = text_response[3:]
            if text_response.endswith("```"):
                text_response = text_response[:-3]
            text_response = text_response.strip()

            predictions = json.loads(text_response)

            if not isinstance(predictions, list):
                predictions = [predictions]

            valid_predictions = []
            for pred in predictions[:3]:
                if isinstance(pred, dict):
                    pred_obj = {
                        "part_num": pred.get("part_num", "unknown"),
                        "part_name": pred.get("part_name", "Unknown Part"),
                        "color_name": pred.get("color_name"),
                        "color_hex": pred.get("color_hex"),
                        "confidence": float(pred.get("confidence", 0.5)),
                    }
                    valid_predictions.append(pred_obj)

            return valid_predictions

        except httpx.HTTPStatusError as e:
            import logging
            logging.getLogger(__name__).warning(
                "Gemini API error %s: %s", e.response.status_code, e.response.text[:200]
            )
            return []
        except httpx.HTTPError as e:
            import logging
            logging.getLogger(__name__).warning("Gemini HTTP error: %s", e)
            return []
        except json.JSONDecodeError as e:
            import logging
            logging.getLogger(__name__).warning("Gemini JSON parse error: %s", e)
            return []
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("Gemini unexpected error: %s", e)
            return []
