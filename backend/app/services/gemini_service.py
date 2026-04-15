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


def _build_grounded_prompt(candidates: List[Dict[str, Any]]) -> str:
    """
    Build a Gemini prompt that converts open-set classification into
    disambiguation. Given candidates from an upstream classifier
    (typically Brickognize), Gemini's job becomes "confirm one of these
    OR override if none match" — a much easier task than identifying
    the part from scratch against the full 10k-part LEGO catalog.

    When no candidates are provided, callers should use SYSTEM_PROMPT
    directly (no grounding).
    """
    lines = [
        "You are a LEGO expert. An initial classifier suggested these candidates:",
        "",
    ]
    for i, c in enumerate(candidates[:5], 1):
        part_num  = c.get("part_num", "unknown")
        part_name = c.get("part_name") or c.get("name") or "Unknown"
        conf_pct  = int(round((c.get("confidence", 0.0) or 0.0) * 100))
        lines.append(f"  {i}. {part_num} — {part_name} ({conf_pct}% confidence)")
    lines += [
        "",
        "Look at the image and decide:",
        "  (a) Confirm one of these candidates (return it with HIGH confidence),",
        "  (b) Rank multiple if you're uncertain between them, or",
        "  (c) Override with a different part if NONE of these match.",
        "",
        "Return a JSON array with up to 3 predictions. Each must have:",
        "- part_num:    BrickLink or Rebrickable part number (string)",
        "- part_name:   Full official LEGO part name (string)",
        "- color_name:  Official LEGO colour name (string)",
        "- color_hex:   Hex colour code if known (string or null)",
        "- confidence:  0.0 to 1.0 (number)",
        "",
        "Return ONLY valid JSON. No markdown, no explanations.",
    ]
    return "\n".join(lines)


async def identify_piece(
    image_bytes: bytes,
    candidates: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Classify a LEGO part from a JPEG/PNG.

    Args:
        image_bytes: raw image bytes
        candidates:  optional upstream-classifier predictions (Brickognize
                     top-K). When provided, a grounded prompt asks Gemini
                     to disambiguate these instead of open-set classify —
                     +10-15% accuracy on visually confusable pairs.
                     Callers gate this behind SCAN_GROUNDED_GEMINI env var.
    """
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    prompt_text = _build_grounded_prompt(candidates) if candidates else SYSTEM_PROMPT

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
                        "text": prompt_text,
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
