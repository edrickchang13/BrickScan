"""
Regression test: legacy POST /api/scan must keep returning the same response
shape (predictions list of {part_num, part_name, confidence, …}).

The streaming endpoints can evolve; this test guards backwards-compat for
mobile clients that haven't been updated to the SSE flow yet.
"""

import base64
import io
from unittest.mock import patch, AsyncMock

import pytest
from PIL import Image


def _tiny_jpeg_b64() -> str:
    img = Image.new("RGB", (32, 32), color=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


@pytest.mark.asyncio
async def test_legacy_scan_returns_expected_shape(client, auth_headers, sample_parts):
    """POST /api/scan returns ScanResponse with predictions list — same shape as v1."""
    fake_predictions = [
        {
            "part_num": "3001",
            "part_name": "Brick 2x4",
            "confidence": 0.91,
            "color_name": "Red",
            "color_hex": "FF0000",
            "color_id": 1,
            "source": "brickognize",
        },
        {
            "part_num": "3002",
            "part_name": "Brick 2x2",
            "confidence": 0.42,
            "color_name": None,
            "color_hex": None,
            "color_id": None,
            "source": "gemini",
        },
    ]

    with patch(
        "app.api.scan.hybrid_predict",
        new=AsyncMock(return_value=fake_predictions),
    ), patch(
        "app.api.scan._maybe_detect_multipiece",
        new=AsyncMock(return_value=[]),
    ):
        resp = client.post(
            "/api/scan",
            json={"image_base64": _tiny_jpeg_b64()},
            headers=auth_headers,
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "predictions" in body
    assert isinstance(body["predictions"], list)
    assert len(body["predictions"]) == 2

    top = body["predictions"][0]
    for required_field in ("part_num", "part_name", "confidence"):
        assert required_field in top, f"Missing field: {required_field}"
    assert top["part_num"] == "3001"
    assert 0.0 <= top["confidence"] <= 1.0

    # New optional fields must be present but won't break old clients
    assert "detections" in body
    assert body["detections"] == []
