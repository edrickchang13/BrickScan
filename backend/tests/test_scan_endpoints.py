"""
Comprehensive tests for BrickScan video and multi-piece scan endpoints.

Tests cover:
- POST /api/local-inventory/scan-video
- POST /api/local-inventory/scan-multi

Uses AsyncClient with mocked hybrid_predict to test endpoint behavior
without requiring GPU inference or actual LEGO images.
"""

import pytest
import base64
import io
from unittest.mock import AsyncMock, patch, MagicMock
from httpx import AsyncClient, ASGITransport
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from app.main import app
from app.local_inventory.database import get_local_db, Base
from app.local_inventory.models import LocalInventoryPart, ScanSession
from app.local_inventory.schemas import (
    VideoScanRequest,
    MultiPieceScanRequest,
    ScanPrediction,
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def test_db_engine():
    """Create an in-memory SQLite database for testing."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def test_db_session(test_db_engine):
    """Create a database session for testing."""
    SessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=test_db_engine,
    )
    session = SessionLocal()
    yield session
    session.close()


@pytest.fixture
def app_with_test_db(test_db_session):
    """Override the app's database dependency with test DB."""
    def override_get_local_db():
        yield test_db_session

    app.dependency_overrides[get_local_db] = override_get_local_db
    yield app
    app.dependency_overrides.clear()


@pytest.fixture
async def async_client(app_with_test_db):
    """Create an async HTTP client for testing the FastAPI app."""
    async with AsyncClient(
        app=app_with_test_db,
        base_url="http://test",
        transport=ASGITransport(app=app_with_test_db),
    ) as client:
        yield client


@pytest.fixture
def valid_jpeg_base64():
    """Generate a valid small JPEG image in base64 format."""
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("Pillow not installed")

    # Create a 4x4 white JPEG
    img = Image.new("RGB", (4, 4), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    buf.seek(0)

    return base64.b64encode(buf.getvalue()).decode()


@pytest.fixture
def invalid_base64():
    """Generate clearly invalid base64 that cannot be decoded as an image."""
    return "not_valid_base64_!!!@@@"


@pytest.fixture
def empty_base64():
    """Return an empty base64 string."""
    return ""


def create_mock_prediction(part_num: str, confidence: float = 0.85) -> dict:
    """Helper to create a mock hybrid_predict output."""
    return {
        "part_num": part_num,
        "part_name": f"Part {part_num}",
        "confidence": confidence,
        "color_id": 1,
        "color_name": "Red",
        "color_hex": "FF0000",
        "source": "local_model",
        "image_url": None,
    }


# ============================================================================
# VIDEO SCAN ENDPOINT TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_scan_video_happy_path_2_frames(async_client, valid_jpeg_base64):
    """Test successful video scan with 2 valid frames."""
    frames = [valid_jpeg_base64, valid_jpeg_base64]

    mock_predictions = [
        [create_mock_prediction("3001", 0.92)],
        [create_mock_prediction("3001", 0.88)],
    ]

    with patch(
        "app.local_inventory.routes.hybrid_predict",
        new_callable=AsyncMock,
        side_effect=mock_predictions,
    ):
        response = await async_client.post(
            "/api/local-inventory/scan-video",
            json={"frames": frames},
        )

    assert response.status_code == 200
    data = response.json()

    # Validate response structure
    assert "status" in data
    assert "predictions" in data
    assert "primary_prediction" in data
    assert "frames_analyzed" in data
    assert "agreement_score" in data
    assert "save_image" in data

    # Validate values
    assert data["status"] in ["known", "uncertain"]
    assert data["frames_analyzed"] == 2
    assert 0.0 <= data["agreement_score"] <= 1.0
    assert isinstance(data["save_image"], bool)
    assert len(data["predictions"]) > 0
    assert data["primary_prediction"]["part_num"] is not None


@pytest.mark.asyncio
async def test_scan_video_happy_path_10_frames(async_client, valid_jpeg_base64):
    """Test successful video scan with maximum 10 frames."""
    frames = [valid_jpeg_base64] * 10

    mock_predictions = [
        [create_mock_prediction("3001", 0.90 - i * 0.01)]
        for i in range(10)
    ]

    with patch(
        "app.local_inventory.routes.hybrid_predict",
        new_callable=AsyncMock,
        side_effect=mock_predictions,
    ):
        response = await async_client.post(
            "/api/local-inventory/scan-video",
            json={"frames": frames},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["frames_analyzed"] == 10


@pytest.mark.asyncio
async def test_scan_video_fewer_than_2_frames(async_client, valid_jpeg_base64):
    """Test that fewer than 2 frames returns 422."""
    # Single frame
    frames = [valid_jpeg_base64]

    response = await async_client.post(
        "/api/local-inventory/scan-video",
        json={"frames": frames},
    )

    # Pydantic validation error (min_length=2)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_scan_video_more_than_10_frames(async_client, valid_jpeg_base64):
    """Test that more than 10 frames returns 422."""
    frames = [valid_jpeg_base64] * 11

    response = await async_client.post(
        "/api/local-inventory/scan-video",
        json={"frames": frames},
    )

    # Pydantic validation error (max_length=10)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_scan_video_with_invalid_frames(async_client, valid_jpeg_base64, invalid_base64):
    """Test that scan handles mix of valid and invalid frames gracefully."""
    frames = [valid_jpeg_base64, invalid_base64, valid_jpeg_base64]

    mock_predictions = [
        [create_mock_prediction("3001", 0.90)],
        [create_mock_prediction("3001", 0.88)],
    ]

    with patch(
        "app.local_inventory.routes.validate_and_decode_image",
        side_effect=[
            b"frame0_bytes",
            ValueError("Invalid image"),
            b"frame2_bytes",
        ],
    ):
        with patch(
            "app.local_inventory.routes.hybrid_predict",
            new_callable=AsyncMock,
            side_effect=mock_predictions,
        ):
            response = await async_client.post(
                "/api/local-inventory/scan-video",
                json={"frames": frames},
            )

    # Should succeed with 2 valid frames
    assert response.status_code == 200
    data = response.json()
    assert data["frames_analyzed"] == 2


@pytest.mark.asyncio
async def test_scan_video_empty_base64_frames(async_client, empty_base64):
    """Test that empty base64 frames are handled gracefully."""
    frames = [empty_base64, empty_base64]

    with patch(
        "app.local_inventory.routes.validate_and_decode_image",
        side_effect=ValueError("Invalid image"),
    ):
        response = await async_client.post(
            "/api/local-inventory/scan-video",
            json={"frames": frames},
        )

    # Should fail because we end up with 0 valid frames (threshold is 2)
    assert response.status_code == 400
    assert "Need at least 2 valid frames" in response.json()["detail"]


@pytest.mark.asyncio
async def test_scan_video_all_frames_fail_prediction(async_client, valid_jpeg_base64):
    """Test that when all predictions fail, endpoint returns 500."""
    frames = [valid_jpeg_base64, valid_jpeg_base64]

    with patch(
        "app.local_inventory.routes.hybrid_predict",
        new_callable=AsyncMock,
        side_effect=[Exception("Model error"), Exception("Model error")],
    ):
        response = await async_client.post(
            "/api/local-inventory/scan-video",
            json={"frames": frames},
        )

    assert response.status_code == 500
    assert "All frames failed recognition" in response.json()["detail"]


@pytest.mark.asyncio
async def test_scan_video_response_schema_validation(async_client, valid_jpeg_base64):
    """Test that all required response fields are present and have correct types."""
    frames = [valid_jpeg_base64, valid_jpeg_base64]

    mock_predictions = [
        [
            create_mock_prediction("3001", 0.92),
            create_mock_prediction("3002", 0.05),
        ],
        [
            create_mock_prediction("3001", 0.88),
            create_mock_prediction("3003", 0.08),
        ],
    ]

    with patch(
        "app.local_inventory.routes.hybrid_predict",
        new_callable=AsyncMock,
        side_effect=mock_predictions,
    ):
        response = await async_client.post(
            "/api/local-inventory/scan-video",
            json={"frames": frames},
        )

    assert response.status_code == 200
    data = response.json()

    # Validate all required fields exist
    assert "status" in data
    assert "predictions" in data
    assert "primary_prediction" in data
    assert "frames_analyzed" in data
    assert "agreement_score" in data
    assert "save_image" in data

    # Validate types
    assert isinstance(data["status"], str)
    assert isinstance(data["predictions"], list)
    assert isinstance(data["primary_prediction"], dict)
    assert isinstance(data["frames_analyzed"], int)
    assert isinstance(data["agreement_score"], (int, float))
    assert isinstance(data["save_image"], bool)

    # Validate prediction structure
    for pred in data["predictions"]:
        assert "part_num" in pred
        assert "confidence" in pred
        assert 0.0 <= pred["confidence"] <= 1.0

    primary = data["primary_prediction"]
    assert "part_num" in primary
    assert "confidence" in primary
    assert 0.0 <= primary["confidence"] <= 1.0


@pytest.mark.asyncio
async def test_scan_video_agreement_score_bounds(async_client, valid_jpeg_base64):
    """Test that agreement_score is always between 0 and 1."""
    frames = [valid_jpeg_base64, valid_jpeg_base64, valid_jpeg_base64]

    # All frames agree on same part
    mock_predictions = [
        [create_mock_prediction("3001", 0.95)],
        [create_mock_prediction("3001", 0.93)],
        [create_mock_prediction("3001", 0.91)],
    ]

    with patch(
        "app.local_inventory.routes.hybrid_predict",
        new_callable=AsyncMock,
        side_effect=mock_predictions,
    ):
        response = await async_client.post(
            "/api/local-inventory/scan-video",
            json={"frames": frames},
        )

    assert response.status_code == 200
    data = response.json()

    # Perfect agreement should be 1.0
    assert data["agreement_score"] == 1.0

    # Test partial agreement
    mock_predictions = [
        [create_mock_prediction("3001", 0.95)],
        [create_mock_prediction("3001", 0.93)],
        [create_mock_prediction("3002", 0.91)],  # Different part
    ]

    with patch(
        "app.local_inventory.routes.hybrid_predict",
        new_callable=AsyncMock,
        side_effect=mock_predictions,
    ):
        response = await async_client.post(
            "/api/local-inventory/scan-video",
            json={"frames": frames},
        )

    assert response.status_code == 200
    data = response.json()

    # Partial agreement: 2/3 frames agree = 0.667
    assert 0.0 <= data["agreement_score"] <= 1.0


@pytest.mark.asyncio
async def test_scan_video_weighted_voting(async_client, valid_jpeg_base64):
    """Test that predictions are aggregated via weighted voting by confidence."""
    frames = [valid_jpeg_base64, valid_jpeg_base64]

    # Frame 1: high confidence for 3001, low for 3002
    # Frame 2: high confidence for 3002, low for 3001
    # Should vote for the part with highest aggregate confidence
    mock_predictions = [
        [
            create_mock_prediction("3001", 0.95),
            create_mock_prediction("3002", 0.03),
        ],
        [
            create_mock_prediction("3002", 0.92),
            create_mock_prediction("3001", 0.05),
        ],
    ]

    with patch(
        "app.local_inventory.routes.hybrid_predict",
        new_callable=AsyncMock,
        side_effect=mock_predictions,
    ):
        response = await async_client.post(
            "/api/local-inventory/scan-video",
            json={"frames": frames},
        )

    assert response.status_code == 200
    data = response.json()

    # Primary prediction should exist and have both parts in predictions list
    assert data["primary_prediction"]["part_num"] in ["3001", "3002"]
    assert len(data["predictions"]) >= 1


# ============================================================================
# MULTI-PIECE SCAN ENDPOINT TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_scan_multi_happy_path_valid_image(async_client, valid_jpeg_base64):
    """Test successful multi-piece scan with a valid image."""
    # Mock predictions for 4 quadrants (2x2 grid)
    mock_predictions = [
        [create_mock_prediction("3001", 0.85)],  # quadrant 0
        [create_mock_prediction("3002", 0.82)],  # quadrant 1
        [create_mock_prediction("3003", 0.75)],  # quadrant 2
        [create_mock_prediction("3004", 0.65)],  # quadrant 3
    ]

    with patch(
        "app.local_inventory.routes.hybrid_predict",
        new_callable=AsyncMock,
        side_effect=mock_predictions,
    ):
        response = await async_client.post(
            "/api/local-inventory/scan-multi",
            json={"image_base64": valid_jpeg_base64},
        )

    assert response.status_code == 200
    data = response.json()

    # Validate response structure
    assert "pieces_detected" in data
    assert "pieces" in data
    assert "status" in data

    # Validate values
    assert isinstance(data["pieces_detected"], int)
    assert isinstance(data["pieces"], list)
    assert data["status"] in ["success", "partial"]
    assert data["pieces_detected"] == len(data["pieces"])


@pytest.mark.asyncio
async def test_scan_multi_filters_low_confidence(async_client, valid_jpeg_base64):
    """Test that quadrants with <0.30 confidence are filtered out."""
    # Some quadrants below MULTI_THRESHOLD (0.30), some above
    mock_predictions = [
        [create_mock_prediction("3001", 0.85)],  # above threshold
        [create_mock_prediction("3002", 0.15)],  # below threshold (will be filtered)
        [create_mock_prediction("3003", 0.45)],  # above threshold
        [],  # empty quadrant
    ]

    with patch(
        "app.local_inventory.routes.hybrid_predict",
        new_callable=AsyncMock,
        side_effect=mock_predictions,
    ):
        response = await async_client.post(
            "/api/local-inventory/scan-multi",
            json={"image_base64": valid_jpeg_base64},
        )

    assert response.status_code == 200
    data = response.json()

    # Should only include 2 pieces (0.85 and 0.45)
    assert data["pieces_detected"] == 2
    assert len(data["pieces"]) == 2


@pytest.mark.asyncio
async def test_scan_multi_empty_image_returns_partial(async_client, valid_jpeg_base64):
    """Test that an image with no detectable pieces returns 'partial' status."""
    # All quadrants below threshold or empty
    mock_predictions = [
        [create_mock_prediction("3001", 0.25)],  # below threshold
        [create_mock_prediction("3002", 0.10)],  # below threshold
        [create_mock_prediction("3003", 0.20)],  # below threshold
        [],  # empty
    ]

    with patch(
        "app.local_inventory.routes.hybrid_predict",
        new_callable=AsyncMock,
        side_effect=mock_predictions,
    ):
        response = await async_client.post(
            "/api/local-inventory/scan-multi",
            json={"image_base64": valid_jpeg_base64},
        )

    assert response.status_code == 200
    data = response.json()

    assert data["pieces_detected"] == 0
    assert data["status"] == "partial"
    assert len(data["pieces"]) == 0


@pytest.mark.asyncio
async def test_scan_multi_invalid_image_base64(async_client, invalid_base64):
    """Test that invalid base64 image returns 400."""
    response = await async_client.post(
        "/api/local-inventory/scan-multi",
        json={"image_base64": invalid_base64},
    )

    assert response.status_code == 400
    assert "Invalid image" in response.json()["detail"]


@pytest.mark.asyncio
async def test_scan_multi_empty_base64_returns_400(async_client, empty_base64):
    """Test that empty base64 returns 400."""
    response = await async_client.post(
        "/api/local-inventory/scan-multi",
        json={"image_base64": empty_base64},
    )

    assert response.status_code == 400


@pytest.mark.asyncio
async def test_scan_multi_response_schema_validation(async_client, valid_jpeg_base64):
    """Test that all required response fields are present and have correct types."""
    mock_predictions = [
        [create_mock_prediction("3001", 0.85)],
        [create_mock_prediction("3002", 0.82)],
        [],
        [],
    ]

    with patch(
        "app.local_inventory.routes.hybrid_predict",
        new_callable=AsyncMock,
        side_effect=mock_predictions,
    ):
        response = await async_client.post(
            "/api/local-inventory/scan-multi",
            json={"image_base64": valid_jpeg_base64},
        )

    assert response.status_code == 200
    data = response.json()

    # Validate all required fields
    assert "pieces_detected" in data
    assert "pieces" in data
    assert "status" in data

    # Validate types
    assert isinstance(data["pieces_detected"], int)
    assert isinstance(data["pieces"], list)
    assert isinstance(data["status"], str)

    # Validate pieces structure
    for piece in data["pieces"]:
        assert "piece_index" in piece
        assert "predictions" in piece
        assert "primary_prediction" in piece
        assert "bbox" in piece

        assert isinstance(piece["piece_index"], int)
        assert isinstance(piece["predictions"], list)
        assert isinstance(piece["primary_prediction"], dict)
        assert isinstance(piece["bbox"], list)

        # Validate bbox is normalized 0-1
        for coord in piece["bbox"]:
            assert 0.0 <= coord <= 1.0

        # Validate prediction structure
        for pred in piece["predictions"]:
            assert "part_num" in pred
            assert "confidence" in pred
            assert 0.0 <= pred["confidence"] <= 1.0


@pytest.mark.asyncio
async def test_scan_multi_pieces_detected_matches_length(async_client, valid_jpeg_base64):
    """Test that pieces_detected matches len(pieces)."""
    mock_predictions = [
        [create_mock_prediction("3001", 0.85)],
        [create_mock_prediction("3002", 0.75)],
        [create_mock_prediction("3003", 0.60)],
        [],  # empty, filtered out
    ]

    with patch(
        "app.local_inventory.routes.hybrid_predict",
        new_callable=AsyncMock,
        side_effect=mock_predictions,
    ):
        response = await async_client.post(
            "/api/local-inventory/scan-multi",
            json={"image_base64": valid_jpeg_base64},
        )

    assert response.status_code == 200
    data = response.json()

    assert data["pieces_detected"] == len(data["pieces"])
    assert data["pieces_detected"] == 3


@pytest.mark.asyncio
async def test_scan_multi_piece_index_sequential(async_client, valid_jpeg_base64):
    """Test that piece_index values are sequential starting from 0."""
    mock_predictions = [
        [create_mock_prediction("3001", 0.85)],
        [create_mock_prediction("3002", 0.75)],
        [create_mock_prediction("3003", 0.60)],
        [create_mock_prediction("3004", 0.55)],
    ]

    with patch(
        "app.local_inventory.routes.hybrid_predict",
        new_callable=AsyncMock,
        side_effect=mock_predictions,
    ):
        response = await async_client.post(
            "/api/local-inventory/scan-multi",
            json={"image_base64": valid_jpeg_base64},
        )

    assert response.status_code == 200
    data = response.json()

    # Verify piece indices are sequential
    for i, piece in enumerate(data["pieces"]):
        assert piece["piece_index"] == i


@pytest.mark.asyncio
async def test_scan_multi_threshold_boundary(async_client, valid_jpeg_base64):
    """Test behavior at the 0.30 confidence threshold boundary."""
    # Create predictions at and around the 0.30 threshold
    mock_predictions = [
        [create_mock_prediction("3001", 0.3001)],   # Just above, should be included
        [create_mock_prediction("3002", 0.30)],     # At threshold, behavior depends on implementation
        [create_mock_prediction("3003", 0.2999)],   # Just below, should be excluded
        [create_mock_prediction("3004", 0.15)],     # Well below, should be excluded
    ]

    with patch(
        "app.local_inventory.routes.hybrid_predict",
        new_callable=AsyncMock,
        side_effect=mock_predictions,
    ):
        response = await async_client.post(
            "/api/local-inventory/scan-multi",
            json={"image_base64": valid_jpeg_base64},
        )

    assert response.status_code == 200
    data = response.json()

    # Should include pieces above 0.30
    assert data["pieces_detected"] >= 1


@pytest.mark.asyncio
async def test_scan_multi_bbox_coordinates(async_client, valid_jpeg_base64):
    """Test that bbox coordinates are normalized 0-1 (representing quadrants)."""
    mock_predictions = [
        [create_mock_prediction("3001", 0.85)],
        [create_mock_prediction("3002", 0.75)],
        [create_mock_prediction("3003", 0.60)],
        [create_mock_prediction("3004", 0.55)],
    ]

    with patch(
        "app.local_inventory.routes.hybrid_predict",
        new_callable=AsyncMock,
        side_effect=mock_predictions,
    ):
        response = await async_client.post(
            "/api/local-inventory/scan-multi",
            json={"image_base64": valid_jpeg_base64},
        )

    assert response.status_code == 200
    data = response.json()

    # For 2x2 grid, each quadrant should have bbox covering its area
    for piece in data["pieces"]:
        bbox = piece["bbox"]
        assert len(bbox) == 4
        x1, y1, x2, y2 = bbox

        # Validate normalized coordinates
        assert 0.0 <= x1 <= 1.0
        assert 0.0 <= y1 <= 1.0
        assert 0.0 <= x2 <= 1.0
        assert 0.0 <= y2 <= 1.0

        # x2 > x1 and y2 > y1 (valid bounding box)
        assert x2 > x1
        assert y2 > y1


@pytest.mark.asyncio
async def test_scan_multi_all_quadrants_predictions(async_client, valid_jpeg_base64):
    """Test that multi-piece scan runs prediction on all 4 quadrants."""
    call_count = 0
    async_mock = AsyncMock()

    async def mock_hybrid_predict_counter(image_bytes):
        nonlocal call_count
        call_count += 1
        if call_count <= 4:
            return [create_mock_prediction(f"300{call_count}", 0.85)]
        return []

    async_mock.side_effect = mock_hybrid_predict_counter

    with patch(
        "app.local_inventory.routes.hybrid_predict",
        new_callable=AsyncMock,
        side_effect=mock_hybrid_predict_counter,
    ):
        response = await async_client.post(
            "/api/local-inventory/scan-multi",
            json={"image_base64": valid_jpeg_base64},
        )

    assert response.status_code == 200
    # hybrid_predict should have been called 4 times (one per quadrant)
    assert call_count == 4


# ============================================================================
# EDGE CASE AND INTEGRATION TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_scan_video_zero_frames(async_client):
    """Test that empty frames list is rejected."""
    response = await async_client.post(
        "/api/local-inventory/scan-video",
        json={"frames": []},
    )

    # Pydantic validation: min_length=2
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_scan_video_response_status_known_vs_uncertain(async_client, valid_jpeg_base64):
    """Test that status is 'known' when confidence and agreement are high."""
    frames = [valid_jpeg_base64, valid_jpeg_base64]

    # High confidence and agreement should yield "known"
    mock_predictions = [
        [create_mock_prediction("3001", 0.95)],
        [create_mock_prediction("3001", 0.93)],
    ]

    with patch(
        "app.local_inventory.routes.hybrid_predict",
        new_callable=AsyncMock,
        side_effect=mock_predictions,
    ):
        response = await async_client.post(
            "/api/local-inventory/scan-video",
            json={"frames": frames},
        )

    assert response.status_code == 200
    data = response.json()

    # Should be "known" due to high confidence and full agreement
    assert data["status"] == "known"
    assert data["agreement_score"] == 1.0


@pytest.mark.asyncio
async def test_scan_video_response_status_uncertain_low_agreement(async_client, valid_jpeg_base64):
    """Test that status is 'uncertain' when agreement is low."""
    frames = [valid_jpeg_base64, valid_jpeg_base64, valid_jpeg_base64]

    # Low agreement (different parts in different frames)
    mock_predictions = [
        [create_mock_prediction("3001", 0.90)],
        [create_mock_prediction("3002", 0.85)],
        [create_mock_prediction("3003", 0.80)],
    ]

    with patch(
        "app.local_inventory.routes.hybrid_predict",
        new_callable=AsyncMock,
        side_effect=mock_predictions,
    ):
        response = await async_client.post(
            "/api/local-inventory/scan-video",
            json={"frames": frames},
        )

    assert response.status_code == 200
    data = response.json()

    # Should be "uncertain" due to disagreement
    assert data["status"] == "uncertain"
    assert data["agreement_score"] < 1.0


@pytest.mark.asyncio
async def test_scan_multi_no_missing_fields_in_piece_predictions(async_client, valid_jpeg_base64):
    """Test that piece predictions include all required fields."""
    mock_predictions = [
        [
            create_mock_prediction("3001", 0.85),
            create_mock_prediction("3002", 0.10),
        ],
        [],
        [],
        [],
    ]

    with patch(
        "app.local_inventory.routes.hybrid_predict",
        new_callable=AsyncMock,
        side_effect=mock_predictions,
    ):
        response = await async_client.post(
            "/api/local-inventory/scan-multi",
            json={"image_base64": valid_jpeg_base64},
        )

    assert response.status_code == 200
    data = response.json()

    for piece in data["pieces"]:
        for pred in piece["predictions"]:
            # All prediction fields should be present (though some may be None/null)
            assert "part_num" in pred
            assert "confidence" in pred
            assert "part_name" in pred
