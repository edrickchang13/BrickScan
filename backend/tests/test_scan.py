"""
Unit tests for the scan endpoint.
Tests image validation, model inference, and confirmation workflows.
"""

import pytest
import base64
from unittest.mock import AsyncMock, patch, MagicMock
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from app.main import app
from app.core.database import get_db, Base
from app.models.user import User
from app.models.scan import Scan
from app.services.image_service import decode_base64_image, validate_and_preprocess_image


# Test fixtures
@pytest.fixture
async def test_db():
    """Create in-memory test database"""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        yield session

    await engine.dispose()


@pytest.fixture
async def test_user(test_db: AsyncSession) -> User:
    """Create test user"""
    user = User(
        email="test@example.com",
        hashed_password="hashed_password",
        username="testuser",
    )
    test_db.add(user)
    await test_db.commit()
    await test_db.refresh(user)
    return user


@pytest.fixture
def valid_base64_image() -> str:
    """Create a valid base64-encoded JPEG image"""
    from PIL import Image
    import io

    # Create a simple 100x100 red image
    img = Image.new("RGB", (100, 100), color="red")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)

    return base64.b64encode(buffer.getvalue()).decode()


@pytest.fixture
def data_url_image(valid_base64_image: str) -> str:
    """Create a data URL format image"""
    return f"data:image/jpeg;base64,{valid_base64_image}"


# Tests
@pytest.mark.asyncio
async def test_scan_returns_predictions(test_db: AsyncSession, test_user: User, valid_base64_image: str):
    """Test that scan endpoint returns ML model predictions"""

    with patch("app.services.ml_service.infer_part") as mock_infer:
        mock_infer.return_value = {
            "part_num": "3003",
            "confidence": 0.92,
            "alternatives": [
                {"part_num": "3003", "confidence": 0.92},
                {"part_num": "3004", "confidence": 0.05},
            ]
        }

        client = AsyncClient(app=app, base_url="http://test")

        response = await client.post(
            "/api/scans/scan",
            json={"image": valid_base64_image},
            headers={"Authorization": f"Bearer {test_user.id}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "part_num" in data
        assert data["confidence"] >= 0
        assert "alternatives" in data


@pytest.mark.asyncio
async def test_scan_invalid_base64_returns_400(test_user: User):
    """Test that invalid base64 returns 400"""

    client = AsyncClient(app=app, base_url="http://test")

    response = await client.post(
        "/api/scans/scan",
        json={"image": "not_valid_base64!!!"},
        headers={"Authorization": f"Bearer {test_user.id}"},
    )

    assert response.status_code == 400
    assert "Invalid" in response.json().get("detail", "")


@pytest.mark.asyncio
async def test_scan_image_too_large_returns_413(test_user: User):
    """Test that oversized image returns 413"""

    # Create a base64 string larger than 10MB limit
    huge_image = "A" * (11 * 1024 * 1024)

    client = AsyncClient(app=app, base_url="http://test")

    response = await client.post(
        "/api/scans/scan",
        json={"image": huge_image},
        headers={"Authorization": f"Bearer {test_user.id}"},
    )

    assert response.status_code == 413


@pytest.mark.asyncio
async def test_scan_confirm_updates_scan_log(
    test_db: AsyncSession,
    test_user: User,
    valid_base64_image: str,
):
    """Test that confirming a scan updates the scan log with actual part"""

    with patch("app.services.ml_service.infer_part") as mock_infer:
        mock_infer.return_value = {
            "part_num": "3003",
            "confidence": 0.92,
        }

        client = AsyncClient(app=app, base_url="http://test")

        # Initial scan
        response = await client.post(
            "/api/scans/scan",
            json={"image": valid_base64_image},
            headers={"Authorization": f"Bearer {test_user.id}"},
        )

        assert response.status_code == 200
        scan_id = response.json()["scan_id"]

        # Confirm scan
        confirm_response = await client.post(
            f"/api/scans/{scan_id}/confirm",
            json={"confirmed_part_num": "3004"},
            headers={"Authorization": f"Bearer {test_user.id}"},
        )

        assert confirm_response.status_code == 200

        # Verify in database
        scan = await test_db.get(Scan, scan_id)
        assert scan.confirmed_part_num == "3004"
        assert scan.prediction == "3003"


@pytest.mark.asyncio
async def test_scan_without_auth_returns_401():
    """Test that scan without authentication returns 401"""

    client = AsyncClient(app=app, base_url="http://test")

    response = await client.post(
        "/api/scans/scan",
        json={"image": "data:image/jpeg;base64,xyz"},
    )

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_scan_preprocesses_image_before_inference(
    test_user: User,
    valid_base64_image: str,
):
    """Test that image is preprocessed (resized, validated) before ML inference"""

    with patch("app.services.ml_service.infer_part") as mock_infer, \
         patch("app.services.image_service.validate_and_preprocess_image") as mock_preprocess:

        mock_preprocess.return_value = b"preprocessed_image_bytes"
        mock_infer.return_value = {"part_num": "3003", "confidence": 0.95}

        client = AsyncClient(app=app, base_url="http://test")

        response = await client.post(
            "/api/scans/scan",
            json={"image": valid_base64_image},
            headers={"Authorization": f"Bearer {test_user.id}"},
        )

        assert response.status_code == 200

        # Verify preprocessing was called
        mock_preprocess.assert_called_once()
        # Verify inference was called with preprocessed image
        mock_infer.assert_called_once_with(b"preprocessed_image_bytes")


@pytest.mark.asyncio
async def test_scan_data_url_format_handled(test_user: User, data_url_image: str):
    """Test that data URL format images are correctly handled"""

    with patch("app.services.ml_service.infer_part") as mock_infer:
        mock_infer.return_value = {"part_num": "3003", "confidence": 0.92}

        client = AsyncClient(app=app, base_url="http://test")

        response = await client.post(
            "/api/scans/scan",
            json={"image": data_url_image},
            headers={"Authorization": f"Bearer {test_user.id}"},
        )

        assert response.status_code == 200


@pytest.mark.asyncio
async def test_decode_base64_image():
    """Test image decoding with various formats"""

    from PIL import Image
    import io

    # Create test image
    img = Image.new("RGB", (50, 50), color="blue")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)

    # Test raw base64
    raw_base64 = base64.b64encode(buffer.getvalue()).decode()
    decoded = decode_base64_image(raw_base64)
    assert len(decoded) > 0
    assert isinstance(decoded, bytes)

    # Test data URL format
    data_url = f"data:image/jpeg;base64,{raw_base64}"
    decoded_from_url = decode_base64_image(data_url)
    assert decoded == decoded_from_url

    # Test invalid base64
    with pytest.raises(ValueError):
        decode_base64_image("invalid!!!base64")


@pytest.mark.asyncio
async def test_validate_and_preprocess_image():
    """Test image validation and preprocessing"""

    from PIL import Image
    import io

    # Create test image
    img = Image.new("RGB", (1000, 1000), color="green")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)
    image_bytes = buffer.getvalue()

    # Should resize and return bytes
    result = validate_and_preprocess_image(image_bytes)
    assert isinstance(result, bytes)
    assert len(result) > 0

    # Result should be valid image
    result_img = Image.open(io.BytesIO(result))
    assert result_img.format == "JPEG"
    assert result_img.size == (512, 512)  # Target size

    # Test oversized image
    huge_bytes = b"x" * (11 * 1024 * 1024)
    with pytest.raises(ValueError, match="too large"):
        validate_and_preprocess_image(huge_bytes)

    # Test invalid image data
    with pytest.raises(ValueError, match="Invalid"):
        validate_and_preprocess_image(b"not an image")
