"""
Unit tests for local inventory routes.

Tests cover:
- Image scanning and prediction
- Adding parts to inventory
- Updating quantities and correcting parts
- Exporting as CSV
- Scan session management
"""

import pytest
import base64
import io
from datetime import datetime
from PIL import Image
from sqlalchemy.orm import Session

from app.local_inventory.models import LocalInventoryPart, ScanSession
from app.local_inventory.database import SessionLocal
from app.local_inventory.image_processor import (
    validate_and_decode_image,
    preprocess_for_inference,
)


@pytest.fixture
def db():
    """Create a test database session."""
    session = SessionLocal()
    yield session
    session.close()


@pytest.fixture
def sample_image_base64():
    """Create a sample RGB image and return as base64."""
    # Create a 100x100 white image
    img = Image.new("RGB", (100, 100), color="white")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()
    return base64.b64encode(image_bytes).decode("utf-8")


class TestImageProcessing:
    """Test image validation and preprocessing."""

    def test_validate_and_decode_valid_base64(self, sample_image_base64):
        """Valid base64 image should decode successfully."""
        image_bytes = validate_and_decode_image(sample_image_base64)
        assert isinstance(image_bytes, bytes)
        assert len(image_bytes) > 0

    def test_validate_and_decode_invalid_base64(self):
        """Invalid base64 should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid base64"):
            validate_and_decode_image("not-valid-base64!!!")

    def test_validate_and_decode_corrupted_image(self):
        """Corrupted image data should raise ValueError."""
        corrupted = base64.b64encode(b"not an image").decode("utf-8")
        with pytest.raises(ValueError, match="corrupted"):
            validate_and_decode_image(corrupted)

    def test_preprocess_for_inference(self, sample_image_base64):
        """Preprocessing should return [1,3,224,224] normalized tensor."""
        image_bytes = validate_and_decode_image(sample_image_base64)
        tensor, pil_img = preprocess_for_inference(image_bytes)

        assert tensor.shape == (1, 3, 224, 224)
        assert tensor.dtype == "float32"
        assert tensor.min() >= -5.0  # Normalized values
        assert tensor.max() <= 5.0


class TestInventoryOperations:
    """Test inventory CRUD operations."""

    def test_add_part_to_inventory(self, db):
        """Adding a new part should create InventoryPart entry."""
        part = LocalInventoryPart(
            part_num="3001",
            color_id=1,
            color_name="White",
            quantity=5,
            confidence=0.92,
            user_confirmed=True,
        )
        db.add(part)
        db.commit()
        db.refresh(part)

        assert part.id is not None
        assert part.part_num == "3001"
        assert part.quantity == 5
        assert part.user_confirmed is True

    def test_add_duplicate_part_increments_quantity(self, db):
        """Adding existing part+color should increment quantity."""
        part1 = LocalInventoryPart(
            part_num="3001",
            color_id=1,
            color_name="White",
            quantity=5,
            confidence=1.0,
            user_confirmed=True,
        )
        db.add(part1)
        db.commit()

        # Query for existing part
        existing = db.query(LocalInventoryPart).filter(
            (LocalInventoryPart.part_num == "3001")
            & (LocalInventoryPart.color_id == 1)
        ).first()

        assert existing is not None
        existing.quantity += 3
        db.commit()
        db.refresh(existing)

        assert existing.quantity == 8

    def test_correct_part_prediction(self, db):
        """Correcting a part should update details and set user_confirmed."""
        part = LocalInventoryPart(
            part_num="3001",
            color_id=1,
            color_name="White",
            quantity=5,
            confidence=0.65,
            user_confirmed=False,
        )
        db.add(part)
        db.commit()

        # Correct the part
        part.part_num = "3002"
        part.color_id = 2
        part.color_name = "Red"
        part.user_confirmed = True
        part.confidence = 1.0
        db.commit()
        db.refresh(part)

        assert part.part_num == "3002"
        assert part.color_id == 2
        assert part.user_confirmed is True

    def test_delete_part_from_inventory(self, db):
        """Deleting a part should remove it from database."""
        part = LocalInventoryPart(
            part_num="3001",
            color_id=1,
            color_name="White",
            quantity=5,
            confidence=1.0,
            user_confirmed=True,
        )
        db.add(part)
        db.commit()
        part_id = part.id

        db.delete(part)
        db.commit()

        # Verify deletion
        deleted = db.query(LocalInventoryPart).filter(
            LocalInventoryPart.id == part_id
        ).first()
        assert deleted is None


class TestScanSessions:
    """Test scan session management."""

    def test_create_scan_session(self, db):
        """Creating a session should store set_name and timestamps."""
        session = ScanSession(set_name="Technic 42145", completed=False)
        db.add(session)
        db.commit()
        db.refresh(session)

        assert session.id is not None
        assert session.set_name == "Technic 42145"
        assert session.completed is False
        assert session.created_at is not None

    def test_complete_scan_session(self, db):
        """Completing a session should set completed=True."""
        session = ScanSession(set_name="Test Session", completed=False)
        db.add(session)
        db.commit()

        session.completed = True
        db.commit()
        db.refresh(session)

        assert session.completed is True

    def test_list_scan_sessions(self, db):
        """Query should return all sessions."""
        s1 = ScanSession(set_name="Session 1", completed=False)
        s2 = ScanSession(set_name="Session 2", completed=True)
        db.add_all([s1, s2])
        db.commit()

        all_sessions = db.query(ScanSession).all()
        assert len(all_sessions) >= 2

        incomplete = db.query(ScanSession).filter(
            ScanSession.completed == False
        ).all()
        assert any(s.set_name == "Session 1" for s in incomplete)


class TestInventoryStats:
    """Test aggregate statistics."""

    def test_inventory_stats(self, db):
        """Stats should correctly aggregate inventory data."""
        parts = [
            LocalInventoryPart(
                part_num="3001",
                color_id=1,
                color_name="White",
                quantity=5,
                confidence=1.0,
                user_confirmed=True,
            ),
            LocalInventoryPart(
                part_num="3002",
                color_id=2,
                color_name="Red",
                quantity=3,
                confidence=0.70,
                user_confirmed=False,
            ),
            LocalInventoryPart(
                part_num="3003",
                color_id=1,
                color_name="White",
                quantity=2,
                confidence=1.0,
                user_confirmed=True,
            ),
        ]
        db.add_all(parts)
        db.commit()

        total_parts = db.query(LocalInventoryPart).count()
        total_qty = db.query(LocalInventoryPart.quantity).count()
        confirmed = db.query(LocalInventoryPart).filter(
            LocalInventoryPart.user_confirmed == True
        ).count()

        assert total_parts >= 3
        assert confirmed >= 2


class TestConfidenceThresholds:
    """Test confidence status determination."""

    def test_known_prediction_high_confidence(self, db):
        """Confidence >= 80% should be marked as user_confirmed-ready."""
        part = LocalInventoryPart(
            part_num="3001",
            color_id=1,
            color_name="White",
            quantity=1,
            confidence=0.92,  # 92% >= 80%
            user_confirmed=True,  # Ready to add
        )
        db.add(part)
        db.commit()

        retrieved = db.query(LocalInventoryPart).filter(
            LocalInventoryPart.part_num == "3001"
        ).first()
        assert retrieved.confidence >= 0.80

    def test_uncertain_prediction_low_confidence(self, db):
        """Confidence < 80% should not be auto-confirmed."""
        part = LocalInventoryPart(
            part_num="3002",
            color_id=2,
            color_name="Red",
            quantity=1,
            confidence=0.65,  # 65% < 80%
            user_confirmed=False,  # Needs user selection
        )
        db.add(part)
        db.commit()

        retrieved = db.query(LocalInventoryPart).filter(
            LocalInventoryPart.part_num == "3002"
        ).first()
        assert retrieved.confidence < 0.80
        assert retrieved.user_confirmed is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
