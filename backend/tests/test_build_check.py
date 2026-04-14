"""Tests for build check service."""
import pytest
import uuid
from xml.etree import ElementTree as ET

from app.models.inventory import InventoryItem
from app.services.build_check import (
    check_build_completeness,
    generate_bricklink_xml,
    calculate_missing_parts,
)


class TestBuildCheckCompleteness:
    """Test build check service."""

    @pytest.mark.asyncio
    async def test_build_check_complete_set(self, db_session, test_user, sample_set, sample_parts, sample_colors):
        """Test that user has all parts for a set."""
        # Add all parts from the set to user's inventory
        for set_part in sample_set.set_parts:
            item = InventoryItem(
                user_id=test_user.id,
                part_id=set_part.part_id,
                color_id=set_part.color_id,
                quantity=set_part.quantity,
            )
            db_session.add(item)

        await db_session.commit()

        # Check build completeness
        result = await check_build_completeness(db_session, test_user.id, sample_set.id)

        assert result["is_complete"] is True
        assert result["total_parts_needed"] == 4
        assert result["total_parts_have"] == 4
        assert result["missing_parts"] == []
        assert result["completion_percentage"] == 100.0

    @pytest.mark.asyncio
    async def test_build_check_missing_parts(self, db_session, test_user, sample_set, sample_parts, sample_colors):
        """Test build check with missing parts."""
        # Only add first 2 parts
        set_parts = list(sample_set.set_parts)[:2]
        for set_part in set_parts:
            item = InventoryItem(
                user_id=test_user.id,
                part_id=set_part.part_id,
                color_id=set_part.color_id,
                quantity=set_part.quantity,
            )
            db_session.add(item)

        await db_session.commit()

        # Check build completeness
        result = await check_build_completeness(db_session, test_user.id, sample_set.id)

        assert result["is_complete"] is False
        assert result["total_parts_needed"] == 4
        assert result["total_parts_have"] == 2
        assert len(result["missing_parts"]) == 2
        assert result["completion_percentage"] == 50.0

    @pytest.mark.asyncio
    async def test_build_check_partial_quantities(self, db_session, test_user, sample_set, sample_parts, sample_colors):
        """Test build check with partial quantities."""
        # Add parts but with insufficient quantities
        for set_part in sample_set.set_parts:
            item = InventoryItem(
                user_id=test_user.id,
                part_id=set_part.part_id,
                color_id=set_part.color_id,
                quantity=set_part.quantity // 2,  # Only half the required quantity
            )
            db_session.add(item)

        await db_session.commit()

        # Check build completeness
        result = await check_build_completeness(db_session, test_user.id, sample_set.id)

        assert result["is_complete"] is False
        assert result["total_parts_needed"] == 4
        assert len(result["missing_parts"]) == 4  # All parts are short on quantity

    @pytest.mark.asyncio
    async def test_build_check_empty_inventory(self, db_session, test_user, sample_set):
        """Test build check with empty inventory."""
        # User has no inventory items
        result = await check_build_completeness(db_session, test_user.id, sample_set.id)

        assert result["is_complete"] is False
        assert result["total_parts_needed"] == 4
        assert result["total_parts_have"] == 0
        assert len(result["missing_parts"]) == 4
        assert result["completion_percentage"] == 0.0

    @pytest.mark.asyncio
    async def test_build_check_nonexistent_set(self, db_session, test_user):
        """Test build check with non-existent set."""
        result = await check_build_completeness(
            db_session,
            test_user.id,
            str(uuid.uuid4())
        )

        assert result is None


class TestCalculateMissingParts:
    """Test missing parts calculation."""

    @pytest.mark.asyncio
    async def test_calculate_missing_parts_no_inventory(self, db_session, test_user, sample_set):
        """Test missing parts calculation with no inventory."""
        missing = await calculate_missing_parts(db_session, test_user.id, sample_set.id)

        assert len(missing) == 4
        for item in missing:
            assert item["quantity_missing"] > 0

    @pytest.mark.asyncio
    async def test_calculate_missing_parts_with_inventory(self, db_session, test_user, sample_set, sample_parts, sample_colors):
        """Test missing parts calculation with partial inventory."""
        # Add only first part
        set_parts = list(sample_set.set_parts)
        first_part = set_parts[0]
        item = InventoryItem(
            user_id=test_user.id,
            part_id=first_part.part_id,
            color_id=first_part.color_id,
            quantity=first_part.quantity - 5,  # 5 short
        )
        db_session.add(item)
        await db_session.commit()

        missing = await calculate_missing_parts(db_session, test_user.id, sample_set.id)

        # 3 parts completely missing, 1 part 5 short
        assert len(missing) == 4
        missing_dict = {m["part_num"]: m["quantity_missing"] for m in missing}
        assert missing_dict[first_part.part.part_num] == 5


class TestBrickLinkXMLGeneration:
    """Test BrickLink XML generation."""

    @pytest.mark.asyncio
    async def test_bricklink_xml_generation(self, db_session, test_user, sample_set, sample_parts, sample_colors):
        """Test BrickLink XML generation for missing parts."""
        # Add only first 2 parts
        set_parts = list(sample_set.set_parts)[:2]
        for set_part in set_parts:
            item = InventoryItem(
                user_id=test_user.id,
                part_id=set_part.part_id,
                color_id=set_part.color_id,
                quantity=set_part.quantity - 3,  # 3 short on each
            )
            db_session.add(item)

        await db_session.commit()

        # Generate XML
        xml_str = await generate_bricklink_xml(db_session, test_user.id, sample_set.id)

        # Parse and validate XML
        root = ET.fromstring(xml_str)
        assert root.tag == "INVENTORY"

        # Should have items for the missing parts
        items = root.findall("ITEM")
        assert len(items) >= 2

        # Validate item structure
        for item in items:
            assert item.find("ITEMTYPE") is not None
            assert item.find("ITEMID") is not None
            assert item.find("COLOR") is not None
            assert item.find("QTY") is not None

    @pytest.mark.asyncio
    async def test_bricklink_xml_with_no_missing_parts(self, db_session, test_user, sample_set, sample_parts, sample_colors):
        """Test BrickLink XML generation when all parts are available."""
        # Add all parts
        for set_part in sample_set.set_parts:
            item = InventoryItem(
                user_id=test_user.id,
                part_id=set_part.part_id,
                color_id=set_part.color_id,
                quantity=set_part.quantity,
            )
            db_session.add(item)

        await db_session.commit()

        # Generate XML
        xml_str = await generate_bricklink_xml(db_session, test_user.id, sample_set.id)

        # Parse and validate XML
        root = ET.fromstring(xml_str)
        assert root.tag == "INVENTORY"

        # Should have no items since all parts are available
        items = root.findall("ITEM")
        assert len(items) == 0
