"""Tests for inventory management endpoints."""
import pytest
import uuid
from fastapi import status

from app.models.inventory import InventoryItem


class TestAddToInventory:
    """Test adding items to inventory."""

    @pytest.mark.asyncio
    async def test_add_to_inventory(self, client, auth_headers, test_user, sample_parts, sample_colors, db_session):
        """Test successfully adding an item to inventory."""
        response = client.post(
            "/api/v1/inventory/items",
            headers=auth_headers,
            json={
                "part_id": sample_parts[0].id,
                "color_id": sample_colors[0].id,
                "quantity": 10,
            },
        )

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["part_id"] == sample_parts[0].id
        assert data["color_id"] == sample_colors[0].id
        assert data["quantity"] == 10

    @pytest.mark.asyncio
    async def test_add_to_inventory_unauthenticated(self, client, sample_parts, sample_colors):
        """Test adding to inventory without authentication."""
        response = client.post(
            "/api/v1/inventory/items",
            json={
                "part_id": sample_parts[0].id,
                "color_id": sample_colors[0].id,
                "quantity": 10,
            },
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    @pytest.mark.asyncio
    async def test_add_to_inventory_invalid_part(self, client, auth_headers, sample_colors):
        """Test adding to inventory with non-existent part."""
        response = client.post(
            "/api/v1/inventory/items",
            headers=auth_headers,
            json={
                "part_id": str(uuid.uuid4()),
                "color_id": sample_colors[0].id,
                "quantity": 10,
            },
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND

    @pytest.mark.asyncio
    async def test_add_to_inventory_invalid_color(self, client, auth_headers, sample_parts):
        """Test adding to inventory with non-existent color."""
        response = client.post(
            "/api/v1/inventory/items",
            headers=auth_headers,
            json={
                "part_id": sample_parts[0].id,
                "color_id": 99999,
                "quantity": 10,
            },
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestGetInventory:
    """Test retrieving inventory."""

    @pytest.mark.asyncio
    async def test_get_inventory_empty(self, client, auth_headers, test_user):
        """Test getting empty inventory."""
        response = client.get(
            "/api/v1/inventory/items",
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["items"] == []
        assert data["total"] == 0

    @pytest.mark.asyncio
    async def test_get_inventory_with_items(self, client, auth_headers, test_user, sample_parts, sample_colors, db_session):
        """Test getting inventory with items."""
        # Add some items
        item = InventoryItem(
            user_id=test_user.id,
            part_id=sample_parts[0].id,
            color_id=sample_colors[0].id,
            quantity=15,
        )
        db_session.add(item)
        await db_session.commit()

        response = client.get(
            "/api/v1/inventory/items",
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["items"]) == 1
        assert data["items"][0]["quantity"] == 15
        assert data["total"] == 1

    @pytest.mark.asyncio
    async def test_get_inventory_unauthenticated(self, client):
        """Test getting inventory without authentication."""
        response = client.get("/api/v1/inventory/items")

        assert response.status_code == status.HTTP_401_UNAUTHORIZED


class TestUpdateInventory:
    """Test updating inventory items."""

    @pytest.mark.asyncio
    async def test_update_inventory_quantity(self, client, auth_headers, test_user, sample_parts, sample_colors, db_session):
        """Test updating inventory item quantity."""
        # Create an initial item
        item = InventoryItem(
            user_id=test_user.id,
            part_id=sample_parts[0].id,
            color_id=sample_colors[0].id,
            quantity=10,
        )
        db_session.add(item)
        await db_session.commit()
        item_id = item.id

        # Update quantity
        response = client.patch(
            f"/api/v1/inventory/items/{item_id}",
            headers=auth_headers,
            json={"quantity": 25},
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["quantity"] == 25

    @pytest.mark.asyncio
    async def test_update_nonexistent_item(self, client, auth_headers):
        """Test updating non-existent inventory item."""
        response = client.patch(
            f"/api/v1/inventory/items/{uuid.uuid4()}",
            headers=auth_headers,
            json={"quantity": 25},
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND

    @pytest.mark.asyncio
    async def test_update_other_users_item(self, client, auth_headers, db_session, sample_parts, sample_colors):
        """Test updating another user's inventory item."""
        # Create another user
        from app.models.user import User
        from app.core.security import hash_password
        from datetime import datetime, timezone

        other_user = User(
            id=str(uuid.uuid4()),
            email="other@example.com",
            hashed_password=hash_password("password"),
            is_active=True,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        db_session.add(other_user)
        await db_session.flush()

        # Create item for other user
        item = InventoryItem(
            user_id=other_user.id,
            part_id=sample_parts[0].id,
            color_id=sample_colors[0].id,
            quantity=10,
        )
        db_session.add(item)
        await db_session.commit()

        # Try to update as different user
        response = client.patch(
            f"/api/v1/inventory/items/{item.id}",
            headers=auth_headers,
            json={"quantity": 25},
        )

        assert response.status_code == status.HTTP_403_FORBIDDEN


class TestDeleteInventoryItem:
    """Test deleting inventory items."""

    @pytest.mark.asyncio
    async def test_delete_inventory_item(self, client, auth_headers, test_user, sample_parts, sample_colors, db_session):
        """Test successfully deleting an inventory item."""
        # Create an item
        item = InventoryItem(
            user_id=test_user.id,
            part_id=sample_parts[0].id,
            color_id=sample_colors[0].id,
            quantity=10,
        )
        db_session.add(item)
        await db_session.commit()
        item_id = item.id

        # Delete it
        response = client.delete(
            f"/api/v1/inventory/items/{item_id}",
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_204_NO_CONTENT

        # Verify it's deleted
        get_response = client.get(
            "/api/v1/inventory/items",
            headers=auth_headers,
        )
        assert len(get_response.json()["items"]) == 0

    @pytest.mark.asyncio
    async def test_delete_nonexistent_item(self, client, auth_headers):
        """Test deleting non-existent item."""
        response = client.delete(
            f"/api/v1/inventory/items/{uuid.uuid4()}",
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestAddSamePartTwice:
    """Test adding the same part twice (should upsert quantity)."""

    @pytest.mark.asyncio
    async def test_add_same_part_twice_upserts_quantity(self, client, auth_headers, test_user, sample_parts, sample_colors, db_session):
        """Test that adding the same part twice upserts the quantity."""
        # Add first time
        response1 = client.post(
            "/api/v1/inventory/items",
            headers=auth_headers,
            json={
                "part_id": sample_parts[0].id,
                "color_id": sample_colors[0].id,
                "quantity": 10,
            },
        )

        assert response1.status_code == status.HTTP_201_CREATED
        item_id_1 = response1.json()["id"]

        # Add second time with same part
        response2 = client.post(
            "/api/v1/inventory/items",
            headers=auth_headers,
            json={
                "part_id": sample_parts[0].id,
                "color_id": sample_colors[0].id,
                "quantity": 15,
            },
        )

        assert response2.status_code == status.HTTP_200_OK
        data = response2.json()
        assert data["id"] == item_id_1  # Same item ID
        assert data["quantity"] == 25  # Quantities combined (10 + 15)

        # Verify only one item in inventory
        get_response = client.get(
            "/api/v1/inventory/items",
            headers=auth_headers,
        )
        assert len(get_response.json()["items"]) == 1
        assert get_response.json()["items"][0]["quantity"] == 25
