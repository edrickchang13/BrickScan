"""
Integration tests covering the complete BrickScan user workflow:
1. Register -> Login
2. Scan a piece -> Add to inventory
3. Search for a LEGO set
4. Check build status
5. Generate BrickLink list

These tests use real test data seeded by conftest.py
"""
import pytest
import pytest_asyncio
from httpx import AsyncClient
from fastapi import status
import json
from datetime import datetime, timezone


@pytest.mark.asyncio
async def test_complete_user_journey(client, async_client, sample_colors, sample_parts, sample_set):
    """Full workflow: register, scan, build check, bricklink list."""

    # 1. Register new user
    register_resp = client.post(
        "/api/v1/auth/register",
        json={
            "email": "journeyuser@example.com",
            "password": "JourneyPassword123!",
            "full_name": "Journey User"
        }
    )
    assert register_resp.status_code == status.HTTP_201_CREATED
    user_data = register_resp.json()
    token = user_data.get("access_token")
    assert token is not None
    headers = {"Authorization": f"Bearer {token}"}

    # 2. Add some parts to inventory
    parts_to_add = [
        {
            "part_num": "3001",
            "color_id": str(sample_colors[0].id),
            "quantity": 10
        },
        {
            "part_num": "3002",
            "color_id": str(sample_colors[1].id),
            "quantity": 20
        },
        {
            "part_num": "3003",
            "color_id": str(sample_colors[2].id),
            "quantity": 15
        },
    ]

    for part in parts_to_add:
        resp = client.post(
            "/api/v1/inventory",
            json=part,
            headers=headers
        )
        assert resp.status_code == status.HTTP_201_CREATED

    # 3. Verify inventory
    inv_resp = client.get(
        "/api/v1/inventory",
        headers=headers
    )
    assert inv_resp.status_code == status.HTTP_200_OK
    inv_data = inv_resp.json()
    assert len(inv_data.get("items", [])) == 3

    # 4. Search for a LEGO set
    search_resp = client.get(
        "/api/v1/sets",
        params={"search": sample_set.name}
    )
    assert search_resp.status_code == status.HTTP_200_OK
    search_data = search_resp.json()
    assert "results" in search_data or "items" in search_data

    # 5. Get set details
    set_resp = client.get(
        f"/api/v1/sets/{sample_set.set_num}"
    )
    assert set_resp.status_code == status.HTTP_200_OK
    set_detail = set_resp.json()
    assert set_detail.get("set_num") == sample_set.set_num

    # 6. Run build check
    check_resp = client.post(
        f"/api/v1/sets/{sample_set.set_num}/compare",
        headers=headers
    )
    assert check_resp.status_code == status.HTTP_200_OK
    result = check_resp.json()
    assert "percent_complete" in result
    assert isinstance(result["percent_complete"], (int, float))
    assert result["percent_complete"] >= 0
    assert result["percent_complete"] <= 100
    assert "missing_parts" in result

    # 7. Generate BrickLink list
    bl_resp = client.post(
        f"/api/v1/sets/{sample_set.set_num}/bricklink",
        params={"condition": "X"},
        headers=headers
    )
    assert bl_resp.status_code == status.HTTP_200_OK
    bl_data = bl_resp.json()
    assert "xml" in bl_data or "content" in bl_data
    xml_content = bl_data.get("xml") or bl_data.get("content")
    assert xml_content is not None
    assert "INVENTORY" in xml_content or "Item" in xml_content


@pytest.mark.asyncio
async def test_inventory_upsert_behavior(client, sample_colors, sample_parts, test_user, auth_headers):
    """Test that adding the same part twice increases quantity."""

    part = {
        "part_num": "3001",
        "color_id": str(sample_colors[0].id),
        "quantity": 5
    }

    # Add part first time
    resp1 = client.post(
        "/api/v1/inventory",
        json=part,
        headers=auth_headers
    )
    assert resp1.status_code == status.HTTP_201_CREATED

    # Add same part again
    resp2 = client.post(
        "/api/v1/inventory",
        json=part,
        headers=auth_headers
    )
    assert resp2.status_code in [status.HTTP_200_OK, status.HTTP_201_CREATED]

    # Verify quantity was summed
    inv_resp = client.get(
        "/api/v1/inventory",
        headers=auth_headers
    )
    assert inv_resp.status_code == status.HTTP_200_OK
    items = inv_resp.json().get("items", [])
    matching = [i for i in items if i.get("part", {}).get("part_num") == "3001"]
    assert len(matching) >= 1
    assert matching[0].get("quantity", 0) >= 10


@pytest.mark.asyncio
async def test_stats_endpoint(client, sample_colors, sample_parts, sample_set, test_user, auth_headers):
    """Test user stats calculation after adding inventory."""

    # Add some inventory items first
    part = {
        "part_num": "3001",
        "color_id": str(sample_colors[0].id),
        "quantity": 25
    }

    resp = client.post(
        "/api/v1/inventory",
        json=part,
        headers=auth_headers
    )
    assert resp.status_code == status.HTTP_201_CREATED

    # Get stats
    stats_resp = client.get(
        "/api/v1/stats/me",
        headers=auth_headers
    )
    assert stats_resp.status_code == status.HTTP_200_OK
    stats = stats_resp.json()

    assert "total_parts" in stats
    assert "total_pieces" in stats
    assert stats["total_parts"] > 0
    assert stats["total_pieces"] >= 25


@pytest.mark.asyncio
async def test_scan_and_add_workflow(client, sample_colors, sample_parts, test_user, auth_headers):
    """Test scanning a piece and adding it to inventory."""

    # Simulate scan prediction
    scan_resp = client.post(
        "/api/v1/scan/predict",
        json={
            "image_data": "data:image/jpeg;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        },
        headers=auth_headers
    )
    assert scan_resp.status_code == status.HTTP_200_OK
    prediction = scan_resp.json()
    assert "part_num" in prediction or "predictions" in prediction

    # Add the predicted part to inventory
    if "part_num" in prediction:
        part_num = prediction["part_num"]
    else:
        part_num = "3001"  # Fallback

    add_resp = client.post(
        "/api/v1/inventory",
        json={
            "part_num": part_num,
            "color_id": str(sample_colors[0].id),
            "quantity": 1
        },
        headers=auth_headers
    )
    assert add_resp.status_code == status.HTTP_201_CREATED


@pytest.mark.asyncio
async def test_build_progress_multiple_sets(client, sample_colors, sample_parts, sample_set, test_user, auth_headers):
    """Test checking build progress for multiple sets."""

    # Add inventory
    part = {
        "part_num": "3001",
        "color_id": str(sample_colors[0].id),
        "quantity": 50
    }

    resp = client.post(
        "/api/v1/inventory",
        json=part,
        headers=auth_headers
    )
    assert resp.status_code == status.HTTP_201_CREATED

    # Check build progress
    check_resp = client.post(
        f"/api/v1/sets/{sample_set.set_num}/compare",
        headers=auth_headers
    )
    assert check_resp.status_code == status.HTTP_200_OK

    result = check_resp.json()
    percent_complete = result.get("percent_complete", 0)
    assert percent_complete >= 0
    assert percent_complete <= 100


@pytest.mark.asyncio
async def test_filter_and_search_sets(client, sample_set):
    """Test filtering and searching LEGO sets."""

    # Search by name
    search_resp = client.get(
        "/api/v1/sets",
        params={"search": sample_set.name[:5]}
    )
    assert search_resp.status_code == status.HTTP_200_OK

    data = search_resp.json()
    results = data.get("results") or data.get("items") or []
    assert len(results) > 0

    # Filter by year
    filter_resp = client.get(
        "/api/v1/sets",
        params={"year": sample_set.year}
    )
    assert filter_resp.status_code == status.HTTP_200_OK


@pytest.mark.asyncio
async def test_export_inventory(client, sample_colors, sample_parts, test_user, auth_headers):
    """Test exporting inventory as CSV."""

    # Add some inventory
    for i, part in enumerate(sample_parts[:3]):
        client.post(
            "/api/v1/inventory",
            json={
                "part_num": part.part_num,
                "color_id": str(sample_colors[i].id),
                "quantity": (i + 1) * 10
            },
            headers=auth_headers
        )

    # Export as CSV
    export_resp = client.get(
        "/api/v1/inventory/export?format=csv",
        headers=auth_headers
    )
    assert export_resp.status_code == status.HTTP_200_OK
    assert "text/csv" in export_resp.headers.get("content-type", "")


@pytest.mark.asyncio
async def test_get_set_parts(client, sample_set):
    """Test retrieving parts for a specific set."""

    parts_resp = client.get(
        f"/api/v1/sets/{sample_set.set_num}/parts"
    )
    assert parts_resp.status_code == status.HTTP_200_OK

    data = parts_resp.json()
    parts_list = data.get("parts") or data.get("items") or []
    assert len(parts_list) > 0

    # Verify part structure
    if parts_list:
        part = parts_list[0]
        assert "part_num" in part or "id" in part
        assert "quantity" in part


@pytest.mark.asyncio
async def test_user_wishlist(client, sample_set, test_user, auth_headers):
    """Test adding and removing sets from wishlist."""

    # Add to wishlist
    add_resp = client.post(
        f"/api/v1/wishlist/{sample_set.set_num}",
        headers=auth_headers
    )
    assert add_resp.status_code in [status.HTTP_201_CREATED, status.HTTP_200_OK]

    # Get wishlist
    get_resp = client.get(
        "/api/v1/wishlist",
        headers=auth_headers
    )
    assert get_resp.status_code == status.HTTP_200_OK

    wishlist = get_resp.json()
    items = wishlist.get("items") or wishlist.get("sets") or []
    assert len(items) > 0

    # Remove from wishlist
    remove_resp = client.delete(
        f"/api/v1/wishlist/{sample_set.set_num}",
        headers=auth_headers
    )
    assert remove_resp.status_code == status.HTTP_204_NO_CONTENT


@pytest.mark.asyncio
async def test_color_management(client, sample_colors):
    """Test color retrieval and filtering."""

    # Get all colors
    colors_resp = client.get("/api/v1/colors")
    assert colors_resp.status_code == status.HTTP_200_OK

    colors_data = colors_resp.json()
    colors_list = colors_data.get("colors") or colors_data.get("items") or []
    assert len(colors_list) > 0

    # Verify color structure
    if colors_list:
        color = colors_list[0]
        assert "name" in color or "id" in color
        assert "hex_code" in color or "hex" in color

    # Get specific color
    if sample_colors:
        color_id = sample_colors[0].id
        color_resp = client.get(f"/api/v1/colors/{color_id}")
        assert color_resp.status_code == status.HTTP_200_OK


@pytest.mark.asyncio
async def test_concurrent_inventory_updates(client, sample_colors, sample_parts, test_user, auth_headers):
    """Test that concurrent inventory updates are handled correctly."""

    import asyncio

    part = {
        "part_num": "3001",
        "color_id": str(sample_colors[0].id),
        "quantity": 1
    }

    # Simulate 5 concurrent additions of the same part
    tasks = [
        asyncio.create_task(
            asyncio.to_thread(
                client.post,
                "/api/v1/inventory",
                json=part,
                headers=auth_headers
            )
        )
        for _ in range(5)
    ]

    responses = await asyncio.gather(*tasks)

    # All should succeed
    assert all(r.status_code in [status.HTTP_201_CREATED, status.HTTP_200_OK] for r in responses)

    # Verify final quantity
    inv_resp = client.get(
        "/api/v1/inventory",
        headers=auth_headers
    )
    items = inv_resp.json().get("items", [])
    matching = [i for i in items if i.get("part", {}).get("part_num") == "3001"]
    assert len(matching) >= 1
    # Quantity should be at least 5 (from concurrent adds)
    assert matching[0].get("quantity", 0) >= 5
