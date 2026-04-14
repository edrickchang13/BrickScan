"""
Example API usage and test cases for BrickScan backend.
These demonstrate how to use the API endpoints.
"""

import httpx
import json
import base64
from pathlib import Path

BASE_URL = "http://localhost:8000"


async def test_auth_flow():
    """Test user registration and login flow."""
    async with httpx.AsyncClient() as client:
        # Register a new user
        register_response = await client.post(
            f"{BASE_URL}/auth/register",
            json={
                "email": "user@example.com",
                "password": "securepassword123"
            }
        )
        assert register_response.status_code == 200
        token = register_response.json()["access_token"]
        print(f"Registered user, token: {token[:20]}...")

        # Login with same credentials
        login_response = await client.post(
            f"{BASE_URL}/auth/login",
            json={
                "email": "user@example.com",
                "password": "securepassword123"
            }
        )
        assert login_response.status_code == 200
        login_token = login_response.json()["access_token"]
        print(f"Logged in, token: {login_token[:20]}...")

        # Get current user
        me_response = await client.get(
            f"{BASE_URL}/auth/me",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert me_response.status_code == 200
        user = me_response.json()
        print(f"Current user: {user['email']}")


async def test_parts_api():
    """Test LEGO parts search and retrieval."""
    async with httpx.AsyncClient() as client:
        # Search for parts
        search_response = await client.get(
            f"{BASE_URL}/parts?search=brick&limit=10&offset=0"
        )
        assert search_response.status_code == 200
        parts = search_response.json()
        print(f"Found {len(parts)} parts matching 'brick'")

        if parts:
            # Get details of first part
            part_num = parts[0]["part_num"]
            detail_response = await client.get(
                f"{BASE_URL}/parts/{part_num}"
            )
            assert detail_response.status_code == 200
            part_detail = detail_response.json()
            print(f"Part {part_num}: {part_detail['name']}")


async def test_sets_api():
    """Test LEGO sets search and retrieval."""
    async with httpx.AsyncClient() as client:
        # Search for sets
        search_response = await client.get(
            f"{BASE_URL}/sets?search=classic&limit=5"
        )
        assert search_response.status_code == 200
        sets = search_response.json()
        print(f"Found {len(sets)} sets matching 'classic'")

        if sets:
            # Get full set details
            set_num = sets[0]["set_num"]
            detail_response = await client.get(
                f"{BASE_URL}/sets/{set_num}"
            )
            assert detail_response.status_code == 200
            set_detail = detail_response.json()
            print(f"Set {set_num}: {set_detail['name']} ({set_detail['num_parts']} parts)")

            # Get parts list
            parts_response = await client.get(
                f"{BASE_URL}/sets/{set_num}/parts"
            )
            assert parts_response.status_code == 200
            parts = parts_response.json()
            print(f"Set contains {len(parts)} part entries")


async def test_inventory_management(token: str):
    """Test inventory CRUD operations."""
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {token}"}

        # Add item to inventory
        add_response = await client.post(
            f"{BASE_URL}/inventory",
            json={
                "part_num": "3001",  # Standard 2x4 brick
                "color_id": "12c3d4e5-f6a7-b8c9-d0e1-f2a3b4c5d6e7",
                "quantity": 5
            },
            headers=headers
        )
        assert add_response.status_code == 200
        item = add_response.json()
        item_id = item["id"]
        print(f"Added {item['quantity']} x {item['part_name']} ({item['color_name']})")

        # Get inventory
        get_response = await client.get(
            f"{BASE_URL}/inventory",
            headers=headers
        )
        assert get_response.status_code == 200
        inventory = get_response.json()
        print(f"Inventory has {len(inventory)} items")

        # Update quantity
        update_response = await client.put(
            f"{BASE_URL}/inventory/{item_id}",
            json={"quantity": 10},
            headers=headers
        )
        assert update_response.status_code == 200
        print(f"Updated quantity to 10")

        # Export inventory as CSV
        export_response = await client.get(
            f"{BASE_URL}/inventory/export",
            headers=headers
        )
        assert export_response.status_code == 200
        csv_data = export_response.text
        print(f"Exported CSV:\n{csv_data[:100]}...")

        # Delete item
        delete_response = await client.delete(
            f"{BASE_URL}/inventory/{item_id}",
            headers=headers
        )
        assert delete_response.status_code == 200
        print("Deleted item from inventory")


async def test_scan_api(token: str):
    """Test piece scanning with image."""
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {token}"}

        # Create a dummy test image (1x1 red pixel JPEG)
        # In real usage, this would be a photo of a LEGO piece
        test_image_data = base64.b64encode(b"fake_jpeg_data").decode()

        # Scan image
        scan_response = await client.post(
            f"{BASE_URL}/scan",
            json={"image_base64": test_image_data},
            headers=headers
        )
        assert scan_response.status_code == 200
        scan_result = scan_response.json()
        predictions = scan_result["predictions"]
        print(f"Got {len(predictions)} predictions from scan")

        for i, pred in enumerate(predictions, 1):
            print(f"  {i}. {pred['part_name']} ({pred['part_num']}) - "
                  f"{pred['color_name']} - {pred['confidence']:.2%}")

        # Confirm scan (if we have a scan_log_id)
        # This would typically come from the scan response with additional metadata
        if predictions:
            confirm_response = await client.post(
                f"{BASE_URL}/scan/confirm",
                json={
                    "scan_log_id": "scan_log_uuid_here",
                    "confirmed_part_num": predictions[0]["part_num"]
                },
                headers=headers
            )
            print(f"Confirmed scan result")


async def test_build_check(token: str, set_num: str = "10128-1"):
    """Test building a set and checking progress."""
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {token}"}

        # Generate wanted list for BrickLink
        wanted_response = await client.post(
            f"{BASE_URL}/bricklink/wanted-list/{set_num}?condition=N",
            headers=headers
        )
        assert wanted_response.status_code == 200
        xml = wanted_response.text
        print(f"Generated BrickLink wanted list:\n{xml[:200]}...")

        # Get color mapping
        colors_response = await client.get(
            f"{BASE_URL}/bricklink/colors"
        )
        assert colors_response.status_code == 200
        color_map = colors_response.json()
        print(f"Color mapping available with {len(color_map)} colors")


async def test_health_check():
    """Test health check endpoint."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        print("API is healthy")


async def run_all_tests():
    """Run all test examples."""
    print("=" * 60)
    print("BrickScan API Test Examples")
    print("=" * 60)

    try:
        # Health check
        print("\n1. Health Check")
        print("-" * 40)
        await test_health_check()

        # Auth flow
        print("\n2. Authentication Flow")
        print("-" * 40)
        await test_auth_flow()

        # Parts API
        print("\n3. LEGO Parts API")
        print("-" * 40)
        await test_parts_api()

        # Sets API
        print("\n4. LEGO Sets API")
        print("-" * 40)
        await test_sets_api()

        # Note: For authenticated endpoints, you'll need to use a real token
        print("\n5. Inventory Management (requires auth token)")
        print("-" * 40)
        print("Skipped - requires valid auth token")

        print("\n6. Piece Scanning (requires auth token)")
        print("-" * 40)
        print("Skipped - requires valid auth token")

        print("\n7. Build Check & BrickLink (requires auth token)")
        print("-" * 40)
        print("Skipped - requires valid auth token")

        print("\n" + "=" * 60)
        print("Tests completed!")
        print("=" * 60)

    except httpx.ConnectError:
        print("ERROR: Could not connect to API at {BASE_URL}")
        print("Make sure the API is running: uvicorn main:app --reload")
    except Exception as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(run_all_tests())
