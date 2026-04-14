#!/usr/bin/env python3
"""
Test script for local inventory module integration.

Tests:
1. GET /api/local-inventory/inventory - should return 200 with empty list
2. POST /api/local-inventory/scan-session/start - should create session
3. POST /api/local-inventory/inventory/add - should add part to inventory

Run with: python test_inventory.py
"""

import asyncio
import json
import base64
import sys
from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

# Import the app
from main import app

client = TestClient(app)


def create_test_image_base64() -> str:
    """Create a minimal test image and encode as base64."""
    img = Image.new("RGB", (224, 224), color="red")
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()
    return base64.b64encode(image_bytes).decode("utf-8")


def test_health_check():
    """Test the health check endpoint."""
    print("\n=== Test 1: Health Check ===")
    response = client.get("/health")
    print(f"GET /health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    print("✓ PASS")


def test_get_empty_inventory():
    """Test GET /api/local-inventory/inventory - should return empty list."""
    print("\n=== Test 2: Get Empty Inventory ===")
    response = client.get("/api/local-inventory/inventory")
    print(f"GET /api/local-inventory/inventory")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Response: {data}")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    assert isinstance(data, list), f"Expected list, got {type(data)}"
    print(f"✓ PASS (empty inventory has {len(data)} items)")


def test_start_scan_session():
    """Test POST /api/local-inventory/scan-session/start."""
    print("\n=== Test 3: Start Scan Session ===")
    payload = {"set_name": "Test Session"}
    response = client.post(
        "/api/local-inventory/scan-session/start",
        json=payload,
    )
    print(f"POST /api/local-inventory/scan-session/start")
    print(f"Payload: {payload}")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Response: {json.dumps(data, indent=2, default=str)}")

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    assert "id" in data, "Response missing 'id' field"
    assert data["set_name"] == "Test Session", "Session name mismatch"
    assert data["completed"] is False, "Session should not be completed"

    session_id = data["id"]
    print(f"✓ PASS (created session: {session_id})")
    return session_id


def test_get_inventory_stats():
    """Test GET /api/local-inventory/inventory/stats."""
    print("\n=== Test 4: Get Inventory Stats ===")
    response = client.get("/api/local-inventory/inventory/stats")
    print(f"GET /api/local-inventory/inventory/stats")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Response: {json.dumps(data, indent=2)}")

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    assert "total_parts" in data, "Missing 'total_parts'"
    assert "total_quantity" in data, "Missing 'total_quantity'"
    assert "user_confirmed" in data, "Missing 'user_confirmed'"
    print("✓ PASS")


def test_add_to_inventory():
    """Test POST /api/local-inventory/inventory/add."""
    print("\n=== Test 5: Add Part to Inventory ===")
    payload = {
        "part_num": "3001",
        "color_id": 1,
        "color_name": "White",
        "quantity": 5,
    }
    response = client.post(
        "/api/local-inventory/inventory/add",
        json=payload,
    )
    print(f"POST /api/local-inventory/inventory/add")
    print(f"Payload: {payload}")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Response: {json.dumps(data, indent=2, default=str)}")

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    assert data["part_num"] == "3001", "Part number mismatch"
    assert data["quantity"] == 5, "Quantity mismatch"
    assert data["user_confirmed"] is True, "Should be user-confirmed"

    print(f"✓ PASS (added part: {data['id']})")
    return data["id"]


def test_get_populated_inventory():
    """Test GET /api/local-inventory/inventory - should return 1 item."""
    print("\n=== Test 6: Get Populated Inventory ===")
    response = client.get("/api/local-inventory/inventory")
    print(f"GET /api/local-inventory/inventory")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Response: {json.dumps(data, indent=2, default=str)}")

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    assert len(data) >= 1, "Should have at least 1 inventory item"
    print(f"✓ PASS (inventory has {len(data)} items)")


def test_get_inventory_stats_populated():
    """Test GET /api/local-inventory/inventory/stats with data."""
    print("\n=== Test 7: Get Inventory Stats (With Data) ===")
    response = client.get("/api/local-inventory/inventory/stats")
    print(f"GET /api/local-inventory/inventory/stats")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Response: {json.dumps(data, indent=2)}")

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    assert data["total_parts"] >= 1, "Should have at least 1 part"
    assert data["total_quantity"] >= 5, "Should have at least 5 total quantity"
    assert data["user_confirmed"] >= 1, "Should have at least 1 confirmed part"
    print("✓ PASS")


def test_list_scan_sessions():
    """Test GET /api/local-inventory/scan-session."""
    print("\n=== Test 8: List Scan Sessions ===")
    response = client.get("/api/local-inventory/scan-session")
    print(f"GET /api/local-inventory/scan-session")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Response: {json.dumps(data, indent=2, default=str)}")

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    assert isinstance(data, list), f"Expected list, got {type(data)}"
    assert len(data) >= 1, "Should have at least 1 session"
    print(f"✓ PASS (found {len(data)} sessions)")


def main():
    """Run all tests."""
    print("=" * 60)
    print("LOCAL INVENTORY MODULE INTEGRATION TESTS")
    print("=" * 60)

    try:
        test_health_check()
        test_get_empty_inventory()

        # Create a session first
        session_id = test_start_scan_session()

        # Add an inventory item
        item_id = test_add_to_inventory()

        # Get updated inventory
        test_get_populated_inventory()
        test_get_inventory_stats_populated()
        test_list_scan_sessions()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        print("\nSummary:")
        print("✓ Health check endpoint works")
        print("✓ GET /api/local-inventory/inventory returns 200")
        print("✓ POST /api/local-inventory/scan-session/start creates session")
        print("✓ POST /api/local-inventory/inventory/add adds parts")
        print("✓ GET /api/local-inventory/inventory/stats returns stats")
        print("✓ GET /api/local-inventory/scan-session lists sessions")
        print(f"\nCreated test session: {session_id}")
        print(f"Added test part: {item_id}")
        return 0

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
