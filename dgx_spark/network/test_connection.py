#!/usr/bin/env python3
"""
Test Connection to DGX Spark Vision Server

Verifies the DGX Spark is reachable and benchmarks inference speed.

Usage:
    python3 test_connection.py --url http://192.168.x.x:8001
    python3 test_connection.py --url http://dgx.local:8001
    python3 test_connection.py --url http://dgx.local:8001 --image lego_brick.jpg --runs 5
"""

import asyncio
import httpx
import argparse
import time
import base64
import json
from pathlib import Path
from typing import Optional


async def test_health(url: str) -> dict:
    """Test if server is reachable and healthy"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{url}/health")
            response.raise_for_status()

        data = response.json()
        return data

    except httpx.ConnectError as e:
        return {
            "status": "unreachable",
            "error": f"Cannot connect: {e}"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


async def test_inference(url: str, image_path: str) -> Optional[dict]:
    """Test a single inference call"""
    try:
        # Read image
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        # Encode as base64
        image_b64 = base64.b64encode(image_bytes).decode()

        # Make request
        async with httpx.AsyncClient(timeout=120.0) as client:
            start = time.time()

            response = await client.post(
                f"{url}/identify",
                json={
                    "image_base64": image_b64,
                    "top_k": 3
                }
            )

            elapsed = time.time() - start
            response.raise_for_status()

        data = response.json()
        data["elapsed_seconds"] = elapsed

        return data

    except Exception as e:
        return {
            "error": str(e)
        }


async def benchmark_inference(
    url: str,
    image_path: str,
    num_runs: int = 5
) -> None:
    """Run inference benchmark and report results"""
    print(f"\nInference Benchmark ({num_runs} runs)")
    print("-" * 50)

    times = []
    errors = 0

    for i in range(num_runs):
        print(f"Run {i+1}/{num_runs}...", end="", flush=True)

        result = await test_inference(url, image_path)

        if result and "error" not in result:
            elapsed = result["elapsed_seconds"]
            times.append(elapsed)

            predictions = result.get("predictions", [])
            top_pred = predictions[0] if predictions else {}

            print(
                f" OK ({elapsed:.2f}s) "
                f"part={top_pred.get('part_num')} "
                f"confidence={top_pred.get('confidence'):.2f}"
            )
        else:
            error_msg = result.get("error", "Unknown error") if result else "No response"
            print(f" FAILED: {error_msg}")
            errors += 1

    if times:
        print()
        print("Results:")
        print(f"  Average time: {sum(times) / len(times):.2f}s")
        print(f"  Min time: {min(times):.2f}s")
        print(f"  Max time: {max(times):.2f}s")
        print(f"  Success rate: {len(times)}/{num_runs}")
    else:
        print(f"All {errors} inferences failed")


async def list_models(url: str) -> None:
    """List available models"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{url}/models")
            response.raise_for_status()

        data = response.json()
        models = data.get("models", [])

        if models:
            print("\nAvailable Models:")
            print("-" * 50)
            for model in models:
                size_gb = model.get("size_gb", 0)
                print(f"  {model['name']} ({size_gb:.1f} GB)")
        else:
            print("No models available")

    except Exception as e:
        print(f"Failed to list models: {e}")


async def main():
    parser = argparse.ArgumentParser(
        description="Test DGX Spark Vision Server"
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8001",
        help="Vision server URL (default: http://localhost:8001)"
    )
    parser.add_argument(
        "--image",
        help="Path to test image (for inference benchmark)"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of inference runs (default: 5)"
    )

    args = parser.parse_args()

    print("=" * 50)
    print("DGX Spark Vision Server - Connection Test")
    print("=" * 50)

    url = args.url.rstrip("/")
    print(f"\nServer URL: {url}")
    print(f"Image: {args.image or 'None (will skip benchmark)'}")
    print()

    # Test health
    print("Testing server health...")
    health = await test_health(url)

    print(f"Status: {health.get('status')}")

    if health.get("status") == "healthy":
        print("✓ Server is healthy and responding")

        # List models
        models = health.get("available_models", [])
        print(f"Available models: {', '.join(models) if models else 'None'}")

    elif health.get("status") == "unreachable":
        print(f"✗ Cannot connect to server")
        print(f"  Error: {health.get('error')}")
        print()
        print("Troubleshooting:")
        print("  1. Check DGX Spark is powered on")
        print("  2. Verify network connectivity:")
        print(f"     ping {url.split('://')[1].split(':')[0]}")
        print("  3. Check vision server is running:")
        print("     ssh ubuntu@<dgx-ip>")
        print("     sudo systemctl status brickscan-vision")
        return

    else:
        print(f"✗ Server error: {health.get('error')}")
        return

    # List available models
    await list_models(url)

    # Run inference benchmark if image provided
    if args.image:
        image_path = Path(args.image)

        if not image_path.exists():
            print(f"\nError: Image file not found: {args.image}")
            return

        if not image_path.is_file():
            print(f"\nError: Not a file: {args.image}")
            return

        print()
        await benchmark_inference(url, str(image_path), args.runs)

    print()
    print("=" * 50)
    print("Test Complete")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
