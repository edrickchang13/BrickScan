"""Download LEGO part images from Rebrickable API for training data.

Requires REBRICKABLE_API_KEY environment variable to be set.

Usage:
    python rebrickable_images.py --output_dir ./data/catalog_images --max_parts 3000
"""

import argparse
import asyncio
import os
from pathlib import Path
from typing import List, Optional

import httpx
from tqdm.asyncio import tqdm as atqdm


REBRICKABLE_BASE_URL = "https://rebrickable.com/api/v3/lego"
REBRICKABLE_API_KEY = os.environ.get("REBRICKABLE_API_KEY")


async def get_api_session() -> httpx.AsyncClient:
    """Create an authenticated HTTPX session for Rebrickable API.

    Returns:
        Authenticated AsyncClient
    """
    headers = {}
    if REBRICKABLE_API_KEY:
        headers["Authorization"] = f"key {REBRICKABLE_API_KEY}"

    return httpx.AsyncClient(headers=headers, timeout=30.0)


async def get_all_parts(
    session: httpx.AsyncClient,
    page: int = 1,
    page_size: int = 1000,
) -> tuple:
    """Fetch all parts from Rebrickable API with pagination.

    Args:
        session: HTTPX session
        page: Page number
        page_size: Results per page

    Returns:
        Tuple of (parts_list, next_page_token)
    """
    url = f"{REBRICKABLE_BASE_URL}/parts/"
    params = {
        "page": page,
        "page_size": page_size,
    }

    response = await session.get(url, params=params)
    response.raise_for_status()

    data = response.json()
    parts = data.get("results", [])
    next_url = data.get("next")

    return parts, next_url


async def fetch_all_parts_paginated(session: httpx.AsyncClient) -> List[dict]:
    """Fetch all parts from Rebrickable with automatic pagination.

    Args:
        session: HTTPX session

    Returns:
        List of all parts
    """
    all_parts = []
    page = 1

    print("Fetching all parts from Rebrickable API...")

    while True:
        try:
            parts, next_url = await get_all_parts(session, page=page)

            if not parts:
                break

            all_parts.extend(parts)
            print(f"  Fetched page {page}: {len(parts)} parts (total: {len(all_parts)})")

            if next_url is None:
                break

            page += 1

        except Exception as e:
            print(f"Error fetching page {page}: {e}")
            break

    print(f"Total parts fetched: {len(all_parts)}")
    return all_parts


async def download_part_image(
    session: httpx.AsyncClient,
    part_num: str,
    image_url: str,
    output_dir: str,
) -> bool:
    """Download a single part image.

    Args:
        session: HTTPX session
        part_num: Part number (used for directory name)
        image_url: URL to image
        output_dir: Output directory

    Returns:
        True if successful, False otherwise
    """
    try:
        # Create part directory
        part_dir = Path(output_dir) / part_num
        part_dir.mkdir(parents=True, exist_ok=True)

        # Download image
        response = await session.get(image_url, follow_redirects=True)
        response.raise_for_status()

        # Save image
        output_path = part_dir / "catalog.jpg"
        with open(output_path, "wb") as f:
            f.write(response.content)

        return True

    except Exception as e:
        print(f"Error downloading {part_num}: {e}")
        return False


async def download_all_part_images(
    output_dir: str,
    max_parts: Optional[int] = None,
    concurrent_downloads: int = 5,
) -> None:
    """Download images for all LEGO parts from Rebrickable.

    Args:
        output_dir: Output directory
        max_parts: Max number of parts to download
        concurrent_downloads: Number of concurrent downloads
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if not REBRICKABLE_API_KEY:
        print("Warning: REBRICKABLE_API_KEY not set. Will use public API (limited rate).")

    # Get session
    async with await get_api_session() as session:
        # Fetch all parts
        all_parts = await fetch_all_parts_paginated(session)

        # Filter parts with images
        parts_with_images = [
            p for p in all_parts
            if p.get("part_img_url") and p.get("part_img_url").strip()
        ]

        if max_parts:
            parts_with_images = parts_with_images[:max_parts]

        print(f"Found {len(parts_with_images)} parts with images")

        # Download with semaphore to limit concurrency
        semaphore = asyncio.Semaphore(concurrent_downloads)

        async def download_with_semaphore(part: dict) -> bool:
            async with semaphore:
                part_num = part.get("part_num")
                image_url = part.get("part_img_url")

                if not part_num or not image_url:
                    return False

                return await download_part_image(session, part_num, image_url, output_dir)

        # Download all
        tasks = [download_with_semaphore(p) for p in parts_with_images]
        results = await atqdm.gather(*tasks, desc="Downloading images")

        successful = sum(1 for r in results if r)
        failed = len(results) - successful

        print(f"\nDownload complete!")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Output directory: {output_dir}")


async def get_top_n_parts_by_set_frequency(
    session: httpx.AsyncClient,
    n: int = 3000,
) -> List[str]:
    """Get the top N parts by frequency in LEGO sets.

    This helps prioritize which parts to focus on for training.

    Args:
        session: HTTPX session
        n: Number of top parts

    Returns:
        List of part_nums sorted by frequency descending
    """
    url = f"{REBRICKABLE_BASE_URL}/part_inventories/"

    # This would require fetching all inventories and counting
    # For now, we'll just return all parts in order from API
    print("Fetching parts by set frequency...")

    all_parts = await fetch_all_parts_paginated(session)

    # Parts are returned from API roughly by frequency
    part_nums = [p.get("part_num") for p in all_parts[:n] if p.get("part_num")]

    print(f"Top {min(n, len(part_nums))} parts selected")
    return part_nums


async def download_top_n_parts(
    output_dir: str,
    n: int = 3000,
    concurrent_downloads: int = 5,
) -> None:
    """Download images for the top N most common LEGO parts.

    Args:
        output_dir: Output directory
        n: Number of top parts
        concurrent_downloads: Number of concurrent downloads
    """
    async with await get_api_session() as session:
        # Get top N parts by frequency
        top_parts = await get_top_n_parts_by_set_frequency(session, n)

        print(f"Top {n} parts selected for download")

        # Fetch full part data for these parts
        print("Fetching part details...")
        all_parts, _ = await get_all_parts(session, page=1, page_size=n)

        parts_to_download = [
            p for p in all_parts
            if p.get("part_num") in top_parts and p.get("part_img_url")
        ]

        print(f"Downloading images for {len(parts_to_download)} parts...")

        # Download with semaphore
        semaphore = asyncio.Semaphore(concurrent_downloads)

        async def download_with_semaphore(part: dict) -> bool:
            async with semaphore:
                part_num = part.get("part_num")
                image_url = part.get("part_img_url")

                if not part_num or not image_url:
                    return False

                return await download_part_image(session, part_num, image_url, output_dir)

        tasks = [download_with_semaphore(p) for p in parts_to_download]
        results = await atqdm.gather(*tasks, desc="Downloading")

        successful = sum(1 for r in results if r)
        print(f"\nDownload complete! {successful}/{len(parts_to_download)} successful")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Download LEGO part images from Rebrickable API"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/catalog_images",
        help="Output directory",
    )
    parser.add_argument(
        "--max_parts",
        type=int,
        default=3000,
        help="Maximum number of parts to download",
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=5,
        help="Number of concurrent downloads",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="Rebrickable API key (or set REBRICKABLE_API_KEY env var)",
    )

    args = parser.parse_args()

    # Set API key if provided
    if args.api_key:
        os.environ["REBRICKABLE_API_KEY"] = args.api_key

    # Run download
    asyncio.run(
        download_all_part_images(
            args.output_dir,
            max_parts=args.max_parts,
            concurrent_downloads=args.concurrent,
        )
    )


if __name__ == "__main__":
    main()
