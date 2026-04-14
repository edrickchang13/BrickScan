#!/usr/bin/env python3
"""
Download official LEGO part images from Rebrickable API.

Downloads high-quality photos of every part×color combination available from
Rebrickable, rate-limited to respect their free tier limits (1 req/sec).

Usage:
  python3 download_rebrickable_images.py --api-key YOUR_KEY --parts-csv parts.csv --output-dir ./rebrickable_images
  python3 download_rebrickable_images.py --api-key YOUR_KEY --renders-dir ./renders --output-dir ./rebrickable_images
"""

import argparse
import csv
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, List, Dict
from io import BytesIO

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError:
    print("ERROR: requests not installed. Run: pip3 install requests")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    Image = None
    print("WARNING: Pillow not installed. Image resizing disabled. Run: pip3 install Pillow")

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
log = logging.getLogger("download_rebrickable_images")


class RebrickableDownloader:
    """Download LEGO part images from Rebrickable API with rate limiting."""

    BASE_URL = "https://rebrickable.com/api/v3/lego"
    MIN_INTERVAL = 1.0  # Rate limit: 1 second between requests (free tier)

    def __init__(self, api_key: str, output_dir: Path, workers: int = 4):
        """
        Initialize downloader.

        Args:
            api_key: Rebrickable API key
            output_dir: Root directory for downloaded images
            workers: Number of concurrent download threads
        """
        self.api_key = api_key
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.workers = workers
        self.last_request_time = 0

        # Create session with retries
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def _rate_limit(self):
        """Enforce rate limiting: max 1 request per second."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.MIN_INTERVAL:
            time.sleep(self.MIN_INTERVAL - elapsed)
        self.last_request_time = time.time()

    def download_part_image(
        self,
        part_num: str,
        color_id: int,
    ) -> Optional[Path]:
        """
        Download official image for a part×color combination.

        Args:
            part_num: LEGO part number (e.g., "3001")
            color_id: Rebrickable color ID (e.g., 1 for black)

        Returns:
            Path to saved image file, or None if download failed
        """
        self._rate_limit()

        # Query API for part color info
        url = f"{self.BASE_URL}/parts/{part_num}/colors/{color_id}/?key={self.api_key}"

        try:
            r = self.session.get(url, timeout=10)
            r.raise_for_status()
        except Exception as e:
            log.debug(f"Failed to fetch metadata for {part_num}/{color_id}: {e}")
            return None

        try:
            data = r.json()
        except Exception as e:
            log.debug(f"Failed to parse JSON for {part_num}/{color_id}: {e}")
            return None

        # Extract image URL from response
        img_url = data.get("image_url")
        if not img_url:
            return None

        # Download image
        self._rate_limit()
        try:
            r = self.session.get(img_url, timeout=10)
            r.raise_for_status()
        except Exception as e:
            log.debug(f"Failed to download image for {part_num}/{color_id}: {e}")
            return None

        # Save to disk
        output_path = self.output_dir / str(part_num) / str(color_id)
        output_path.mkdir(parents=True, exist_ok=True)

        image_path = output_path / f"rebrickable_{part_num}_{color_id}.jpg"

        try:
            # Load image and resize if PIL available
            if Image:
                img = Image.open(BytesIO(r.content)).convert("RGB")
                img.thumbnail((400, 400), Image.Resampling.LANCZOS)
                img.save(image_path, format="JPEG", quality=90)
            else:
                # Just write raw bytes
                with open(image_path, 'wb') as f:
                    f.write(r.content)

            return image_path

        except Exception as e:
            log.warning(f"Failed to save image for {part_num}/{color_id}: {e}")
            return None

    def download_all_parts(
        self,
        parts_list: List[Dict[str, int]],
    ) -> Dict[str, int]:
        """
        Download all images for a list of part×color combinations.

        Args:
            parts_list: List of dicts with 'part_num' and 'color_id' keys

        Returns:
            Summary dict with 'success', 'skipped', 'failed' counts
        """
        success = 0
        skipped = 0
        failed = 0

        iterator = tqdm(parts_list, desc="Downloading") if tqdm else parts_list

        for item in iterator:
            part_num = item['part_num']
            color_id = item['color_id']

            # Check if already exists
            existing = (
                self.output_dir / str(part_num) / str(color_id) /
                f"rebrickable_{part_num}_{color_id}.jpg"
            )
            if existing.exists():
                skipped += 1
                continue

            result = self.download_part_image(part_num, color_id)
            if result:
                success += 1
            else:
                failed += 1

        return {
            'success': success,
            'skipped': skipped,
            'failed': failed,
            'total': len(parts_list),
        }

    def download_parts_for_training_set(self, renders_dir: Path) -> Dict[str, int]:
        """
        Auto-discover all parts in a renders directory and download their
        official images for all available colors.

        Args:
            renders_dir: Directory containing renders (subdirs are part numbers)

        Returns:
            Summary dict with counts
        """
        # Discover all parts
        parts = set()
        for part_subdir in renders_dir.iterdir():
            if part_subdir.is_dir() and not part_subdir.name.startswith('.'):
                parts.add(part_subdir.name)

        log.info(f"Discovered {len(parts)} parts in {renders_dir}")

        # For each part, download all available colors
        # Query Rebrickable's parts list to find all available colors for each part
        all_tasks = []

        for part_num in sorted(parts):
            log.info(f"Fetching available colors for part {part_num}...")
            self._rate_limit()

            # Query part info to get available colors
            url = f"{self.BASE_URL}/parts/{part_num}/?key={self.api_key}"
            try:
                r = self.session.get(url, timeout=10)
                r.raise_for_status()
                data = r.json()

                # Get color count
                color_count = data.get('num_colors', 0)
                log.info(f"  {part_num}: {color_count} colors available")

                # Add tasks for all colors
                for color_id in range(1, color_count + 1):
                    all_tasks.append({'part_num': part_num, 'color_id': color_id})

            except Exception as e:
                log.warning(f"Failed to fetch colors for {part_num}: {e}")

        log.info(f"Total download tasks: {len(all_tasks)}")

        return self.download_all_parts(all_tasks)


def main():
    parser = argparse.ArgumentParser(
        description="Download official LEGO part images from Rebrickable"
    )
    parser.add_argument(
        "--api-key",
        required=True,
        help="Rebrickable API key (get from https://rebrickable.com/api/)"
    )
    parser.add_argument(
        "--parts-csv",
        default=None,
        help="CSV file with 'part_num' and 'color_id' columns"
    )
    parser.add_argument(
        "--renders-dir",
        default=None,
        help="Alternatively, auto-discover parts from a renders directory"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for downloaded images"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of concurrent downloads (respects rate limit, default: 4)"
    )

    args = parser.parse_args()

    if not args.parts_csv and not args.renders_dir:
        print("ERROR: Must provide either --parts-csv or --renders-dir")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    downloader = RebrickableDownloader(args.api_key, output_dir, workers=args.workers)

    # Load parts list
    parts_list = []

    if args.parts_csv:
        log.info(f"Loading parts from {args.parts_csv}...")
        with open(args.parts_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    parts_list.append({
                        'part_num': row['part_num'],
                        'color_id': int(row['color_id']),
                    })
                except (KeyError, ValueError) as e:
                    log.warning(f"Skipped row: {e}")

        log.info(f"Loaded {len(parts_list)} part×color combinations")

    if args.renders_dir:
        log.info(f"Auto-discovering parts from {args.renders_dir}...")
        renders_dir = Path(args.renders_dir)
        if not renders_dir.exists():
            log.error(f"Renders directory not found: {renders_dir}")
            sys.exit(1)

        stats = downloader.download_parts_for_training_set(renders_dir)
        log.info("=" * 60)
        log.info(f"Download complete!")
        log.info(f"Success: {stats['success']}")
        log.info(f"Skipped (already exist): {stats['skipped']}")
        log.info(f"Failed: {stats['failed']}")
        log.info(f"Total: {stats['total']}")
        log.info(f"Output: {output_dir}")
        return

    # Download parts from CSV
    if parts_list:
        stats = downloader.download_all_parts(parts_list)
        log.info("=" * 60)
        log.info(f"Download complete!")
        log.info(f"Success: {stats['success']}")
        log.info(f"Skipped (already exist): {stats['skipped']}")
        log.info(f"Failed: {stats['failed']}")
        log.info(f"Total: {stats['total']}")
        log.info(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
