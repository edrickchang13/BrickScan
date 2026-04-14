#!/usr/bin/env python3
"""
Fetch LEGO part names from Rebrickable API and save as part_names.json.

Usage:
    python fetch_part_names.py --api-key YOUR_KEY --labels-dir ml/models/ --output backend/models/part_names.json

This creates a JSON mapping: {"3001": "Brick 2 x 4", "3003": "Brick 2 x 2", ...}
Only fetches names for parts in the label encoder (the 2007 parts we can classify).
"""

import argparse
import json
import os
import time
import requests
from pathlib import Path


def fetch_part_names(api_key: str, labels_dir: str, output_path: str):
    """Fetch part names from Rebrickable for all parts in the model."""

    labels_path = Path(labels_dir) / "part_labels.json"
    if not labels_path.exists():
        print(f"ERROR: part_labels.json not found at {labels_path}")
        return

    with open(labels_path) as f:
        labels = json.load(f)

    idx2part = labels.get("idx2part", {})
    part_nums = list(idx2part.values())
    print(f"Fetching names for {len(part_nums)} parts...")

    part_names = {}
    headers = {"Authorization": f"key {api_key}"}

    for i, part_num in enumerate(part_nums):
        try:
            url = f"https://rebrickable.com/api/v3/lego/parts/{part_num}/"
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                part_names[part_num] = data.get("name", "")
            elif resp.status_code == 404:
                part_names[part_num] = ""  # Unknown part
            else:
                print(f"  HTTP {resp.status_code} for part {part_num}")
                part_names[part_num] = ""
        except Exception as e:
            print(f"  Error fetching {part_num}: {e}")
            part_names[part_num] = ""

        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(part_nums)}")
            time.sleep(1)  # Rate limiting: max ~100 req/min on free tier
        else:
            time.sleep(0.1)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(part_names, f, indent=2)

    filled = sum(1 for v in part_names.values() if v)
    print(f"\nSaved {len(part_names)} entries ({filled} with names) to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", required=True, help="Rebrickable API key")
    parser.add_argument("--labels-dir", required=True, help="Directory with part_labels.json")
    parser.add_argument("--output", required=True, help="Output path for part_names.json")
    args = parser.parse_args()

    fetch_part_names(args.api_key, args.labels_dir, args.output)
