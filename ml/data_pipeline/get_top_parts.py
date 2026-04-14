"""
Get Top N LEGO Parts by Frequency

Queries Rebrickable data to find which parts appear most often across all sets.
These are the parts to prioritize for training data generation.

The script uses CSV data from Rebrickable (https://rebrickable.com/downloads/).
Data files needed:
- inventories.csv: Maps inventory IDs to set numbers
- inventory_parts.csv: Lists all parts in each inventory with quantities

Usage:
    python get_top_parts.py \
        --inventory_parts ./rebrickable_data/inventory_parts.csv \
        --inventories ./rebrickable_data/inventories.csv \
        --output top_3000_parts.txt \
        --limit 3000

To download Rebrickable data:
    Visit https://rebrickable.com/downloads/ and download:
    1. inventories.csv (full LEGO database)
    2. inventory_parts.csv (parts in each set)
    Extract and provide paths to this script.
"""

import csv
import sys
from pathlib import Path
from collections import Counter
from typing import Dict, Set, Tuple


def build_frequency_table_from_csv(
    inventory_csv: str,
    inventories_csv: str,
) -> Counter:
    """
    Build part frequency table from CSV files.

    Counts how many unique LEGO sets contain each part.
    This is more meaningful than counting total quantity,
    since some sets have many duplicates of common parts.

    File formats:
    - inventories.csv columns: id, set_num, version
    - inventory_parts.csv columns: inventory_id, part_num, color_id, quantity, is_spare

    Args:
        inventory_csv: Path to inventory_parts.csv
        inventories_csv: Path to inventories.csv

    Returns:
        Counter object mapping part_num -> frequency (number of sets containing it)
    """
    print(f"Loading inventory data from CSV files...")
    print(f"  Inventories: {inventories_csv}")
    print(f"  Parts: {inventory_csv}")

    # Build inventory_id -> set_num mapping
    inventory_to_set: Dict[str, str] = {}

    try:
        with open(inventories_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                inventory_id = row.get('id')
                set_num = row.get('set_num')
                if inventory_id and set_num:
                    inventory_to_set[inventory_id] = set_num
    except FileNotFoundError:
        print(f"ERROR: {inventories_csv} not found")
        sys.exit(1)
    except KeyError as e:
        print(f"ERROR: Missing column in {inventories_csv}: {e}")
        sys.exit(1)

    print(f"Loaded {len(inventory_to_set)} inventories")

    # Count part frequencies across all sets
    # Use set of part_nums per set to avoid counting duplicates
    part_counter = Counter()
    set_parts: Dict[str, Set[str]] = {}

    try:
        with open(inventory_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Skip spare parts
                is_spare = row.get('is_spare', 'f').lower()
                if is_spare in ('true', 't', '1', 'yes'):
                    continue

                inventory_id = row.get('inventory_id')
                part_num = row.get('part_num')

                if not inventory_id or not part_num:
                    continue

                set_num = inventory_to_set.get(inventory_id)
                if not set_num:
                    continue

                # Track unique parts per set
                if set_num not in set_parts:
                    set_parts[set_num] = set()

                set_parts[set_num].add(part_num)

    except FileNotFoundError:
        print(f"ERROR: {inventory_csv} not found")
        sys.exit(1)
    except KeyError as e:
        print(f"ERROR: Missing column in {inventory_csv}: {e}")
        sys.exit(1)

    print(f"Parsed {len(set_parts)} unique sets")

    # Count how many SETS each part appears in
    total_part_instances = 0
    for set_num, parts in set_parts.items():
        for part_num in parts:
            part_counter[part_num] += 1
            total_part_instances += 1

    print(f"Found {len(part_counter)} unique parts across all sets")
    print(f"Total part instances: {total_part_instances}")

    return part_counter


def get_top_parts_from_csv(
    inventory_parts_csv: str,
    inventories_csv: str,
    output_file: str,
    limit: int = 3000,
) -> list:
    """
    Main function: find top N parts and save to file.

    Output format: one part per line, tab-separated part_num and frequency count

    Args:
        inventory_parts_csv: Path to inventory_parts.csv from Rebrickable
        inventories_csv: Path to inventories.csv from Rebrickable
        output_file: Output file path
        limit: Number of top parts to return

    Returns:
        List of part numbers in order of frequency
    """
    print("\n" + "=" * 70)
    print("BrickScan: Get Top LEGO Parts for Training Data Generation")
    print("=" * 70 + "\n")

    counter = build_frequency_table_from_csv(
        inventory_parts_csv,
        inventories_csv,
    )

    # Get top N parts
    top_parts = counter.most_common(limit)

    print(f"\nTop 20 most common parts:")
    print(f"{'Rank':<6} {'Part #':<12} {'Sets':<6}")
    print("-" * 30)
    for i, (part_num, count) in enumerate(top_parts[:20], 1):
        print(f"{i:<6} {part_num:<12} {count:<6}")

    # Save to output file
    print(f"\nSaving {len(top_parts)} parts to {output_file}...")

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for part_num, count in top_parts:
                f.write(f"{part_num}\t{count}\n")
    except IOError as e:
        print(f"ERROR: Could not write to {output_file}: {e}")
        sys.exit(1)

    print(f"Success! Saved {len(top_parts)} parts to {output_file}")
    print(f"\nExample command to render these parts:")
    print(f"  blender --background --python ldraw_renderer.py -- \\")
    print(f"    --parts_dir ./ldraw/parts \\")
    print(f"    --output_dir ./output/synthetic_data \\")
    print(f"    --top_parts_file {output_file} \\")
    print(f"    --num_renders 100 \\")
    print(f"    --colors common \\")
    print(f"    --gpu")

    return [p[0] for p in top_parts]


def generate_statistics(counter: Counter, top_n: int = 100):
    """
    Generate statistics about the parts distribution.

    Useful for understanding dataset composition and coverage.
    """
    total_parts = len(counter)
    total_appearances = sum(counter.values())
    top_parts = counter.most_common(top_n)
    top_appearances = sum(c for _, c in top_parts)

    print(f"\nStatistics:")
    print(f"  Total unique parts: {total_parts}")
    print(f"  Total part appearances: {total_appearances}")
    print(f"  Top {top_n} parts cover: {top_appearances / total_appearances * 100:.1f}% of all instances")
    print(f"  Average appearances per part: {total_appearances / total_parts:.2f}")

    # Distribution analysis
    coverage_levels = {10: 0, 20: 0, 50: 0, 100: 0}
    cumulative = 0

    for part_num, count in counter.most_common():
        cumulative += count
        percentage = cumulative / total_appearances * 100

        for level in sorted(coverage_levels.keys()):
            if percentage >= level:
                coverage_levels[level] += 1

    print(f"\n  Parts needed for coverage:")
    for level in [10, 20, 50, 100]:
        print(f"    {level}% coverage: {coverage_levels[level]} parts")


def main():
    """Command line entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract top LEGO parts from Rebrickable data for BrickScan training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download Rebrickable data first from https://rebrickable.com/downloads/
  python get_top_parts.py \\
    --inventory_parts ./rebrickable_data/inventory_parts.csv \\
    --inventories ./rebrickable_data/inventories.csv \\
    --output ./top_3000_parts.txt \\
    --limit 3000

  # Then render these parts
  blender --background --python ldraw_renderer.py -- \\
    --parts_dir /path/to/ldraw/parts \\
    --output_dir ./synthetic_data \\
    --top_parts_file ./top_3000_parts.txt \\
    --num_renders 100
        """
    )

    parser.add_argument(
        "--inventory_parts",
        type=str,
        default="./rebrickable_data/inventory_parts.csv",
        help="Path to inventory_parts.csv from Rebrickable (default: ./rebrickable_data/inventory_parts.csv)"
    )
    parser.add_argument(
        "--inventories",
        type=str,
        default="./rebrickable_data/inventories.csv",
        help="Path to inventories.csv from Rebrickable (default: ./rebrickable_data/inventories.csv)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./top_3000_parts.txt",
        help="Output file path (default: ./top_3000_parts.txt)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3000,
        help="Number of top parts to extract (default: 3000)"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print distribution statistics"
    )

    args = parser.parse_args()

    # Validate input files
    if not Path(args.inventory_parts).exists():
        print(f"ERROR: {args.inventory_parts} not found")
        print("Download from: https://rebrickable.com/downloads/")
        sys.exit(1)

    if not Path(args.inventories).exists():
        print(f"ERROR: {args.inventories} not found")
        print("Download from: https://rebrickable.com/downloads/")
        sys.exit(1)

    # Extract top parts
    top_parts = get_top_parts_from_csv(
        args.inventory_parts,
        args.inventories,
        args.output,
        args.limit
    )

    # Optional: print statistics
    if args.stats:
        counter = build_frequency_table_from_csv(
            args.inventory_parts,
            args.inventories,
        )
        generate_statistics(counter, min(100, args.limit))


if __name__ == "__main__":
    main()
