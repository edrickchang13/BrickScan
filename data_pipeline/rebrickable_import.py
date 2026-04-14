"""Import Rebrickable CSV database dumps into PostgreSQL."""
import asyncio
import csv
import asyncpg
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import time

DATABASE_URL = os.environ.get('DATABASE_URL', 'postgresql://brickscan_user:brickscan_password@localhost/brickscan')


async def get_connection():
    """Create and return a PostgreSQL connection."""
    return await asyncpg.connect(DATABASE_URL.replace('postgresql+asyncpg://', 'postgresql://'))


async def import_colors(conn, csv_path: str) -> int:
    """
    Import colors.csv (id, name, rgb, is_trans)
    Upsert into colors table.
    Returns count of imported rows.
    """
    print(f"Importing colors from {csv_path}...")
    count = 0

    if not Path(csv_path).exists():
        print(f"Warning: {csv_path} not found, skipping colors import")
        return 0

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                color_id = int(row['id'])
                name = row['name']
                rgb = row.get('rgb', None)
                is_transparent = row.get('is_trans', '0') == 't'

                await conn.execute(
                    '''
                    INSERT INTO colors (id, name, rgb, is_transparent)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (id) DO UPDATE
                    SET name = EXCLUDED.name,
                        rgb = EXCLUDED.rgb,
                        is_transparent = EXCLUDED.is_transparent,
                        updated_at = now()
                    ''',
                    color_id, name, rgb, is_transparent
                )
                count += 1
            except Exception as e:
                print(f"Error importing color {row.get('id')}: {e}")

    print(f"Imported {count} colors")
    return count


async def import_part_categories(conn, csv_path: str) -> int:
    """Import part_categories.csv (id, name)"""
    print(f"Importing part categories from {csv_path}...")
    count = 0

    if not Path(csv_path).exists():
        print(f"Warning: {csv_path} not found, skipping part categories import")
        return 0

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                cat_id = int(row['id'])
                name = row['name']

                await conn.execute(
                    '''
                    INSERT INTO part_categories (id, name)
                    VALUES ($1, $2)
                    ON CONFLICT (id) DO UPDATE
                    SET name = EXCLUDED.name,
                        updated_at = now()
                    ''',
                    cat_id, name
                )
                count += 1
            except Exception as e:
                print(f"Error importing part category {row.get('id')}: {e}")

    print(f"Imported {count} part categories")
    return count


async def import_parts(conn, csv_path: str) -> int:
    """
    Import parts.csv (part_num, name, part_cat_id, part_material)
    Parts have no year in this file — that comes from part_relationships.
    Upsert. Skip parts with no image (we check later).
    """
    print(f"Importing parts from {csv_path}...")
    count = 0

    if not Path(csv_path).exists():
        print(f"Warning: {csv_path} not found, skipping parts import")
        return 0

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                part_num = row['part_num'].strip()
                name = row['name'].strip()
                part_cat_id = int(row['part_cat_id'])
                material = row.get('part_material', '').strip() or None

                # Generate a deterministic ID from part_num for consistency
                import uuid
                part_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"part_{part_num}"))

                await conn.execute(
                    '''
                    INSERT INTO parts (id, part_num, name, part_category_id, material)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (part_num) DO UPDATE
                    SET name = EXCLUDED.name,
                        part_category_id = EXCLUDED.part_category_id,
                        material = EXCLUDED.material,
                        updated_at = now()
                    ''',
                    part_id, part_num, name, part_cat_id, material
                )
                count += 1
            except Exception as e:
                print(f"Error importing part {row.get('part_num')}: {e}")

    print(f"Imported {count} parts")
    return count


async def import_themes(conn, csv_path: str) -> int:
    """Import themes.csv (id, name, parent_id)"""
    print(f"Importing themes from {csv_path}...")
    count = 0

    if not Path(csv_path).exists():
        print(f"Warning: {csv_path} not found, skipping themes import")
        return 0

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                theme_id = int(row['id'])
                name = row['name'].strip()
                parent_id = None
                if row.get('parent_id', '').strip():
                    try:
                        parent_id = int(row['parent_id'])
                    except ValueError:
                        pass

                await conn.execute(
                    '''
                    INSERT INTO themes (id, name, parent_id)
                    VALUES ($1, $2, $3)
                    ON CONFLICT (id) DO UPDATE
                    SET name = EXCLUDED.name,
                        parent_id = EXCLUDED.parent_id,
                        updated_at = now()
                    ''',
                    theme_id, name, parent_id
                )
                count += 1
            except Exception as e:
                print(f"Error importing theme {row.get('id')}: {e}")

    print(f"Imported {count} themes")
    return count


async def import_sets(conn, csv_path: str) -> int:
    """Import sets.csv (set_num, name, year, theme_id, num_parts, img_url)"""
    print(f"Importing sets from {csv_path}...")
    count = 0

    if not Path(csv_path).exists():
        print(f"Warning: {csv_path} not found, skipping sets import")
        return 0

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                set_num = row['set_num'].strip()
                name = row['name'].strip()
                year = None
                if row.get('year', '').strip():
                    try:
                        year = int(row['year'])
                    except ValueError:
                        pass
                theme_id = int(row['theme_id'])
                num_parts = int(row['num_parts'])
                img_url = row.get('img_url', '').strip() or None

                import uuid
                set_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"set_{set_num}"))

                await conn.execute(
                    '''
                    INSERT INTO lego_sets (id, set_num, name, year, theme_id, num_parts, image_url)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (set_num) DO UPDATE
                    SET name = EXCLUDED.name,
                        year = EXCLUDED.year,
                        theme_id = EXCLUDED.theme_id,
                        num_parts = EXCLUDED.num_parts,
                        image_url = EXCLUDED.image_url,
                        updated_at = now()
                    ''',
                    set_id, set_num, name, year, theme_id, num_parts, img_url
                )
                count += 1
            except Exception as e:
                print(f"Error importing set {row.get('set_num')}: {e}")

    print(f"Imported {count} sets")
    return count


async def import_set_parts(conn, csv_path: str, inventories_csv: str) -> int:
    """
    Import inventory_parts.csv (inventory_id, part_num, color_id, quantity, is_spare)
    Note: inventory_parts links via inventories.csv → sets.
    Need to join: inventories.csv (id, set_num, version) to get set_num
    Import inventories.csv first, build inventory_id → set_num dict
    Then import inventory_parts using that mapping
    """
    print(f"Importing set parts from {csv_path}...")
    count = 0

    # First, load inventory mappings
    inventory_map: Dict[int, str] = {}

    if Path(inventories_csv).exists():
        print(f"Loading inventory mappings from {inventories_csv}...")
        with open(inventories_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    inventory_id = int(row['id'])
                    set_num = row['set_num'].strip()
                    inventory_map[inventory_id] = set_num
                except Exception as e:
                    print(f"Error loading inventory mapping {row.get('id')}: {e}")
        print(f"Loaded {len(inventory_map)} inventory mappings")
    else:
        print(f"Warning: {inventories_csv} not found, skipping inventory mappings")

    if not Path(csv_path).exists():
        print(f"Warning: {csv_path} not found, skipping set parts import")
        return 0

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                inventory_id = int(row['inventory_id'])
                part_num = row['part_num'].strip()
                color_id = int(row['color_id'])
                quantity = int(row['quantity'])
                is_spare = row.get('is_spare', '0') == 't'

                # Look up the set_num from inventory_id
                set_num = inventory_map.get(inventory_id)
                if not set_num:
                    continue

                # Get the actual set_id and part_id from the database
                set_row = await conn.fetchrow(
                    'SELECT id FROM lego_sets WHERE set_num = $1',
                    set_num
                )
                if not set_row:
                    continue
                set_id = set_row['id']

                part_row = await conn.fetchrow(
                    'SELECT id FROM parts WHERE part_num = $1',
                    part_num
                )
                if not part_row:
                    continue
                part_id = part_row['id']

                import uuid
                sp_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"set_part_{set_id}_{part_id}_{color_id}"))

                await conn.execute(
                    '''
                    INSERT INTO set_parts (id, set_id, part_id, color_id, quantity, is_spare)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (set_id, part_id, color_id) DO UPDATE
                    SET quantity = EXCLUDED.quantity,
                        is_spare = EXCLUDED.is_spare,
                        updated_at = now()
                    ''',
                    sp_id, set_id, part_id, color_id, quantity, is_spare
                )
                count += 1
            except Exception as e:
                print(f"Error importing set part {row.get('inventory_id')}: {e}")

    print(f"Imported {count} set parts")
    return count


async def main(data_dir: str):
    """
    Main import function.
    data_dir should contain all the Rebrickable CSV files.
    Connects to PostgreSQL.
    Runs all imports in order (colors → categories → parts → themes → sets → set_parts)
    Prints timing and row counts for each.
    """
    print(f"Starting Rebrickable data import from {data_dir}")
    print(f"Database: {DATABASE_URL}")

    conn = await get_connection()

    try:
        results = {}

        # Import colors
        start = time.time()
        results['colors'] = await import_colors(
            conn,
            os.path.join(data_dir, 'colors.csv')
        )
        print(f"  Time: {time.time() - start:.2f}s\n")

        # Import part categories
        start = time.time()
        results['part_categories'] = await import_part_categories(
            conn,
            os.path.join(data_dir, 'part_categories.csv')
        )
        print(f"  Time: {time.time() - start:.2f}s\n")

        # Import parts
        start = time.time()
        results['parts'] = await import_parts(
            conn,
            os.path.join(data_dir, 'parts.csv')
        )
        print(f"  Time: {time.time() - start:.2f}s\n")

        # Import themes
        start = time.time()
        results['themes'] = await import_themes(
            conn,
            os.path.join(data_dir, 'themes.csv')
        )
        print(f"  Time: {time.time() - start:.2f}s\n")

        # Import sets
        start = time.time()
        results['sets'] = await import_sets(
            conn,
            os.path.join(data_dir, 'sets.csv')
        )
        print(f"  Time: {time.time() - start:.2f}s\n")

        # Import set parts
        start = time.time()
        results['set_parts'] = await import_set_parts(
            conn,
            os.path.join(data_dir, 'inventory_parts.csv'),
            os.path.join(data_dir, 'inventories.csv')
        )
        print(f"  Time: {time.time() - start:.2f}s\n")

        # Print summary
        print("\n" + "="*50)
        print("Import Summary:")
        print("="*50)
        for table, count in results.items():
            print(f"{table:20s}: {count:8d} rows")
        print("="*50)

    finally:
        await conn.close()


if __name__ == '__main__':
    data_dir = sys.argv[1] if len(sys.argv) > 1 else './rebrickable_data'
    asyncio.run(main(data_dir))
