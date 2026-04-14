"""Quick verification script to check the import worked."""
import asyncio
import asyncpg
import os
from collections import Counter

DATABASE_URL = os.environ.get('DATABASE_URL', 'postgresql://brickscan_user:brickscan_password@localhost/brickscan')


async def get_connection():
    """Create and return a PostgreSQL connection."""
    return await asyncpg.connect(DATABASE_URL)


async def verify_import():
    """Verify that the data import was successful."""
    conn = await get_connection()

    try:
        print("\n" + "="*60)
        print("BRICKSCAN DATA IMPORT VERIFICATION")
        print("="*60 + "\n")

        # Count rows in each table
        print("Table Row Counts:")
        print("-"*60)
        tables = [
            'colors',
            'part_categories',
            'parts',
            'themes',
            'lego_sets',
            'set_parts',
            'inventory_items',
            'scan_logs',
            'users'
        ]

        for table in tables:
            try:
                count = await conn.fetchval(f'SELECT COUNT(*) FROM {table}')
                print(f"  {table:25s}: {count:8d} rows")
            except Exception as e:
                print(f"  {table:25s}: ERROR - {e}")

        print("\n" + "-"*60)

        # Find a famous set (Millennium Falcon)
        print("\nSearching for 'Millennium Falcon' set...")
        print("-"*60)
        sets = await conn.fetch(
            """
            SELECT id, set_num, name, year, num_parts FROM lego_sets
            WHERE name ILIKE '%millennium%' OR name ILIKE '%falcon%'
            LIMIT 5
            """
        )

        if sets:
            for s in sets:
                print(f"\nFound: {s['name']} ({s['set_num']})")
                print(f"  Year: {s['year']}")
                print(f"  Total Parts: {s['num_parts']}")

                # Get parts in this set
                parts = await conn.fetch(
                    """
                    SELECT p.part_num, p.name, c.name as color_name, sp.quantity
                    FROM set_parts sp
                    JOIN parts p ON sp.part_id = p.id
                    JOIN colors c ON sp.color_id = c.id
                    WHERE sp.set_id = $1
                    ORDER BY sp.quantity DESC
                    LIMIT 10
                    """,
                    s['id']
                )

                if parts:
                    print(f"\n  Top 10 Parts:")
                    for p in parts:
                        print(f"    {p['part_num']:10s} - {p['name']:40s} ({p['color_name']:15s}) x{p['quantity']}")
        else:
            print("  Not found in database")

        print("\n" + "-"*60)

        # Find top 10 most common parts across all sets
        print("\nTop 10 Most Common Parts Across All Sets:")
        print("-"*60)
        top_parts = await conn.fetch(
            """
            SELECT p.part_num, p.name, COUNT(*) as set_count, SUM(sp.quantity) as total_quantity
            FROM set_parts sp
            JOIN parts p ON sp.part_id = p.id
            GROUP BY p.id, p.part_num, p.name
            ORDER BY total_quantity DESC
            LIMIT 10
            """
        )

        for i, p in enumerate(top_parts, 1):
            print(f"  {i:2d}. {p['part_num']:10s} - {p['name']:40s}")
            print(f"      In {p['set_count']} sets, total quantity: {p['total_quantity']}")

        print("\n" + "-"*60)

        # Check for data quality issues
        print("\nData Quality Checks:")
        print("-"*60)

        # Parts with no category
        parts_no_cat = await conn.fetchval(
            'SELECT COUNT(*) FROM parts WHERE part_category_id IS NULL'
        )
        print(f"  Parts with no category: {parts_no_cat}")

        # Sets with 0 parts
        sets_zero = await conn.fetchval(
            'SELECT COUNT(*) FROM lego_sets WHERE num_parts = 0'
        )
        print(f"  Sets with 0 parts: {sets_zero}")

        # Sets with no theme
        sets_no_theme = await conn.fetchval(
            'SELECT COUNT(*) FROM lego_sets WHERE theme_id IS NULL'
        )
        print(f"  Sets with no theme: {sets_no_theme}")

        # Parts with no image
        parts_no_image = await conn.fetchval(
            'SELECT COUNT(*) FROM parts WHERE image_url IS NULL'
        )
        print(f"  Parts with no image: {parts_no_image}")

        # Colors with no RGB
        colors_no_rgb = await conn.fetchval(
            'SELECT COUNT(*) FROM colors WHERE rgb IS NULL'
        )
        print(f"  Colors with no RGB: {colors_no_rgb}")

        # Inventory items for non-existent parts (orphaned)
        orphaned = await conn.fetchval(
            '''
            SELECT COUNT(*) FROM inventory_items ii
            WHERE NOT EXISTS (SELECT 1 FROM parts p WHERE p.id = ii.part_id)
            '''
        )
        print(f"  Orphaned inventory items: {orphaned}")

        print("\n" + "-"*60)

        # Sample sets by theme
        print("\nSample Sets by Theme:")
        print("-"*60)
        themes = await conn.fetch(
            """
            SELECT t.name, COUNT(s.id) as set_count, ARRAY_AGG(s.name LIMIT 3) as sample_sets
            FROM themes t
            LEFT JOIN lego_sets s ON s.theme_id = t.id
            GROUP BY t.id, t.name
            HAVING COUNT(s.id) > 0
            ORDER BY set_count DESC
            LIMIT 10
            """
        )

        for t in themes:
            print(f"\n  {t['name']} ({t['set_count']} sets)")
            if t['sample_sets']:
                for sample in t['sample_sets'][:3]:
                    print(f"    - {sample}")

        print("\n" + "="*60)
        print("Verification Complete!")
        print("="*60 + "\n")

    finally:
        await conn.close()


if __name__ == '__main__':
    asyncio.run(verify_import())
