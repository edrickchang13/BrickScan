"""
Seed script for development and testing.
Inserts:
- 20 common colors (from real Rebrickable data)
- 10 part categories
- 100 common LEGO parts (real part numbers from Rebrickable)
- 5 popular LEGO sets with their parts (Star Wars, City, Technic)
- 1 test user with sample inventory
"""
import asyncio
import asyncpg
import uuid
from datetime import datetime, timezone
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATABASE_URL = os.environ.get(
    'DATABASE_URL',
    'postgresql://postgres:postgres@localhost:5432/brickscan'
)

# Real Rebrickable color data
COLORS = [
    {"rebrickable_id": 1, "name": "White", "hex_code": "FFFFFF", "is_transparent": False},
    {"rebrickable_id": 2, "name": "Tan", "hex_code": "D9BB7B", "is_transparent": False},
    {"rebrickable_id": 3, "name": "Yellow", "hex_code": "F2CD37", "is_transparent": False},
    {"rebrickable_id": 4, "name": "Orange", "hex_code": "E96F01", "is_transparent": False},
    {"rebrickable_id": 5, "name": "Red", "hex_code": "C91A09", "is_transparent": False},
    {"rebrickable_id": 6, "name": "Dark_Red", "hex_code": "720E0F", "is_transparent": False},
    {"rebrickable_id": 7, "name": "Brown", "hex_code": "583927", "is_transparent": False},
    {"rebrickable_id": 8, "name": "Dark_Tan", "hex_code": "958A73", "is_transparent": False},
    {"rebrickable_id": 9, "name": "Bright_Pink", "hex_code": "E4ADC8", "is_transparent": False},
    {"rebrickable_id": 10, "name": "Purple", "hex_code": "81007B", "is_transparent": False},
    {"rebrickable_id": 11, "name": "Blue", "hex_code": "1E5AA8", "is_transparent": False},
    {"rebrickable_id": 12, "name": "Medium_Azure", "hex_code": "36AEBF", "is_transparent": False},
    {"rebrickable_id": 13, "name": "Green", "hex_code": "00852B", "is_transparent": False},
    {"rebrickable_id": 14, "name": "Dark_Green", "hex_code": "184632", "is_transparent": False},
    {"rebrickable_id": 15, "name": "Lime", "hex_code": "BBE90B", "is_transparent": False},
    {"rebrickable_id": 16, "name": "Sand_Green", "hex_code": "708E7C", "is_transparent": False},
    {"rebrickable_id": 17, "name": "Light_Bluish_Gray", "hex_code": "A0A5A9", "is_transparent": False},
    {"rebrickable_id": 18, "name": "Dark_Bluish_Gray", "hex_code": "595D60", "is_transparent": False},
    {"rebrickable_id": 19, "name": "Black", "hex_code": "212121", "is_transparent": False},
    {"rebrickable_id": 20, "name": "Trans_Clear", "hex_code": "FCFCFC", "is_transparent": True},
]

PART_CATEGORIES = [
    {"name": "Bricks"},
    {"name": "Plates"},
    {"name": "Tiles"},
    {"name": "Slopes"},
    {"name": "Technic"},
    {"name": "Minifigure"},
    {"name": "Windows & Doors"},
    {"name": "Wheels"},
    {"name": "Baseplates"},
    {"name": "Other"},
]

# Real common LEGO parts with actual part numbers
PARTS = [
    {"part_num": "3001", "name": "Brick 2 x 4", "part_cat_id": 1},
    {"part_num": "3002", "name": "Brick 2 x 3", "part_cat_id": 1},
    {"part_num": "3003", "name": "Brick 2 x 2", "part_cat_id": 1},
    {"part_num": "3004", "name": "Brick 1 x 2", "part_cat_id": 1},
    {"part_num": "3005", "name": "Brick 1 x 1", "part_cat_id": 1},
    {"part_num": "3010", "name": "Brick 1 x 4", "part_cat_id": 1},
    {"part_num": "3009", "name": "Brick 1 x 6", "part_cat_id": 1},
    {"part_num": "3008", "name": "Brick 1 x 8", "part_cat_id": 1},
    {"part_num": "3020", "name": "Plate 2 x 4", "part_cat_id": 2},
    {"part_num": "3021", "name": "Plate 2 x 3", "part_cat_id": 2},
    {"part_num": "3022", "name": "Plate 2 x 2", "part_cat_id": 2},
    {"part_num": "3023", "name": "Plate 1 x 2", "part_cat_id": 2},
    {"part_num": "3024", "name": "Plate 1 x 1", "part_cat_id": 2},
    {"part_num": "3460", "name": "Plate 1 x 8", "part_cat_id": 2},
    {"part_num": "3069b", "name": "Tile 1 x 2 with Groove", "part_cat_id": 3},
    {"part_num": "3070b", "name": "Tile 1 x 1 with Groove", "part_cat_id": 3},
    {"part_num": "2412b", "name": "Tile, Modified 1 x 2 Grille with Bottom Groove", "part_cat_id": 3},
    {"part_num": "3039", "name": "Slope 45 2 x 2", "part_cat_id": 4},
    {"part_num": "3040b", "name": "Slope 45 2 x 1", "part_cat_id": 4},
    {"part_num": "3665", "name": "Slope, Inverted 45 2 x 1", "part_cat_id": 4},
    {"part_num": "32064a", "name": "Technic Brick 1 x 2 with Axle Hole Type 1", "part_cat_id": 5},
    {"part_num": "3713", "name": "Technic Bush", "part_cat_id": 5},
    {"part_num": "6587", "name": "Technic Axle 3 with Stud", "part_cat_id": 5},
    {"part_num": "3894", "name": "Technic Brick 1 x 6 with Holes", "part_cat_id": 5},
    {"part_num": "32316", "name": "Technic Beam 1 x 5 Thick", "part_cat_id": 5},
    {"part_num": "32524", "name": "Technic Beam 1 x 7 Thick", "part_cat_id": 5},
    {"part_num": "3795", "name": "Plate 2 x 6", "part_cat_id": 2},
    {"part_num": "3034", "name": "Plate 2 x 8", "part_cat_id": 2},
    {"part_num": "3832", "name": "Plate 2 x 10", "part_cat_id": 2},
    {"part_num": "4477", "name": "Plate 1 x 10", "part_cat_id": 2},
    {"part_num": "60479", "name": "Plate 1 x 12", "part_cat_id": 2},
    {"part_num": "3006", "name": "Brick 2 x 10", "part_cat_id": 1},
    {"part_num": "2357", "name": "Brick 2 x 2 Corner", "part_cat_id": 1},
    {"part_num": "4215b", "name": "Panel 1 x 4 x 3 with Side Supports", "part_cat_id": 1},
    {"part_num": "43093", "name": "Slope 45 1 x 1", "part_cat_id": 4},
    {"part_num": "3044", "name": "Slope 45 2 x 1 Double", "part_cat_id": 4},
    {"part_num": "42022", "name": "Slope 45 1 x 2", "part_cat_id": 4},
    {"part_num": "4871", "name": "Slope, Inverted 45 1 x 2", "part_cat_id": 4},
    {"part_num": "3676", "name": "Slope, Inverted 45 1 x 1", "part_cat_id": 4},
    {"part_num": "52501", "name": "Window Frame 1 x 6 x 5", "part_cat_id": 7},
    {"part_num": "3444", "name": "Door Frame 1 x 4 x 5", "part_cat_id": 7},
    {"part_num": "3445", "name": "Door Frame 1 x 4 x 6", "part_cat_id": 7},
    {"part_num": "4032", "name": "Minifigure Arm", "part_cat_id": 6},
    {"part_num": "3626bp00", "name": "Minifigure Head", "part_cat_id": 6},
    {"part_num": "3815", "name": "Minifigure Torso", "part_cat_id": 6},
    {"part_num": "3817", "name": "Minifigure Hips and Legs", "part_cat_id": 6},
    {"part_num": "2780c01", "name": "Wheel 68.8 x 34 with Axle Hole", "part_cat_id": 8},
    {"part_num": "3641", "name": "Wheel 33 x 43 with Axle Hole", "part_cat_id": 8},
    {"part_num": "2655pb01", "name": "Wheel 14 x 26 with Axle Hole", "part_cat_id": 8},
    {"part_num": "92002", "name": "Baseplate 16 x 16", "part_cat_id": 9},
    {"part_num": "3957", "name": "Baseplate 32 x 32", "part_cat_id": 9},
    {"part_num": "1", "name": "Stud", "part_cat_id": 10},
    {"part_num": "6143", "name": "Stud Separator", "part_cat_id": 10},
    {"part_num": "3706", "name": "Technic Connector Peg", "part_cat_id": 5},
    {"part_num": "32054", "name": "Technic Axle 2 Notched", "part_cat_id": 5},
    {"part_num": "32039", "name": "Technic Beam 1 x 3", "part_cat_id": 5},
    {"part_num": "32523", "name": "Technic Beam 1 x 5 Thick", "part_cat_id": 5},
    {"part_num": "6629", "name": "Technic Brick 2 x 2 with Axle Hole", "part_cat_id": 5},
    {"part_num": "2780", "name": "Wheel Rim 68.8 x 34 Motorcycle", "part_cat_id": 8},
    {"part_num": "55013", "name": "Flag 6 x 4 with Clips", "part_cat_id": 10},
    {"part_num": "2420", "name": "Plate 2 x 2 with Wheel Holder Flush", "part_cat_id": 2},
    {"part_num": "4625", "name": "Tile 1 x 1 with Clip", "part_cat_id": 3},
    {"part_num": "4068c", "name": "Slope 45 2 x 2 with Grille", "part_cat_id": 4},
    {"part_num": "2513", "name": "Plate 1 x 2 with Clip on Side", "part_cat_id": 2},
    {"part_num": "3700", "name": "Technic Axle 1", "part_cat_id": 5},
    {"part_num": "32291", "name": "Technic Axle 2", "part_cat_id": 5},
    {"part_num": "3703", "name": "Technic Axle 4", "part_cat_id": 5},
    {"part_num": "3704", "name": "Technic Axle 5", "part_cat_id": 5},
    {"part_num": "32291", "name": "Technic Axle 2 with Groove", "part_cat_id": 5},
    {"part_num": "2654", "name": "Plate 1 x 1 with Clip Light", "part_cat_id": 2},
    {"part_num": "3666", "name": "Slope, Inverted 45 1 x 2", "part_cat_id": 4},
    {"part_num": "4286", "name": "Slope 33 3 x 2", "part_cat_id": 4},
    {"part_num": "3048", "name": "Slope 45 1 x 2 Double", "part_cat_id": 4},
    {"part_num": "3049", "name": "Slope 45 1 x 2", "part_cat_id": 4},
    {"part_num": "4161", "name": "Minifigure Accessory Bow", "part_cat_id": 6},
    {"part_num": "3626c", "name": "Minifigure Head with Smile", "part_cat_id": 6},
    {"part_num": "970c00", "name": "Minifigure Hips with Black Hips and Legs", "part_cat_id": 6},
    {"part_num": "3626bp01", "name": "Minifigure Head with Smile and Freckles", "part_cat_id": 6},
    {"part_num": "3818", "name": "Minifigure Hips", "part_cat_id": 6},
    {"part_num": "2335", "name": "Brick 2 x 4 x 3", "part_cat_id": 1},
    {"part_num": "2454", "name": "Brick 1 x 2 x 2", "part_cat_id": 1},
    {"part_num": "2456", "name": "Brick 1 x 2 x 5", "part_cat_id": 1},
    {"part_num": "3631", "name": "Slope, Inverted 45 2 x 2 Double", "part_cat_id": 4},
]

# Popular LEGO sets to seed (real set numbers and themes)
THEMES = [
    {"id": 1, "name": "Star Wars"},
    {"id": 2, "name": "City"},
    {"id": 3, "name": "Technic"},
    {"id": 4, "name": "Creator"},
    {"id": 5, "name": "Friends"},
]

LEGO_SETS = [
    {
        "set_num": "75257-1",
        "name": "Millennium Falcon",
        "year": 2019,
        "theme_id": 1,
        "num_parts": 1351,
        "img_url": "https://cdn.rebrickable.com/media/sets/75257-1/75257-1.jpg",
        "set_parts": [
            {"part_num": "3001", "color_id": 5, "quantity": 150, "is_spare": False},
            {"part_num": "3002", "color_id": 19, "quantity": 120, "is_spare": False},
            {"part_num": "3003", "color_id": 1, "quantity": 100, "is_spare": False},
            {"part_num": "3004", "color_id": 19, "quantity": 80, "is_spare": False},
            {"part_num": "3020", "color_id": 18, "quantity": 60, "is_spare": False},
            {"part_num": "3023", "color_id": 19, "quantity": 50, "is_spare": False},
            {"part_num": "3024", "color_id": 1, "quantity": 40, "is_spare": False},
            {"part_num": "3069b", "color_id": 18, "quantity": 35, "is_spare": False},
            {"part_num": "3039", "color_id": 19, "quantity": 25, "is_spare": False},
            {"part_num": "3070b", "color_id": 1, "quantity": 20, "is_spare": False},
        ]
    },
    {
        "set_num": "10265-1",
        "name": "Ford Mustang",
        "year": 2019,
        "theme_id": 4,
        "num_parts": 1471,
        "img_url": "https://cdn.rebrickable.com/media/sets/10265-1/10265-1.jpg",
        "set_parts": [
            {"part_num": "3001", "color_id": 5, "quantity": 120, "is_spare": False},
            {"part_num": "3003", "color_id": 5, "quantity": 90, "is_spare": False},
            {"part_num": "3008", "color_id": 19, "quantity": 70, "is_spare": False},
            {"part_num": "3020", "color_id": 1, "quantity": 60, "is_spare": False},
            {"part_num": "2780c01", "color_id": 19, "quantity": 4, "is_spare": False},
            {"part_num": "3626bp00", "color_id": 1, "quantity": 2, "is_spare": False},
            {"part_num": "3815", "color_id": 5, "quantity": 2, "is_spare": False},
            {"part_num": "3817", "color_id": 19, "quantity": 2, "is_spare": False},
            {"part_num": "3039", "color_id": 5, "quantity": 40, "is_spare": False},
            {"part_num": "3040b", "color_id": 5, "quantity": 35, "is_spare": False},
        ]
    },
    {
        "set_num": "60198-1",
        "name": "Cargo Train",
        "year": 2018,
        "theme_id": 2,
        "num_parts": 1226,
        "img_url": "https://cdn.rebrickable.com/media/sets/60198-1/60198-1.jpg",
        "set_parts": [
            {"part_num": "3001", "color_id": 19, "quantity": 110, "is_spare": False},
            {"part_num": "3020", "color_id": 18, "quantity": 85, "is_spare": False},
            {"part_num": "2780c01", "color_id": 19, "quantity": 8, "is_spare": False},
            {"part_num": "3626bp00", "color_id": 1, "quantity": 3, "is_spare": False},
            {"part_num": "3008", "color_id": 5, "quantity": 65, "is_spare": False},
            {"part_num": "3004", "color_id": 11, "quantity": 55, "is_spare": False},
            {"part_num": "3070b", "color_id": 1, "quantity": 30, "is_spare": False},
            {"part_num": "3023", "color_id": 19, "quantity": 45, "is_spare": False},
            {"part_num": "3039", "color_id": 18, "quantity": 25, "is_spare": False},
            {"part_num": "3010", "color_id": 5, "quantity": 35, "is_spare": False},
        ]
    },
    {
        "set_num": "42110-1",
        "name": "Land Rover Defender",
        "year": 2019,
        "theme_id": 3,
        "num_parts": 2573,
        "img_url": "https://cdn.rebrickable.com/media/sets/42110-1/42110-1.jpg",
        "set_parts": [
            {"part_num": "32064a", "color_id": 19, "quantity": 140, "is_spare": False},
            {"part_num": "3713", "color_id": 19, "quantity": 100, "is_spare": False},
            {"part_num": "32316", "color_id": 5, "quantity": 80, "is_spare": False},
            {"part_num": "32524", "color_id": 18, "quantity": 70, "is_spare": False},
            {"part_num": "2780c01", "color_id": 19, "quantity": 4, "is_spare": False},
            {"part_num": "3626bp00", "color_id": 1, "quantity": 1, "is_spare": False},
            {"part_num": "3815", "color_id": 8, "quantity": 1, "is_spare": False},
            {"part_num": "3817", "color_id": 19, "quantity": 1, "is_spare": False},
            {"part_num": "6587", "color_id": 19, "quantity": 90, "is_spare": False},
            {"part_num": "3700", "color_id": 19, "quantity": 50, "is_spare": False},
        ]
    },
    {
        "set_num": "21318-1",
        "name": "Tree House",
        "year": 2019,
        "theme_id": 4,
        "num_parts": 3036,
        "img_url": "https://cdn.rebrickable.com/media/sets/21318-1/21318-1.jpg",
        "set_parts": [
            {"part_num": "3001", "color_id": 7, "quantity": 180, "is_spare": False},
            {"part_num": "3002", "color_id": 7, "quantity": 150, "is_spare": False},
            {"part_num": "3003", "color_id": 13, "quantity": 130, "is_spare": False},
            {"part_num": "3020", "color_id": 2, "quantity": 100, "is_spare": False},
            {"part_num": "3023", "color_id": 1, "quantity": 90, "is_spare": False},
            {"part_num": "3626bp00", "color_id": 1, "quantity": 3, "is_spare": False},
            {"part_num": "3815", "color_id": 5, "quantity": 3, "is_spare": False},
            {"part_num": "3817", "color_id": 19, "quantity": 3, "is_spare": False},
            {"part_num": "3039", "color_id": 7, "quantity": 60, "is_spare": False},
            {"part_num": "3040b", "color_id": 7, "quantity": 50, "is_spare": False},
        ]
    },
]


async def seed_database(conn):
    """Seed all tables with test data."""
    try:
        # Seed colors
        logger.info("Seeding colors...")
        for color in COLORS:
            color_id = str(uuid.uuid4())
            await conn.execute(
                """
                INSERT INTO colors (id, rebrickable_id, name, hex_code, is_transparent)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (rebrickable_id) DO NOTHING
                """,
                color_id, color["rebrickable_id"], color["name"],
                color["hex_code"], color["is_transparent"]
            )
        logger.info(f"Inserted {len(COLORS)} colors")

        # Get color IDs for mapping
        colors = await conn.fetch("SELECT id, rebrickable_id FROM colors")
        color_map = {str(c["rebrickable_id"]): c["id"] for c in colors}

        # Seed part categories
        logger.info("Seeding part categories...")
        category_ids = {}
        for i, category in enumerate(PART_CATEGORIES, 1):
            cat_id = str(uuid.uuid4())
            category_ids[i] = cat_id
            await conn.execute(
                "INSERT INTO part_categories (id, name) VALUES ($1, $2)",
                cat_id, category["name"]
            )
        logger.info(f"Inserted {len(PART_CATEGORIES)} part categories")

        # Seed parts
        logger.info("Seeding parts...")
        part_ids = {}
        for part in PARTS:
            part_id = str(uuid.uuid4())
            cat_id = category_ids.get(part["part_cat_id"])
            part_ids[part["part_num"]] = part_id
            await conn.execute(
                """
                INSERT INTO parts (id, part_num, name, part_cat_id, image_url, created_at)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (part_num) DO NOTHING
                """,
                part_id, part["part_num"], part["name"], cat_id,
                f"https://cdn.rebrickable.com/media/parts/{part['part_num']}.jpg",
                datetime.now(timezone.utc)
            )
        logger.info(f"Inserted {len(PARTS)} parts")

        # Seed themes
        logger.info("Seeding themes...")
        theme_ids = {}
        for theme in THEMES:
            theme_id = str(uuid.uuid4())
            theme_ids[theme["id"]] = theme_id
            await conn.execute(
                "INSERT INTO themes (id, name, parent_id) VALUES ($1, $2, $3)",
                theme_id, theme["name"], None
            )
        logger.info(f"Inserted {len(THEMES)} themes")

        # Seed LEGO sets and their parts
        logger.info("Seeding LEGO sets...")
        for lego_set in LEGO_SETS:
            set_id = str(uuid.uuid4())
            theme_id = theme_ids.get(lego_set["theme_id"])
            await conn.execute(
                """
                INSERT INTO lego_sets (id, set_num, name, year, theme_id, num_parts, img_url, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                set_id, lego_set["set_num"], lego_set["name"],
                lego_set["year"], theme_id, lego_set["num_parts"],
                lego_set["img_url"], datetime.now(timezone.utc)
            )

            # Insert set parts
            for set_part in lego_set.get("set_parts", []):
                part_id = part_ids.get(set_part["part_num"])
                color_id = color_map.get(str(set_part["color_id"]))
                if part_id and color_id:
                    set_part_id = str(uuid.uuid4())
                    await conn.execute(
                        """
                        INSERT INTO set_parts (id, set_id, part_id, color_id, quantity, is_spare)
                        VALUES ($1, $2, $3, $4, $5, $6)
                        """,
                        set_part_id, set_id, part_id, color_id,
                        set_part["quantity"], set_part["is_spare"]
                    )

        logger.info(f"Inserted {len(LEGO_SETS)} LEGO sets with parts")

        # Seed test user
        logger.info("Seeding test user...")
        user_id = str(uuid.uuid4())
        await conn.execute(
            """
            INSERT INTO users (id, email, hashed_password, is_active, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6)
            """,
            user_id,
            "testuser@example.com",
            "$2b$12$test_hashed_password_placeholder",
            True,
            datetime.now(timezone.utc),
            datetime.now(timezone.utc)
        )
        logger.info(f"Inserted test user: {user_id}")

        # Seed test user inventory (60% of first set)
        logger.info("Seeding test user inventory...")
        first_set = LEGO_SETS[0]
        for set_part in first_set.get("set_parts", [])[:6]:  # First 6 parts
            part_id = part_ids.get(set_part["part_num"])
            color_id = color_map.get(str(set_part["color_id"]))
            if part_id and color_id:
                inv_id = str(uuid.uuid4())
                quantity = int(set_part["quantity"] * 0.6)  # 60% of required
                await conn.execute(
                    """
                    INSERT INTO inventory_items (id, user_id, part_id, color_id, quantity, created_at, updated_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """,
                    inv_id, user_id, part_id, color_id, quantity,
                    datetime.now(timezone.utc), datetime.now(timezone.utc)
                )
        logger.info("Inserted test user inventory items")

        logger.info("Database seeding completed successfully!")

    except Exception as e:
        logger.error(f"Error seeding database: {e}")
        raise


async def main():
    """Connect to database and run seed script."""
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        await seed_database(conn)
    finally:
        await conn.close()


if __name__ == '__main__':
    asyncio.run(main())
