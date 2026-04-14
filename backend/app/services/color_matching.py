"""
Color matching service for mapping vision API color names to Rebrickable colors.

Vision models return color names like "red", "dark grey", "transparent blue"
which need to be mapped to official LEGO color names and IDs in the database.
"""

import logging
from typing import Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# Comprehensive mapping of color name variations to official LEGO color names
COLOR_ALIASES = {
    # Grays and neutrals
    "gray": "Light Bluish Gray",
    "grey": "Light Bluish Gray",
    "light gray": "Light Bluish Gray",
    "light grey": "Light Bluish Gray",
    "dark gray": "Dark Bluish Gray",
    "dark grey": "Dark Bluish Gray",
    "silver": "Light Bluish Gray",
    "light silver": "Light Bluish Gray",
    "dark silver": "Dark Bluish Gray",
    "white": "White",
    "black": "Black",
    "dark charcoal": "Dark Bluish Gray",
    "medium stone gray": "Light Bluish Gray",
    # Standard primary colors
    "red": "Red",
    "bright red": "Bright Red",
    "dark red": "Dark Red",
    "blue": "Blue",
    "bright blue": "Bright Blue",
    "dark blue": "Dark Blue",
    "medium blue": "Medium Blue",
    "yellow": "Yellow",
    "bright yellow": "Bright Yellow",
    "green": "Green",
    "dark green": "Dark Green",
    "bright green": "Bright Green",
    "lime": "Lime",
    "lime green": "Lime",
    "orange": "Orange",
    "dark orange": "Dark Orange",
    "bright orange": "Bright Orange",
    "brown": "Brown",
    "dark brown": "Dark Brown",
    "reddish brown": "Reddish Brown",
    "purple": "Purple",
    "dark purple": "Dark Purple",
    "magenta": "Magenta",
    "pink": "Bright Pink",
    "bright pink": "Bright Pink",
    "light pink": "Light Pink",
    # LEGO specific colors
    "sand green": "Sand Green",
    "sand blue": "Sand Blue",
    "dark tan": "Dark Tan",
    "tan": "Tan",
    "light tan": "Tan",
    "medium tan": "Tan",
    "pearl": "Pearl Light Gold",
    "pearl gold": "Pearl Light Gold",
    "metallic": "Dark Metallic Gray",
    "metallic gray": "Dark Metallic Gray",
    "metallic grey": "Dark Metallic Gray",
    "chrome": "Dark Metallic Gray",
    "gold": "Pearl Light Gold",
    "dark gold": "Dark Tan",
    "silver metallic": "Dark Metallic Gray",
    "copper": "Dark Orange",
    "bronze": "Reddish Brown",
    "maroon": "Dark Red",
    "navy": "Dark Blue",
    "teal": "Medium Azure",
    "aqua": "Medium Azure",
    "cyan": "Medium Azure",
    "azure": "Medium Azure",
    "light azure": "Medium Azure",
    "medium azure": "Medium Azure",
    "dark azure": "Dark Blue",
    "turquoise": "Medium Azure",
    "mint": "Bright Green",
    "light green": "Bright Green",
    "forest green": "Dark Green",
    "olive": "Olive Green",
    "olive green": "Olive Green",
    "khaki": "Tan",
    "beige": "Tan",
    "cream": "Tan",
    "ivory": "White",
    "off-white": "White",
    # Transparent colors
    "transparent": "Trans-Clear",
    "clear": "Trans-Clear",
    "trans-clear": "Trans-Clear",
    "transparent clear": "Trans-Clear",
    "transparent red": "Trans-Red",
    "trans-red": "Trans-Red",
    "transparent blue": "Trans-Dark Blue",
    "trans-blue": "Trans-Dark Blue",
    "transparent dark blue": "Trans-Dark Blue",
    "trans-dark blue": "Trans-Dark Blue",
    "transparent green": "Trans-Green",
    "trans-green": "Trans-Green",
    "transparent yellow": "Trans-Yellow",
    "trans-yellow": "Trans-Yellow",
    "transparent orange": "Trans-Orange",
    "trans-orange": "Trans-Orange",
    "transparent pink": "Trans-Light Pink",
    "trans-pink": "Trans-Light Pink",
    "transparent purple": "Trans-Purple",
    "trans-purple": "Trans-Purple",
    "transparent brown": "Trans-Brown",
    "trans-brown": "Trans-Brown",
    "transparent lime": "Trans-Neon Green",
    "trans-lime": "Trans-Neon Green",
    "transparent light blue": "Trans-Light Blue",
    "trans-light blue": "Trans-Light Blue",
}


def normalize_color_name(raw_color: str) -> str:
    """
    Convert any color name to official LEGO color name.

    Performs case-insensitive lookup in COLOR_ALIASES.
    If no alias match, returns the color in title case.

    Args:
    - raw_color: Raw color name from vision model (e.g., "dark red", "BLUE")

    Returns:
    - Official LEGO color name
    """
    if not raw_color:
        return "Unknown"

    # Normalize input: lowercase and strip whitespace
    normalized = raw_color.lower().strip()

    # Look up in aliases
    if normalized in COLOR_ALIASES:
        return COLOR_ALIASES[normalized]

    # If no exact match, try to find partial matches
    # (e.g., "bright red" if "bright" + "red" are separate)
    for alias, official in COLOR_ALIASES.items():
        if normalized in alias or alias in normalized:
            return official

    # Fall back to title case of original
    return raw_color.title()


async def find_color_id_by_name(
    color_name: str, db: AsyncSession
) -> Optional[int]:
    """
    Look up color ID in database by name.

    Attempts exact match first, then fuzzy matching if no exact match.

    Args:
    - color_name: Official LEGO color name
    - db: AsyncSession for database access

    Returns:
    - Color ID (integer), or None if not found
    """
    if not color_name:
        return None

    try:
        from app.models.part import Color

        # Try exact match first
        result = await db.execute(
            select(Color.id).where(Color.name == color_name)
        )
        color_id = result.scalar_one_or_none()

        if color_id:
            return color_id

        # Try case-insensitive match
        result = await db.execute(
            select(Color.id).where(
                Color.name.ilike(f"%{color_name}%")
            )
        )
        color_id = result.scalar_one_or_none()

        return color_id

    except Exception as e:
        logger.error(f"Error looking up color '{color_name}': {e}")
        return None


def get_color_similarity(name1: str, name2: str) -> float:
    """
    Calculate similarity between two color names (0.0 to 1.0).

    Simple implementation using string matching:
    - 1.0 for exact match
    - 0.75 for substring match (e.g., "red" in "dark red")
    - 0.5 for partial match (e.g., "dar" in "dark")
    - 0.0 for no match

    Args:
    - name1: First color name
    - name2: Second color name

    Returns:
    - Similarity score between 0.0 and 1.0
    """
    if not name1 or not name2:
        return 0.0

    n1 = name1.lower().strip()
    n2 = name2.lower().strip()

    # Exact match
    if n1 == n2:
        return 1.0

    # Substring match (one contains the other)
    if n1 in n2 or n2 in n1:
        return 0.75

    # Partial word match (e.g., "dar" in "dark")
    if len(n1) >= 3 and n1[:3] in n2:
        return 0.5
    if len(n2) >= 3 and n2[:3] in n1:
        return 0.5

    # No match
    return 0.0


def resolve_color_ambiguity(
    color_names: list[str],
) -> str:
    """
    When vision model returns multiple possible colors, pick the best one.

    Uses COLOR_ALIASES to prefer official LEGO colors.

    Args:
    - color_names: List of possible color names

    Returns:
    - Best matching official LEGO color name
    """
    if not color_names:
        return "Unknown"

    if len(color_names) == 1:
        return normalize_color_name(color_names[0])

    # Score each color: prefer if it's in aliases and exact match
    best_color = color_names[0]
    best_score = 0

    for color in color_names:
        normalized = normalize_color_name(color)

        # Score: higher if normalized != color (was in aliases)
        score = 1.0 if normalized != color.title() else 0.5

        if score > best_score:
            best_score = score
            best_color = normalized

    return best_color


def batch_normalize_colors(raw_colors: list[str]) -> dict[str, str]:
    """
    Normalize a batch of raw color names.

    Args:
    - raw_colors: List of raw color names from vision model

    Returns:
    - Dictionary mapping raw colors to official names
    """
    return {raw: normalize_color_name(raw) for raw in raw_colors}
