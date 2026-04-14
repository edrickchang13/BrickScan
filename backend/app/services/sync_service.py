"""
Rebrickable data sync service.
Synchronizes LEGO sets and parts from Rebrickable API to database.
Runs periodically (weekly) to pick up new releases.
"""

import asyncio
import httpx
import logging
from datetime import datetime, timedelta
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

REBRICKABLE_API_URL = "https://rebrickable.com/api/v3"


class SyncError(Exception):
    """Raised when sync operations fail"""
    pass


async def sync_new_sets(
    db: AsyncSession,
    rebrickable_api_key: str,
    since_date: Optional[datetime] = None,
) -> dict:
    """
    Fetch sets added since last sync and insert to database.

    If since_date is None, syncs last 30 days.

    Args:
    - db: AsyncSession for database access
    - rebrickable_api_key: Rebrickable API key
    - since_date: Only fetch sets released after this date

    Returns:
    - Dictionary with: count (int), errors (list)

    Raises:
    - SyncError: If API call or database insert fails
    """
    if since_date is None:
        since_date = datetime.utcnow() - timedelta(days=30)

    try:
        from app.models.lego_set import LegoSet
        from sqlalchemy import select

        synced_count = 0
        errors = []

        # Format date for API: YYYY-MM-DD
        date_str = since_date.strftime("%Y-%m-%d")
        url = f"{REBRICKABLE_API_URL}/lego/sets/"
        params = {
            "key": rebrickable_api_key,
            "min_year__gte": since_date.year,
            "ordering": "-year",
            "page_size": 1000,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            page = 1
            while True:
                params["page"] = page

                logger.info(f"Fetching sets page {page} from Rebrickable...")
                response = await client.get(url, params=params)
                response.raise_for_status()

                data = response.json()
                sets_data = data.get("results", [])

                if not sets_data:
                    break

                # Insert each set into database
                for set_data in sets_data:
                    try:
                        set_num = set_data["set_num"]

                        # Check if set already exists
                        existing = await db.execute(
                            select(LegoSet).where(LegoSet.set_num == set_num)
                        )
                        if existing.scalar_one_or_none():
                            continue

                        # Create new set record
                        lego_set = LegoSet(
                            set_num=set_num,
                            name=set_data.get("name", ""),
                            year=set_data.get("year", 0),
                            theme=set_data.get("theme_id", 0),
                            num_parts=set_data.get("num_parts", 0),
                            img_url=set_data.get("set_img_url", ""),
                        )
                        db.add(lego_set)
                        synced_count += 1

                    except Exception as e:
                        errors.append(f"Set {set_data.get('set_num', 'unknown')}: {str(e)}")
                        logger.warning(f"Error syncing set: {e}")

                # Commit batch
                await db.commit()

                # Check for next page
                if not data.get("next"):
                    break

                page += 1
                await asyncio.sleep(0.5)  # Rate limit API calls

        logger.info(f"Synced {synced_count} new sets from Rebrickable")
        return {"count": synced_count, "errors": errors}

    except httpx.HTTPError as e:
        logger.error(f"Rebrickable API error: {e}")
        raise SyncError(f"API call failed: {e}")
    except Exception as e:
        await db.rollback()
        logger.error(f"Sync error: {e}")
        raise SyncError(f"Sync failed: {e}")


async def sync_set_inventory(
    db: AsyncSession,
    rebrickable_api_key: str,
    set_num: str,
) -> bool:
    """
    Sync parts list for a specific set from Rebrickable.

    Called when user looks up a set not yet in our database.

    Args:
    - db: AsyncSession
    - rebrickable_api_key: Rebrickable API key
    - set_num: LEGO set number (e.g., "10307")

    Returns:
    - True if successful, False if set not found

    Raises:
    - SyncError: If API or database error occurs
    """
    try:
        from app.models.inventory_part import InventoryPart
        from sqlalchemy import select

        url = f"{REBRICKABLE_API_URL}/lego/sets/{set_num}/parts/"
        params = {"key": rebrickable_api_key, "page_size": 1000}

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, params=params)

            if response.status_code == 404:
                logger.warning(f"Set {set_num} not found on Rebrickable")
                return False

            response.raise_for_status()
            data = response.json()
            parts_data = data.get("results", [])

            # Insert each part into inventory_parts
            for part_data in parts_data:
                try:
                    # Check if part already in inventory
                    existing = await db.execute(
                        select(InventoryPart).where(
                            (InventoryPart.set_num == set_num)
                            & (InventoryPart.part_num == part_data["part"]["part_num"])
                            & (InventoryPart.color_id == part_data["color"]["id"])
                        )
                    )
                    if existing.scalar_one_or_none():
                        continue

                    # Create inventory part record
                    inv_part = InventoryPart(
                        set_num=set_num,
                        part_num=part_data["part"]["part_num"],
                        color_id=part_data["color"]["id"],
                        quantity=part_data["quantity"],
                        is_spare=part_data.get("is_spare", False),
                    )
                    db.add(inv_part)

                except Exception as e:
                    logger.warning(f"Error syncing part for {set_num}: {e}")

            await db.commit()
            logger.info(f"Synced {len(parts_data)} parts for set {set_num}")
            return True

    except httpx.HTTPError as e:
        logger.error(f"Rebrickable API error for set {set_num}: {e}")
        raise SyncError(f"API call failed: {e}")
    except Exception as e:
        await db.rollback()
        logger.error(f"Sync error for set {set_num}: {e}")
        raise SyncError(f"Sync failed: {e}")


async def ensure_set_exists(
    db: AsyncSession,
    rebrickable_api_key: str,
    set_num: str,
) -> bool:
    """
    Check if set exists in database; if not, fetch and insert it.

    Used to lazily populate the database when users scan sets
    that aren't yet in the system.

    Args:
    - db: AsyncSession
    - rebrickable_api_key: Rebrickable API key
    - set_num: LEGO set number

    Returns:
    - True if set exists in DB (or was successfully added)
    - False if set not found on Rebrickable
    """
    try:
        from app.models.lego_set import LegoSet
        from sqlalchemy import select

        # Check if already in database
        existing = await db.execute(
            select(LegoSet).where(LegoSet.set_num == set_num)
        )
        if existing.scalar_one_or_none():
            return True

        # Fetch from Rebrickable
        url = f"{REBRICKABLE_API_URL}/lego/sets/{set_num}/"
        params = {"key": rebrickable_api_key}

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, params=params)

            if response.status_code == 404:
                logger.warning(f"Set {set_num} not found on Rebrickable")
                return False

            response.raise_for_status()
            set_data = response.json()

            # Create set record
            lego_set = LegoSet(
                set_num=set_num,
                name=set_data.get("name", ""),
                year=set_data.get("year", 0),
                theme=set_data.get("theme_id", 0),
                num_parts=set_data.get("num_parts", 0),
                img_url=set_data.get("set_img_url", ""),
            )
            db.add(lego_set)
            await db.commit()

            logger.info(f"Added set {set_num} from Rebrickable")

            # Also sync its parts
            await sync_set_inventory(db, rebrickable_api_key, set_num)

            return True

    except httpx.HTTPError as e:
        logger.error(f"Rebrickable API error for set {set_num}: {e}")
        raise SyncError(f"API call failed: {e}")
    except Exception as e:
        await db.rollback()
        logger.error(f"Error ensuring set exists: {e}")
        raise SyncError(f"Failed to ensure set exists: {e}")


async def sync_colors(
    db: AsyncSession,
    rebrickable_api_key: str,
) -> int:
    """
    Sync all LEGO colors from Rebrickable.

    Should be run once during initial setup.

    Args:
    - db: AsyncSession
    - rebrickable_api_key: Rebrickable API key

    Returns:
    - Number of colors synced

    Raises:
    - SyncError: If sync fails
    """
    try:
        from app.models.part import Color
        from sqlalchemy import select

        url = f"{REBRICKABLE_API_URL}/lego/colors/"
        params = {"key": rebrickable_api_key, "page_size": 1000}
        synced_count = 0

        async with httpx.AsyncClient(timeout=30.0) as client:
            page = 1
            while True:
                params["page"] = page

                logger.info(f"Fetching colors page {page}...")
                response = await client.get(url, params=params)
                response.raise_for_status()

                data = response.json()
                colors_data = data.get("results", [])

                if not colors_data:
                    break

                for color_data in colors_data:
                    # Check if color exists
                    existing = await db.execute(
                        select(Color).where(Color.id == color_data["id"])
                    )
                    if existing.scalar_one_or_none():
                        continue

                    color = Color(
                        id=color_data["id"],
                        name=color_data.get("name", ""),
                        rgb=color_data.get("rgb", ""),
                    )
                    db.add(color)
                    synced_count += 1

                await db.commit()

                if not data.get("next"):
                    break

                page += 1
                await asyncio.sleep(0.5)

        logger.info(f"Synced {synced_count} colors")
        return synced_count

    except Exception as e:
        await db.rollback()
        logger.error(f"Color sync error: {e}")
        raise SyncError(f"Color sync failed: {e}")
