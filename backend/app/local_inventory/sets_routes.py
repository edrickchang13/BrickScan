"""
Sets search and build-check routes backed by Rebrickable public CSV data.

Data is downloaded ONCE from cdn.rebrickable.com on first use and cached in
~/brickscan_sets.db (persisted via Docker volume across restarts).

No API key required — Rebrickable publishes full database dumps publicly.

Endpoints:
  GET  /api/local-inventory/sets          - Search / browse sets
  GET  /api/local-inventory/sets/status   - Download progress
  GET  /api/local-inventory/sets/{num}    - Set detail + parts list
  POST /api/local-inventory/builds/check  - Compare inventory vs set
"""

import os
import gzip
import csv
import io
import json
import sqlite3
import asyncio
import logging
from pathlib import Path
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from fastapi import Depends

from app.local_inventory.database import get_local_db
from app.local_inventory.models import LocalInventoryPart

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/local-inventory",
    tags=["sets"],
)

# ── Paths ────────────────────────────────────────────────────────────────────

DATA_DIR = Path(os.path.expanduser("~"))
SETS_DB_PATH = DATA_DIR / "brickscan_sets.db"
MODELS_DIR = Path(os.path.expanduser("~")) / "brickscan" / "ml" / "models"
CDN = "https://cdn.rebrickable.com/media/downloads"

# ── Download state ────────────────────────────────────────────────────────────

_download_lock = asyncio.Lock()
_status = {
    "sets_ready": False,
    "parts_ready": False,
    "sets_count": 0,
    "parts_count": 0,
    "downloading": False,
    "error": None,
}


# ── Part name lookup (from our generated part_labels.json) ───────────────────

_part_names: dict[str, str] = {}

def _load_part_names():
    """Load part number → name from our generated JSON."""
    global _part_names
    if _part_names:
        return
    # Try several locations
    candidates = [
        Path("/app/models/part_names.json"),
        MODELS_DIR / "part_names.json",
        DATA_DIR / "brickscan_part_names.json",
    ]
    for path in candidates:
        if path.exists():
            try:
                with open(path) as f:
                    _part_names = json.load(f)
                logger.info(f"Loaded {len(_part_names)} part names from {path}")
                return
            except Exception as e:
                logger.warning(f"Could not load part names from {path}: {e}")

def _part_name(part_num: str) -> str:
    if not _part_names:
        _load_part_names()
    return _part_names.get(part_num, f"Part {part_num}")


# ── SQLite helpers ────────────────────────────────────────────────────────────

def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(SETS_DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def _init_schema(conn: sqlite3.Connection):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sets (
            set_num   TEXT PRIMARY KEY,
            name      TEXT,
            year      INTEGER,
            theme_id  TEXT,
            theme_name TEXT,
            num_parts INTEGER,
            img_url   TEXT
        );
        CREATE TABLE IF NOT EXISTS themes (
            id        TEXT PRIMARY KEY,
            name      TEXT,
            parent_id TEXT
        );
        CREATE TABLE IF NOT EXISTS inventories (
            id      TEXT PRIMARY KEY,
            set_num TEXT,
            version INTEGER
        );
        CREATE TABLE IF NOT EXISTS inventory_parts (
            inventory_id TEXT,
            part_num     TEXT,
            color_id     TEXT,
            quantity     INTEGER,
            is_spare     TEXT,
            PRIMARY KEY (inventory_id, part_num, color_id)
        );
        CREATE TABLE IF NOT EXISTS colors (
            id   TEXT PRIMARY KEY,
            name TEXT,
            rgb  TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_sets_name   ON sets(name);
        CREATE INDEX IF NOT EXISTS idx_sets_year   ON sets(year DESC);
        CREATE INDEX IF NOT EXISTS idx_inv_set     ON inventories(set_num);
        CREATE INDEX IF NOT EXISTS idx_iparts_inv  ON inventory_parts(inventory_id);
    """)
    conn.commit()


# ── Downloader ────────────────────────────────────────────────────────────────

async def _fetch_gz(url: str, timeout: float = 120.0) -> str:
    """Download a gzipped CSV and return as decoded string."""
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()
    return gzip.decompress(resp.content).decode("utf-8", errors="replace")


async def _ensure_sets_ready():
    """Download themes, colors, sets CSVs if needed. Fast (~5 MB total)."""
    global _status
    if _status["sets_ready"]:
        return

    async with _download_lock:
        if _status["sets_ready"]:
            return

        _status["downloading"] = True
        _status["error"] = None

        try:
            conn = _get_conn()
            _init_schema(conn)

            # Check if already populated
            count = conn.execute("SELECT COUNT(*) FROM sets").fetchone()[0]
            if count > 0:
                _status["sets_ready"] = True
                _status["sets_count"] = count
                conn.close()
                return

            logger.info("Downloading Rebrickable public data (themes, colors, sets)…")

            # 1. Themes
            text = await _fetch_gz(f"{CDN}/themes.csv.gz")
            reader = csv.DictReader(io.StringIO(text))
            rows = [(r["id"], r["name"], r.get("parent_id", "")) for r in reader]
            conn.executemany("INSERT OR IGNORE INTO themes VALUES (?,?,?)", rows)
            conn.commit()
            logger.info(f"  themes: {len(rows)}")

            # Build quick theme lookup
            theme_map = {r[0]: r[1] for r in rows}

            # 2. Colors
            text = await _fetch_gz(f"{CDN}/colors.csv.gz")
            reader = csv.DictReader(io.StringIO(text))
            rows = [(r["id"], r["name"], r.get("rgb", "")) for r in reader]
            conn.executemany("INSERT OR IGNORE INTO colors VALUES (?,?,?)", rows)
            conn.commit()
            logger.info(f"  colors: {len(rows)}")

            # 3. Sets
            text = await _fetch_gz(f"{CDN}/sets.csv.gz", timeout=180.0)
            reader = csv.DictReader(io.StringIO(text))
            batch = []
            for r in reader:
                tid = r.get("theme_id", "")
                tname = theme_map.get(tid, "")
                batch.append((
                    r["set_num"],
                    r["name"],
                    int(r["year"]) if r.get("year") else None,
                    tid,
                    tname,
                    int(r["num_parts"]) if r.get("num_parts") else 0,
                    r.get("img_url", ""),
                ))
            conn.executemany("INSERT OR IGNORE INTO sets VALUES (?,?,?,?,?,?,?)", batch)
            conn.commit()
            conn.close()

            _status["sets_ready"] = True
            _status["sets_count"] = len(batch)
            logger.info(f"  sets: {len(batch)} — sets data ready!")

        except Exception as e:
            _status["error"] = str(e)
            logger.error(f"Failed to load sets data: {e}")
            raise
        finally:
            _status["downloading"] = False


async def _ensure_parts_ready():
    """Download inventories + inventory_parts CSVs if needed. Larger (~40 MB compressed)."""
    global _status
    if _status["parts_ready"]:
        return

    # Ensure sets (and schema) are ready first
    await _ensure_sets_ready()

    async with _download_lock:
        if _status["parts_ready"]:
            return

        _status["downloading"] = True
        _status["error"] = None

        try:
            conn = _get_conn()
            count = conn.execute("SELECT COUNT(*) FROM inventories").fetchone()[0]
            if count > 0:
                parts_count = conn.execute("SELECT COUNT(*) FROM inventory_parts").fetchone()[0]
                _status["parts_ready"] = True
                _status["parts_count"] = parts_count
                conn.close()
                return

            logger.info("Downloading Rebrickable inventories + parts (large, one-time)…")

            # 4. Inventories
            text = await _fetch_gz(f"{CDN}/inventories.csv.gz", timeout=120.0)
            reader = csv.DictReader(io.StringIO(text))
            rows = [(r["id"], r["set_num"], int(r.get("version", 1))) for r in reader]
            conn.executemany("INSERT OR IGNORE INTO inventories VALUES (?,?,?)", rows)
            conn.commit()
            logger.info(f"  inventories: {len(rows)}")

            # 5. Inventory parts — stream in batches
            text = await _fetch_gz(f"{CDN}/inventory_parts.csv.gz", timeout=600.0)
            reader = csv.DictReader(io.StringIO(text))

            batch, total = [], 0
            BATCH_SIZE = 20_000
            for r in reader:
                batch.append((
                    r["inventory_id"],
                    r["part_num"],
                    r["color_id"],
                    int(r["quantity"]),
                    r.get("is_spare", "f"),
                ))
                if len(batch) >= BATCH_SIZE:
                    conn.executemany(
                        "INSERT OR IGNORE INTO inventory_parts VALUES (?,?,?,?,?)", batch
                    )
                    conn.commit()
                    total += len(batch)
                    batch = []

            if batch:
                conn.executemany(
                    "INSERT OR IGNORE INTO inventory_parts VALUES (?,?,?,?,?)", batch
                )
                conn.commit()
                total += len(batch)

            conn.close()
            _status["parts_ready"] = True
            _status["parts_count"] = total
            logger.info(f"  inventory_parts: {total} — full parts data ready!")

        except Exception as e:
            _status["error"] = str(e)
            logger.error(f"Failed to load parts data: {e}")
            raise
        finally:
            _status["downloading"] = False


# ── Background task ───────────────────────────────────────────────────────────

async def _background_download():
    """Called at startup to pre-cache everything."""
    try:
        await _ensure_sets_ready()
        await _ensure_parts_ready()
    except Exception as e:
        logger.error(f"Background download failed: {e}")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/sets/status")
async def sets_status(background_tasks: BackgroundTasks = None):
    """Return current download progress; trigger download on first call."""
    if not _status["sets_ready"] and not _status["downloading"]:
        # Kick off the download automatically so the client doesn't have to
        # call /sets first (the old chicken-and-egg bug).
        if background_tasks:
            background_tasks.add_task(_background_download)
        else:
            asyncio.create_task(_background_download())
    return _status


@router.get("/sets")
async def search_sets(
    q: str = "",
    theme: str = "",
    limit: int = 24,
    background_tasks: BackgroundTasks = None,
):
    """
    Search LEGO sets or browse by theme.

    Returns up to `limit` sets ordered by year descending.
    On first call, triggers background download of Rebrickable data.
    """
    if not _status["sets_ready"]:
        if background_tasks:
            background_tasks.add_task(_ensure_sets_ready)
        # Start it now (will return quickly if already downloading)
        asyncio.create_task(_ensure_sets_ready())
        # Return empty while loading
        return []

    conn = _get_conn()
    try:
        if q.strip():
            rows = conn.execute("""
                SELECT set_num, name, year, theme_name, num_parts, img_url
                FROM sets
                WHERE name LIKE ? OR set_num LIKE ?
                ORDER BY year DESC
                LIMIT ?
            """, (f"%{q}%", f"%{q}%", limit)).fetchall()
        elif theme and theme.lower() not in ("", "all"):
            rows = conn.execute("""
                SELECT set_num, name, year, theme_name, num_parts, img_url
                FROM sets
                WHERE theme_name LIKE ?
                ORDER BY year DESC
                LIMIT ?
            """, (f"%{theme}%", limit)).fetchall()
        else:
            # Default: recent sets
            rows = conn.execute("""
                SELECT set_num, name, year, theme_name, num_parts, img_url
                FROM sets
                ORDER BY year DESC
                LIMIT ?
            """, (limit,)).fetchall()

        return [
            {
                "setNum": r["set_num"],
                "name": r["name"],
                "year": r["year"],
                "theme": r["theme_name"] or "Other",
                "numParts": r["num_parts"],
                "imageUrl": r["img_url"] or "",
            }
            for r in rows
        ]
    finally:
        conn.close()


@router.get("/sets/{set_num}")
async def get_set(set_num: str, background_tasks: BackgroundTasks = None):
    """Get a specific set with its full parts list."""
    await _ensure_sets_ready()

    conn = _get_conn()
    try:
        s = conn.execute(
            "SELECT * FROM sets WHERE set_num = ?", (set_num,)
        ).fetchone()
    finally:
        conn.close()

    if not s:
        raise HTTPException(status_code=404, detail=f"Set '{set_num}' not found")

    # Try to get parts — trigger background download if not ready
    parts = []
    if _status["parts_ready"]:
        conn = _get_conn()
        try:
            inv = conn.execute(
                "SELECT id FROM inventories WHERE set_num = ? ORDER BY version DESC LIMIT 1",
                (set_num,)
            ).fetchone()

            if inv:
                rows = conn.execute("""
                    SELECT ip.part_num, ip.color_id, ip.quantity, ip.is_spare,
                           c.name AS color_name, c.rgb AS color_rgb
                    FROM inventory_parts ip
                    LEFT JOIN colors c ON c.id = ip.color_id
                    WHERE ip.inventory_id = ? AND LOWER(ip.is_spare) != 'true'
                    ORDER BY ip.part_num
                """, (inv["id"],)).fetchall()

                parts = [
                    {
                        "partNum": r["part_num"],
                        "partName": _part_name(r["part_num"]),
                        "colorId": r["color_id"],
                        "colorName": r["color_name"] or "Unknown",
                        "colorHex": f"#{r['color_rgb']}" if r["color_rgb"] else "#cccccc",
                        "quantity": r["quantity"],
                        "imageUrl": f"https://img.bricklink.com/ItemImage/PN/11/{r['part_num']}.png",
                    }
                    for r in rows
                ]
        finally:
            conn.close()
    else:
        # Trigger background parts download
        if background_tasks:
            background_tasks.add_task(_ensure_parts_ready)
        else:
            asyncio.create_task(_ensure_parts_ready())

    return {
        "setNum": s["set_num"],
        "name": s["name"],
        "year": s["year"],
        "theme": s["theme_name"] or "Other",
        "numParts": s["num_parts"],
        "imageUrl": s["img_url"] or "",
        "parts": parts,
        "partsLoading": not _status["parts_ready"],
    }


@router.post("/builds/check")
async def build_check(
    body: dict,
    db: Session = Depends(get_local_db),
    background_tasks: BackgroundTasks = None,
):
    """
    Compare user's local inventory vs parts required to build a LEGO set.

    Returns % complete, have/missing part lists.
    """
    set_num = body.get("setNum", "").strip()
    if not set_num:
        raise HTTPException(status_code=400, detail="setNum is required")

    await _ensure_sets_ready()

    if not _status["parts_ready"]:
        if background_tasks:
            background_tasks.add_task(_ensure_parts_ready)
        else:
            asyncio.create_task(_ensure_parts_ready())
        raise HTTPException(
            status_code=503,
            detail="Parts data is still downloading. Please try again in a moment.",
        )

    conn = _get_conn()
    try:
        s = conn.execute(
            "SELECT set_num, name, num_parts FROM sets WHERE set_num = ?",
            (set_num,)
        ).fetchone()
        if not s:
            raise HTTPException(status_code=404, detail=f"Set '{set_num}' not found")

        inv = conn.execute(
            "SELECT id FROM inventories WHERE set_num = ? ORDER BY version DESC LIMIT 1",
            (set_num,)
        ).fetchone()
        if not inv:
            raise HTTPException(status_code=404, detail="No parts data for this set")

        set_parts = conn.execute("""
            SELECT ip.part_num, ip.quantity,
                   c.name AS color_name, c.rgb AS color_rgb
            FROM inventory_parts ip
            LEFT JOIN colors c ON c.id = ip.color_id
            WHERE ip.inventory_id = ? AND LOWER(ip.is_spare) != 'true'
        """, (inv["id"],)).fetchall()
    finally:
        conn.close()

    # Load local inventory (all parts, summed by part_num)
    inventory_rows = db.query(LocalInventoryPart).all()
    owned: dict[str, int] = {}
    for item in inventory_rows:
        owned[item.part_num] = owned.get(item.part_num, 0) + item.quantity

    have_parts, missing_parts = [], []
    total_needed = 0
    total_have = 0

    for sp in set_parts:
        needed = sp["quantity"]
        total_needed += needed
        you_have = min(owned.get(sp["part_num"], 0), needed)
        total_have += you_have

        color_hex = f"#{sp['color_rgb']}" if sp["color_rgb"] else "#cccccc"
        entry = {
            "partNum": sp["part_num"],
            "partName": _part_name(sp["part_num"]),
            "colorName": sp["color_name"] or "Unknown",
            "colorHex": color_hex,
            "imageUrl": f"https://img.bricklink.com/ItemImage/PN/11/{sp['part_num']}.png",
        }

        if you_have >= needed:
            have_parts.append({**entry, "quantity": needed})
        else:
            missing_parts.append({
                **entry,
                "quantityNeeded": needed - you_have,
                "quantityHave": you_have,
            })

    pct = round(total_have / total_needed * 100) if total_needed > 0 else 0

    return {
        "setNum": set_num,
        "setName": s["name"],
        "percentComplete": pct,
        "have": total_have,
        "total": total_needed,
        "missing": total_needed - total_have,
        "haveParts": have_parts,
        "missingParts": missing_parts,
    }
