"""
Collection-analytics aggregations — group the user's local inventory by
theme, year, decade, part-category. Powers the CollectionAnalyticsScreen
in the mobile app.

Why a separate module from set_completion:
  - set_completion answers "what can I build?" (joins inventory ↔ sets ↔ inventory_parts)
  - this module answers "what do I have?" (joins inventory ↔ part_categories
    + per-part appearances in sets ↔ themes)

Data sources:
  parts.csv             — (part_num, name, part_cat_id, part_material)
  part_categories.csv   — (id, name)
  themes.csv            — (id, name, parent_id)            ~600 rows
  sets.csv              — (set_num, name, year, theme_id, num_parts, img_url)
  inventory_parts.csv   — used here only to map part_num → set appearances
                          (via reverse-lookup) so we can aggregate by theme
                          without the user's inventory tracking which set
                          each brick came from.

We compute three aggregations per request:
  - by_part_category   — (cat_id, cat_name, total_quantity)
  - by_theme           — (theme_id, theme_name, total_quantity, distinct_parts)
                         (a brick contributes to a theme if any set in that
                          theme uses that brick — pro-rated when ambiguous)
  - by_year_decade     — (decade, total_quantity, distinct_parts)

The theme/year aggregations are *imputed* — a brick on its own doesn't know
which set it came from. We use Rebrickable inventory_parts to find which
sets use a given (part_num, color_id) and credit each set proportionally.
"""
from __future__ import annotations

import csv
import logging
import os
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_BULK_DIR = Path(
    os.environ.get(
        "REBRICKABLE_BULK_DIR",
        Path(__file__).resolve().parent.parent.parent / "data" / "rebrickable_bulk",
    )
)


@dataclass
class CategoryAggregate:
    cat_id: int
    cat_name: str
    total_quantity: int
    distinct_parts: int


@dataclass
class ThemeAggregate:
    theme_id: int
    theme_name: str
    total_quantity: int
    distinct_parts: int


@dataclass
class DecadeAggregate:
    decade: int                  # e.g. 1990, 2000, 2010
    total_quantity: int
    distinct_parts: int


_LOCK = threading.Lock()
_LOADED = False
_PART_CAT: Dict[str, Tuple[int, str]] = {}        # part_num → (cat_id, cat_name)
_THEMES_BY_ID: Dict[int, str] = {}                # theme_id → theme_name
# part_num → set of (set_num) that use this part — for theme/year credit
_PART_TO_SETS: Dict[str, set] = {}
# set_num → (theme_id, year)
_SET_META: Dict[str, Tuple[Optional[int], Optional[int]]] = {}


def _int_or_none(s) -> Optional[int]:
    if s is None or s == "":
        return None
    try:
        return int(s)
    except (TypeError, ValueError):
        return None


def _root_theme(theme_id: int, themes_with_parent: Dict[int, Optional[int]]) -> int:
    """Walk theme parents to the root theme — Brickify-style "Star Wars"
    grouping rather than fine sub-themes ("Star Wars Episode 1", etc).
    Caps at 8 hops for safety against cycles in the data."""
    cur = theme_id
    for _ in range(8):
        parent = themes_with_parent.get(cur)
        if parent is None:
            return cur
        cur = parent
    return cur


def _load_bulk(bulk_dir: Path = DEFAULT_BULK_DIR) -> None:
    global _LOADED, _PART_CAT, _THEMES_BY_ID, _PART_TO_SETS, _SET_META
    with _LOCK:
        if _LOADED:
            return
        if not bulk_dir.exists():
            logger.warning(
                "Rebrickable bulk dir missing: %s — analytics disabled", bulk_dir,
            )
            _LOADED = True
            return
        t0 = time.time()

        # 1. parts.csv → part_num → cat_id (cat_name resolved via part_categories)
        part_to_cat_id: Dict[str, int] = {}
        if (bulk_dir / "parts.csv").exists():
            with open(bulk_dir / "parts.csv", newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    pn = row.get("part_num", "").strip()
                    cid = _int_or_none(row.get("part_cat_id"))
                    if pn and cid is not None:
                        part_to_cat_id[pn] = cid

        cat_id_to_name: Dict[int, str] = {}
        if (bulk_dir / "part_categories.csv").exists():
            with open(bulk_dir / "part_categories.csv", newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    cid = _int_or_none(row.get("id"))
                    if cid is not None:
                        cat_id_to_name[cid] = row.get("name", f"Category {cid}")

        for pn, cid in part_to_cat_id.items():
            _PART_CAT[pn] = (cid, cat_id_to_name.get(cid, f"Category {cid}"))

        # 2. themes.csv → resolve to root themes (collapse sub-themes)
        themes_with_parent: Dict[int, Optional[int]] = {}
        themes_raw_names: Dict[int, str] = {}
        if (bulk_dir / "themes.csv").exists():
            with open(bulk_dir / "themes.csv", newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    tid = _int_or_none(row.get("id"))
                    if tid is None:
                        continue
                    themes_with_parent[tid] = _int_or_none(row.get("parent_id"))
                    themes_raw_names[tid] = row.get("name", f"Theme {tid}")
        # Build root-theme map: theme_id → root theme name (for grouping)
        for tid, _name in themes_raw_names.items():
            root = _root_theme(tid, themes_with_parent)
            _THEMES_BY_ID[tid] = themes_raw_names.get(root, themes_raw_names.get(tid, f"Theme {root}"))

        # 3. sets.csv → set_num → (root_theme_id, year)
        if (bulk_dir / "sets.csv").exists():
            with open(bulk_dir / "sets.csv", newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    set_num = row.get("set_num", "").strip()
                    if not set_num:
                        continue
                    tid = _int_or_none(row.get("theme_id"))
                    root_tid = _root_theme(tid, themes_with_parent) if tid is not None else None
                    _SET_META[set_num] = (root_tid, _int_or_none(row.get("year")))

        # 4. inventory_parts.csv → part_num → set_num set
        # Only keep latest inventory per set (otherwise a re-release inflates counts).
        inv_to_set: Dict[int, str] = {}
        if (bulk_dir / "inventories.csv").exists():
            latest: Dict[str, Tuple[int, int]] = {}
            with open(bulk_dir / "inventories.csv", newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    inv_id = _int_or_none(row.get("id"))
                    set_num = row.get("set_num", "").strip()
                    version = _int_or_none(row.get("version")) or 1
                    if inv_id is None or not set_num:
                        continue
                    cur = latest.get(set_num)
                    if cur is None or version > cur[0]:
                        latest[set_num] = (version, inv_id)
            inv_to_set = {iid: sn for sn, (_, iid) in latest.items()}

        if (bulk_dir / "inventory_parts.csv").exists():
            with open(bulk_dir / "inventory_parts.csv", newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    inv_id = _int_or_none(row.get("inventory_id"))
                    if inv_id is None:
                        continue
                    set_num = inv_to_set.get(inv_id)
                    if not set_num:
                        continue
                    pn = row.get("part_num", "").strip()
                    if not pn:
                        continue
                    _PART_TO_SETS.setdefault(pn, set()).add(set_num)

        _LOADED = True
        dt = time.time() - t0
        logger.info(
            "Loaded analytics catalogue: %d parts→cat, %d themes, %d sets, %d parts→sets, %.1fs",
            len(_PART_CAT), len(_THEMES_BY_ID), len(_SET_META), len(_PART_TO_SETS), dt,
        )


def aggregate_inventory(
    user_inventory: List[Tuple[str, Optional[int], int]],
) -> Dict[str, list]:
    """
    Compute the three breakdowns for a user's local inventory.

    Theme/year aggregations are imputed: if a brick appears in N sets (across
    M themes), we credit each theme with quantity / M to avoid double-counting.

    Returns:
        {
          "by_part_category": [CategoryAggregate, ...],
          "by_theme":         [ThemeAggregate, ...],
          "by_year_decade":   [DecadeAggregate, ...],
        }
    """
    _load_bulk()

    # --- by_part_category --------------------------------------------------
    cat_total: Dict[int, int] = defaultdict(int)
    cat_distinct: Dict[int, int] = defaultdict(int)
    cat_names: Dict[int, str] = {}
    for pn, _cid, qty in user_inventory:
        meta = _PART_CAT.get(pn)
        if not meta:
            continue
        cat_id, cat_name = meta
        cat_total[cat_id] += qty
        cat_distinct[cat_id] += 1
        cat_names[cat_id] = cat_name
    by_cat = [
        CategoryAggregate(
            cat_id=cid, cat_name=cat_names[cid],
            total_quantity=cat_total[cid], distinct_parts=cat_distinct[cid],
        )
        for cid in cat_total
    ]
    by_cat.sort(key=lambda x: -x.total_quantity)

    # --- by_theme + by_decade ---------------------------------------------
    theme_total: Dict[int, float] = defaultdict(float)
    theme_distinct: Dict[int, int] = defaultdict(int)
    decade_total: Dict[int, float] = defaultdict(float)
    decade_distinct: Dict[int, int] = defaultdict(int)

    for pn, _cid, qty in user_inventory:
        sets_for_part = _PART_TO_SETS.get(pn)
        if not sets_for_part:
            continue
        # Crediting strategy: each set the part appears in gets qty / N share.
        # That keeps totals honest even when a single brick is shared across
        # 200 sets (e.g. a 2x4 brick).
        share = qty / len(sets_for_part)

        themes_seen_for_this_part: set = set()
        decades_seen_for_this_part: set = set()
        for set_num in sets_for_part:
            meta = _SET_META.get(set_num)
            if not meta:
                continue
            tid, year = meta
            if tid is not None:
                theme_total[tid] += share
                themes_seen_for_this_part.add(tid)
            if year is not None:
                decade = (year // 10) * 10
                decade_total[decade] += share
                decades_seen_for_this_part.add(decade)
        for tid in themes_seen_for_this_part:
            theme_distinct[tid] += 1
        for decade in decades_seen_for_this_part:
            decade_distinct[decade] += 1

    by_theme = [
        ThemeAggregate(
            theme_id=tid,
            theme_name=_THEMES_BY_ID.get(tid, f"Theme {tid}"),
            total_quantity=int(round(theme_total[tid])),
            distinct_parts=theme_distinct[tid],
        )
        for tid in theme_total
    ]
    by_theme.sort(key=lambda x: -x.total_quantity)

    by_decade = [
        DecadeAggregate(
            decade=d,
            total_quantity=int(round(decade_total[d])),
            distinct_parts=decade_distinct[d],
        )
        for d in decade_total
    ]
    by_decade.sort(key=lambda x: x.decade)

    return {
        "by_part_category": by_cat,
        "by_theme": by_theme,
        "by_year_decade": by_decade,
    }


def is_loaded() -> bool:
    return _LOADED and bool(_PART_CAT)
