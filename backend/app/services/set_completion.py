"""
Set-completion engine — given a user's local inventory, compute which
Rebrickable sets they can build (or are close to completing).

Closes the "what can you build with these bricks?" feature gap from
Brickit/Brickify. Pure algorithm — no ML, no GPU, just CSVs + SQL.

Data sources (loaded once at startup, cached in-memory; the bulk CSVs are
~50MB total which is fine to hold in process memory):

  inventories.csv     — (id, version, set_num)        ~38K rows
  inventory_parts.csv — (inventory_id, part_num, color_id, quantity, is_spare)
                        ~1.5M rows
  sets.csv            — (set_num, name, year, theme_id, num_parts, img_url)

For each set we keep only the latest (max version) inventory and aggregate
to (set_num) → list of (part_num, color_id, qty_required) tuples.

At query time we:
  1. Load the user's inventory (part_num, color_id, qty_have)
  2. For each set, score by qty_have / qty_required for shared parts
  3. Buildable = 100% completion; Buildable-Loose = 100% ignoring color
  4. Return top N sets sorted by completion %

Color matching is exact by default; pass `color_match='loose'` to ignore
colour and just match shapes (Brickit's "buildable somehow" mode).
"""
from __future__ import annotations

import csv
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Default location — the bulk CSVs live in backend/data/rebrickable_bulk/.
# Override via REBRICKABLE_BULK_DIR env var if you mount them elsewhere.
import os
DEFAULT_BULK_DIR = Path(
    os.environ.get(
        "REBRICKABLE_BULK_DIR",
        Path(__file__).resolve().parent.parent.parent / "data" / "rebrickable_bulk",
    )
)


@dataclass
class SetSummary:
    set_num: str
    name: str
    year: Optional[int]
    theme_id: Optional[int]
    num_parts: Optional[int]
    img_url: Optional[str]


@dataclass
class SetCompletionResult:
    set_num: str
    name: str
    year: Optional[int]
    theme_id: Optional[int]
    num_parts: int
    img_url: Optional[str]
    # 0.0–1.0 fraction of distinct (part, colour) pairs the user has any
    # quantity of. We use this rather than the absolute-quantity ratio
    # because that's what users intuitively mean by "I can build this".
    distinct_completion: float
    # 0.0–1.0 — sum(min(have, need)) / sum(need). The honest version of
    # completion that respects "I need 4 of these but only have 2".
    quantity_completion: float
    matched_pairs: int
    total_pairs: int
    missing: List[Tuple[str, Optional[int], int]]   # (part_num, color_id, qty_short)


# In-memory caches — loaded lazily on first query
_LOCK = threading.Lock()
_LOADED = False
_SETS_BY_NUM: Dict[str, SetSummary] = {}
# set_num → list of (part_num, color_id, qty_required)
_SET_PARTS: Dict[str, List[Tuple[str, Optional[int], int]]] = {}


def _load_bulk(bulk_dir: Path = DEFAULT_BULK_DIR) -> None:
    """Idempotent loader. Safe to call from multiple threads."""
    global _LOADED, _SETS_BY_NUM, _SET_PARTS
    with _LOCK:
        if _LOADED:
            return
        if not bulk_dir.exists():
            logger.warning("Rebrickable bulk dir missing: %s — set completion disabled", bulk_dir)
            _LOADED = True
            return
        t0 = time.time()

        # 1. Load sets metadata
        sets_path = bulk_dir / "sets.csv"
        if sets_path.exists():
            with open(sets_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    set_num = row.get("set_num", "").strip()
                    if not set_num:
                        continue
                    _SETS_BY_NUM[set_num] = SetSummary(
                        set_num=set_num,
                        name=row.get("name", set_num),
                        year=_int_or_none(row.get("year")),
                        theme_id=_int_or_none(row.get("theme_id")),
                        num_parts=_int_or_none(row.get("num_parts")),
                        img_url=row.get("img_url") or None,
                    )

        # 2. Inventories.csv: (id, version, set_num) → keep only the latest
        # version per set so we don't double-count partial-version inventories.
        inventories_path = bulk_dir / "inventories.csv"
        latest_inv_id_by_set: Dict[str, Tuple[int, int]] = {}  # set_num → (version, inv_id)
        if inventories_path.exists():
            with open(inventories_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    set_num = row.get("set_num", "").strip()
                    inv_id = _int_or_none(row.get("id"))
                    version = _int_or_none(row.get("version")) or 1
                    if not set_num or inv_id is None:
                        continue
                    cur = latest_inv_id_by_set.get(set_num)
                    if cur is None or version > cur[0]:
                        latest_inv_id_by_set[set_num] = (version, inv_id)
        inv_id_to_set = {inv_id: set_num for set_num, (_, inv_id) in latest_inv_id_by_set.items()}

        # 3. inventory_parts.csv: aggregate parts per (latest) inventory_id
        parts_path = bulk_dir / "inventory_parts.csv"
        per_set: Dict[str, List[Tuple[str, Optional[int], int]]] = defaultdict(list)
        if parts_path.exists():
            with open(parts_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    inv_id = _int_or_none(row.get("inventory_id"))
                    set_num = inv_id_to_set.get(inv_id) if inv_id is not None else None
                    if not set_num:
                        continue
                    # Skip spare parts — they're optional extras, not core.
                    if (row.get("is_spare") or "f").lower() in ("t", "true", "1"):
                        continue
                    part_num = row.get("part_num", "").strip()
                    color_id = _int_or_none(row.get("color_id"))
                    qty = _int_or_none(row.get("quantity")) or 1
                    if part_num:
                        per_set[set_num].append((part_num, color_id, qty))
        _SET_PARTS = dict(per_set)
        _LOADED = True
        dt = time.time() - t0
        logger.info(
            "Loaded set-completion catalogue: %d sets, %d sets w/ inventory parts, %.1fs",
            len(_SETS_BY_NUM), len(_SET_PARTS), dt,
        )


def _int_or_none(s) -> Optional[int]:
    if s is None or s == "":
        return None
    try:
        return int(s)
    except (TypeError, ValueError):
        return None


def buildable_sets(
    user_inventory: List[Tuple[str, Optional[int], int]],
    *,
    color_match: str = "exact",  # "exact" | "loose"
    min_completion: float = 0.50,
    limit: int = 50,
    min_set_size: int = 5,
    theme_id: Optional[int] = None,
) -> List[SetCompletionResult]:
    """
    Compute completion % for every set in the catalogue against the given
    inventory. Returns top `limit` sets above `min_completion`.

    Args:
        user_inventory: list of (part_num, color_id, qty_have).  Pass color_id
            as None for color-agnostic entries — those will still match in
            color_match='loose' mode but won't contribute under 'exact'.
        color_match: 'exact' requires (part, colour) pair equality;
            'loose' ignores colour and matches by part_num only.
        min_completion: 0.0–1.0 — drop sets below this distinct_completion.
        limit: max results to return (sorted desc by distinct_completion).
        min_set_size: ignore tiny sets (single-figure polybags etc).
        theme_id: when set, restrict to sets in this theme.

    Empty list when bulk catalogue isn't loaded (CSVs missing).
    """
    _load_bulk()
    if not _SET_PARTS:
        return []

    # Build a lookup over the user's inventory keyed for fast matching
    if color_match == "loose":
        # part_num → total qty across all colours
        by_part: Dict[str, int] = defaultdict(int)
        for pn, _cid, q in user_inventory:
            by_part[pn] += q
        have = {("loose", pn): q for pn, q in by_part.items()}
    else:
        have: Dict[Tuple[str, Optional[int]], int] = defaultdict(int)
        for pn, cid, q in user_inventory:
            have[(pn, cid)] += q

    results: List[SetCompletionResult] = []
    for set_num, parts in _SET_PARTS.items():
        if len(parts) < min_set_size:
            continue
        meta = _SETS_BY_NUM.get(set_num)
        if theme_id is not None and (meta is None or meta.theme_id != theme_id):
            continue

        matched_pairs = 0
        qty_have_sum = 0
        qty_need_sum = 0
        missing: List[Tuple[str, Optional[int], int]] = []
        for part_num, color_id, qty_required in parts:
            qty_need_sum += qty_required
            if color_match == "loose":
                got = have.get(("loose", part_num), 0)
            else:
                got = have.get((part_num, color_id), 0)
            if got > 0:
                matched_pairs += 1
            contributing = min(got, qty_required)
            qty_have_sum += contributing
            short = qty_required - contributing
            if short > 0:
                missing.append((part_num, color_id, short))

        total_pairs = len(parts)
        distinct = matched_pairs / total_pairs if total_pairs else 0.0
        quantity = qty_have_sum / qty_need_sum if qty_need_sum else 0.0
        if distinct < min_completion:
            continue

        results.append(SetCompletionResult(
            set_num=set_num,
            name=meta.name if meta else set_num,
            year=meta.year if meta else None,
            theme_id=meta.theme_id if meta else None,
            num_parts=meta.num_parts if meta and meta.num_parts else total_pairs,
            img_url=meta.img_url if meta else None,
            distinct_completion=distinct,
            quantity_completion=quantity,
            matched_pairs=matched_pairs,
            total_pairs=total_pairs,
            missing=missing[:20],  # cap to keep response sizes reasonable
        ))

    results.sort(key=lambda r: (-r.distinct_completion, -r.quantity_completion, -r.num_parts))
    return results[:limit]


def is_loaded() -> bool:
    """Health-check helper: returns True iff the catalogue was loaded
    successfully (i.e. CSVs were present at startup)."""
    return _LOADED and bool(_SET_PARTS)
