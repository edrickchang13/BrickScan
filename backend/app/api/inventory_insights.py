"""
Inventory-insights API — closes Brickify's "what can I build" + "what
do I have, by theme/year" feature gaps.

Two endpoints:
  GET  /api/inventory/buildable-sets    → top sets the user can complete
  GET  /api/inventory/analytics         → category / theme / decade aggregates

Both read from the user's local SQLite inventory and join in-process against
the Rebrickable bulk CSVs (loaded once into memory at startup of the
underlying service modules).
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.local_inventory.database import get_local_db
from app.local_inventory.models import LocalInventoryPart
from app.services import set_completion, inventory_analytics

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/inventory", tags=["inventory-insights"])


# ── Schema ──────────────────────────────────────────────────────────────────

class BuildableMissingPart(BaseModel):
    part_num: str
    color_id: Optional[int]
    quantity_short: int


class BuildableSet(BaseModel):
    set_num: str
    name: str
    year: Optional[int]
    theme_id: Optional[int]
    num_parts: int
    img_url: Optional[str]
    distinct_completion: float
    quantity_completion: float
    matched_pairs: int
    total_pairs: int
    missing: List[BuildableMissingPart]


class BuildableSetsResponse(BaseModel):
    catalog_loaded: bool
    color_match: str
    sets: List[BuildableSet]


class CategoryAgg(BaseModel):
    cat_id: int
    cat_name: str
    total_quantity: int
    distinct_parts: int


class ThemeAgg(BaseModel):
    theme_id: int
    theme_name: str
    total_quantity: int
    distinct_parts: int


class DecadeAgg(BaseModel):
    decade: int
    total_quantity: int
    distinct_parts: int


class AnalyticsResponse(BaseModel):
    catalog_loaded: bool
    total_quantity: int
    distinct_parts: int
    by_part_category: List[CategoryAgg]
    by_theme: List[ThemeAgg]
    by_year_decade: List[DecadeAgg]


# ── Helpers ─────────────────────────────────────────────────────────────────

def _load_user_inventory(db: Session) -> List[Tuple[str, Optional[int], int]]:
    """Pull every (part_num, color_id, quantity) row from the user's local
    inventory into a flat list. Stays small (<10K parts in practice)."""
    rows = (
        db.query(
            LocalInventoryPart.part_num,
            LocalInventoryPart.color_id,
            LocalInventoryPart.quantity,
        )
        .all()
    )
    return [(r[0], r[1], r[2] or 0) for r in rows]


# ── Endpoints ───────────────────────────────────────────────────────────────

@router.get("/buildable-sets", response_model=BuildableSetsResponse)
def get_buildable_sets(
    color_match: str = Query("loose", regex="^(exact|loose)$"),
    min_completion: float = Query(0.50, ge=0.0, le=1.0),
    limit: int = Query(50, ge=1, le=200),
    min_set_size: int = Query(5, ge=1),
    theme_id: Optional[int] = None,
    db: Session = Depends(get_local_db),
):
    """
    Compute completion-percent of every Rebrickable set against the user's
    local inventory and return the top `limit` sets above `min_completion`.

    Defaults to color_match='loose' (match by shape only) which mirrors what
    most users mean by "what can I build" — they're happy to swap a red 2x4
    for a blue one. Pass color_match='exact' for strict colour matching.
    """
    inv = _load_user_inventory(db)
    raw = set_completion.buildable_sets(
        inv,
        color_match=color_match,
        min_completion=min_completion,
        limit=limit,
        min_set_size=min_set_size,
        theme_id=theme_id,
    )
    sets = [
        BuildableSet(
            set_num=r.set_num,
            name=r.name,
            year=r.year,
            theme_id=r.theme_id,
            num_parts=r.num_parts,
            img_url=r.img_url,
            distinct_completion=r.distinct_completion,
            quantity_completion=r.quantity_completion,
            matched_pairs=r.matched_pairs,
            total_pairs=r.total_pairs,
            missing=[
                BuildableMissingPart(part_num=pn, color_id=cid, quantity_short=qs)
                for (pn, cid, qs) in r.missing
            ],
        )
        for r in raw
    ]
    return BuildableSetsResponse(
        catalog_loaded=set_completion.is_loaded(),
        color_match=color_match,
        sets=sets,
    )


@router.get("/analytics", response_model=AnalyticsResponse)
def get_inventory_analytics(db: Session = Depends(get_local_db)):
    """
    Three breakdowns of the user's collection:
      - by_part_category  (Bricks, Plates, Tiles, Minifig accessories, ...)
      - by_theme          (Star Wars, Technic, City, ...)
      - by_year_decade    (1990s, 2000s, ...)

    Theme + year aggregations are imputed because a brick on its own doesn't
    know which set it came from — see inventory_analytics.aggregate_inventory
    for the credit-distribution algorithm.
    """
    inv = _load_user_inventory(db)
    total_qty = sum(q for _, _, q in inv)
    distinct = len({(p, c) for p, c, _ in inv})
    agg = inventory_analytics.aggregate_inventory(inv)
    return AnalyticsResponse(
        catalog_loaded=inventory_analytics.is_loaded(),
        total_quantity=total_qty,
        distinct_parts=distinct,
        by_part_category=[
            CategoryAgg(cat_id=c.cat_id, cat_name=c.cat_name,
                        total_quantity=c.total_quantity, distinct_parts=c.distinct_parts)
            for c in agg["by_part_category"]
        ],
        by_theme=[
            ThemeAgg(theme_id=t.theme_id, theme_name=t.theme_name,
                     total_quantity=t.total_quantity, distinct_parts=t.distinct_parts)
            for t in agg["by_theme"]
        ],
        by_year_decade=[
            DecadeAgg(decade=d.decade,
                      total_quantity=d.total_quantity, distinct_parts=d.distinct_parts)
            for d in agg["by_year_decade"]
        ],
    )
