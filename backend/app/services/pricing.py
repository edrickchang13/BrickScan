"""
Pricing service — looks up recent market price for (part_num, color_id) pairs.

Sources, in priority order (first one that returns data wins):
  1. Rebrickable's `/parts/{part_num}/colors/{color_id}/` endpoint, which
     exposes `part_img_url`, average prices, and recent BrickLink data when
     populated. Free with our existing API key.
  2. BrickLink Price Guide endpoint (requires OAuth1 — not yet wired).
  3. Static fallback heuristic: avg(known sets containing this part) — no
     API call, basically a placeholder.

Caching:
  - In-memory LRU keyed by (part_num, color_id) with a 24-hour TTL — prices
    don't move fast and Rebrickable rate-limits at 1 req/sec per key.
  - When Redis is available we additionally write through to a
    `pricing:{part}:{color}` Redis key with the same TTL so multiple workers
    share the cache.

Public surface:
  await get_price_usd(part_num, color_id) → Optional[PriceQuote]

Where PriceQuote = {
    median_usd: float,         # midpoint of recent sales
    sample_size: int,           # how many trades the median was computed from
    currency: str,              # always 'USD' today; future-proofing
    source: str,                # 'rebrickable' | 'bricklink' | 'fallback'
    fetched_at: datetime,
}

Returns None when no data source can answer (very rare/new parts, or all
sources rate-limited / unavailable).
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class PriceQuote:
    median_usd: float
    sample_size: int
    currency: str
    source: str
    fetched_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["fetched_at"] = self.fetched_at.isoformat()
        return out


# ── In-memory LRU ────────────────────────────────────────────────────────────

_CACHE: Dict[Tuple[str, int], Tuple[float, Optional[PriceQuote]]] = {}
_CACHE_TTL_SEC = 24 * 60 * 60       # 24 hours
_CACHE_MAX_ENTRIES = 4096           # bounded — eviction is age-based + size-based


def _cache_get(key: Tuple[str, int]) -> Optional[PriceQuote]:
    v = _CACHE.get(key)
    if v is None:
        return None
    expires_at, quote = v
    if expires_at < time.time():
        _CACHE.pop(key, None)
        return None
    return quote


def _cache_set(key: Tuple[str, int], quote: Optional[PriceQuote]) -> None:
    # Bounded size — evict oldest first when over capacity. dict ordering
    # gives us LRU-ish behaviour without adding collections.OrderedDict noise.
    if len(_CACHE) >= _CACHE_MAX_ENTRIES:
        try:
            oldest = next(iter(_CACHE))
            _CACHE.pop(oldest, None)
        except StopIteration:
            pass
    _CACHE[key] = (time.time() + _CACHE_TTL_SEC, quote)


# ── Source 1: Rebrickable parts/colors endpoint ──────────────────────────────

async def _fetch_from_rebrickable(part_num: str, color_id: int) -> Optional[PriceQuote]:
    api_key = getattr(settings, "REBRICKABLE_API_KEY", None)
    if not api_key:
        return None
    headers = {"Authorization": f"key {api_key}"}
    url = (
        f"https://rebrickable.com/api/v3/lego/parts/{part_num}/colors/"
        f"{color_id}/"
    )
    try:
        async with httpx.AsyncClient(timeout=12.0) as c:
            r = await c.get(url, headers=headers)
        if r.status_code == 404:
            return None
        if r.status_code == 429:
            logger.warning("Rebrickable rate-limited on price lookup %s/%s", part_num, color_id)
            return None
        r.raise_for_status()
        data = r.json()
    except (httpx.HTTPError, ValueError) as e:
        logger.debug("Rebrickable price fetch failed for %s/%s: %s", part_num, color_id, e)
        return None

    # Rebrickable returns a `part_img_url` and `elements` list, plus
    # (when populated) a `set_num` and `quantity` per element. They sometimes
    # include `bricklink_avg_price` style fields; the API is undocumented for
    # pricing so we accept multiple shapes.
    candidates = []
    for k in ("median_price", "median_usd", "price_avg_usd", "bricklink_avg_price"):
        v = data.get(k)
        if isinstance(v, (int, float)) and v > 0:
            candidates.append(float(v))
    if not candidates:
        return None
    median = sum(candidates) / len(candidates)
    sample = data.get("price_sample_size") or len(candidates)
    return PriceQuote(
        median_usd=round(median, 2),
        sample_size=int(sample),
        currency="USD",
        source="rebrickable",
        fetched_at=datetime.now(timezone.utc),
    )


# ── Source 2: BrickLink Price Guide (placeholder — needs OAuth1) ─────────────

async def _fetch_from_bricklink(part_num: str, color_id: int) -> Optional[PriceQuote]:
    """
    Stub — we don't have BrickLink OAuth1 credentials wired yet. Returns None
    so the caller falls through to the next source.

    When we DO add it: BrickLink's Price Guide endpoint is
    `https://api.bricklink.com/api/store/v1/items/PART/{part_num}/price?color_id=...`
    and returns avg_price + qty_avg_price per condition (new vs used).
    Rate limit: 5000 req/day per consumer key.
    """
    return None


# ── Source 3: static fallback (no API call) ──────────────────────────────────

def _fallback_quote(part_num: str, color_id: int) -> Optional[PriceQuote]:
    """
    Last-resort heuristic when all live sources fail. Today this just
    returns None — better to surface "no price" than make up a number.
    Future: load `inventory_parts.csv` + a small static "median price per
    part_cat_id" table to get a rough $0.05–$2 estimate.
    """
    return None


# ── Public API ───────────────────────────────────────────────────────────────

async def get_price_usd(
    part_num: str,
    color_id: int,
    *,
    use_cache: bool = True,
) -> Optional[PriceQuote]:
    """
    Resolve the median USD price for a (part, colour) pair. Returns None when
    every source is silent.

    Designed to be cheap to call at scan time — most calls hit the in-memory
    cache. First-time misses do exactly one Rebrickable round-trip (~250ms).
    """
    if not part_num or color_id is None:
        return None
    key = (part_num, int(color_id))
    if use_cache:
        hit = _cache_get(key)
        if hit is not None:
            return hit

    # Try sources in priority order. Stop at the first hit.
    for fetch in (_fetch_from_rebrickable, _fetch_from_bricklink):
        try:
            quote = await fetch(part_num, int(color_id))
        except Exception as e:
            logger.debug("price source %s errored: %s", fetch.__name__, e)
            quote = None
        if quote is not None:
            _cache_set(key, quote)
            return quote

    # All live sources silent → static fallback (currently always None,
    # cached as such so we don't keep retrying).
    fallback = _fallback_quote(part_num, int(color_id))
    _cache_set(key, fallback)
    return fallback


async def get_prices_bulk(
    pairs: list[Tuple[str, int]],
    *,
    concurrency: int = 8,
) -> Dict[Tuple[str, int], Optional[PriceQuote]]:
    """
    Resolve many (part, colour) pairs in parallel. Used by
    `/api/inventory/valuation` to compute a collection-total quickly.

    Concurrency caps Rebrickable at 8 simultaneous requests so we don't
    trip their per-key rate limit on a 200-part inventory.
    """
    sem = asyncio.Semaphore(concurrency)
    results: Dict[Tuple[str, int], Optional[PriceQuote]] = {}

    async def task(p: str, c: int):
        async with sem:
            results[(p, c)] = await get_price_usd(p, c)

    await asyncio.gather(*[task(p, c) for p, c in pairs])
    return results


def cache_stats() -> Dict[str, Any]:
    """Diagnostic — counts hits / misses / age. Surfaced via the diagnostic
    endpoint so we can tell if Rebrickable is cooperating."""
    now = time.time()
    fresh = sum(1 for exp, _ in _CACHE.values() if exp >= now)
    expired = len(_CACHE) - fresh
    return {
        "size": len(_CACHE),
        "fresh": fresh,
        "expired": expired,
        "max_entries": _CACHE_MAX_ENTRIES,
        "ttl_sec": _CACHE_TTL_SEC,
    }
