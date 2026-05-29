"""
Phase 4 live-scan TELEMETRY ingest route — the server-side sink for the
OUTER-loop (real device) closed-loop eval.

The closed-loop eval is two loops sharing one telemetry schema
(`brickscan.livescan.telemetry/v1`):
  - INNER loop: ml/scripts/livescan_harness.py replays recorded frames offline.
  - OUTER loop: the app on a tethered iPhone records the SAME schema for a live
    sweep (mobile/src/ml/liveScanTelemetry.ts).

The app can persist a session locally (a JSON file in its document dir) AND/OR
POST it here so device sweeps are collected centrally and can be diffed against
the inner-loop baseline. This endpoint is deliberately tiny and dependency-free:
it validates the envelope loosely and journals the raw doc to
`backend/data/livescan_telemetry/<session>.json`. No DB, no model — it never
blocks a scan and degrades to a clear error if the body is malformed.

Register after the existing routers in main.py:

    from app.local_inventory.telemetry_routes import telemetry_router
    app.include_router(telemetry_router)

Endpoints
---------
  POST /api/local-inventory/telemetry/livescan
        Body: a LiveScanTelemetryDoc (schema "brickscan.livescan.telemetry/v1").
        Journals it to disk keyed by meta.session_id; returns the saved path +
        a one-line digest of the aggregate so the caller can confirm receipt.

  GET  /api/local-inventory/telemetry/livescan/sessions
        List journaled device sessions (filename, size, mtime) for quick review.
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# backend/ root (this file is backend/app/local_inventory/telemetry_routes.py).
_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
TELEMETRY_DIR = _BACKEND_DIR / "data" / "livescan_telemetry"

SCHEMA = "brickscan.livescan.telemetry/v1"
# Keep session ids filesystem-safe (the app builds them as "<platform>-<ms>").
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")

telemetry_router = APIRouter(
    prefix="/api/local-inventory",
    tags=["telemetry"],
)


class TelemetryDoc(BaseModel):
    """Loose envelope — we journal the whole doc, only validating the bits we
    key/summarize on so a future schema rev with extra fields still ingests."""

    schema_: str = Field(alias="schema")
    source: str
    meta: Dict[str, Any] = Field(default_factory=dict)
    aggregate: Dict[str, Any] = Field(default_factory=dict)
    tracks: List[Dict[str, Any]] = Field(default_factory=list)

    class Config:
        populate_by_name = True
        extra = "allow"  # never drop unknown fields; round-trip the full doc


class TelemetryAck(BaseModel):
    ok: bool
    session_id: str
    saved_path: str
    n_tracks: int
    digest: str


def _safe_session_id(raw: Optional[str]) -> str:
    sid = (raw or "").strip() or f"session-{int(time.time() * 1000)}"
    sid = _SAFE.sub("_", sid)
    return sid[:128]


@telemetry_router.post("/telemetry/livescan", response_model=TelemetryAck)
async def ingest_livescan_telemetry(doc: TelemetryDoc) -> TelemetryAck:
    if doc.schema_ != SCHEMA:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"unexpected schema {doc.schema_!r}; expected {SCHEMA!r}",
        )
    session_id = _safe_session_id(doc.meta.get("session_id"))
    TELEMETRY_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TELEMETRY_DIR / f"{session_id}.json"

    # Persist the full doc (model_dump by alias to round-trip "schema").
    payload = doc.model_dump(by_alias=True)
    out_path.write_text(json.dumps(payload, indent=2))

    agg = doc.aggregate or {}
    digest = (
        f"source={doc.source} tracks={len(doc.tracks)} "
        f"commit_rate={agg.get('commit_rate')}% "
        f"fused_top1={agg.get('fused_top1')} "
        f"lat_p90={(agg.get('retrieval_latency_ms') or {}).get('p90')}ms "
        f"errors={agg.get('errors')}"
    )
    logger.info("livescan telemetry ingested: %s -> %s | %s",
                session_id, out_path, digest)
    return TelemetryAck(
        ok=True,
        session_id=session_id,
        saved_path=str(out_path),
        n_tracks=len(doc.tracks),
        digest=digest,
    )


@telemetry_router.get("/telemetry/livescan/sessions")
async def list_livescan_sessions() -> Dict[str, Any]:
    if not TELEMETRY_DIR.exists():
        return {"count": 0, "sessions": []}
    sessions = []
    for p in sorted(TELEMETRY_DIR.glob("*.json")):
        st = p.stat()
        sessions.append({
            "session_id": p.stem,
            "bytes": st.st_size,
            "mtime": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(st.st_mtime)),
        })
    return {"count": len(sessions), "sessions": sessions}
