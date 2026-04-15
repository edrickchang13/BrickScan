"""
In-process job registry for streaming scan progress.

The scan pipeline is split into two phases:
  1. POST /api/scan/start  — creates a job, returns scan_id, spawns worker
  2. GET  /api/scan/stream/{scan_id}  — SSE stream of progress events
  3. GET  /api/scan/result/{scan_id}  — final ScanResponse once done

Each job owns an asyncio.Queue. The worker publishes events; the SSE handler
consumes them. A job is auto-deleted 5 minutes after creation regardless of
whether it was streamed or not, so abandoned scans don't leak.

Single-backend assumption: jobs live in a process-local dict. If we ever
horizontally scale the backend, swap this for Redis pub/sub — the public API
(create / publish / subscribe / store_result / get_result) stays the same.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, Optional

logger = logging.getLogger(__name__)

JOB_TTL_SECONDS = 300  # 5 min
SENTINEL_DONE = object()


@dataclass
class ScanJob:
    scan_id: str
    created_at: float = field(default_factory=time.time)
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    finished: bool = False


_jobs: Dict[str, ScanJob] = {}
_jobs_lock = asyncio.Lock()


async def create_job() -> str:
    """Create a new scan job and return its scan_id."""
    scan_id = str(uuid.uuid4())
    async with _jobs_lock:
        _jobs[scan_id] = ScanJob(scan_id=scan_id)
        await _gc_expired_locked()
    return scan_id


async def publish(scan_id: str, event: Dict[str, Any]) -> None:
    """Publish a progress event to the job's stream. No-op if job is gone."""
    job = _jobs.get(scan_id)
    if job is None:
        return
    await job.queue.put(event)


async def subscribe(scan_id: str) -> AsyncIterator[Dict[str, Any]]:
    """
    Async iterator over events for a job. Terminates when worker calls finish().
    Yields {} once when waiting > 15s with no event, so SSE stays alive through proxies.
    """
    job = _jobs.get(scan_id)
    if job is None:
        raise KeyError(scan_id)
    while True:
        try:
            event = await asyncio.wait_for(job.queue.get(), timeout=15.0)
        except asyncio.TimeoutError:
            yield {"stage": "heartbeat", "percent": -1, "message": ""}
            continue
        if event is SENTINEL_DONE:
            return
        yield event


async def finish(scan_id: str, result: Optional[Dict[str, Any]] = None,
                 error: Optional[str] = None) -> None:
    """Mark the job complete. Stores final result for /scan/result/{id} and
    closes the SSE stream by enqueueing a sentinel."""
    job = _jobs.get(scan_id)
    if job is None:
        return
    job.result = result
    job.error = error
    job.finished = True
    await job.queue.put(SENTINEL_DONE)


async def get_result(scan_id: str) -> Optional[ScanJob]:
    """Fetch the completed job (or the in-progress job) by id. None if expired."""
    return _jobs.get(scan_id)


async def _gc_expired_locked() -> None:
    """Drop jobs older than JOB_TTL_SECONDS. Caller must hold _jobs_lock."""
    now = time.time()
    expired = [sid for sid, j in _jobs.items() if now - j.created_at > JOB_TTL_SECONDS]
    for sid in expired:
        del _jobs[sid]
    if expired:
        logger.debug("Expired %d scan job(s)", len(expired))
