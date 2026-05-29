# BrickScan Backend — Prod-Readiness Punch List

What is missing or half-wired for **real multi-user production**, as found by
reading the backend during Phase 8 prep. Each item is concrete: file, what's
wrong, why it matters, and the fix. IDs (M1…) are referenced from `DEPLOY.md`.

Severity: **P0** = blocks/breaks multi-user prod · **P1** = real gap, ship-soon ·
**P2** = hygiene/observability.

The single-box happy path in `DEPLOY.md` works **today** for a small,
single-instance deployment. These items are what stand between that and robust
multi-user production.

---

## P0 — must address before multi-user / multi-instance

### M1 — SSE scan jobs are in-process; breaks with >1 worker or replica
- **Where:** `app/services/scan_jobs.py` (process-local `dict` of `asyncio.Queue`).
- **Problem:** the streaming scan flow is 3 requests — `POST /api/scan/start`
  (creates job), `GET /api/scan/stream/{id}` (SSE), `GET /api/scan/result/{id}`.
  All must hit the **same** process. With multiple gunicorn workers or replicas,
  follow-ups land elsewhere → "scan not found". The module docstring already
  calls this out.
- **Why it matters:** any horizontal scaling, and stateless PaaS like Cloud Run,
  silently break streamed scans. It's the reason `WEB_CONCURRENCY=1` is the
  default.
- **Fix:** back the registry with Redis pub/sub (Redis is already a dependency).
  Keep the public API (`create` / `publish` / `subscribe` / `store_result` /
  `get_result`) identical so callers don't change. **Then** you can raise
  `WEB_CONCURRENCY` and add replicas. *(The synchronous `POST /api/scan` is
  unaffected and scales now.)*

### M4 — S3 scan-image storage is configured but NOT wired (and boto3 isn't installed)
- **Where:** `app/services/image_service.py::save_scan_image_to_s3`,
  `app/core/config.py` (`S3_BUCKET`, `AWS_*` are **required** fields).
- **Problem:** the function takes `s3_client=None` and returns `None` ("S3 not
  configured, skipping"); **nothing constructs a client**, and `boto3`/`aioboto3`
  is **not** in `requirements.txt`. Yet `S3_BUCKET`/`AWS_*` are required-to-boot,
  so every deploy must set dummy values for a feature that does nothing.
- **Why it matters:** the "collect scan images for training" capability the
  config implies doesn't exist; it's a trap (looks wired, isn't), and it forces
  meaningless required secrets.
- **Fix (choose):** (a) implement it — add `aioboto3`, build a client from the
  settings, call `save_scan_image_to_s3` from the scan path, and make the AWS
  vars **optional** (default `""`) so S3 is opt-in; **or** (b) remove the S3
  surface and drop `S3_BUCKET`/`AWS_*` from `Settings` until it's real. Don't
  ship it half-on.

---

## P1 — real gaps to close soon

### M7 — Auth/JWT is partial
- **Where:** `app/core/security.py`, `app/api/auth.py`, `app/models/user.py`,
  and the admin gate in `app/api/scan.py` (`/admin/trigger-retrain`).
- **Problems:**
  1. **No roles in the token, no role column.** `create_access_token` puts only
     `{"sub": user_id}`. The admin gate reads `current_user.get("role", "user")`
     and 403s non-admins — but `role` is never set and `User` has no
     `role`/`is_admin` column, so **the admin path is unreachable** (always
     defaults to "user").
  2. **No refresh token.** Only a 30-min access token; clients must re-login.
     `/auth/refresh` is referenced in the (unwired) rate-limit middleware but
     doesn't exist.
  3. **No token revocation / logout / blocklist.** A leaked token is valid until
     expiry.
  4. `is_active` exists on `User` but is **not checked** at login or in
     `get_current_user` — a deactivated user keeps working until token expiry.
- **Why it matters:** admin features can't be used; no session management for a
  multi-user app.
- **Fix:** add a `role`/`is_admin` column + migration; include it in the token
  claims; check `is_active` at login and in `get_current_user`; add a refresh
  token (and a revocation list in Redis if you need logout).

### M_RL — Rate limiting exists but is NOT wired in
- **Where:** `app/middleware/rate_limit.py` (full `RateLimiter` + middleware fn).
- **Problem:** it's never instantiated and never `app.add_middleware`'d in
  `main.py` — dead code. Also its path checks target `"/api/scans/scan"`, but the
  scan router mounts at `/api/scan` (prefix `/api` + `scan.router`), so even if
  wired the scan-specific limit wouldn't match.
- **Why it matters:** no protection against scan-spam (each scan can call Gemini —
  cost) or auth brute-force.
- **Fix:** wire it as middleware (or per-route dependencies) using the existing
  Redis client; correct the scan path to `/api/scan`; verify the auth paths
  (`/auth/login`, `/auth/register`) match the real routes (they do).

### M_LOG — Structured logging built but never initialized
- **Where:** `app/core/logging_config.py::setup_logging` (has a JSON formatter for
  prod, colored for dev) — but `main.py` never calls it.
- **Problem:** in prod you get default uvicorn/gunicorn logging, not the JSON
  formatter, so logs won't carry `request_id`/`user_id`/`duration_ms` and won't
  parse cleanly in a log aggregator.
- **Fix:** call `setup_logging(debug=…, log_level=…)` in `main.py` (lifespan or
  module top), driven by an `ENVIRONMENT`/`LOG_LEVEL` env var. Optionally add a
  middleware that emits `log_request`/`log_response` with timing.

### M_ADMIN — `/admin/*` router is unmounted and partially broken
- **Where:** `app/api/admin.py`; `main.py` includes do **not** add `admin.router`.
- **Problems:** the whole `/admin/*` surface (stats, scan-logs, model-status) is
  unreachable. And `get_model_status` references `settings.ML_MODEL_VERSION` and
  `settings.ML_CONFIDENCE_THRESHOLD`, **neither of which exists** in `config.py`
  (it has `CONFIDENCE_THRESHOLD`, no `ML_MODEL_VERSION`) → it would `AttributeError`
  if mounted. `model-status` is also hardcoded stub data.
- **Why it matters:** there's no working admin/ops surface; the code implies one
  that isn't real.
- **Fix:** decide whether admin is in scope. If yes: mount the router, fix the
  config attribute names, and replace the stubbed `model-status` with real values
  (and a working `check_admin` once M7 adds roles). If no: delete it so it's not
  mistaken for functional.

---

## P2 — observability & hygiene

### M2 — No readiness probe (only liveness)
- **Where:** `main.py` has `GET /health` → `{"status":"ok"}`; nothing checks DB/Redis.
- **Fix:** add a `/readyz` that pings the DB and Redis so the orchestrator gates
  traffic on real dependencies:
  ```python
  from sqlalchemy import text
  @app.get("/readyz")
  async def readyz():
      checks = {}
      try:
          async with engine.connect() as c:
              await c.execute(text("SELECT 1"))
          checks["db"] = "ok"
      except Exception as e:
          checks["db"] = f"error: {e}"
      return ({"status": "ready", **checks}
              if all(v == "ok" for v in checks.values())
              else (JSONResponse({"status": "not-ready", **checks}, status_code=503)))
  ```

### M3 — No way to confirm the student tier loaded at runtime
- **Where:** `app/services/student_retrieval.py` exposes `is_available()` /
  `gallery_size()`, but nothing surfaces them over HTTP.
- **Why it matters:** the tier self-disables silently when artifacts are missing
  (the central provisioning risk in `DEPLOY.md` §5). Today you can only tell from
  startup logs.
- **Fix:** a tiny ops endpoint, e.g.:
  ```python
  from app.services import student_retrieval
  @app.get("/ml/status")
  async def ml_status():
      return {"student_retrieval": {
          "available": student_retrieval.is_available(),
          "gallery_size": student_retrieval.gallery_size()}}
  ```
  Fold the other tiers' availability in too. (Consider gating behind admin once
  M7 lands.)

### M_METRICS — No metrics / tracing
- **Problem:** no `/metrics` (Prometheus), no request timing, no error tracking
  (e.g. Sentry). Per-scan source/latency/cost is invisible.
- **Fix:** add `prometheus-fastapi-instrumentator` (or equivalent) and a Sentry
  DSN env var. Track cascade source distribution and Gemini call rate (cost).

### M6 — `create_all` on startup can mask a missing migration
- **Where:** `main.py` lifespan runs `Base.metadata.create_all`.
- **Problem:** it creates missing tables but never alters existing ones or runs
  alembic. A forgotten `alembic upgrade head` can go unnoticed until a column is
  actually used.
- **Fix:** treat `alembic upgrade head` as the source of truth (it's in the
  deploy steps). Optionally gate `create_all` behind a dev-only flag so prod
  relies solely on migrations.

### M5 — Duplicate/dev compose files
- **Where:** root `docker-compose.yml` (canonical dev) **and**
  `backend/docker-compose.yml` (older `version: "3.8"`, whole-dir bind-mount,
  no adminer).
- **Fix:** delete `backend/docker-compose.yml` (superseded by the root dev compose
  and the new `docker-compose.prod.yml`) to remove ambiguity.

### M_DB — DB defaults & connection sizing for managed Postgres
- **Where:** `docker-compose.yml` hardcodes dev creds (`brickscan_user/
  brickscan_password`); `app/core/database.py` sets `pool_size=20, max_overflow=10`.
- **Notes for prod:** the prod compose already parameterizes creds. With a managed
  Postgres, 30 connections/worker can exceed plan limits — size `pool_size` to the
  provider's cap (and to `WEB_CONCURRENCY`). `pool_pre_ping=True` is already set
  (good for managed DBs that drop idle conns).

---

## Quick triage order

1. **M1** (Redis-back scan jobs) — unlocks scaling; do first if you expect >1
   concurrent user on streamed scans or use Cloud Run.
2. **M7 + M_RL** (auth roles/refresh + rate limiting) — needed before real users.
3. **M4** (decide S3 in/out) and **M_LOG** (init logging) — close the traps.
4. **M2/M3/M_METRICS** (readiness, ml-status, metrics) — operability.
5. **M5/M6/M_ADMIN/M_DB** — hygiene.

Items resolved in prep (Phase 8): prod Dockerfile + `.dockerignore`, prod compose,
artifact provisioning script, complete `.env.example`, container health check,
and this documentation. None are committed — the lead integrates.
