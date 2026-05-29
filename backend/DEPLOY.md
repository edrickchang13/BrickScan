# BrickScan Backend — Production Deploy Runbook

This is the runbook for taking the BrickScan backend from the dev
`docker-compose` stack to a real, internet-facing deployment. It covers the
serving model, hosting options, secrets, the **model-artifact provisioning
step** (the one genuinely non-obvious part), database migrations, and CI/CD.

> **Scope.** Everything here is *prep*. The steps marked **[USER ACTION]** need
> hosting, credentials, or DNS that only you can provide — those are the line
> this runbook stops at. Nothing in this repo has been deployed; no secrets
> have been generated. A companion **prod-readiness punch list** lives at the
> bottom — read it before opening this to real multi-user traffic.

---

## 0. TL;DR — single-box deploy

```bash
# on the server, in the repo root
cp backend/.env.example backend/.env        # [USER ACTION] fill in real secrets
export POSTGRES_PASSWORD='…'                 # [USER ACTION] strong password
./scripts/provision_artifacts.sh            # stage student.onnx + gallery_index.json → ./artifacts
docker compose -f docker-compose.prod.yml up -d --build
docker compose -f docker-compose.prod.yml exec backend alembic upgrade head
curl -fsS http://127.0.0.1:8000/health      # {"status":"ok"}
```

Then put a TLS reverse proxy in front (§4.1). That's the whole happy path; the
rest of this document explains each step and the alternatives.

---

## 1. What's in the box vs. what dev gives you

The dev stack (`docker-compose.yml` + `backend/Dockerfile`) is **not** suitable
for production as-is:

| Dev (`Dockerfile`, `docker-compose.yml`) | Prod (`Dockerfile.prod`, `docker-compose.prod.yml`) |
|---|---|
| `uvicorn --reload`, single process | `gunicorn` + uvicorn workers, no reload |
| Source bind-mounted (`./backend/app:/app/app`, etc.) | Code baked into the image |
| Installs **dev** deps (pytest, aiosqlite) | Runtime deps only, multi-stage build |
| Runs as root | Non-root `appuser` (uid 1000) |
| `adminer` DB UI exposed on :8080 | No adminer |
| DB/Redis ports published to host | DB/Redis internal-only; backend bound to `127.0.0.1` |
| Secrets partly hardcoded in compose | Secrets via `backend/.env` (`env_file`) + `${VAR:?}` |
| No `HEALTHCHECK` | Container `HEALTHCHECK` on `/health` |

New files this runbook adds (none committed yet — your lead integrates):

- `backend/Dockerfile.prod` — multi-stage, gunicorn, non-root, healthcheck.
- `backend/.dockerignore` — keeps tests/docs/caches/secrets/big blobs out of the image.
- `docker-compose.prod.yml` — single-host prod stack (repo root).
- `scripts/provision_artifacts.sh` — stages the gitignored model artifacts.
- `backend/.env.example` — rewritten to be complete (see §3).

> Note: there are currently **two** dev compose files — the canonical
> `docker-compose.yml` at the repo root and an older `backend/docker-compose.yml`
> (`version: "3.8"`, whole-dir bind-mount). The prod path uses neither; consider
> deleting `backend/docker-compose.yml` to avoid confusion (punch list M5).

---

## 2. Serving decision — model inference runs **in the backend container**

**Decision: serve all ML in-process via `onnxruntime` (CPU) inside the backend
container. The `dgx_spark/` Ollama/LLaVA vision server is shelved and is NOT
part of the serving path.**

The recognition cascade (`app/services/hybrid_recognition.py`) is a graceful
fall-through:

1. **Brickognize API** (primary, network) — fast ConvNeXt-T.
2. **Gemini 2.5 Flash** (network, needs `GEMINI_API_KEY`) — only when Brickognize
   is below 0.80 confidence (or `SCAN_ALWAYS_RUN_GEMINI=true`).
3. **Student retrieval** (local, onnxruntime CPU) — the validated 90.1% top-1 /
   95.6% recall@3 engine: FastViT-SA24 student encoder + int8 gallery k-NN.
4. **Contrastive k-NN / visual-search** (local, needs embedding caches).
5. **Distilled MobileNetV3 → legacy EfficientNet** (local, final fallbacks).

Every local tier **self-disables and returns `[]`** when its model files are
absent. That is the key operational risk: **if the artifacts aren't provisioned,
the server doesn't crash — it silently degrades to API-only recognition.** Verify
provisioning explicitly (§5.4).

### Latency (CPU, in-container)

These are the relevant figures; **none were measured on the target host** (the
dev stack wasn't booted during prep — measure on your box before promising SLAs):

- Student gallery k-NN: ~5–10 ms (one `(1, ~12.5k) × 768` int8→f32 matmul,
  pure numpy) — per the module docstring and the int8/latency proof from ML phase.
- Student ONNX embed (FastViT-SA24, 224², CPU): the dominant local cost — expect
  tens of ms per image on a modern x86 core; budget conservatively.
- The cascade also makes **network** calls (Brickognize, sometimes Gemini) which
  dominate wall-clock latency. The local student tier runs off the event loop via
  `asyncio.to_thread`, so it doesn't block other awaited work.
- `onnxruntime` auto-selects CUDA if present (`_ort_providers()` in
  `student_retrieval.py`); on a GPU host the embed cost drops, but **CPU is the
  assumed prod path** and is sufficient because the network tiers are primary.

### Sizing

- CPU: 2 vCPU minimum; 4 vCPU comfortable for the ONNX tiers under light load.
- RAM: budget ~1.5–2 GB for the backend container — the gallery dequantises to a
  float32 matrix (~12.5k × 768 × 4 B ≈ 38 MB) plus the loaded ONNX sessions
  (student 86 MB + legacy classifier + color), held in a process-wide singleton.
- Disk: ~250 MB image + ~100 MB artifacts volume.

---

## 3. Environment & secrets

`backend/.env.example` is the source of truth and has been rewritten to be
complete and annotated (`[REQUIRED]` vs `[optional]`). Copy it to `backend/.env`
and fill real values. **[USER ACTION]**

**Audit result — what was missing from the old `.env.example` and is now added:**

- `CORS_ALLOWED_ORIGINS` — without it the app serves **localhost-only CORS**
  (see `main.py`), so a hosted client can't call the API. Effectively required.
- `STUDENT_ONNX_PATH` / `STUDENT_GALLERY_PATH` — needed in prod (defaults point
  into `mobile/`, absent from a backend deploy). See §5.
- `ML_MODEL_TYPE` and the `SCAN_*` cascade feature flags (read via `os.environ`
  in `hybrid_recognition.py`, not in `Settings`).
- `WEB_CONCURRENCY` / `GUNICORN_TIMEOUT` (consumed by `Dockerfile.prod`).
- Fixed a stale value: old `.env.example` had `ML_MODEL_PATH=…/lego_detector.onnx`
  but both `config.py` and compose use `lego_classifier.onnx`.

**Required-or-won't-boot.** `app/core/config.py` (pydantic `Settings`) declares
these with **no default**, so the app raises on startup if any is unset:
`DATABASE_URL`, `REDIS_URL`, `SECRET_KEY`, `REBRICKABLE_API_KEY`, the four
`BRICKLINK_*`, `GEMINI_API_KEY`, `S3_BUCKET`, `AWS_ACCESS_KEY_ID`,
`AWS_SECRET_ACCESS_KEY`. (S3/AWS are required by config but otherwise unused —
punch list M4. Set placeholders to satisfy startup.)

**Secrets hygiene (verified):**

- No real secrets are committed. `.env` and `.env.*` are gitignored; only
  `.env.example` (placeholders) is tracked. Source scan found **no** hardcoded
  API keys — the only hardcoded credentials are the **dev** Postgres
  `brickscan_user/brickscan_password` in `docker-compose.yml`, which the prod
  compose does not use.
- **[USER ACTION]** Generate a real `SECRET_KEY` (`openssl rand -hex 32`). The
  placeholder signs JWTs — shipping it means anyone can forge tokens.
- **[USER ACTION]** Use a strong `POSTGRES_PASSWORD` (prod compose requires it
  via `${POSTGRES_PASSWORD:?}` and fails loudly if unset).
- Prefer your host's secret store (Fly secrets, Cloud Run Secret Manager, etc.)
  over a plaintext `.env` where available — see §4.

---

## 4. Hosting options

All four assume the image from `backend/Dockerfile.prod`. Pick one.

### 4.1 Your own box / VPS (recommended first deploy) — `docker-compose.prod.yml`

App + Postgres + Redis on one host. Simplest, full control, cheapest.

```bash
cp backend/.env.example backend/.env         # [USER ACTION]
export POSTGRES_PASSWORD='…'                  # [USER ACTION]
./scripts/provision_artifacts.sh             # §5
docker compose -f docker-compose.prod.yml up -d --build
docker compose -f docker-compose.prod.yml exec backend alembic upgrade head
```

The backend binds to `127.0.0.1:8000` only. **[USER ACTION]** Put a TLS reverse
proxy in front. Caddy is the least effort (auto-HTTPS):

```
# /etc/caddy/Caddyfile
api.brickscan.example {
    reverse_proxy 127.0.0.1:8000
    # SSE scans stream progress — don't buffer the response.
    reverse_proxy 127.0.0.1:8000 {
        flush_interval -1
    }
}
```

For nginx, set `proxy_buffering off;` and a long `proxy_read_timeout` (≥120s) on
the scan routes — the `/api/scan/stream/{id}` endpoint is a long-lived
`text/event-stream` with heartbeats (`app/services/scan_jobs.py`).

> **[USER ACTION]** DB backups are on you. `docker compose exec db pg_dump -U
> $POSTGRES_USER brickscan | gzip > backup.sql.gz` on a cron, off-host storage.

### 4.2 Fly.io

Good fit: persistent volume for artifacts, managed TLS, easy secrets.

- `fly launch --dockerfile backend/Dockerfile.prod` (build context `backend/`).
- Managed Postgres: `fly postgres create` then `fly postgres attach` (sets
  `DATABASE_URL` — convert the scheme to `postgresql+asyncpg://`). Redis via
  Upstash (`fly redis create`).
- Secrets: `fly secrets set SECRET_KEY=… GEMINI_API_KEY=… …` (not a `.env`).
- Artifacts: create a volume (`fly volumes create artifacts --size 1`), mount at
  `/artifacts`, and upload `student.onnx` + `gallery_index.json` into it (e.g.
  `fly sftp shell`). Set `STUDENT_ONNX_PATH=/artifacts/student.onnx` etc.
- `WEB_CONCURRENCY=1` and a **single machine** until §6/M1 (in-process SSE jobs).

### 4.3 Google Cloud Run

Works, with two caveats that come straight from the architecture:

- **Statelessness vs. SSE.** Cloud Run can route consecutive requests to
  different instances, which **breaks the `/start`→`/stream`→`/result` scan flow**
  (in-process job registry, §6). Use `min-instances=1, max-instances=1` and
  session affinity, **or** finish M1 (Redis-backed jobs) first. The synchronous
  `POST /api/scan` (no streaming) is unaffected.
- **Artifacts.** Cloud Run has no persistent disk. Either (a) bake artifacts into
  the image (drop the `data/gallery_index.json` and model excludes from
  `.dockerignore`, `COPY` `student.onnx` in — image grows ~100 MB), or (b) fetch
  from GCS into the container at startup (an entrypoint that runs
  `provision_artifacts.sh` with `*_URL` env pointing at GCS, writing to a
  writable `/tmp` path, and set the `STUDENT_*` env to match).
- Managed Postgres = Cloud SQL (use the connector / unix socket); Redis =
  Memorystore. Secrets = Secret Manager mounted as env vars.

### 4.4 Other PaaS (Render, Railway, etc.)

Same shape as Fly: container from `Dockerfile.prod`, managed Postgres + Redis
add-ons, secrets in the dashboard, a persistent disk mounted at `/artifacts`.
Honor `WEB_CONCURRENCY=1` + single instance until M1.

---

## 5. Model-artifact provisioning (the non-obvious step)

The backend's student tier needs two files that are **not** in a normal backend
checkout. This is the single most error-prone part of the deploy.

### 5.1 What's tracked vs. not

| Artifact | Size | In git? | Where |
|---|---|---|---|
| `lego_classifier.onnx(.data)` | ~18 MB | ✅ tracked | `backend/models/` |
| `color_classifier.onnx(.data)` | ~6 MB | ✅ tracked | `backend/models/` |
| `yolo_lego.int8.onnx` | 58 MB | ✅ tracked | `backend/models/`, `mobile/assets/models/` |
| `gallery_index.json` | 13 MB | ✅ tracked **(at mobile path)** | `mobile/assets/models/` |
| `gallery_index.json` (backend default path) | 13 MB | ❌ **gitignored** | `backend/data/` (a server copy) |
| **`student.onnx`** | **86 MB** | ❌ **gitignored** (>100 MB GitHub limit) | `mobile/assets/models/` only |

So the committed bundles (`lego_classifier`, `color_classifier`) ship **in the
image** already. The gap is the **student tier**: `student.onnx` is in git
nowhere, and the gallery the backend reads by default
(`backend/data/gallery_index.json`) is gitignored. The mobile and backend gallery
files are **byte-identical** (verified — same sha256), so either is fine to
provision.

### 5.2 Provisioning script

`scripts/provision_artifacts.sh` stages both into `./artifacts/`, which the prod
compose mounts read-only at `/artifacts`. It verifies sha256 against the
known-good hashes baked into the script.

```bash
# Local source — copies from the in-repo mobile bundle (works in a full clone):
./scripts/provision_artifacts.sh

# Remote source — fetch from object storage / a release asset:
STUDENT_ONNX_URL="https://…/student.onnx" \
GALLERY_URL="https://…/gallery_index.json" \
./scripts/provision_artifacts.sh
```

### 5.3 How to get `student.onnx` onto the server — pick one **[USER ACTION]**

`student.onnx` isn't in git, so it must travel out-of-band:

1. **Object storage (recommended).** Upload `student.onnx` to S3/GCS/R2 once,
   then `STUDENT_ONNX_URL=… ./scripts/provision_artifacts.sh` on the server (or
   in a Cloud Run startup hook). Keep it private; use a signed URL or instance
   creds.
2. **Git LFS.** `git lfs track "mobile/assets/models/student.onnx"`, migrate the
   blob to LFS, and it clones normally. Note GitHub LFS bandwidth/storage quotas;
   the file already exceeds the 100 MB *non-LFS* limit, which is why it's
   gitignored today.
3. **Rebuild from source.** Re-export from the trained checkpoint via the ML
   export scripts (`ml/export/…`) and re-quantize the gallery. Heaviest; only if
   you've lost the blob.
4. **scp/rsync directly** to the server's `./artifacts/` for a one-off.

The 13 MB `gallery_index.json` is small enough to live in LFS or be copied from
the tracked `mobile/assets/models/gallery_index.json` with no fuss.

### 5.4 Verify the tier actually loaded

There is **no runtime status endpoint** for this yet (punch list M3). Confirm via
the startup log line:

```bash
docker compose -f docker-compose.prod.yml logs backend | grep student_retrieval
# Expect: student_retrieval: loaded student.onnx (CPUExecutionProvider, …)
#         student_retrieval: loaded gallery N exemplars × 768D, … unique parts
```

If you instead see `… not found — student tier disabled`, the paths/volume are
wrong and scans are running API-only.

---

## 6. Scaling constraint you must know before adding workers/replicas

`app/services/scan_jobs.py` keeps an **in-process** job registry (a dict of
`asyncio.Queue` per scan). The streaming scan flow is three requests:

```
POST /api/scan/start      → creates job, returns scan_id   (worker A)
GET  /api/scan/stream/{id} → SSE progress                  (must be worker A)
GET  /api/scan/result/{id} → final result                  (must be worker A)
```

With **more than one** gunicorn worker or replica, the follow-up requests can hit
a worker that doesn't have the job → "scan not found". Therefore:

- `Dockerfile.prod` and `.env.example` default `WEB_CONCURRENCY=1`. **This is
  intentional, not a placeholder.**
- Scale **within** the single worker via async concurrency first.
- To scale **out** (multiple workers/replicas) you MUST move `scan_jobs` to Redis
  pub/sub (the module docstring says exactly this; the public API stays the
  same) — punch list **M1** — **or** restrict to sticky sessions + a single
  worker per instance and accept the per-instance ceiling. The plain
  `POST /api/scan` (synchronous, no SSE) is **not** affected and scales freely.

---

## 7. Database migrations (alembic)

The schema is managed by alembic (`backend/alembic/`, three revisions through
`003_add_scans`). `env.py` reads `DATABASE_URL` from the environment and converts
the async URL to sync for migrations — so no extra config is needed.

```bash
docker compose -f docker-compose.prod.yml exec backend alembic upgrade head   # apply
docker compose -f docker-compose.prod.yml exec backend alembic current        # check
docker compose -f docker-compose.prod.yml exec backend alembic downgrade -1   # roll back one
```

> **Caveat.** `main.py`'s lifespan also calls `Base.metadata.create_all` on
> startup, which creates any *missing* tables but does **not** run migrations or
> alter existing ones. **Treat `alembic upgrade head` as the source of truth** and
> run it on every deploy that ships a new revision. The `create_all` is a dev
> convenience; it can mask a forgotten migration (punch list M6).

Run `alembic upgrade head` **after** the container is up and the DB is healthy,
as a release step (it's in the §0 happy path and should be in CI/CD §8).

---

## 8. CI/CD

**There is currently NO CI/CD in this repo** — no `.github/workflows/`, no GitLab
CI, nothing. The Phase 8 brief referred to "existing GitHub Actions"; that does
not match the tree as of this writing. Below is a ready-to-use starter workflow —
**[USER ACTION]** to add it (and wire registry/host secrets).

Suggested pipeline:

1. **Lint + test** on PR — mirror the Makefile targets:
   `ruff check`, `black --check`, `pytest` (the suite needs `requirements-dev.txt`
   and is pinned to `pytest<9` for a reason — see that file).
2. **Build + push** `Dockerfile.prod` to a registry (GHCR) on merge to `main`.
3. **Deploy** — `fly deploy`, `gcloud run deploy`, or SSH `docker compose pull &&
   up -d` — then run `alembic upgrade head` as a release step.

Starter (`.github/workflows/backend.yml`):

```yaml
name: backend
on:
  pull_request:
    paths: ["backend/**"]
  push:
    branches: [main]
    paths: ["backend/**"]
jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:16
        env: { POSTGRES_PASSWORD: postgres, POSTGRES_DB: brickscan, POSTGRES_USER: postgres }
        ports: ["5432:5432"]
        options: >-
          --health-cmd "pg_isready -U postgres" --health-interval 10s
          --health-timeout 5s --health-retries 5
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - working-directory: backend
        run: pip install -r requirements-dev.txt
      - working-directory: backend
        run: |
          ruff check app/ tests/
          black --check app/ tests/
      - working-directory: backend
        env:
          # REQUIRED: conftest.py does `from main import app`, which builds
          # Settings() at import time. Those vars have no defaults in config.py,
          # so without them the suite fails to import. The actual test DB is
          # in-memory aiosqlite (conftest.py), not the postgres service above —
          # that service is here only if you add DB-backed integration tests.
          SECRET_KEY: test-secret
          REBRICKABLE_API_KEY: x
          BRICKLINK_CONSUMER_KEY: x
          BRICKLINK_CONSUMER_SECRET: x
          BRICKLINK_TOKEN: x
          BRICKLINK_TOKEN_SECRET: x
          GEMINI_API_KEY: x
          S3_BUCKET: x
          AWS_ACCESS_KEY_ID: x
          AWS_SECRET_ACCESS_KEY: x
          DATABASE_URL: postgresql+asyncpg://postgres:postgres@localhost:5432/brickscan
          REDIS_URL: redis://localhost:6379/0
        run: pytest -q
```

> Verified against `backend/tests/conftest.py`: it sets up the in-memory SQLite
> session and overrides `get_db`, but does **not** inject the `Settings`-required
> env vars — and `from main import app` instantiates `Settings()` at import
> time, so the env block above is required for the suite to even import. (The
> suite itself was not executed during prep.)

---

## 9. Health checks

- `GET /health` → `{"status": "ok"}`. **Liveness only** — it does not check DB,
  Redis, or model readiness. The prod container `HEALTHCHECK` and the §4.1 proxy
  use it.
- There is **no readiness endpoint** (no `/readyz`, no `/metrics`). For real prod
  you want a readiness probe that checks the DB session and (optionally) reports
  whether the student tier loaded. Punch list **M2/M3** includes a drop-in
  `/readyz`.
- `scripts/health_check.sh` exists and checks the **dev** compose services (it
  hardcodes `docker-compose.yml`); adapt the compose filename for prod use.

---

## 10. What the USER must do to actually deploy — checklist

- [ ] **[USER ACTION]** Choose a host (§4) and provision it.
- [ ] **[USER ACTION]** `cp backend/.env.example backend/.env`; fill **all**
      `[REQUIRED]` vars; generate `SECRET_KEY` and a strong `POSTGRES_PASSWORD`.
- [ ] **[USER ACTION]** Decide how `student.onnx` reaches the host (§5.3) and run
      `scripts/provision_artifacts.sh` (or bake into the image for Cloud Run).
- [ ] **[USER ACTION]** Stand up Postgres + Redis (compose, or managed).
- [ ] **[USER ACTION]** `docker compose -f docker-compose.prod.yml up -d --build`.
- [ ] **[USER ACTION]** `alembic upgrade head`.
- [ ] **[USER ACTION]** Terminate TLS at a reverse proxy; set `CORS_ALLOWED_ORIGINS`
      and point the mobile/web client's API URL at the host.
- [ ] **[USER ACTION]** Verify: `/health` ok, student tier loaded in logs (§5.4),
      one real scan end-to-end.
- [ ] **[USER ACTION]** Set up DB backups and log shipping.
- [ ] Review and triage the **prod-readiness punch list** below.

---

See `PROD_READINESS.md` (same directory) for the detailed punch list of what is
missing or half-wired for real multi-user production.
