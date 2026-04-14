import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.database import engine, Base
from app.api import auth, parts, sets, inventory, scan, bricklink
from app.local_inventory import routes as local_inventory_routes
from app.local_inventory import sets_routes
from app.local_inventory.feedback_routes import feedback_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Main database initialized successfully")
    except Exception as e:
        logger.warning(
            f"Main database unavailable ({e}). "
            "Running in local-only mode — auth/sets/inventory endpoints require PostgreSQL."
        )
    yield
    try:
        await engine.dispose()
    except Exception:
        pass


app = FastAPI(
    title="BrickScan Backend API",
    version="1.0.0",
    description="LEGO piece scanning and inventory management API",
    lifespan=lifespan,
)

import os

# CORS: wildcard origin with credentials is insecure (CSRF/credential leak).
# Read allowed origins from CORS_ALLOWED_ORIGINS env var (comma-separated).
# In development we allow localhost + USB link-local automatically so that
# Metro-served bundles from the phone can call the API without manual config.
_cors_env = os.getenv("CORS_ALLOWED_ORIGINS", "").strip()
if _cors_env:
    _cors_origins = [o.strip() for o in _cors_env.split(",") if o.strip()]
else:
    # Dev defaults — restrict to local networks
    _cors_origins = [
        "http://localhost:8081",
        "http://localhost:19000",
        "http://127.0.0.1:8081",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_origin_regex=r"^https?://(127\.0\.0\.1|localhost|169\.254\.\d+\.\d+|192\.168\.\d+\.\d+|10\.\d+\.\d+\.\d+):\d+$",
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Accept", "X-Requested-With"],
)

app.include_router(auth.router)
# /api/parts/*, /api/sets/*, /api/inventory/*, /api/scan/* — matches mobile
# src/services/api.ts path conventions. /auth and /bricklink stay un-prefixed
# because mobile calls them without /api/.
app.include_router(parts.router, prefix="/api")
app.include_router(sets.router, prefix="/api")
app.include_router(inventory.router, prefix="/api")
app.include_router(scan.router, prefix="/api")
app.include_router(bricklink.router)
app.include_router(feedback_router)  # YOLO + active learning
app.include_router(local_inventory_routes.router)
app.include_router(sets_routes.router)


@app.get("/health")
async def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
