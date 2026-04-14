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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(parts.router)
app.include_router(sets.router)
app.include_router(inventory.router)
app.include_router(scan.router)
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
