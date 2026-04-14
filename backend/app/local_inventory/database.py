"""
Local SQLite database setup for offline inventory storage.

Creates SQLite database at ~/brickscan_inventory.db with tables:
- local_inventory_parts: Scanned parts with quantities and confidence
- scan_sessions: Named scanning sessions

Uses SQLAlchemy with synchronous engine for simplicity (local-only, no concurrency).
Tables are auto-created on first access.

Note: This is separate from the main PostgreSQL database (in app.core.database).
The local inventory system is device-local and not synced to the backend.
"""

import os
import logging
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from typing import Generator

logger = logging.getLogger(__name__)

# Create a separate Base for local inventory models (SQLite compatibility)
Base = declarative_base()

# Determine database path: ~/brickscan_inventory.db
_DB_PATH = os.path.expanduser("~/brickscan_inventory.db")
_DB_URL = f"sqlite:///{_DB_PATH}"

# Create sync engine (SQLite handles sync better than async)
engine = create_engine(
    _DB_URL,
    echo=False,
    connect_args={"check_same_thread": False},  # SQLite thread safety
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


def init_db() -> None:
    """
    Initialize the local inventory database.

    Creates all tables if they don't exist.
    Safe to call multiple times (idempotent).
    """
    try:
        from app.local_inventory.models import LocalInventoryPart, ScanSession

        Base.metadata.create_all(bind=engine)
        logger.info(f"Local inventory database initialized at {_DB_PATH}")
    except Exception as e:
        logger.error(f"Failed to initialize local inventory database: {e}")
        raise


def get_local_db() -> Generator[Session, None, None]:
    """
    Dependency injection for FastAPI endpoints.

    Yields a SQLAlchemy session scoped to a single request.
    Automatically closes on completion.

    Usage:
        @router.get("/api/inventory")
        async def list_inventory(db: Session = Depends(get_local_db)):
            items = db.query(LocalInventoryPart).all()
            return items
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_db_path() -> str:
    """Return the full path to the local inventory database."""
    return _DB_PATH
