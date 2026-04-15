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
from sqlalchemy import create_engine, text
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

    Creates all tables if they don't exist, then runs idempotent ALTER TABLE
    statements to add any columns introduced after a table was first created.
    Safe to call multiple times.
    """
    try:
        # Import ALL models so Base.metadata knows about them before create_all.
        from app.local_inventory.models import (  # noqa: F401
            LocalInventoryPart,
            ScanSession,
            ScanFeedback,
            FeedbackEvalSnapshot,
        )

        Base.metadata.create_all(bind=engine)
        _apply_scan_feedback_column_additions()
        logger.info(f"Local inventory database initialized at {_DB_PATH}")
    except Exception as e:
        logger.error(f"Failed to initialize local inventory database: {e}")
        raise


def _apply_scan_feedback_column_additions() -> None:
    """
    Add columns introduced after the scan_feedback table was first created.

    SQLite supports ALTER TABLE ADD COLUMN; we check PRAGMA table_info first
    so this is idempotent. No-op when columns already exist. New installs skip
    this entirely because create_all() already emits the full current schema.
    """
    required_columns = {
        "feedback_type":           "VARCHAR(30)",
        "correct_rank":            "INTEGER",
        "predictions_shown_json":  "TEXT",
        "time_to_confirm_ms":      "INTEGER",
    }
    with engine.begin() as conn:
        existing = {
            row[1] for row in conn.execute(text("PRAGMA table_info(scan_feedback)"))
        }
        if not existing:
            # Table doesn't exist yet — create_all() will emit the full schema.
            return
        for col, ddl_type in required_columns.items():
            if col not in existing:
                conn.execute(text(f"ALTER TABLE scan_feedback ADD COLUMN {col} {ddl_type}"))
                logger.info("Migrated scan_feedback: added column %s", col)


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
