"""pytest fixtures for BrickScan backend tests."""
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient
from httpx import AsyncClient
import uuid
from datetime import datetime, timezone

from app.main import app
from app.models import Base
from app.models.user import User
from app.models.color import Color
from app.models.part import Part, PartCategory
from app.models.set import LEGOSet
from app.models.set_part import SetPart
from app.models.theme import Theme
from app.database import get_db
from app.core.security import hash_password, create_access_token


@pytest_asyncio.fixture
async def db_session():
    """Create a test database session with SQLite."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with AsyncSession(engine, expire_on_commit=False) as session:
        yield session

    await engine.dispose()


@pytest.fixture
def client(db_session):
    """Create a TestClient with overridden database dependency."""
    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
async def sample_colors(db_session):
    """Insert sample colors into test database."""
    colors = [
        Color(id=1, name="Red", rgb="FF0000", is_transparent=False),
        Color(id=2, name="Blue", rgb="0000FF", is_transparent=False),
        Color(id=3, name="Yellow", rgb="FFFF00", is_transparent=False),
        Color(id=4, name="Green", rgb="00FF00", is_transparent=False),
        Color(id=5, name="Black", rgb="000000", is_transparent=False),
        Color(id=16, name="Transparent Clear", rgb="FFFFFF", is_transparent=True),
    ]

    for color in colors:
        db_session.add(color)

    await db_session.commit()
    return colors


@pytest.fixture
async def sample_part_category(db_session):
    """Insert a sample part category."""
    category = PartCategory(id=1, name="Brick")
    db_session.add(category)
    await db_session.commit()
    return category


@pytest.fixture
async def sample_parts(db_session, sample_part_category):
    """Insert sample parts into test database."""
    parts = [
        Part(
            id=str(uuid.uuid4()),
            part_num="3001",
            name="Brick 2x4",
            part_category_id=1,
            material="Plastic",
        ),
        Part(
            id=str(uuid.uuid4()),
            part_num="3002",
            name="Brick 2x2",
            part_category_id=1,
            material="Plastic",
        ),
        Part(
            id=str(uuid.uuid4()),
            part_num="3003",
            name="Brick 1x2",
            part_category_id=1,
            material="Plastic",
        ),
        Part(
            id=str(uuid.uuid4()),
            part_num="3004",
            name="Brick 1x1",
            part_category_id=1,
            material="Plastic",
        ),
    ]

    for part in parts:
        db_session.add(part)

    await db_session.commit()
    return parts


@pytest.fixture
async def sample_theme(db_session):
    """Insert a sample theme."""
    theme = Theme(id=1, name="Star Wars", parent_id=None)
    db_session.add(theme)
    await db_session.commit()
    return theme


@pytest.fixture
async def sample_set(db_session, sample_theme, sample_parts, sample_colors):
    """Insert a sample LEGO set with known parts list."""
    lego_set = LEGOSet(
        id=str(uuid.uuid4()),
        set_num="75105",
        name="Millennium Falcon",
        year=2015,
        theme_id=1,
        num_parts=1175,
        image_url="https://example.com/75105.jpg",
    )
    db_session.add(lego_set)
    await db_session.flush()

    # Add set parts
    set_parts = [
        SetPart(
            id=str(uuid.uuid4()),
            set_id=lego_set.id,
            part_id=sample_parts[0].id,
            color_id=1,
            quantity=100,
            is_spare=False,
        ),
        SetPart(
            id=str(uuid.uuid4()),
            set_id=lego_set.id,
            part_id=sample_parts[1].id,
            color_id=5,
            quantity=50,
            is_spare=False,
        ),
        SetPart(
            id=str(uuid.uuid4()),
            set_id=lego_set.id,
            part_id=sample_parts[2].id,
            color_id=2,
            quantity=75,
            is_spare=False,
        ),
        SetPart(
            id=str(uuid.uuid4()),
            set_id=lego_set.id,
            part_id=sample_parts[3].id,
            color_id=16,
            quantity=20,
            is_spare=True,
        ),
    ]

    for part in set_parts:
        db_session.add(part)

    await db_session.commit()
    return lego_set


@pytest.fixture
async def test_user(db_session):
    """Create a test user."""
    user = User(
        id=str(uuid.uuid4()),
        email="test@example.com",
        hashed_password=hash_password("testpassword123"),
        full_name="Test User",
        is_active=True,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    db_session.add(user)
    await db_session.commit()
    return user


@pytest.fixture
async def auth_headers(test_user):
    """Generate auth headers for a test user."""
    token = create_access_token(str(test_user.id))
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
async def async_client(client, db_session):
    """Create an async HTTP client for testing."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac
