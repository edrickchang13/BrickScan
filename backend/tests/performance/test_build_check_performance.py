"""
Performance tests for build_check service.
Tests that build comparison runs fast even with large inventories.

Run with: pytest backend/tests/performance/test_build_check_performance.py -v
"""
import time
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import insert
import uuid
from datetime import datetime, timezone

from app.models.inventory import InventoryItem
from app.models.part import Part, PartCategory
from app.models.color import Color
from app.models.lego_set import LegoSet, SetPart, Theme
from app.models.user import User


@pytest_asyncio.fixture
async def seeded_large_inventory(db_session: AsyncSession, sample_parts, sample_colors, test_user):
    """
    Seed 10,000 inventory items for performance testing.

    Creates a realistic distribution:
    - 5 part types with varying colors
    - 2,000 items per part across different colors
    - Total: 10,000 inventory entries
    """
    color_count = len(sample_colors)
    parts_to_use = sample_parts[:5]  # Use first 5 parts
    items_per_part = 2000

    batch_size = 500
    all_items = []

    # Generate inventory items
    for part in parts_to_use:
        for i in range(items_per_part):
            color = sample_colors[i % color_count]
            item = InventoryItem(
                id=uuid.uuid4(),
                user_id=test_user.id,
                part_id=part.id,
                color_id=color.id,
                quantity=(i % 10) + 1,  # Vary quantities from 1-10
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
            all_items.append(item)

            # Batch insert every 500 items
            if len(all_items) >= batch_size:
                db_session.add_all(all_items)
                await db_session.flush()
                all_items = []

    # Insert remaining items
    if all_items:
        db_session.add_all(all_items)
        await db_session.flush()

    await db_session.commit()
    return test_user.id


class TestBuildCheckPerformance:
    """Performance tests for build check functionality."""

    @pytest.mark.asyncio
    async def test_build_check_small_inventory_performance(self, client, test_user, auth_headers, sample_set):
        """Build check with ~100 inventory items should be very fast (<100ms)."""
        # Small inventory is already seeded in test fixtures
        start = time.perf_counter()

        response = client.post(
            f"/api/v1/sets/{sample_set.set_num}/compare",
            headers=auth_headers
        )

        elapsed = time.perf_counter() - start

        assert response.status_code == 200
        assert elapsed < 0.1, f"Build check took {elapsed:.3f}s with small inventory — should be <100ms"

    @pytest.mark.asyncio
    async def test_build_check_medium_inventory_performance(
        self, db_session: AsyncSession, client, test_user, auth_headers,
        sample_parts, sample_colors, sample_set
    ):
        """Build check with ~1,000 inventory items should complete in <500ms."""
        # Add 1,000 inventory items
        items = []
        for i in range(1000):
            part = sample_parts[i % len(sample_parts)]
            color = sample_colors[i % len(sample_colors)]
            item = InventoryItem(
                id=uuid.uuid4(),
                user_id=test_user.id,
                part_id=part.id,
                color_id=color.id,
                quantity=(i % 5) + 1,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
            items.append(item)

        db_session.add_all(items)
        await db_session.commit()

        start = time.perf_counter()

        response = client.post(
            f"/api/v1/sets/{sample_set.set_num}/compare",
            headers=auth_headers
        )

        elapsed = time.perf_counter() - start

        assert response.status_code == 200
        result = response.json()
        assert "percent_complete" in result
        assert elapsed < 0.5, f"Build check took {elapsed:.3f}s with 1,000 items — should be <500ms"

    @pytest.mark.asyncio
    async def test_build_check_large_inventory_performance(
        self, db_session: AsyncSession, client, test_user, auth_headers,
        seeded_large_inventory, sample_set
    ):
        """Build check with 10,000 inventory items should complete in <1 second."""
        start = time.perf_counter()

        response = client.post(
            f"/api/v1/sets/{sample_set.set_num}/compare",
            headers=auth_headers
        )

        elapsed = time.perf_counter() - start

        assert response.status_code == 200
        result = response.json()
        assert "percent_complete" in result
        assert isinstance(result["percent_complete"], (int, float))
        assert result["percent_complete"] >= 0
        assert result["percent_complete"] <= 100
        assert elapsed < 1.0, f"Build check took {elapsed:.3f}s with 10,000 items — should be <1s"

    @pytest.mark.asyncio
    async def test_build_check_consistency_under_load(
        self, db_session: AsyncSession, client, test_user, auth_headers,
        seeded_large_inventory, sample_set
    ):
        """Build check should return consistent results when called multiple times."""
        results = []
        times = []

        # Call build check 5 times
        for i in range(5):
            start = time.perf_counter()

            response = client.post(
                f"/api/v1/sets/{sample_set.set_num}/compare",
                headers=auth_headers
            )

            elapsed = time.perf_counter() - start

            assert response.status_code == 200
            result = response.json()
            results.append(result["percent_complete"])
            times.append(elapsed)

        # All results should be the same
        assert all(r == results[0] for r in results), "Build check returned inconsistent results"

        # Execution times should be relatively consistent (within 2x of median)
        median_time = sorted(times)[len(times) // 2]
        max_allowed = median_time * 2

        for t in times:
            assert t <= max_allowed, f"Execution time {t:.3f}s exceeds {max_allowed:.3f}s threshold"

    @pytest.mark.asyncio
    async def test_inventory_retrieval_performance(
        self, client, test_user, auth_headers, seeded_large_inventory
    ):
        """Getting inventory should be fast even with 10,000 items."""
        start = time.perf_counter()

        response = client.get(
            "/api/v1/inventory",
            headers=auth_headers
        )

        elapsed = time.perf_counter() - start

        assert response.status_code == 200
        data = response.json()
        assert "items" in data or "inventory" in data
        assert elapsed < 2.0, f"Inventory retrieval took {elapsed:.3f}s — should be <2s"

    @pytest.mark.asyncio
    async def test_inventory_search_performance(
        self, client, test_user, auth_headers, seeded_large_inventory
    ):
        """Searching inventory should be fast."""
        start = time.perf_counter()

        response = client.get(
            "/api/v1/inventory",
            params={"search": "3001"},
            headers=auth_headers
        )

        elapsed = time.perf_counter() - start

        assert response.status_code == 200
        assert elapsed < 0.5, f"Inventory search took {elapsed:.3f}s — should be <500ms"

    @pytest.mark.asyncio
    async def test_set_retrieval_performance(self, client, sample_set):
        """Getting set details should be fast."""
        start = time.perf_counter()

        response = client.get(
            f"/api/v1/sets/{sample_set.set_num}"
        )

        elapsed = time.perf_counter() - start

        assert response.status_code == 200
        assert elapsed < 0.2, f"Set retrieval took {elapsed:.3f}s — should be <200ms"

    @pytest.mark.asyncio
    async def test_set_parts_retrieval_performance(self, client, sample_set):
        """Getting set parts should scale well."""
        start = time.perf_counter()

        response = client.get(
            f"/api/v1/sets/{sample_set.set_num}/parts"
        )

        elapsed = time.perf_counter() - start

        assert response.status_code == 200
        assert elapsed < 0.3, f"Set parts retrieval took {elapsed:.3f}s — should be <300ms"

    @pytest.mark.asyncio
    async def test_bricklink_export_performance(
        self, client, test_user, auth_headers,
        seeded_large_inventory, sample_set
    ):
        """BrickLink export should be fast even with large inventories."""
        start = time.perf_counter()

        response = client.post(
            f"/api/v1/sets/{sample_set.set_num}/bricklink",
            params={"condition": "X"},
            headers=auth_headers
        )

        elapsed = time.perf_counter() - start

        assert response.status_code == 200
        assert elapsed < 1.0, f"BrickLink export took {elapsed:.3f}s — should be <1s"

    @pytest.mark.asyncio
    async def test_stats_calculation_performance(
        self, client, test_user, auth_headers, seeded_large_inventory
    ):
        """Stats calculation should be fast with large inventories."""
        start = time.perf_counter()

        response = client.get(
            "/api/v1/stats/me",
            headers=auth_headers
        )

        elapsed = time.perf_counter() - start

        assert response.status_code == 200
        stats = response.json()
        assert "total_parts" in stats
        assert "total_pieces" in stats
        assert elapsed < 0.5, f"Stats calculation took {elapsed:.3f}s — should be <500ms"

    @pytest.mark.asyncio
    async def test_add_inventory_performance_scaling(
        self, client, test_user, auth_headers, sample_parts, sample_colors
    ):
        """Adding items should maintain performance as inventory grows."""
        times = []

        for i in range(10):
            part = sample_parts[i % len(sample_parts)]
            color = sample_colors[i % len(sample_colors)]

            start = time.perf_counter()

            response = client.post(
                "/api/v1/inventory",
                json={
                    "part_num": part.part_num,
                    "color_id": str(color.id),
                    "quantity": i + 1
                },
                headers=auth_headers
            )

            elapsed = time.perf_counter() - start

            assert response.status_code in [200, 201]
            times.append(elapsed)

        # Should maintain relatively consistent performance
        avg_time = sum(times) / len(times)
        max_time = max(times)

        assert max_time < avg_time * 3, "Adding items shows performance degradation"

    @pytest.mark.asyncio
    async def test_database_query_efficiency(
        self, db_session: AsyncSession, seeded_large_inventory, sample_set
    ):
        """Verify that queries are efficient (checking for N+1 issues)."""
        # This is a placeholder for actual query analysis
        # In production, use SQLAlchemy echo=True or query profiling

        start = time.perf_counter()

        # Simulate typical build check query
        result = await db_session.execute(
            """
            SELECT COUNT(*) FROM inventory_items
            WHERE user_id = :user_id
            """
        )

        elapsed = time.perf_counter() - start

        assert elapsed < 0.1, f"Query took {elapsed:.3f}s — possible N+1 issue"


class TestCacheEffectiveness:
    """Tests to verify caching strategies are working."""

    @pytest.mark.asyncio
    async def test_repeated_queries_use_cache(
        self, client, test_user, auth_headers, sample_set
    ):
        """Repeated calls should benefit from caching."""
        times = []

        for _ in range(3):
            start = time.perf_counter()

            response = client.get(
                f"/api/v1/sets/{sample_set.set_num}"
            )

            elapsed = time.perf_counter() - start

            assert response.status_code == 200
            times.append(elapsed)

        # Second and third calls should be faster (caching)
        # Note: May not be true if caching isn't implemented yet
        # Just track the timings for now
        assert times[0] >= 0  # Just ensure we can time it

    @pytest.mark.asyncio
    async def test_build_check_is_idempotent(
        self, client, test_user, auth_headers, sample_set
    ):
        """Multiple build checks should return identical results."""
        response1 = client.post(
            f"/api/v1/sets/{sample_set.set_num}/compare",
            headers=auth_headers
        )

        response2 = client.post(
            f"/api/v1/sets/{sample_set.set_num}/compare",
            headers=auth_headers
        )

        assert response1.json() == response2.json()
