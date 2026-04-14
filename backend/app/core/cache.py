"""
Centralized caching layer using Redis.
Provides: get, set, delete, invalidate_pattern for distributed caching.
"""

import json
import logging
from typing import Any, Optional, Callable, Coroutine
import redis.asyncio as aioredis

logger = logging.getLogger(__name__)


class Cache:
    """
    Redis-based cache with JSON serialization.
    Supports TTL, pattern invalidation, and cache-aside pattern.
    """

    def __init__(self, redis_url: str):
        """
        Initialize cache with Redis connection.

        Args:
        - redis_url: Redis connection URL (e.g., "redis://localhost:6379/0")
        """
        self.redis = aioredis.from_url(
            redis_url,
            encoding="utf-8",
            decode_responses=True,
        )

    async def get(self, key: str) -> Optional[Any]:
        """
        Get cached value by key.

        Returns None if key not found or expired.

        Args:
        - key: Cache key

        Returns:
        - Cached value, or None if not found/expired
        """
        try:
            value = await self.redis.get(key)
            if value is None:
                return None

            # Attempt JSON deserialization
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                # Value is not JSON, return as-is
                return value

        except Exception as e:
            logger.error(f"Cache get error for key '{key}': {e}")
            return None

    async def set(
        self, key: str, value: Any, ttl_seconds: int = 3600
    ) -> bool:
        """
        Set cached value with optional TTL.

        Args:
        - key: Cache key
        - value: Value to cache (will be JSON serialized if not string)
        - ttl_seconds: Time to live in seconds (default 1 hour)

        Returns:
        - True if successful, False on error
        """
        try:
            # Serialize to JSON if not already a string
            if isinstance(value, str):
                cached_value = value
            else:
                cached_value = json.dumps(value)

            # Set with expiry
            await self.redis.setex(key, ttl_seconds, cached_value)
            return True

        except Exception as e:
            logger.error(f"Cache set error for key '{key}': {e}")
            return False

    async def delete(self, key: str) -> bool:
        """
        Delete a cache key.

        Args:
        - key: Cache key to delete

        Returns:
        - True if key existed and was deleted, False otherwise
        """
        try:
            result = await self.redis.delete(key)
            return result > 0

        except Exception as e:
            logger.error(f"Cache delete error for key '{key}': {e}")
            return False

    async def invalidate_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching a glob pattern.

        Useful for invalidating related cache entries.
        Examples:
        - "sets:*" - all set-related cache
        - "user:123:*" - all cache for user 123
        - "*:inventory:*" - all inventory cache

        Args:
        - pattern: Glob pattern (supports *, ?, [...])

        Returns:
        - Number of keys deleted
        """
        try:
            # Use SCAN to find matching keys (more efficient than KEYS)
            deleted_count = 0

            async for key in self.redis.scan_iter(match=pattern):
                await self.redis.delete(key)
                deleted_count += 1

            if deleted_count > 0:
                logger.info(f"Invalidated {deleted_count} cache keys matching '{pattern}'")

            return deleted_count

        except Exception as e:
            logger.error(f"Cache invalidate_pattern error for '{pattern}': {e}")
            return 0

    async def get_or_set(
        self,
        key: str,
        fetch_fn: Callable[..., Coroutine],
        ttl_seconds: int = 3600,
    ) -> Any:
        """
        Cache-aside pattern: fetch from cache, or call fetch_fn if miss.

        Automatically caches the result from fetch_fn.

        Args:
        - key: Cache key
        - fetch_fn: Async callable that fetches the value
        - ttl_seconds: TTL for cached result

        Returns:
        - Cached or freshly fetched value

        Example:
        ```python
        cache = get_cache()
        user = await cache.get_or_set(
            f"user:{user_id}",
            lambda: db.query(User).filter(User.id == user_id).first(),
            ttl_seconds=3600
        )
        ```
        """
        try:
            # Try to get from cache
            cached = await self.get(key)
            if cached is not None:
                logger.debug(f"Cache hit for key '{key}'")
                return cached

            # Cache miss: fetch from source
            logger.debug(f"Cache miss for key '{key}', fetching...")
            value = await fetch_fn()

            # Store in cache
            if value is not None:
                await self.set(key, value, ttl_seconds)

            return value

        except Exception as e:
            logger.error(f"Cache get_or_set error for key '{key}': {e}")
            # On error, try to fetch directly
            return await fetch_fn()

    async def exists(self, key: str) -> bool:
        """
        Check if a key exists in cache.

        Args:
        - key: Cache key

        Returns:
        - True if key exists, False otherwise
        """
        try:
            result = await self.redis.exists(key)
            return result > 0
        except Exception as e:
            logger.error(f"Cache exists error for key '{key}': {e}")
            return False

    async def ttl(self, key: str) -> int:
        """
        Get remaining TTL for a key in seconds.

        Args:
        - key: Cache key

        Returns:
        - TTL in seconds, -1 if key has no expiry, -2 if key doesn't exist
        """
        try:
            return await self.redis.ttl(key)
        except Exception as e:
            logger.error(f"Cache ttl error for key '{key}': {e}")
            return -2

    async def flush_all(self) -> bool:
        """
        Flush all cache (use with caution!).

        Returns:
        - True if successful
        """
        try:
            await self.redis.flushdb()
            logger.warning("Cache flushed completely")
            return True
        except Exception as e:
            logger.error(f"Cache flush_all error: {e}")
            return False

    async def close(self):
        """Close Redis connection."""
        try:
            await self.redis.close()
        except Exception as e:
            logger.error(f"Error closing cache connection: {e}")


# Singleton instance
_cache: Optional[Cache] = None


def get_cache() -> Cache:
    """
    Get or initialize the global cache instance.

    Returns:
    - Cache singleton

    Example:
    ```python
    from app.core.cache import get_cache

    cache = get_cache()
    value = await cache.get("key")
    ```
    """
    global _cache
    if _cache is None:
        from app.core.config import settings

        _cache = Cache(settings.REDIS_URL)
    return _cache


async def init_cache(redis_url: str) -> Cache:
    """
    Initialize cache with custom Redis URL.

    Args:
    - redis_url: Redis connection URL

    Returns:
    - Cache instance
    """
    global _cache
    _cache = Cache(redis_url)
    return _cache
