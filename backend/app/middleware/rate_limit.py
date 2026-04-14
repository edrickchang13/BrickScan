"""
Redis-based rate limiting middleware.
Enforces per-endpoint and per-user/IP limits.

Rate limits:
- Scan endpoint: 20 requests/minute per user
- General API: 120 requests/minute per user/IP
- Auth endpoints: 10 requests/minute per IP
"""

import time
import logging
from typing import Optional
from fastapi import Request, HTTPException, status

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Redis-based sliding window rate limiter.
    Uses Redis ZADD/ZCOUNT for efficient per-window tracking.
    """

    def __init__(self, redis_client):
        """
        Args:
        - redis_client: aioredis async client instance
        """
        self.redis = redis_client
        self.SCAN_LIMIT = 20
        self.SCAN_WINDOW = 60
        self.GENERAL_LIMIT = 120
        self.GENERAL_WINDOW = 60
        self.AUTH_LIMIT = 10
        self.AUTH_WINDOW = 60

    async def check_rate_limit(
        self, key: str, limit: int, window_seconds: int
    ) -> bool:
        """
        Check if request is allowed using sliding window rate limiting.

        Uses Redis ZADD to track timestamps in a sorted set.
        Removes expired entries and checks if current count <= limit.

        Args:
        - key: Rate limit key (e.g., "user:user_id:scan")
        - limit: Max requests allowed in window
        - window_seconds: Time window in seconds

        Returns:
        - True if request is allowed
        - False if rate limited
        """
        now = time.time()
        window_start = now - window_seconds

        try:
            pipe = self.redis.pipeline()

            # Add current request timestamp to sorted set
            pipe.zadd(key, {str(now): now})

            # Remove timestamps outside the window
            pipe.zremrangebyscore(key, 0, window_start)

            # Count requests in current window
            pipe.zcard(key)

            # Set expiry to prevent orphaned keys
            pipe.expire(key, window_seconds * 2)

            results = await pipe.execute()
            request_count = results[2]

            return request_count <= limit

        except Exception as e:
            logger.error(f"Rate limiter error for key {key}: {e}")
            # On error, allow request to pass (fail open)
            return True

    async def scan_rate_limit(self, user_id: str) -> None:
        """
        Enforce scan endpoint rate limit (20 req/min per user).

        Args:
        - user_id: User making the request

        Raises:
        - HTTPException with 429 status if rate limited
        """
        key = f"ratelimit:scan:{user_id}"
        allowed = await self.check_rate_limit(
            key, self.SCAN_LIMIT, self.SCAN_WINDOW
        )

        if not allowed:
            logger.warning(f"Scan rate limit exceeded for user {user_id}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limited: maximum {self.SCAN_LIMIT} scans per minute",
                headers={"Retry-After": str(self.SCAN_WINDOW)},
            )

    async def general_rate_limit(self, identifier: str) -> None:
        """
        Enforce general API rate limit (120 req/min per user/IP).

        Args:
        - identifier: User ID or IP address

        Raises:
        - HTTPException with 429 status if rate limited
        """
        key = f"ratelimit:api:{identifier}"
        allowed = await self.check_rate_limit(
            key, self.GENERAL_LIMIT, self.GENERAL_WINDOW
        )

        if not allowed:
            logger.warning(f"API rate limit exceeded for {identifier}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limited: maximum {self.GENERAL_LIMIT} requests per minute",
                headers={"Retry-After": str(self.GENERAL_WINDOW)},
            )

    async def auth_rate_limit(self, ip_address: str) -> None:
        """
        Enforce auth endpoint rate limit (10 req/min per IP).
        Protects against brute force attacks.

        Args:
        - ip_address: Client IP address

        Raises:
        - HTTPException with 429 status if rate limited
        """
        key = f"ratelimit:auth:{ip_address}"
        allowed = await self.check_rate_limit(
            key, self.AUTH_LIMIT, self.AUTH_WINDOW
        )

        if not allowed:
            logger.warning(f"Auth rate limit exceeded for IP {ip_address}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Too many login attempts. Try again in {self.AUTH_WINDOW} seconds.",
                headers={"Retry-After": str(self.AUTH_WINDOW)},
            )

    def get_client_ip(self, request: Request) -> str:
        """
        Extract client IP from request, handling X-Forwarded-For header.

        Args:
        - request: FastAPI Request object

        Returns:
        - Client IP address string
        """
        # Check X-Forwarded-For header (for requests behind proxy)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            # X-Forwarded-For can contain multiple IPs; use the first one
            return forwarded_for.split(",")[0].strip()

        # Fall back to direct client address
        return request.client.host if request.client else "unknown"


# Middleware function for integration into FastAPI
async def rate_limit_middleware(
    request: Request, call_next, rate_limiter: RateLimiter, current_user: Optional[str] = None
):
    """
    FastAPI middleware that applies rate limiting based on endpoint and user.

    Args:
    - request: FastAPI Request
    - call_next: Next middleware/route handler
    - rate_limiter: RateLimiter instance
    - current_user: Optional authenticated user ID (from dependency)

    Returns:
    - Response from next handler, or 429 if rate limited
    """
    path = request.url.path

    # Auth endpoints: check by IP
    if path in ["/auth/login", "/auth/register", "/auth/refresh"]:
        client_ip = rate_limiter.get_client_ip(request)
        await rate_limiter.auth_rate_limit(client_ip)

    # Scan endpoint: check by user
    elif path == "/api/scans/scan" and request.method == "POST":
        if current_user:
            await rate_limiter.scan_rate_limit(current_user)
        else:
            client_ip = rate_limiter.get_client_ip(request)
            await rate_limiter.general_rate_limit(client_ip)

    # General API: check by user or IP
    else:
        identifier = current_user or rate_limiter.get_client_ip(request)
        await rate_limiter.general_rate_limit(identifier)

    return await call_next(request)
