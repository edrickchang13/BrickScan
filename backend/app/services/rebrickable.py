import httpx
import redis.asyncio as redis
import json
from typing import Optional, List, Dict, Any
from app.core.config import settings

redis_client = None


async def get_redis():
    global redis_client
    if redis_client is None:
        redis_client = await redis.from_url(settings.REDIS_URL)
    return redis_client


async def get_set(set_num: str) -> Optional[Dict[str, Any]]:
    cache = await get_redis()
    cache_key = f"rebrickable:set:{set_num}"

    cached = await cache.get(cache_key)
    if cached:
        return json.loads(cached)

    headers = {"Authorization": f"key {settings.REBRICKABLE_API_KEY}"}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                f"https://rebrickable.com/api/v3/lego/sets/{set_num}/",
                headers=headers,
                timeout=10.0,
            )
            response.raise_for_status()
            data = response.json()
            await cache.setex(cache_key, 86400, json.dumps(data))
            return data
        except httpx.HTTPError:
            return None


async def get_set_parts(set_num: str) -> Optional[List[Dict[str, Any]]]:
    cache = await get_redis()
    cache_key = f"rebrickable:set_parts:{set_num}"

    cached = await cache.get(cache_key)
    if cached:
        return json.loads(cached)

    headers = {"Authorization": f"key {settings.REBRICKABLE_API_KEY}"}
    all_parts = []

    async with httpx.AsyncClient() as client:
        try:
            url = f"https://rebrickable.com/api/v3/lego/sets/{set_num}/parts/"
            while url:
                response = await client.get(url, headers=headers, timeout=10.0)
                response.raise_for_status()
                data = response.json()
                all_parts.extend(data.get("results", []))
                url = data.get("next")

            await cache.setex(cache_key, 86400, json.dumps(all_parts))
            return all_parts
        except httpx.HTTPError:
            return None


async def search_sets(query: str, theme: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
    cache = await get_redis()
    cache_key = f"rebrickable:search_sets:{query}:{theme or 'all'}"

    cached = await cache.get(cache_key)
    if cached:
        return json.loads(cached)

    headers = {"Authorization": f"key {settings.REBRICKABLE_API_KEY}"}
    params = {"search": query}
    if theme:
        params["theme"] = theme

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                "https://rebrickable.com/api/v3/lego/sets/",
                headers=headers,
                params=params,
                timeout=10.0,
            )
            response.raise_for_status()
            data = response.json()
            results = data.get("results", [])
            await cache.setex(cache_key, 3600, json.dumps(results))
            return results
        except httpx.HTTPError:
            return None


async def search_parts(query: str) -> Optional[List[Dict[str, Any]]]:
    cache = await get_redis()
    cache_key = f"rebrickable:search_parts:{query}"

    cached = await cache.get(cache_key)
    if cached:
        return json.loads(cached)

    headers = {"Authorization": f"key {settings.REBRICKABLE_API_KEY}"}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                "https://rebrickable.com/api/v3/lego/parts/",
                headers=headers,
                params={"search": query},
                timeout=10.0,
            )
            response.raise_for_status()
            data = response.json()
            results = data.get("results", [])
            await cache.setex(cache_key, 3600, json.dumps(results))
            return results
        except httpx.HTTPError:
            return None


async def get_part(part_num: str) -> Optional[Dict[str, Any]]:
    cache = await get_redis()
    cache_key = f"rebrickable:part:{part_num}"

    cached = await cache.get(cache_key)
    if cached:
        return json.loads(cached)

    headers = {"Authorization": f"key {settings.REBRICKABLE_API_KEY}"}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                f"https://rebrickable.com/api/v3/lego/parts/{part_num}/",
                headers=headers,
                timeout=10.0,
            )
            response.raise_for_status()
            data = response.json()
            await cache.setex(cache_key, 86400, json.dumps(data))
            return data
        except httpx.HTTPError:
            return None
