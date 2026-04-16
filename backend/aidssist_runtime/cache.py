from __future__ import annotations

import json
import time
from functools import lru_cache

try:
    from redis import Redis
except ImportError:
    Redis = None

from .config import get_settings
from .metrics import observe_cache


class CacheStore:
    def __init__(self):
        settings = get_settings()
        self.default_ttl = settings.cache_ttl_seconds
        self._local_cache: dict[str, tuple[float | None, str]] = {}
        self._redis = None

        if Redis is not None and settings.redis_url:
            try:
                self._redis = Redis.from_url(settings.redis_url, decode_responses=True)
                self._redis.ping()
            except Exception:
                self._redis = None

    def get_json(self, key: str):
        if self._redis is not None:
            value = self._redis.get(key)
            if value is None:
                observe_cache("get", "miss")
                return None
            observe_cache("get", "hit")
            return json.loads(value)

        cached = self._local_cache.get(key)
        if cached is None:
            observe_cache("get", "miss")
            return None

        expires_at, payload = cached
        if expires_at is not None and expires_at < time.time():
            self._local_cache.pop(key, None)
            observe_cache("get", "miss")
            return None

        observe_cache("get", "hit")
        return json.loads(payload)

    def set_json(self, key: str, value, ttl_seconds: int | None = None) -> None:
        ttl = int(ttl_seconds or self.default_ttl)
        payload = json.dumps(value, default=str)

        if self._redis is not None:
            self._redis.set(name=key, value=payload, ex=ttl)
            observe_cache("set", "stored")
            return

        expires_at = time.time() + ttl if ttl > 0 else None
        self._local_cache[key] = (expires_at, payload)
        observe_cache("set", "stored")

    def delete(self, key: str) -> None:
        if self._redis is not None:
            self._redis.delete(key)
            observe_cache("delete", "stored")
            return

        self._local_cache.pop(key, None)
        observe_cache("delete", "stored")

    def delete_prefix(self, prefix: str) -> int:
        deleted = 0
        if self._redis is not None:
            cursor = 0
            pattern = f"{prefix}*"
            while True:
                cursor, keys = self._redis.scan(cursor=cursor, match=pattern, count=100)
                if keys:
                    deleted += self._redis.delete(*keys)
                if cursor == 0:
                    break
            observe_cache("delete_prefix", "stored")
            return deleted

        for key in list(self._local_cache.keys()):
            if key.startswith(prefix):
                self._local_cache.pop(key, None)
                deleted += 1
        observe_cache("delete_prefix", "stored")
        return deleted


@lru_cache(maxsize=1)
def get_cache_store() -> CacheStore:
    return CacheStore()
