"""
cache.py — Recommendation cache with Redis primary and local dict fallback.

Supports:
- Redis TTL-based caching (24h default)
- Thread-safe local dict fallback
- Cache hit/miss/refresh logging
- Cache invalidation

Phase 9 — API and Cache
"""

import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import (
    CACHE_TTL_SECONDS,
    REDIS_DB,
    REDIS_HOST,
    REDIS_PASSWORD,
    REDIS_PORT,
    USE_REDIS,
)
from src.logging_utils import api_logger as logger


class LocalCache:
    """
    Thread-safe in-memory dict cache with TTL support.
    Used as fallback when Redis is unavailable.
    """

    def __init__(self, ttl_seconds: int = CACHE_TTL_SECONDS):
        self.ttl = ttl_seconds
        self._store: Dict[str, Tuple[Any, float]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key in self._store:
                value, expiry = self._store[key]
                if time.time() < expiry:
                    return value
                del self._store[key]
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        effective_ttl = ttl if ttl is not None else self.ttl
        expiry = time.time() + effective_ttl
        with self._lock:
            self._store[key] = (value, expiry)

    def delete(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def flush(self) -> None:
        with self._lock:
            self._store.clear()

    def size(self) -> int:
        with self._lock:
            return len(self._store)


class RecommendationCache:
    """
    Two-tier recommendation cache: Redis → local dict fallback.

    Cache key format:
      rec:{user_id}:{n}         → personalized recommendations
      sim:{movie_id}:{n}        → similar items
      batch:{hash}              → batch prediction results

    Logs all cache hits, misses, and refreshes.
    """

    def __init__(self):
        self._redis = None
        self._local = LocalCache(ttl_seconds=CACHE_TTL_SECONDS)
        self._backend = "local"
        self._stats = {"hits": 0, "misses": 0, "sets": 0, "errors": 0}
        self._init_redis()

    def _init_redis(self) -> None:
        """Attempt to connect to Redis. Fall back to local if unavailable."""
        if not USE_REDIS:
            logger.info("Redis disabled via config. Using local dict cache.")
            self._backend = "local"
            return

        try:
            import redis
            client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                password=REDIS_PASSWORD,
                socket_connect_timeout=2,
                socket_timeout=2,
                decode_responses=True,
            )
            client.ping()
            self._redis = client
            self._backend = "redis"
            logger.info(f"Redis cache connected: {REDIS_HOST}:{REDIS_PORT}/db{REDIS_DB}")
            logger.log_cache_event("backend_init", "redis", "Redis connected successfully")
        except Exception as e:
            logger.warning(f"Redis unavailable ({e}). Falling back to local dict cache.")
            logger.log_cache_event("backend_init", "local", "Redis unavailable — using local cache")
            self._backend = "local"

    def _make_rec_key(self, user_id: int, n: int) -> str:
        return f"rec:{user_id}:{n}"

    def _make_sim_key(self, movie_id: int, n: int) -> str:
        return f"sim:{movie_id}:{n}"

    def get_recommendations(self, user_id: int, n: int) -> Optional[List[int]]:
        """Retrieve cached recommendations for a user."""
        key = self._make_rec_key(user_id, n)
        value = self._get(key)
        if value is not None:
            self._stats["hits"] += 1
            logger.log_cache_event("hit", key, f"user={user_id}")
            return value
        self._stats["misses"] += 1
        logger.log_cache_event("miss", key, f"user={user_id}")
        return None

    def set_recommendations(self, user_id: int, n: int, recs: List[int]) -> None:
        """Cache recommendations for a user."""
        key = self._make_rec_key(user_id, n)
        self._set(key, recs)
        self._stats["sets"] += 1
        logger.log_cache_event("set", key, f"user={user_id} n_recs={len(recs)}")

    def get_similar_items(self, movie_id: int, n: int) -> Optional[List[int]]:
        """Retrieve cached similar items."""
        key = self._make_sim_key(movie_id, n)
        value = self._get(key)
        if value is not None:
            self._stats["hits"] += 1
            logger.log_cache_event("hit", key, f"movie={movie_id}")
            return value
        self._stats["misses"] += 1
        logger.log_cache_event("miss", key, f"movie={movie_id}")
        return None

    def set_similar_items(self, movie_id: int, n: int, items: List[int]) -> None:
        """Cache similar items for a movie."""
        key = self._make_sim_key(movie_id, n)
        self._set(key, items)
        self._stats["sets"] += 1

    def invalidate_user(self, user_id: int) -> None:
        """Invalidate all cached recommendations for a user."""
        for n in [10, 20, 50]:
            key = self._make_rec_key(user_id, n)
            self._delete(key)
        logger.log_cache_event("invalidate", f"user:{user_id}", "User cache invalidated")

    def flush_all(self) -> None:
        """Clear all cached recommendations."""
        if self._backend == "redis" and self._redis:
            try:
                # Only flush recommendation keys, not entire Redis DB
                keys = self._redis.keys("rec:*") + self._redis.keys("sim:*")
                if keys:
                    self._redis.delete(*keys)
                logger.info(f"Redis cache flushed: {len(keys)} keys deleted")
            except Exception as e:
                logger.log_error(e, "Redis flush failed")
        else:
            self._local.flush()
            logger.info("Local cache flushed")
        logger.log_cache_event("flush", "all", "Full cache flush")

    def get_stats(self) -> Dict:
        """Return cache statistics."""
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total if total > 0 else 0.0
        return {
            **self._stats,
            "hit_rate": round(hit_rate, 4),
            "backend": self._backend,
            "local_size": self._local.size(),
        }

    # --- Internal get/set/delete ---
    def _get(self, key: str) -> Optional[Any]:
        if self._backend == "redis" and self._redis:
            try:
                raw = self._redis.get(key)
                if raw:
                    return json.loads(raw)
            except Exception as e:
                self._stats["errors"] += 1
                logger.warning(f"Redis GET failed for {key}: {e}")
        return self._local.get(key)

    def _set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        effective_ttl = ttl or CACHE_TTL_SECONDS
        if self._backend == "redis" and self._redis:
            try:
                self._redis.setex(key, effective_ttl, json.dumps(value))
                return
            except Exception as e:
                self._stats["errors"] += 1
                logger.warning(f"Redis SET failed for {key}: {e}")
        self._local.set(key, value, ttl=effective_ttl)

    def _delete(self, key: str) -> None:
        if self._backend == "redis" and self._redis:
            try:
                self._redis.delete(key)
            except Exception:
                pass
        self._local.delete(key)


# Module-level singleton
_cache: Optional[RecommendationCache] = None


def get_cache() -> RecommendationCache:
    """Get or create the global cache singleton."""
    global _cache
    if _cache is None:
        _cache = RecommendationCache()
    return _cache
