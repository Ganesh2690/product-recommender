"""
test_cache.py — Tests for the two-tier recommendation cache (Redis + local dict fallback).
"""

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# -------------------------------------------------------------------------
# LocalCache tests
# -------------------------------------------------------------------------

class TestLocalCache:
    @pytest.fixture
    def cache(self):
        from src.serving.cache import LocalCache
        return LocalCache(ttl_seconds=2)  # 2-second TTL for testing

    def test_get_miss(self, cache):
        assert cache.get("missing_key") is None

    def test_set_and_get(self, cache):
        cache.set("key1", [1, 2, 3])
        result = cache.get("key1")
        assert result == [1, 2, 3]

    def test_overwrite(self, cache):
        cache.set("key2", "original")
        cache.set("key2", "updated")
        assert cache.get("key2") == "updated"

    def test_ttl_expiry(self, cache):
        from src.serving.cache import LocalCache
        short_cache = LocalCache(ttl_seconds=1)  # 1-second TTL
        short_cache.set("exp_key", "value")
        assert short_cache.get("exp_key") == "value"
        time.sleep(1.1)
        assert short_cache.get("exp_key") is None, "Entry should have expired"

    def test_delete(self, cache):
        cache.set("del_key", "some_value")
        cache.delete("del_key")
        assert cache.get("del_key") is None

    def test_delete_nonexistent_key_no_error(self, cache):
        # Should not raise
        cache.delete("ghost_key")

    def test_flush(self, cache):
        cache.set("a", 1)
        cache.set("b", 2)
        cache.flush()
        assert cache.get("a") is None
        assert cache.get("b") is None

    def test_set_custom_ttl(self, cache):
        from src.serving.cache import LocalCache
        c = LocalCache(ttl_seconds=100)
        c.set("custom_ttl_key", "val", ttl=1)
        assert c.get("custom_ttl_key") == "val"
        time.sleep(1.1)
        assert c.get("custom_ttl_key") is None


# -------------------------------------------------------------------------
# RecommendationCache tests (with Redis forced off via env)
# -------------------------------------------------------------------------

class TestRecommendationCache:
    @pytest.fixture
    def rec_cache(self):
        """RecommendationCache forced to use local fallback by disabling Redis."""
        import src.serving.cache as cache_mod
        # Reset singleton so each test gets a fresh instance
        cache_mod._cache = None
        # Disable Redis via monkeypatched config
        import unittest.mock as mock
        with mock.patch("src.serving.cache.USE_REDIS", False):
            from src.serving.cache import RecommendationCache
            return RecommendationCache()

    def test_get_miss(self, rec_cache):
        result = rec_cache.get_recommendations(9999, 10)
        assert result is None

    def test_set_and_get(self, rec_cache):
        recs = [1, 2, 3]
        rec_cache.set_recommendations(1, 5, recs)
        result = rec_cache.get_recommendations(1, 5)
        assert result == recs

    def test_cache_miss_hit_counting(self, rec_cache):
        """After a miss and then a hit, stats should reflect both."""
        rec_cache.get_recommendations(99998, 10)  # miss
        rec_cache.set_recommendations(99997, 10, [1, 2])
        rec_cache.get_recommendations(99997, 10)  # hit
        stats = rec_cache.get_stats()
        assert isinstance(stats, dict)
        assert stats["misses"] >= 1
        assert stats["hits"] >= 1

    def test_flush(self, rec_cache):
        rec_cache.set_recommendations(88888, 10, [1, 2])
        rec_cache.flush_all()
        assert rec_cache.get_recommendations(88888, 10) is None

    def test_stats_returns_dict(self, rec_cache):
        stats = rec_cache.get_stats()
        assert isinstance(stats, dict)

    def test_stats_has_expected_keys(self, rec_cache):
        stats = rec_cache.get_stats()
        keys = set(stats.keys())
        assert "hits" in keys
        assert "misses" in keys
        assert "hit_rate" in keys


# -------------------------------------------------------------------------
# RecommendationCache with Redis available (mocked)
# -------------------------------------------------------------------------

class TestRecommendationCacheWithRedis:
    @pytest.fixture
    def mock_redis_client(self):
        redis_mock = MagicMock()
        redis_mock.ping.return_value = True
        redis_mock.get.return_value = None  # cache miss by default
        redis_mock.setex.return_value = True
        redis_mock.delete.return_value = 1
        redis_mock.keys.return_value = []
        return redis_mock

    @pytest.fixture
    def cache_with_redis(self, mock_redis_client):
        import src.serving.cache as cache_mod
        cache_mod._cache = None
        with patch("src.serving.cache.USE_REDIS", True), \
             patch("redis.Redis", return_value=mock_redis_client):
            from src.serving.cache import RecommendationCache
            return RecommendationCache()

    def test_redis_miss_returns_none(self, cache_with_redis, mock_redis_client):
        mock_redis_client.get.return_value = None
        result = cache_with_redis.get_recommendations(1, 10)
        assert result is None

    def test_redis_hit_returns_value(self, cache_with_redis, mock_redis_client):
        import json
        recs = [1, 5, 9]
        mock_redis_client.get.return_value = json.dumps(recs)
        result = cache_with_redis.get_recommendations(1, 10)
        assert result == recs

    def test_redis_set_called_on_cache_set(self, cache_with_redis, mock_redis_client):
        recs = [1, 2, 3]
        cache_with_redis.set_recommendations(2, 10, recs)
        assert mock_redis_client.setex.called

    def test_redis_connection_failure_fallback(self):
        """When Redis ping fails, should fall back to local cache without raising."""
        import src.serving.cache as cache_mod
        cache_mod._cache = None
        bad_redis = MagicMock()
        bad_redis.ping.side_effect = Exception("connection refused")
        with patch("src.serving.cache.USE_REDIS", True), \
             patch("redis.Redis", return_value=bad_redis):
            from src.serving.cache import RecommendationCache
            cache = RecommendationCache()
            # Should use local cache fallback
            cache.set_recommendations(77, 10, [99])
            result = cache.get_recommendations(77, 10)
            assert result == [99]


# -------------------------------------------------------------------------
# Cache key generation (internal method tests)
# -------------------------------------------------------------------------

class TestCacheKeyGeneration:
    @pytest.fixture
    def cache(self):
        import src.serving.cache as cache_mod
        cache_mod._cache = None
        with patch("src.serving.cache.USE_REDIS", False):
            from src.serving.cache import RecommendationCache
            return RecommendationCache()

    def test_user_recs_key_unique_per_user(self, cache):
        key1 = cache._make_rec_key(user_id=1, n=10)
        key2 = cache._make_rec_key(user_id=2, n=10)
        assert key1 != key2

    def test_user_recs_key_unique_per_n(self, cache):
        key1 = cache._make_rec_key(user_id=1, n=5)
        key2 = cache._make_rec_key(user_id=1, n=10)
        assert key1 != key2

    def test_similar_items_key_unique_per_item(self, cache):
        key1 = cache._make_sim_key(movie_id=10, n=5)
        key2 = cache._make_sim_key(movie_id=11, n=5)
        assert key1 != key2
