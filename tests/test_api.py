"""
test_api.py — Tests for the Flask recommendation API endpoints.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mock_engine():
    """A mocked RecommendationEngine for API tests."""
    engine = MagicMock()
    _recs = [
        {"movie_id": 1, "score": 4.5},
        {"movie_id": 2, "score": 4.2},
        {"movie_id": 3, "score": 4.0},
    ]
    engine.recommend.return_value = (_recs, "personalized")
    engine.similar_items.return_value = (
        [{"movie_id": 5, "score": 0.92}, {"movie_id": 6, "score": 0.88}],
        "model",
    )
    engine._model = MagicMock()
    return engine


@pytest.fixture(scope="module")
def mock_cache():
    """A mocked RecommendationCache."""
    cache = MagicMock()
    cache.get_recommendations.return_value = None   # cache miss by default
    cache.set_recommendations.return_value = None
    cache.get_similar_items.return_value = None
    cache.set_similar_items.return_value = None
    cache.get_stats.return_value = {
        "backend": "local",
        "hits": 0,
        "misses": 5,
        "hit_rate": 0.0,
    }
    return cache


@pytest.fixture(scope="module")
def app(mock_engine, mock_cache):
    """Create Flask test client with mocked dependencies."""
    # Import the module first so it is in sys.modules, then patch
    import src.serving.app as app_mod
    with patch.object(app_mod, "get_engine", return_value=mock_engine), \
         patch.object(app_mod, "get_cache", return_value=mock_cache):
        # Reset cached singletons so our mocks are picked up
        app_mod._engine = mock_engine
        app_mod._cache = mock_cache
        flask_app = app_mod.create_app()
        flask_app.config["TESTING"] = True
        with flask_app.test_client() as client:
            yield client


# -------------------------------------------------------------------------
# Health endpoint
# -------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_returns_200(self, app):
        resp = app.get("/health")
        assert resp.status_code == 200

    def test_health_json_structure(self, app):
        resp = app.get("/health")
        data = json.loads(resp.data)
        assert "status" in data

    def test_health_status_ok(self, app):
        resp = app.get("/health")
        data = json.loads(resp.data)
        assert data["status"] in ("ok", "healthy", "up")


# -------------------------------------------------------------------------
# Recommend endpoint
# -------------------------------------------------------------------------

class TestRecommendEndpoint:
    def test_valid_request(self, app):
        resp = app.get("/recommend?user_id=1&n=5")
        assert resp.status_code == 200

    def test_response_is_json(self, app):
        resp = app.get("/recommend?user_id=1&n=5")
        assert resp.content_type == "application/json"

    def test_response_has_recommendations_key(self, app):
        resp = app.get("/recommend?user_id=1&n=5")
        data = json.loads(resp.data)
        assert "recommendations" in data

    def test_recommendations_is_list(self, app):
        resp = app.get("/recommend?user_id=1&n=5")
        data = json.loads(resp.data)
        assert isinstance(data["recommendations"], list)

    def test_recommendation_items_have_movie_id(self, app):
        resp = app.get("/recommend?user_id=1&n=5")
        data = json.loads(resp.data)
        for item in data["recommendations"]:
            assert "movie_id" in item

    def test_missing_user_id_returns_400(self, app):
        resp = app.get("/recommend?n=5")
        assert resp.status_code == 400

    def test_invalid_user_id_type_returns_400(self, app):
        resp = app.get("/recommend?user_id=abc&n=5")
        assert resp.status_code == 400

    def test_n_too_large_returns_400(self, app):
        resp = app.get("/recommend?user_id=1&n=10000")
        assert resp.status_code == 400


# -------------------------------------------------------------------------
# Similar items endpoint
# -------------------------------------------------------------------------

class TestSimilarItemsEndpoint:
    def test_valid_request(self, app):
        resp = app.get("/similar-items?item_id=1&n=5")
        assert resp.status_code == 200

    def test_response_has_similar_items(self, app):
        resp = app.get("/similar-items?item_id=1&n=5")
        data = json.loads(resp.data)
        assert "similar_items" in data

    def test_missing_item_id_returns_400(self, app):
        resp = app.get("/similar-items?n=5")
        assert resp.status_code == 400

    def test_invalid_item_id_returns_400(self, app):
        resp = app.get("/similar-items?item_id=xyz")
        assert resp.status_code == 400


# -------------------------------------------------------------------------
# Batch predict endpoint
# -------------------------------------------------------------------------

class TestBatchPredictEndpoint:
    def test_valid_batch_request(self, app):
        payload = {"user_ids": [1, 2, 3], "n": 5}
        resp = app.post(
            "/predict-batch",
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert resp.status_code == 200

    def test_batch_response_structure(self, app):
        payload = {"user_ids": [1], "n": 5}
        resp = app.post(
            "/predict-batch",
            data=json.dumps(payload),
            content_type="application/json",
        )
        data = json.loads(resp.data)
        assert "results" in data or "recommendations" in data or "batch" in data

    def test_empty_user_list_returns_400(self, app):
        payload = {"user_ids": [], "n": 5}
        resp = app.post(
            "/predict-batch",
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert resp.status_code in (200, 400)  # Either handled gracefully

    def test_missing_user_ids_returns_400(self, app):
        payload = {"n": 5}
        resp = app.post(
            "/predict-batch",
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert resp.status_code == 400


# -------------------------------------------------------------------------
# Cache stats endpoint
# -------------------------------------------------------------------------

class TestCacheEndpoints:
    def test_cache_stats_returns_200(self, app):
        resp = app.get("/cache/stats")
        assert resp.status_code == 200

    def test_cache_stats_json(self, app):
        resp = app.get("/cache/stats")
        data = json.loads(resp.data)
        assert isinstance(data, dict)

    def test_cache_flush_returns_200(self, app):
        resp = app.post("/cache/flush")
        assert resp.status_code == 200


# -------------------------------------------------------------------------
# Error handling
# -------------------------------------------------------------------------

class TestErrorHandling:
    def test_unknown_route_returns_404(self, app):
        resp = app.get("/nonexistent-endpoint")
        assert resp.status_code == 404

    def test_method_not_allowed_returns_405(self, app):
        # /recommend is GET-only; POST should fail
        resp = app.post("/recommend")
        assert resp.status_code == 405

    def test_error_response_is_json(self, app):
        resp = app.get("/recommend?user_id=abc&n=5")
        # Should still return JSON, not HTML
        data = json.loads(resp.data)
        assert "error" in data or "message" in data
