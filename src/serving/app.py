"""
app.py — Flask REST API for the Recommender System.

Endpoints:
  GET  /health                           — liveness check
  GET  /recommend?user_id=<id>&n=10      — personalized top-N recommendations
  GET  /similar-items?item_id=<id>&n=10  — item similarity
  POST /predict-batch                    — batch recommendations for multiple users

Phase 9 — Flask API and Cache implementation.
Historical accuracy: Flask was the dominant ML API framework in 2016–2017.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from flask import Flask, jsonify, request

from src.config import FLASK_DEBUG, FLASK_HOST, FLASK_PORT, TOP_N
from src.logging_utils import api_logger as logger
from src.models.registry import get_engine
from src.serving.cache import get_cache
from src.serving.schemas import (
    error_response,
    recommendation_response,
    similar_items_response,
    validate_item_id,
    validate_n,
    validate_user_id,
)

app = Flask(__name__, template_folder="../../templates")
app.config["JSON_SORT_KEYS"] = False

# Admin dashboard
try:
    from src.serving.admin import admin_bp
    app.register_blueprint(admin_bp, url_prefix="/admin")
except Exception:
    pass


# ---------------------------------------------------------------------------
# Startup: load model and cache
# ---------------------------------------------------------------------------
_engine = None
_cache = None


def _get_engine():
    global _engine
    if _engine is None:
        logger.info("Initializing recommendation engine...")
        _engine = get_engine()
    return _engine


def _get_cache():
    global _cache
    if _cache is None:
        _cache = get_cache()
    return _cache


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    """
    Liveness and readiness check.

    Returns:
        200 OK with status and model info
    """
    start = time.time()
    try:
        engine = _get_engine()
        model_loaded = engine._model is not None
        cache_stats = _get_cache().get_stats()
        latency_ms = (time.time() - start) * 1000

        response = {
            "status": "ok",
            "model_loaded": model_loaded,
            "cache_backend": cache_stats["backend"],
            "cache_hit_rate": cache_stats["hit_rate"],
            "latency_ms": round(latency_ms, 2),
        }
        logger.info(f"[GET /health] status=ok latency={latency_ms:.1f}ms model={model_loaded}")
        return jsonify(response), 200
    except Exception as e:
        logger.log_error(e, "Health check failed")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/recommend", methods=["GET"])
def recommend():
    """
    Get personalized top-N recommendations for a user.

    Query params:
      user_id (required): integer user ID
      n (optional): number of recommendations (default: 10, max: 100)

    Returns:
        200: {user_id, recommendations, count, source, latency_ms}
        400: {error, status_code}
    """
    start = time.time()

    user_id, err = validate_user_id(request.args.get("user_id"))
    if err:
        logger.warning(f"[GET /recommend] Bad request: {err}")
        return jsonify(error_response(err)), 400

    n, n_err = validate_n(request.args.get("n"), default=TOP_N)
    if n_err:
        logger.warning(f"[GET /recommend] Bad request: {n_err}")
        return jsonify(error_response(n_err)), 400

    cache = _get_cache()

    # Check cache first
    cached_recs = cache.get_recommendations(user_id, n)
    if cached_recs is not None:
        latency_ms = (time.time() - start) * 1000
        logger.info(f"[GET /recommend] user={user_id} n={n} source=cache latency={latency_ms:.1f}ms")
        return jsonify(recommendation_response(user_id, cached_recs, "cache", latency_ms)), 200

    # Generate recommendations
    engine = _get_engine()
    try:
        recs, source = engine.recommend(user_id, n=n)
    except Exception as e:
        logger.log_error(e, f"recommend() failed for user {user_id}")
        return jsonify(error_response("Recommendation generation failed", 500)), 500

    # Cache result
    cache.set_recommendations(user_id, n, recs)

    latency_ms = (time.time() - start) * 1000
    logger.info(f"[GET /recommend] user={user_id} n={n} source={source} latency={latency_ms:.1f}ms")

    return jsonify(recommendation_response(user_id, recs, source, latency_ms)), 200


@app.route("/similar-items", methods=["GET"])
def similar_items():
    """
    Get similar items for a given movie.

    Query params:
      item_id (required): integer movie ID
      n (optional): number of similar items (default: 10)

    Returns:
        200: {movie_id, similar_items, count, source, latency_ms}
        400: {error, status_code}
    """
    start = time.time()

    movie_id, err = validate_item_id(request.args.get("item_id"))
    if err:
        logger.warning(f"[GET /similar-items] Bad request: {err}")
        return jsonify(error_response(err)), 400

    n, _ = validate_n(request.args.get("n"), default=10)
    cache = _get_cache()

    # Check cache
    cached_sims = cache.get_similar_items(movie_id, n)
    if cached_sims is not None:
        latency_ms = (time.time() - start) * 1000
        return jsonify(similar_items_response(movie_id, cached_sims, "cache", latency_ms)), 200

    # Generate
    engine = _get_engine()
    try:
        items, source = engine.similar_items(movie_id, n=n)
    except Exception as e:
        logger.log_error(e, f"similar_items() failed for movie {movie_id}")
        return jsonify(error_response("Similar items lookup failed", 500)), 500

    cache.set_similar_items(movie_id, n, items)

    latency_ms = (time.time() - start) * 1000
    logger.info(f"[GET /similar-items] movie={movie_id} n={n} source={source} latency={latency_ms:.1f}ms")
    return jsonify(similar_items_response(movie_id, items, source, latency_ms)), 200


@app.route("/predict-batch", methods=["POST"])
def predict_batch():
    """
    Generate recommendations for multiple users in one call.

    Request body (JSON):
      {
        "user_ids": [1, 2, 3, ...],
        "n": 10  (optional)
      }

    Returns:
        200: {results: {user_id: [recs]}, latency_ms}
        400: {error, status_code}
    """
    start = time.time()

    data = request.get_json(silent=True)
    if not data or "user_ids" not in data:
        return jsonify(error_response("Request body must contain 'user_ids' list")), 400

    user_ids = data.get("user_ids", [])
    if not isinstance(user_ids, list) or not user_ids:
        return jsonify(error_response("'user_ids' must be a non-empty list")), 400
    if len(user_ids) > 500:
        return jsonify(error_response("Maximum 500 users per batch request")), 400

    n = int(data.get("n", TOP_N))
    if n <= 0 or n > 100:
        n = TOP_N

    engine = _get_engine()
    cache = _get_cache()
    results = {}

    for uid in user_ids:
        try:
            uid = int(uid)
        except (ValueError, TypeError):
            results[str(uid)] = {"error": "invalid user_id"}
            continue

        cached = cache.get_recommendations(uid, n)
        if cached is not None:
            results[uid] = cached
            continue

        try:
            recs, _ = engine.recommend(uid, n=n)
            cache.set_recommendations(uid, n, recs)
            results[uid] = recs
        except Exception as e:
            results[uid] = {"error": str(e)}

    latency_ms = (time.time() - start) * 1000
    logger.info(f"[POST /predict-batch] users={len(user_ids)} latency={latency_ms:.1f}ms")
    return jsonify({"results": results, "latency_ms": round(latency_ms, 2)}), 200


@app.route("/cache/stats", methods=["GET"])
def cache_stats():
    """Return cache statistics."""
    stats = _get_cache().get_stats()
    return jsonify(stats), 200


@app.route("/cache/flush", methods=["POST"])
def cache_flush():
    """Flush all cached recommendations (admin endpoint)."""
    _get_cache().flush_all()
    logger.info("[POST /cache/flush] Cache flushed")
    return jsonify({"status": "flushed"}), 200


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------
@app.errorhandler(404)
def not_found(e):
    return jsonify(error_response(f"Endpoint not found: {request.path}", 404)), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify(error_response(f"Method {request.method} not allowed", 405)), 405


@app.errorhandler(500)
def internal_error(e):
    logger.log_error(e, "Unhandled 500 error")
    return jsonify(error_response("Internal server error", 500)), 500


# ---------------------------------------------------------------------------
# Application factory (for testing)
# ---------------------------------------------------------------------------
def create_app() -> Flask:
    """Return the configured Flask application instance."""
    return app


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger.start_phase("Phase 9 - API", "Starting Flask recommendation API")
    logger.info(f"Server starting on {FLASK_HOST}:{FLASK_PORT} (debug={FLASK_DEBUG})")
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)
