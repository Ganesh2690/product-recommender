"""
admin.py — Admin dashboard Blueprint for the Recommender System.

Routes:
  GET /admin/             — KPI dashboard
  GET /admin/churn        — Churn prediction summary
  GET /admin/models       — Model registry
  GET /admin/ab           — A/B experiment results
  GET /admin/cache        — Cache stats

Secured with HTTP Basic Auth (credentials from env: ADMIN_USER / ADMIN_PASSWORD).
"""

import json
import os
import sys
from pathlib import Path
from functools import wraps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from flask import Blueprint, Response, jsonify, render_template, request

from src.config import ARTIFACTS_DIR
from src.logging_utils import get_logger

logger = get_logger("admin", "api")

admin_bp = Blueprint("admin", __name__, template_folder="../../templates/admin")

CHURN_DIR = ARTIFACTS_DIR / "churn"
MODELS_DIR = ARTIFACTS_DIR / "models"


# ---------------------------------------------------------------------------
# Basic Auth
# ---------------------------------------------------------------------------

def _check_auth(username: str, password: str) -> bool:
    expected_user = os.getenv("ADMIN_USER", "admin")
    expected_pass = os.getenv("ADMIN_PASSWORD", "changeme")
    return username == expected_user and password == expected_pass


def _requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not _check_auth(auth.username, auth.password):
            return Response(
                "Authentication required.",
                401,
                {"WWW-Authenticate": 'Basic realm="Admin"'},
            )
        return f(*args, **kwargs)
    return decorated


# ---------------------------------------------------------------------------
# Helper: safe JSON load
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict:
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return {}


def _load_benchmark() -> dict:
    """Load benchmark results — check both ARTIFACTS_DIR and MODELS_DIR."""
    from src.config import MODELS_DIR
    for candidate in [
        ARTIFACTS_DIR / "benchmark_results.json",
        MODELS_DIR / "benchmark_results.json",
        ARTIFACTS_DIR / "models" / "benchmark_results.json",
    ]:
        data = _load_json(candidate)
        if data:
            return data
    return {}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@admin_bp.route("/")
@_requires_auth
def dashboard():
    """Main KPI dashboard."""
    benchmark = _load_benchmark()
    churn_eval = _load_json(CHURN_DIR / "evaluation_report.json")
    shap_imp = _load_json(CHURN_DIR / "shap_global_importance.json")
    drift = _load_json(CHURN_DIR / "drift_report.json")

    # Model version
    version_file = ARTIFACTS_DIR / "current_model_version.txt"
    current_model = version_file.read_text().strip() if version_file.exists() else "Not deployed"

    # Cache stats via the serving cache
    try:
        from src.serving.cache import get_cache
        cache_stats = get_cache().get_stats()
    except Exception:
        cache_stats = {}

    context = {
        "current_model": Path(current_model).name if current_model != "Not deployed" else current_model,
        "benchmark": benchmark,
        "churn_eval": churn_eval,
        "shap_importance": list(shap_imp.items())[:8] if shap_imp else [],
        "drift": drift,
        "cache_stats": cache_stats,
        "target_hr": 0.35,
        "target_p10": 0.10,
        "target_auc": 0.88,
        "target_f1": 0.75,
    }
    return render_template("dashboard.html", **context)


@admin_bp.route("/churn")
@_requires_auth
def churn_view():
    """Churn prediction summary page."""
    churn_eval = _load_json(CHURN_DIR / "evaluation_report.json")
    tuning = _load_json(CHURN_DIR / "tuning_results.json")
    drift = _load_json(CHURN_DIR / "drift_report.json")
    sf_result = _load_json(CHURN_DIR / "sf_push_result.json")

    return render_template(
        "churn.html",
        churn_eval=churn_eval,
        tuning=tuning,
        drift=drift,
        sf_result=sf_result,
    )


@admin_bp.route("/models")
@_requires_auth
def models_view():
    """Model registry."""
    try:
        from src.models.registry import ModelRegistry
        registry = ModelRegistry()
        model_list = registry.list_models()[-20:]  # most recent 20
    except Exception as e:
        model_list = []
        logger.warning(f"Failed to load model registry: {e}")

    version_file = ARTIFACTS_DIR / "current_model_version.txt"
    current = version_file.read_text().strip() if version_file.exists() else ""
    return render_template("models.html", models=model_list, current=current)


@admin_bp.route("/api/kpi")
@_requires_auth
def api_kpi():
    """JSON API endpoint for dashboard KPI widgets."""
    benchmark = _load_benchmark()
    churn_eval = _load_json(CHURN_DIR / "evaluation_report.json")

    hr10 = benchmark.get("hybrid", {}).get("hr@10", benchmark.get("hr@10", None))
    p10 = benchmark.get("hybrid", {}).get("precision@10", benchmark.get("precision@10", None))

    return jsonify({
        "recommender": {
            "hr_at_10": hr10,
            "precision_at_10": p10,
            "target_hr_at_10": 0.35,
            "target_precision_at_10": 0.10,
            "meets_hr": hr10 >= 0.35 if hr10 else None,
            "meets_p10": p10 >= 0.10 if p10 else None,
        },
        "churn": {
            "auc_roc": churn_eval.get("auc_roc"),
            "f1_score": churn_eval.get("f1_score"),
            "targets_met": churn_eval.get("targets_met", {}),
        },
    })
