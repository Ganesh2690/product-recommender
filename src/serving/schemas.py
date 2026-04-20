"""
schemas.py — Request/response schema validation for the Flask API.

Simple validation helpers (no Pydantic — historically accurate for 2016–2017 Flask APIs).
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import MAX_RECOMMENDATIONS, TOP_N


def validate_user_id(user_id_str: Optional[str]) -> Tuple[Optional[int], Optional[str]]:
    """
    Validate and parse a user_id query parameter.

    Returns:
        (user_id, error_message) — error_message is None if valid
    """
    if user_id_str is None:
        return None, "Missing required parameter: user_id"
    try:
        uid = int(user_id_str)
        if uid <= 0:
            return None, "user_id must be a positive integer"
        return uid, None
    except (ValueError, TypeError):
        return None, f"Invalid user_id: '{user_id_str}' — must be an integer"


def validate_item_id(item_id_str: Optional[str]) -> Tuple[Optional[int], Optional[str]]:
    """Validate and parse an item_id / movie_id query parameter."""
    if item_id_str is None:
        return None, "Missing required parameter: item_id"
    try:
        iid = int(item_id_str)
        if iid <= 0:
            return None, "item_id must be a positive integer"
        return iid, None
    except (ValueError, TypeError):
        return None, f"Invalid item_id: '{item_id_str}' — must be an integer"


def validate_n(n_str: Optional[str], default: int = TOP_N) -> Tuple[int, Optional[str]]:
    """Validate and parse the n (number of recommendations) parameter."""
    if n_str is None:
        return default, None
    try:
        n = int(n_str)
        if n <= 0:
            return default, "n must be a positive integer"
        if n > MAX_RECOMMENDATIONS:
            return default, f"n must be at most {MAX_RECOMMENDATIONS}"
        return n, None
    except (ValueError, TypeError):
        return default, f"Invalid n: '{n_str}' — must be an integer"


def recommendation_response(
    user_id: int,
    recommendations: list,
    source: str,
    latency_ms: float,
) -> Dict[str, Any]:
    """Build a standardized recommendation response."""
    return {
        "user_id": user_id,
        "recommendations": recommendations,
        "count": len(recommendations),
        "source": source,
        "latency_ms": round(latency_ms, 2),
    }


def similar_items_response(
    movie_id: int,
    similar_items: list,
    source: str,
    latency_ms: float,
) -> Dict[str, Any]:
    """Build a standardized similar-items response."""
    return {
        "movie_id": movie_id,
        "similar_items": similar_items,
        "count": len(similar_items),
        "source": source,
        "latency_ms": round(latency_ms, 2),
    }


def error_response(message: str, status_code: int = 400) -> Dict[str, Any]:
    """Build a standardized error response."""
    return {
        "error": message,
        "status_code": status_code,
    }
