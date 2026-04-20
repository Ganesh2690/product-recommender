"""
validation.py — Input and data validation utilities.
"""

import sys
from pathlib import Path
from typing import Any, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.logging_utils import get_logger

logger = get_logger("validation", "master")


def validate_positive_int(value: Any, name: str = "value") -> int:
    """Validate that a value is a positive integer."""
    try:
        v = int(value)
        if v <= 0:
            raise ValueError(f"{name} must be positive, got {v}")
        return v
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid {name}: {value}") from e


def validate_rating(rating: Any) -> float:
    """Validate a rating value is in [1, 5] range."""
    try:
        r = float(rating)
        if not (1.0 <= r <= 5.0):
            raise ValueError(f"Rating must be in [1, 5], got {r}")
        return r
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid rating: {rating}") from e


def validate_list_of_ints(value: Any, name: str = "list") -> List[int]:
    """Validate that a value is a list of integers."""
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list, got {type(value).__name__}")
    result = []
    for item in value:
        try:
            result.append(int(item))
        except (TypeError, ValueError):
            raise ValueError(f"All items in {name} must be integers, got: {item}")
    return result


def check_required_files(paths: List[Path]) -> bool:
    """Check that all required files exist."""
    all_ok = True
    for p in paths:
        if not p.exists():
            logger.error(f"Required file missing: {p}")
            all_ok = False
        else:
            logger.debug(f"File exists: {p}")
    return all_ok
