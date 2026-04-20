"""
io.py — I/O utilities for the Recommender System.

Provides safe file read/write helpers with logging.
"""

import json
import pickle
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.logging_utils import get_logger

logger = get_logger("io_utils", "master")


def save_pickle(obj: Any, path: Path) -> None:
    """Save an object to a pickle file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "wb") as f:
        pickle.dump(obj, f)
    logger.log_artifact(str(path), f"Pickle file saved: {path.name}")


def load_pickle(path: Path) -> Any:
    """Load an object from a pickle file."""
    if not path.exists():
        raise FileNotFoundError(f"Pickle file not found: {path}")
    with open(str(path), "rb") as f:
        obj = pickle.load(f)
    logger.debug(f"Loaded pickle: {path.name}")
    return obj


def save_json(data: Any, path: Path, indent: int = 2) -> None:
    """Save a dict/list to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)
    logger.log_artifact(str(path), f"JSON file saved: {path.name}")


def load_json(path: Path) -> Any:
    """Load a JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(str(path), encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path) -> Path:
    """Create directory (and parents) if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path
