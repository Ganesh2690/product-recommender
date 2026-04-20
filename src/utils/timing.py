"""
timing.py — Runtime measurement utilities.
"""

import contextlib
import time
from typing import Optional

from src.logging_utils import get_logger

logger = get_logger("timing", "master")


class Timer:
    """Context manager for measuring elapsed time."""

    def __init__(self, name: str = "operation", log: bool = True):
        self.name = name
        self.log = log
        self.elapsed: float = 0.0
        self._start: Optional[float] = None

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self._start
        if self.log:
            logger.log_runtime(self.name, self.elapsed)

    @property
    def elapsed_ms(self) -> float:
        return self.elapsed * 1000


def measure_latency_ms(func, *args, **kwargs):
    """
    Call func(*args, **kwargs) and return (result, latency_ms).
    """
    start = time.time()
    result = func(*args, **kwargs)
    elapsed_ms = (time.time() - start) * 1000
    return result, elapsed_ms
