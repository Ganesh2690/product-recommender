"""src/serving package init"""
from . import app  # noqa: F401 — required so mock.patch("src.serving.app.*") can resolve the submodule
