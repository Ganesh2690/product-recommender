"""
conftest.py — Root pytest configuration.

Ensures the project root is always in sys.path before any test is
collected or imported, so all `src.*` absolute imports resolve correctly
regardless of how pytest is invoked (editable install, local dev, CI).
"""

import sys
from pathlib import Path

# Insert project root at position 0 so `import src.*` always works.
_project_root = str(Path(__file__).resolve().parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
