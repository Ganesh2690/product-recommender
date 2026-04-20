#!/usr/bin/env env python3
"""
scripts/check_logs.py
Logging health check — verifies all required log files exist, are non-empty,
and that the run_log.jsonl has valid JSONL entries.

Exit code 0 = healthy, 1 = issues found.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = ROOT / "logs"

REQUIRED_LOG_FILES = [
    "master_execution_log.md",
    "decision_log.md",
    "run_log.jsonl",
    "data_pipeline.log",
    "model_training.log",
    "evaluation.log",
    "api.log",
    "test.log",
    "checkpoint_status.md",
]

issues: list[str] = []


def check_file_exists_and_nonempty(path: Path) -> bool:
    if not path.exists():
        issues.append(f"MISSING: {path.relative_to(ROOT)}")
        return False
    if path.stat().st_size == 0:
        issues.append(f"EMPTY:   {path.relative_to(ROOT)}")
        return False
    return True


def check_jsonl(path: Path) -> None:
    """Verify every line in a .jsonl file is valid JSON."""
    if not path.exists():
        return
    bad_lines = []
    with path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as exc:
                bad_lines.append(f"  line {i}: {exc}")
    if bad_lines:
        issues.append(f"INVALID JSONL in {path.name}:\n" + "\n".join(bad_lines))


def check_docs() -> None:
    """Verify key documentation files exist."""
    docs_dir = ROOT / "docs"
    expected = [
        "IMPLEMENTATION_JOURNAL.md",
        "ARCHITECTURE_DECISIONS.md",
        "DATA_DICTIONARY.md",
    ]
    for name in expected:
        p = docs_dir / name
        if not p.exists():
            issues.append(f"MISSING doc: docs/{name}")


def main() -> int:
    print("=" * 60)
    print("  Logging Health Check")
    print("=" * 60)

    # Check all required log files
    for name in REQUIRED_LOG_FILES:
        p = LOGS_DIR / name
        check_file_exists_and_nonempty(p)

    # Check JSONL validity
    check_jsonl(LOGS_DIR / "run_log.jsonl")

    # Check docs
    check_docs()

    if issues:
        print(f"\n[FAIL] Found {len(issues)} issue(s):\n")
        for issue in issues:
            print(f"  - {issue}")
        print()
        return 1
    else:
        print("\n[PASS] All log files present and valid.\n")
        print(f"  Checked {len(REQUIRED_LOG_FILES)} log files in {LOGS_DIR}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
