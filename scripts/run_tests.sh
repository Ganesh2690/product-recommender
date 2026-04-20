#!/usr/bin/env bash
# scripts/run_tests.sh — Run the full pytest suite with coverage.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

echo "============================================================"
echo "  Running test suite"
echo "============================================================"

python -m pytest tests/ -v --tb=short "$@"

echo ""
echo "Tests complete."
