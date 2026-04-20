#!/usr/bin/env bash
# scripts/run_train.sh — Training-only pipeline (assumes data already downloaded).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

echo "============================================================"
echo "  Training Pipeline"
echo "============================================================"

echo ""
echo "[1/2] Training all models..."
python -m src.pipelines.train_pipeline

echo ""
echo "[2/2] Evaluating models and writing reports..."
python -m src.evaluation.benchmark
python -m src.evaluation.report

echo ""
echo "Training pipeline complete."
