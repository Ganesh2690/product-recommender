#!/usr/bin/env bash
# scripts/run_all.sh — Run the complete data-to-evaluation pipeline.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

echo "============================================================"
echo "  Personalized Product Recommender — Full Pipeline"
echo "============================================================"

echo ""
echo "[1/7] Downloading MovieLens 1M dataset..."
python -m src.data.download

echo ""
echo "[2/7] Validating dataset schema and integrity..."
python -m src.data.validate

echo ""
echo "[3/7] Preprocessing features and building user-item matrix..."
python -m src.data.preprocess

echo ""
echo "[4/7] Creating time-aware train/val/test splits..."
python -m src.data.split

echo ""
echo "[5/7] Training all models (Popularity, ItemCF, SVD, Hybrid)..."
python -m src.pipelines.train_pipeline

echo ""
echo "[6/7] Running offline evaluation benchmark..."
python -m src.evaluation.benchmark
python -m src.evaluation.report

echo ""
echo "[7/7] Generating batch recommendations and pre-warming cache..."
python -m src.pipelines.batch_recommend --prewarm

echo ""
echo "============================================================"
echo "  Pipeline complete. Check logs/ for execution details."
echo "============================================================"
