#!/usr/bin/env bash
# scripts/run_api.sh — Start the Flask recommendation API.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

# Load .env if it exists
if [ -f "$ROOT_DIR/.env" ]; then
    export $(grep -v '^#' "$ROOT_DIR/.env" | xargs)
fi

HOST="${FLASK_HOST:-0.0.0.0}"
PORT="${FLASK_PORT:-5000}"
ENV="${FLASK_ENV:-development}"

echo "Starting Flask API on $HOST:$PORT (env=$ENV)..."

if [ "$ENV" = "production" ]; then
    echo "Using Gunicorn (production mode)..."
    exec gunicorn \
        -c "$ROOT_DIR/deployment/gunicorn.conf.py" \
        "src.serving.app:create_app()"
else
    echo "Using Flask dev server..."
    exec python -m src.serving.app
fi
