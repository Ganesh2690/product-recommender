# deployment/Dockerfile
# Multi-stage build for the Product Recommender Flask API.
# Base image pinned to Python 3.11 slim for reproducibility.

# ---- Build stage ---------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ---- Runtime stage -------------------------------------------------------
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy source code
COPY src/ src/
COPY deployment/gunicorn.conf.py deployment/gunicorn.conf.py

# Non-root user for security
RUN useradd --no-create-home --shell /bin/false appuser
RUN chown -R appuser:appuser /app
USER appuser

# Create runtime directories
RUN mkdir -p logs models data/processed

# Expose API port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')" || exit 1

CMD ["gunicorn", "-c", "deployment/gunicorn.conf.py", "src.serving.app:create_app()"]
