.PHONY: help install install-dev download preprocess train evaluate serve test lint clean

PYTHON ?= python
PYTEST ?= pytest
PIP    ?= pip

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "  install        Install production dependencies"
	@echo "  install-dev    Install development dependencies"
	@echo "  download       Download MovieLens 1M dataset"
	@echo "  preprocess     Run data preprocessing and split"
	@echo "  train          Train all recommender models"
	@echo "  evaluate       Run offline evaluation benchmark"
	@echo "  serve          Start Flask API (development mode)"
	@echo "  batch          Generate batch recommendations"
	@echo "  promote        Promote best model to production"
	@echo "  test           Run pytest test suite"
	@echo "  lint           Run flake8 linting"
	@echo "  clean          Remove generated artifacts"
	@echo "  all            download -> train -> evaluate -> test"

# -----------------------------------------------------------------------
# Installation
# -----------------------------------------------------------------------
install:
	$(PIP) install -r requirements.txt

install-dev:
	$(PIP) install -r requirements.txt
	$(PIP) install pytest pytest-cov flake8

# -----------------------------------------------------------------------
# Data pipeline
# -----------------------------------------------------------------------
download:
	$(PYTHON) -m src.data.download

validate:
	$(PYTHON) -m src.data.validate

preprocess:
	$(PYTHON) -m src.data.preprocess

split:
	$(PYTHON) -m src.data.split

data-pipeline: download validate preprocess split

# -----------------------------------------------------------------------
# Model training & evaluation
# -----------------------------------------------------------------------
train:
	$(PYTHON) -m src.pipelines.train_pipeline

evaluate:
	$(PYTHON) -m src.evaluation.benchmark

report:
	$(PYTHON) -m src.evaluation.report

promote:
	$(PYTHON) -m src.pipelines.promote_model --model svd

batch:
	$(PYTHON) -m src.pipelines.batch_recommend --prewarm

# -----------------------------------------------------------------------
# API
# -----------------------------------------------------------------------
serve:
	FLASK_ENV=development $(PYTHON) -m src.serving.app

serve-prod:
	gunicorn -c deployment/gunicorn.conf.py "src.serving.app:create_app()"

# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------
test:
	$(PYTEST) tests/ -v --tb=short

test-cov:
	$(PYTEST) tests/ --cov=src --cov-report=html --cov-report=term-missing

# -----------------------------------------------------------------------
# Logs
# -----------------------------------------------------------------------
check-logs:
	$(PYTHON) scripts/check_logs.py

# -----------------------------------------------------------------------
# Linting
# -----------------------------------------------------------------------
lint:
	flake8 src/ tests/ --max-line-length=120 --ignore=E501,W503

# -----------------------------------------------------------------------
# Docker
# -----------------------------------------------------------------------
docker-build:
	docker build -f deployment/Dockerfile -t product-recommender:latest .

docker-up:
	docker-compose -f deployment/docker-compose.yml up -d

docker-down:
	docker-compose -f deployment/docker-compose.yml down

# -----------------------------------------------------------------------
# Full pipeline
# -----------------------------------------------------------------------
all: data-pipeline train evaluate test

# -----------------------------------------------------------------------
# Clean
# -----------------------------------------------------------------------
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage coverage.xml
