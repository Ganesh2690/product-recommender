---
applyTo: "**"
---
# GitHub Copilot Instructions — Product Recommender

## Project Context

This is a **Personalized Product Recommender System** built in the 2016–2017 ML engineering style using the MovieLens 1M dataset. The codebase uses hand-crafted collaborative filtering, SVD/SVD++ matrix factorization, and a weighted hybrid model.

## Architecture Overview

- `src/data/` — data download, validation, preprocessing, and time-aware splitting
- `src/models/` — popularity baseline, item-CF, SVD/SVD++, hybrid, model registry
- `src/evaluation/` — ranking metrics (P@K, HR@K, NDCG@K) and rating metrics (RMSE, MAE)
- `src/serving/` — Flask API, Redis two-tier cache, request/response schemas
- `src/pipelines/` — end-to-end training pipeline, batch recommendations, model promotion
- `src/utils/` — IO helpers, timing utilities, input validation

## Coding Conventions

- Python 3.11, type hints on public function signatures
- All modules use `src.logging_utils.ProjectLogger` for structured logging
- All log calls must include `phase` context
- Configuration must be sourced from `src/config.py` (no magic numbers in code)
- Time-aware splits only — never random train/test splits (prevents temporal leakage)
- Model save/load uses `pickle` via `src/utils/io.py`
- Flask endpoints must validate inputs through `src/serving/schemas.py`
- All new models must implement `fit()`, `recommend()`, `save()`, `load()` interface
- Hybrid blending weight is `alpha` (CF weight) and lives in `src/config.py`

## Testing Requirements

- Unit tests in `tests/` using `pytest`
- Mocked Flask test client for API tests
- No real Redis connection required in tests (LocalCache fallback)
- No internet access required in tests (mock `src.data.download`)

## Do NOT

- Add hard-coded file paths — always use `src/config.py` path constants
- Skip logging — all phase starts/ends must be logged
- Use random splits — always time-aware per user
- Import pandas as anything other than `pd`
- Skip input validation in API endpoints
