# Personalized Product Recommender System

A production-style recommender system built in the 2016–2017 ML engineering idiom, using the MovieLens 1M dataset. Implements popularity baseline, item-item collaborative filtering, SVD/SVD++ matrix factorization, and a weighted hybrid model, surfaced via a Flask REST API with Redis caching.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Architecture](#3-architecture)
4. [Dataset](#4-dataset)
5. [Models Implemented](#5-models-implemented)
6. [Evaluation Metrics & Thresholds](#6-evaluation-metrics--thresholds)
7. [Quick Start](#7-quick-start)
8. [Running the Full Pipeline](#8-running-the-full-pipeline)
9. [API Reference](#9-api-reference)
10. [Configuration](#10-configuration)
11. [Testing](#11-testing)
12. [Deployment](#12-deployment)
13. [Engineering Decisions](#13-engineering-decisions)

---

## 1. Project Overview

This system recommends movies to users based on their historical ratings. It mirrors the production ML engineering practices common in 2016–2017: hand-engineered collaborative filtering, matrix factorization with Surprise, time-aware train/val/test splits, a Flask API, and a Redis recommendation cache.

Post-2017 tooling (MLflow experiment tracking, Docker, GitHub Actions) is included and explicitly labelled as modernisation.

---

## 2. Repository Structure

```
.
├── data/
│   ├── raw/          # Downloaded MovieLens 1M zip
│   └── processed/    # Parquet files, sparse matrix, id maps
├── deployment/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── gunicorn.conf.py
├── docs/
│   ├── ARCHITECTURE_DECISIONS.md
│   ├── DATA_DICTIONARY.md
│   ├── EXPERIMENT_REPORT.md
│   ├── IMPLEMENTATION_JOURNAL.md
│   └── SUCCESS_METRICS_REPORT.md
├── logs/
│   ├── master_execution_log.md
│   ├── decision_log.md
│   ├── run_log.jsonl
│   ├── checkpoint_status.md
│   ├── data_pipeline.log
│   ├── model_training.log
│   ├── evaluation.log
│   ├── api.log
│   └── test.log
├── models/           # Saved model pickles (git-ignored)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline_model.ipynb
│   ├── 03_matrix_factorization.ipynb
│   └── 04_evaluation.ipynb
├── scripts/
│   ├── run_all.sh
│   ├── run_train.sh
│   ├── run_api.sh
│   ├── run_tests.sh
│   └── check_logs.py
├── src/
│   ├── config.py
│   ├── logging_utils.py
│   ├── data/
│   │   ├── download.py
│   │   ├── validate.py
│   │   ├── preprocess.py
│   │   └── split.py
│   ├── models/
│   │   ├── popularity.py
│   │   ├── item_cf.py
│   │   ├── matrix_factorization.py
│   │   ├── hybrid.py
│   │   └── registry.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── benchmark.py
│   │   └── report.py
│   ├── serving/
│   │   ├── app.py
│   │   ├── cache.py
│   │   └── schemas.py
│   ├── pipelines/
│   │   ├── train_pipeline.py
│   │   ├── batch_recommend.py
│   │   └── promote_model.py
│   └── utils/
│       ├── io.py
│       ├── timing.py
│       └── validation.py
├── tests/
│   ├── test_data_pipeline.py
│   ├── test_models.py
│   ├── test_metrics.py
│   ├── test_api.py
│   └── test_cache.py
├── .env.example
├── .gitignore
├── AGENTS.md
├── Makefile
├── pyproject.toml
├── README.md
└── requirements.txt
```

---

## 3. Architecture

```
[MovieLens 1M] ──► download.py ──► validate.py ──► preprocess.py ──► split.py
                                                                          │
                                                                  [train / val / test]
                                                                          │
                                          ┌───────────────────────────────┤
                                          │                               │
                               train_pipeline.py                  evaluate.py
                                          │                               │
                         ┌────────────────┼────────────────┐     Precision@K, HR@K
                    popularity.py   item_cf.py   svd.py     │     NDCG@K, RMSE, MAE
                                          │           hybrid.py          │
                                    registry.py                   promote_model.py
                                          │
                                   batch_recommend.py
                                          │
                              ┌───────────┴──────────┐
                          Redis cache            Local dict fallback
                              │
                         Flask API (app.py)
                              │
                    /recommend   /similar-items   /predict-batch
```

---

## 4. Dataset

**MovieLens 1M** — 1,000,209 ratings from 6,040 users on 3,706 movies.

| File        | Description                                  |
|-------------|----------------------------------------------|
| ratings.dat | UserID::MovieID::Rating::Timestamp           |
| movies.dat  | MovieID::Title::Genres                       |
| users.dat   | UserID::Gender::Age::Occupation::Zip-code    |

Source: `https://files.grouplens.org/datasets/movielens/ml-1m.zip`

Run `make download` or `python -m src.data.download` to fetch it automatically.

---

## 5. Models Implemented

| Model                    | Algorithm                                    | Cold-start |
|--------------------------|----------------------------------------------|-----------|
| PopularityRecommender    | Global popularity rank                        | Yes (fallback) |
| ItemCFRecommender        | Item-item cosine similarity (mean-centred)   | No        |
| SVDRecommender           | SVD / SVD++ (Surprise library)               | No        |
| HybridRecommender        | α × CF + (1−α) × Content (genre vectors)     | Partial   |

---

## 6. Evaluation Metrics & Thresholds

Rankings evaluated at K = 5, 10, 20.

| Metric       | Success Threshold |
|--------------|-------------------|
| HR@10        | ≥ 0.35            |
| Precision@10 | ≥ 0.10            |
| Recall@10    | ≥ 0.20            |
| NDCG@10      | ≥ 0.25            |
| RMSE         | ≤ 0.95            |
| MAE          | ≤ 0.75            |

---

## 7. Quick Start

```bash
# 1. Clone and install
git clone <repo_url>
cd "Product recommender"
pip install -r requirements.txt

# 2. Copy environment file
cp .env.example .env

# 3. Run the full pipeline
make all

# 4. Start the API
make serve
```

---

## 8. Running the Full Pipeline

```bash
# Individual stages
make download       # Download + checksum MovieLens 1M
make validate       # Schema and integrity checks
make preprocess     # Feature engineering
make split          # Time-aware train/val/test split
make train          # Train all models
make evaluate       # Offline benchmark
make promote        # Promote best model if thresholds met
make batch          # Generate batch recommendations + cache pre-warm
make test           # Run pytest suite
```

Or in one command:
```bash
make all
```

Logs are written to `logs/`. Use `python scripts/check_logs.py` to verify logging health.

---

## 9. API Reference

Base URL: `http://localhost:5000`

### `GET /health`
Returns API health status.

### `GET /recommend?user_id=<int>&n=<int>`
Returns top-N recommendations for a user.
```json
{
  "user_id": 42,
  "recommendations": [
    {"movie_id": 318, "score": 4.72, "title": "Shawshank Redemption, The (1994)"}
  ],
  "model": "hybrid",
  "latency_ms": 12.4
}
```

### `GET /similar-items?item_id=<int>&n=<int>`
Returns N most similar items.

### `POST /predict-batch`
Batch recommendations body: `{"user_ids": [1, 2, 3], "n": 10}`

### `GET /cache/stats`
Returns cache hit/miss statistics.

### `POST /cache/flush`
Flushes the recommendation cache.

---

## 10. Configuration

All configuration lives in `src/config.py`. Key settings:

| Variable          | Default              | Description                    |
|-------------------|----------------------|--------------------------------|
| `N_FACTORS`       | 100                  | SVD latent dimensions          |
| `N_EPOCHS`        | 20                   | SVD training epochs            |
| `ALPHA`           | 0.8                  | Hybrid CF weight               |
| `TRAIN_RATIO`     | 0.6                  | Train split ratio              |
| `VAL_RATIO`       | 0.2                  | Validation split ratio         |
| `TEST_RATIO`      | 0.2                  | Test split ratio               |
| `REDIS_URL`       | redis://localhost:6379/0 | Redis connection           |

Override any variable with environment variables (loaded via `python-dotenv`).

---

## 11. Testing

```bash
make test           # Run all tests (verbose)
make test-cov       # Run with coverage report
```

Test files:

| File                        | Covers                              |
|-----------------------------|-------------------------------------|
| `tests/test_data_pipeline.py` | Download, validation, preprocessing, split |
| `tests/test_models.py`      | All 4 models + registry             |
| `tests/test_metrics.py`     | P@K, R@K, HR@K, NDCG@K, RMSE, MAE  |
| `tests/test_api.py`         | Flask endpoints, error handling     |
| `tests/test_cache.py`       | LocalCache, RecommendationCache, Redis fallback |

---

## 12. Deployment

### Docker

```bash
make docker-build
make docker-up     # Starts Flask + Redis via docker-compose
```

### Manual (Gunicorn)

```bash
gunicorn -c deployment/gunicorn.conf.py "src.serving.app:create_app()"
```

### CI/CD

GitHub Actions workflow at `.github/workflows/ci.yml` runs on every push:
- `pip install -r requirements.txt`
- `pytest tests/`
- Lint with `flake8`

---

## 13. Engineering Decisions

See `docs/ARCHITECTURE_DECISIONS.md` for full ADRs. Summary:

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Dataset  | MovieLens 1M | Widely used, explicit ratings, genre features |
| CF library | Surprise | Implements SVD/SVD++, 2016-era standard |
| Split strategy | Time-aware chronological | Prevents temporal leakage |
| Cache | Redis + local dict fallback | Zero-dependency fallback, production-ready |
| API | Flask + Gunicorn | 2016-era production Python web stack |
| MF algorithm | SVD (primary), SVD++ (improved) | Best RMSE/HR trade-off on MovieLens |
| Versioning | Timestamp pickles + MLflow | MLflow is explicit post-2017 modernisation |
| Hybrid | Weighted blend α=0.8 | Simple, interpretable, tunable |
