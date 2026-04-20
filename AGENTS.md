# AGENTS.md — Agent Instructions for Product Recommender

## Project Purpose

Build and maintain a **Personalized Product Recommender System** (2016–2017 ML engineering style).
Dataset: MovieLens 1M. Models: Popularity Baseline → Item-CF → SVD/SVD++ → Weighted Hybrid.

---

## Execution Order

When running the full pipeline, ALWAYS follow this order:

1. `python -m src.data.download`
2. `python -m src.data.validate`
3. `python -m src.data.preprocess`
4. `python -m src.data.split`
5. `python -m src.pipelines.train_pipeline`
6. `python -m src.evaluation.benchmark`
7. `python -m src.evaluation.report`
8. `python -m src.pipelines.promote_model --model svd`
9. `python -m src.pipelines.batch_recommend --prewarm`
10. `pytest tests/ -v`
11. `python -m src.serving.app` (manual verification)

---

## Logging Requirements

- Log EVERY phase start and end using `ProjectLogger.start_phase()` / `end_phase()`
- Write metrics with `log_metric()` after each evaluation step
- Write artifacts with `log_artifact()` after each model save
- Append JSONL events to `logs/run_log.jsonl`
- Update `logs/checkpoint_status.md` after each phase

---

## Code Generation Rules

1. **No TODOs** — every function must be fully implemented
2. **No magic numbers** — use `src/config.py` constants
3. **No random splits** — always time-aware chronological splits
4. **All models** must implement the standard interface:
   - `fit(ratings_df: pd.DataFrame) -> None`
   - `recommend(user_id: int, n: int, seen_items: set) -> list[dict]`
   - `save(path: str) -> None`
   - `load(path: str) -> Model` (classmethod)
5. **All API endpoints** must validate inputs via `src/serving/schemas.py`
6. **All imports** must use `src.*` absolute paths (no relative imports to parent)

---

## Success Criteria

The system is considered complete when ALL of the following are met:

| Check | Target |
|-------|--------|
| HR@10 | ≥ 0.35 |
| Precision@10 | ≥ 0.10 |
| Recall@10 | ≥ 0.20 |
| NDCG@10 | ≥ 0.25 |
| RMSE | ≤ 0.95 |
| MAE | ≤ 0.75 |
| pytest | 100% pass |
| All log files | present & non-empty |
| `/health` endpoint | returns 200 |

---

## File Ownership Map

| File | Responsible phase |
|------|------------------|
| `src/data/download.py` | Phase 2 — Data Acquisition |
| `src/data/validate.py` | Phase 2 — Data Acquisition |
| `src/data/preprocess.py` | Phase 3 — EDA & Preprocessing |
| `src/data/split.py` | Phase 3 — EDA & Preprocessing |
| `src/models/popularity.py` | Phase 4 — Popularity Baseline |
| `src/models/item_cf.py` | Phase 5 — Item-CF |
| `src/models/matrix_factorization.py` | Phase 6 — Matrix Factorization |
| `src/models/hybrid.py` | Phase 7 — Hybrid Model |
| `src/models/registry.py` | Phase 7 — Hybrid Model |
| `src/evaluation/metrics.py` | Phase 8 — Evaluation |
| `src/evaluation/benchmark.py` | Phase 8 — Evaluation |
| `src/evaluation/report.py` | Phase 8 — Evaluation |
| `src/serving/app.py` | Phase 9 — Serving |
| `src/serving/cache.py` | Phase 9 — Serving |
| `src/serving/schemas.py` | Phase 9 — Serving |
| `src/pipelines/train_pipeline.py` | Phase 10 — Pipelines |
| `src/pipelines/batch_recommend.py` | Phase 10 — Pipelines |
| `src/pipelines/promote_model.py` | Phase 10 — Pipelines |
| `tests/` | Phase 11 — Testing |
