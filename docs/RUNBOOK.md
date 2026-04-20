# Runbook — Personalized Product Recommender System

## Quick Reference

| Component | Command |
|-----------|---------|
| Full pipeline | `py -3.13 scripts/run_all.sh` |
| Data only | `py -3.13 -m src.data.download && py -3.13 -m src.data.preprocess && py -3.13 -m src.data.split` |
| Train | `py -3.13 -m src.pipelines.train_pipeline` |
| Evaluate | `py -3.13 -m src.evaluation.benchmark` |
| Promote model | `py -3.13 -m src.pipelines.promote_model --model svd` |
| Batch recs | `py -3.13 -m src.pipelines.batch_recommend --prewarm` |
| Tests | `py -3.13 -m pytest tests/ -v` |
| API server | `py -3.13 -m src.serving.app` |
| Churn features | `py -3.13 -m src.churn.feature_engineering` |
| Churn model | `py -3.13 -m src.churn.train_churn_model` |
| SHAP | `py -3.13 -m src.churn.shap_explanations` |
| Salesforce push | `py -3.13 -m src.churn.salesforce_webhook` |

---

## 1. Initial Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
# Edit .env — set SF_CLIENT_ID, SF_CLIENT_SECRET, etc.

# Check logs directory
python scripts/check_logs.py
```

---

## 2. Retraining Procedure

1. Ensure new data is available in `data/raw/`
2. Run preprocessing: `py -3.13 -m src.data.preprocess`
3. Run split: `py -3.13 -m src.data.split`
4. Run training: `py -3.13 -m src.pipelines.train_pipeline`
5. Evaluate: `py -3.13 -m src.evaluation.benchmark`
6. If metrics meet targets, promote: `py -3.13 -m src.pipelines.promote_model --model hybrid`
7. Regenerate batch cache: `py -3.13 -m src.pipelines.batch_recommend --prewarm`

---

## 3. API Health Check

```bash
curl http://localhost:5000/health
# Expected: {"status": "ok", "model_loaded": true, ...}

curl "http://localhost:5000/recommend?user_id=1&n=10"
# Expected: {"user_id": 1, "recommendations": [...], "source": "personalized"}
```

---

## 4. Churn Pipeline (Weekly)

Automated via Airflow DAG at `dags/churn_scoring_dag.py`.  
Manual run:
```bash
py -3.13 -m src.churn.feature_engineering
py -3.13 -m src.churn.train_churn_model
py -3.13 -m src.churn.shap_explanations
py -3.13 -m src.churn.salesforce_webhook
```

---

## 5. Drift Monitoring

Drift check runs automatically in the Airflow DAG (`task_check_drift`).  
Alert thresholds:
- PSI > 0.10 → moderate drift → log warning, monitor
- PSI > 0.25 → severe drift → DAG raises ValueError, triggers email alert

Manual check:
```bash
py -3.13 -c "
from dags.churn_scoring_dag import task_check_drift
task_check_drift(task_instance=type('TI', (), {'xcom_push': lambda *a, **kw: None})())
"
```

Drift report saved to: `data/artifacts/churn/drift_report.json`

---

## 6. Common Issues

### "No production model registered"
```
FileNotFoundError: No production model registered. Run training pipeline first.
```
**Fix**: Run `py -3.13 -m src.pipelines.promote_model --model svd`

### Redis unavailable
The API falls back to local in-memory `LocalCache`. Check `USE_REDIS=false` is set in `.env` for development.

### scikit-surprise fails to install
Expected on Python 3.13+. The `_ScipySVD` fallback in `matrix_factorization.py` handles this automatically. Metrics may be slightly lower than with Surprise.

### UnicodeEncodeError on Windows
Ensure `PYTHONIOENCODING=utf-8` is set. Or use `py -3.13 -X utf8 ...`.

---

## 7. Rollback

To roll back to a previous model version:
```bash
# List available versions
py -3.13 -c "from src.models.registry import ModelRegistry; r=ModelRegistry(); print([e['artifact_path'] for e in r.list_models()])"

# Manually promote older artifact
py -3.13 -m src.pipelines.promote_model --path data/artifacts/models/svd_20260101_000000.pkl
```

---

## 8. Log Locations

| Log | Path |
|-----|------|
| Main run log | `logs/run_log.jsonl` |
| Checkpoint status | `logs/checkpoint_status.md` |
| Benchmark results | `data/artifacts/benchmark_results.json` |
| Churn eval report | `data/artifacts/churn/evaluation_report.json` |
| Drift report | `data/artifacts/churn/drift_report.json` |
| Current model | `data/artifacts/current_model_version.txt` |
