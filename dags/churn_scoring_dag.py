"""
churn_scoring_dag.py — Airflow DAG for weekly batch churn scoring.

Schedule: Every Monday at 02:00 UTC
Pipeline:
  1. build_features  — rebuild user_churn_features.parquet from latest ratings
  2. score_churn     — run XGBoost model → churn_predictions.parquet
  3. run_shap        — compute SHAP factor tables for flagged users
  4. push_salesforce — push high-risk users to Salesforce CRM as Tasks
  5. check_drift     — run PSI / KS drift check on feature distributions
  6. notify_slack    — post summary digest to #ml-alerts Slack channel

All tasks use PythonOperator to keep the codebase dependency-free of
Airflow-specific constructs in the model code itself.

Phase: Churn Prediction (deliverable 3 — Airflow DAG)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Airflow imports — guarded so the file is importable even without Airflow
# ---------------------------------------------------------------------------
try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from airflow.operators.email import EmailOperator
    from airflow.utils.dates import days_ago
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False
    log.warning("Apache Airflow not installed. DAG definition is for documentation only.")

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Task functions
# ---------------------------------------------------------------------------

def task_build_features(**context) -> str:
    """Build/refresh churn feature store from latest interaction data."""
    import sys
    sys.path.insert(0, str(WORKSPACE_ROOT))
    from src.churn.feature_engineering import build_churn_features, save_feature_store

    features_df = build_churn_features()
    out_path = save_feature_store(features_df)
    log.info(f"Feature store updated: {out_path}")
    context["task_instance"].xcom_push(key="n_users", value=len(features_df))
    return str(out_path)


def task_score_churn(**context) -> str:
    """Run XGBoost model inference over the feature store."""
    import sys
    import json
    import pickle
    import numpy as np
    import pandas as pd

    sys.path.insert(0, str(WORKSPACE_ROOT))
    from src.config import ARTIFACTS_DIR, PROCESSED_DATA_DIR

    feat_path = PROCESSED_DATA_DIR / "feature_store" / "user_churn_features.parquet"
    model_path = ARTIFACTS_DIR / "churn" / "xgboost_churn_model.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Churn model not found at {model_path}")

    features_df = pd.read_parquet(str(feat_path))

    FEATURE_COLS = [
        "total_ratings", "avg_rating", "std_rating", "rating_sessions",
        "days_since_first", "days_since_last", "avg_session_gap_days",
        "pct_high_rating", "genre_diversity",
    ]

    with open(str(model_path), "rb") as f:
        model = pickle.load(f)

    X = features_df[FEATURE_COLS].fillna(0).values
    probs = model.predict_proba(X)[:, 1]

    predictions_df = pd.DataFrame({
        "user_id": features_df["user_id"].values,
        "churn_probability": np.round(probs, 5),
        "churn_flag": (probs >= 0.5).astype(int),
        "scored_at": datetime.utcnow().isoformat(),
    })

    out_path = ARTIFACTS_DIR / "churn" / "churn_predictions.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_parquet(str(out_path), index=False)

    n_flagged = int((probs >= 0.5).sum())
    context["task_instance"].xcom_push(key="n_flagged", value=n_flagged)
    log.info(f"Scored {len(predictions_df):,} users. Flagged: {n_flagged:,}")
    return str(out_path)


def task_run_shap(**context) -> str:
    """Compute SHAP explanations for flagged users."""
    import sys
    sys.path.insert(0, str(WORKSPACE_ROOT))
    from src.churn.shap_explanations import generate_shap_explanations

    shap_df = generate_shap_explanations(flagged_only=True, churn_threshold=0.5)
    context["task_instance"].xcom_push(key="n_explained", value=len(shap_df))

    from src.config import ARTIFACTS_DIR
    out_path = ARTIFACTS_DIR / "churn" / "shap_explanations.parquet"
    return str(out_path)


def task_push_salesforce(**context) -> dict:
    """Push high-risk churn users to Salesforce CRM."""
    import sys
    sys.path.insert(0, str(WORKSPACE_ROOT))
    from src.churn.salesforce_webhook import push_churn_alerts

    result = push_churn_alerts(churn_threshold=0.7)
    context["task_instance"].xcom_push(key="sf_result", value=result)
    return result


def task_check_drift(**context) -> dict:
    """
    PSI + KS drift check on feature distributions vs. training baseline.

    Raises ValueError if severe drift detected on any feature (PSI > 0.25).
    """
    import sys
    import json
    import numpy as np
    import pandas as pd
    from scipy import stats

    sys.path.insert(0, str(WORKSPACE_ROOT))
    from src.config import ARTIFACTS_DIR, PROCESSED_DATA_DIR

    feat_path = PROCESSED_DATA_DIR / "feature_store" / "user_churn_features.parquet"
    baseline_path = ARTIFACTS_DIR / "churn" / "feature_baseline.parquet"

    FEATURE_COLS = [
        "total_ratings", "avg_rating", "rating_sessions", "days_since_last",
    ]

    current_df = pd.read_parquet(str(feat_path))

    if not baseline_path.exists():
        # First run — save baseline
        current_df[["user_id"] + FEATURE_COLS].to_parquet(str(baseline_path), index=False)
        log.info("Baseline saved (first drift check run).")
        return {"status": "baseline_created"}

    baseline_df = pd.read_parquet(str(baseline_path))

    results = {}
    severe = []

    def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
        """Population Stability Index."""
        eps = 1e-8
        bins_range = np.linspace(
            min(expected.min(), actual.min()),
            max(expected.max(), actual.max()) + eps,
            bins + 1,
        )
        exp_counts, _ = np.histogram(expected, bins=bins_range)
        act_counts, _ = np.histogram(actual, bins=bins_range)
        exp_pct = (exp_counts / len(expected)) + eps
        act_pct = (act_counts / len(actual)) + eps
        return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))

    for col in FEATURE_COLS:
        exp = baseline_df[col].dropna().values
        act = current_df[col].dropna().values
        psi = _psi(exp, act)
        ks_stat, ks_p = stats.ks_2samp(exp, act)
        results[col] = {
            "psi": round(psi, 4),
            "ks_stat": round(float(ks_stat), 4),
            "ks_pvalue": round(float(ks_p), 4),
            "drift_level": "severe" if psi > 0.25 else ("moderate" if psi > 0.1 else "none"),
        }
        if psi > 0.25:
            severe.append(col)
            log.warning(f"SEVERE DRIFT detected on '{col}': PSI={psi:.4f}")

    drift_report = {
        "features": results,
        "severe_features": severe,
        "n_severe": len(severe),
    }
    report_path = ARTIFACTS_DIR / "churn" / "drift_report.json"
    report_path.write_text(json.dumps(drift_report, indent=2))

    context["task_instance"].xcom_push(key="drift_report", value=drift_report)

    if severe:
        raise ValueError(
            f"Drift alert: {len(severe)} features with PSI > 0.25: {severe}. "
            "Review model freshness and trigger retraining."
        )

    return drift_report


def task_notify_slack(**context) -> None:
    """Post a Slack digest to #ml-alerts with key pipeline metrics."""
    import os
    import json
    import urllib.request

    slack_url = os.getenv("SLACK_WEBHOOK_URL", "")
    if not slack_url:
        log.warning("SLACK_WEBHOOK_URL not set — skipping notification.")
        return

    ti = context["task_instance"]
    n_users = ti.xcom_pull(key="n_users", task_ids="build_features") or "?"
    n_flagged = ti.xcom_pull(key="n_flagged", task_ids="score_churn") or "?"
    sf_result = ti.xcom_pull(key="sf_result", task_ids="push_salesforce") or {}

    text = (
        f":robot_face: *Weekly Churn Scoring Complete*\n"
        f">Users scored: {n_users:,}\n"
        f">Flagged (p>=0.5): {n_flagged:,}\n"
        f">Salesforce alerts sent: {sf_result.get('sent', '?')}\n"
        f">Drift check: {'OK' if not sf_result.get('drift') else 'ALERT'}"
    )

    payload = json.dumps({"text": text}).encode()
    req = urllib.request.Request(
        slack_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10):
            log.info("Slack notification sent.")
    except Exception as e:
        log.warning(f"Slack notification failed: {e}")


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

if AIRFLOW_AVAILABLE:
    default_args = {
        "owner": "ml-team",
        "depends_on_past": False,
        "retries": 2,
        "retry_delay": timedelta(minutes=5),
        "email_on_failure": True,
        "email": ["ml-alerts@example.com"],
    }

    with DAG(
        dag_id="churn_scoring_weekly",
        description="Weekly batch churn scoring: features → XGBoost → SHAP → Salesforce → drift",
        schedule_interval="0 2 * * 1",   # Every Monday 02:00 UTC
        start_date=days_ago(1),
        default_args=default_args,
        catchup=False,
        tags=["churn", "ml", "batch"],
        max_active_runs=1,
        doc_md=__doc__,
    ) as dag:

        build_features = PythonOperator(
            task_id="build_features",
            python_callable=task_build_features,
        )

        score_churn = PythonOperator(
            task_id="score_churn",
            python_callable=task_score_churn,
        )

        run_shap = PythonOperator(
            task_id="run_shap",
            python_callable=task_run_shap,
        )

        push_salesforce = PythonOperator(
            task_id="push_salesforce",
            python_callable=task_push_salesforce,
        )

        check_drift = PythonOperator(
            task_id="check_drift",
            python_callable=task_check_drift,
        )

        notify_slack = PythonOperator(
            task_id="notify_slack",
            python_callable=task_notify_slack,
            trigger_rule="all_done",   # run even if upstream failed
        )

        # DAG topology
        build_features >> score_churn >> run_shap >> push_salesforce >> check_drift >> notify_slack
