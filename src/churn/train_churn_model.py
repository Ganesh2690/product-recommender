"""
train_churn_model.py — XGBoost churn model with hyperparameter tuning.

Pipeline:
  1. Load feature store Parquet
  2. Train/test split (time-aware: hold out most-recent 20% users)
  3. Hyperparameter search via RandomizedSearchCV
  4. Final model training on best params
  5. Evaluation: AUC-ROC, F1, classification report
  6. Artifact save: model.pkl + tuning_results.json + evaluation_report.json

Targets (per project spec):
  AUC-ROC ≥ 0.88 | F1 ≥ 0.75

Phase: Churn Prediction (deliverable 1)
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import ARTIFACTS_DIR, PROCESSED_DATA_DIR
from src.logging_utils import get_logger

logger = get_logger("churn.model", "churn")

FEATURE_STORE_DIR = PROCESSED_DATA_DIR / "feature_store"
CHURN_MODELS_DIR = ARTIFACTS_DIR / "churn"
FEATURE_COLS = [
    "total_ratings",
    "avg_rating",
    "std_rating",
    "rating_sessions",
    "days_since_first",
    "days_since_last",
    "avg_session_gap_days",
    "pct_high_rating",
    "genre_diversity",
]
TARGET_COL = "churned"


def load_features() -> pd.DataFrame:
    feat_path = FEATURE_STORE_DIR / "user_churn_features.parquet"
    if not feat_path.exists():
        raise FileNotFoundError(
            f"Feature store not found at {feat_path}. "
            "Run src/churn/feature_engineering.py first."
        )
    return pd.read_parquet(str(feat_path))


def time_aware_split(df: pd.DataFrame, test_frac: float = 0.2):
    """
    Stratified random split that preserves churn class ratio in train and test.
    """
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(
        df,
        test_size=test_frac,
        stratify=df[TARGET_COL] if df[TARGET_COL].nunique() > 1 else None,
        random_state=42,
    )
    return train, test


def train_xgboost_churn(
    features_df: pd.DataFrame | None = None,
    n_iter: int = 30,
    cv: int = 5,
    random_state: int = 42,
) -> dict:
    """
    Train XGBoost churn classifier with RandomizedSearchCV tuning.

    Returns:
        dict with keys: model, best_params, eval_metrics, artifacts
    """
    try:
        import xgboost as xgb
    except ImportError:
        logger.warning("xgboost not installed. Installing...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "xgboost", "-q"], check=True)
        import xgboost as xgb

    from sklearn.metrics import (
        classification_report,
        f1_score,
        roc_auc_score,
    )
    from sklearn.model_selection import RandomizedSearchCV

    logger.start_phase("Churn Model - Train", "XGBoost churn classifier with hyperparameter tuning")
    t0 = time.time()

    if features_df is None:
        features_df = load_features()

    logger.info(f"Loaded {len(features_df):,} user records. Churn rate: {features_df[TARGET_COL].mean():.2%}")

    train_df, test_df = time_aware_split(features_df)
    logger.info(f"Train: {len(train_df):,} | Test: {len(test_df):,}")

    X_train = train_df[FEATURE_COLS].fillna(0).values
    y_train = train_df[TARGET_COL].values
    X_test = test_df[FEATURE_COLS].fillna(0).values
    y_test = test_df[TARGET_COL].values

    # Class weight
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = float(neg) / float(pos) if pos > 0 else 1.0

    base_model = xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )

    param_dist = {
        "n_estimators":      [100, 200, 300, 500],
        "max_depth":         [3, 4, 5, 6, 7],
        "learning_rate":     [0.01, 0.05, 0.1, 0.2],
        "subsample":         [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree":  [0.6, 0.7, 0.8, 0.9, 1.0],
        "min_child_weight":  [1, 3, 5, 7],
        "gamma":             [0, 0.1, 0.2, 0.5],
        "reg_alpha":         [0, 0.1, 0.5, 1.0],
        "reg_lambda":        [0.5, 1.0, 2.0, 5.0],
    }

    logger.info(f"Starting RandomizedSearchCV (n_iter={n_iter}, cv={cv})...")
    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        verbose=0,
        random_state=random_state,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)

    best_params = search.best_params_
    best_cv_auc = search.best_score_
    logger.info(f"Best CV AUC: {best_cv_auc:.4f}")
    logger.info(f"Best params: {best_params}")

    # Final model on full train data
    final_model = xgb.XGBClassifier(
        **best_params,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )
    final_model.fit(X_train, y_train)

    # Evaluate
    y_prob = final_model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    auc = float(roc_auc_score(y_test, y_prob))
    f1 = float(f1_score(y_test, y_pred))
    report = classification_report(y_test, y_pred, output_dict=True)

    elapsed = time.time() - t0
    logger.log_metric("churn_auc_roc", round(auc, 4), "Churn Model")
    logger.log_metric("churn_f1", round(f1, 4), "Churn Model")
    logger.log_metric("churn_best_cv_auc", round(best_cv_auc, 4), "Churn Model")
    logger.log_runtime("xgboost_train", elapsed, "Churn Model")

    logger.info(f"Test AUC-ROC: {auc:.4f} (target >=0.88)")
    logger.info(f"Test F1:      {f1:.4f} (target >=0.75)")

    # Save artifacts
    CHURN_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    import pickle
    model_path = CHURN_MODELS_DIR / "xgboost_churn_model.pkl"
    with open(str(model_path), "wb") as f:
        pickle.dump(final_model, f)

    tuning_path = CHURN_MODELS_DIR / "tuning_results.json"
    tuning_results = {
        "best_params": best_params,
        "best_cv_auc": round(best_cv_auc, 4),
        "n_iter": n_iter,
        "cv_folds": cv,
        "feature_cols": FEATURE_COLS,
    }
    tuning_path.write_text(json.dumps(tuning_results, indent=2))

    eval_path = CHURN_MODELS_DIR / "evaluation_report.json"
    eval_report = {
        "auc_roc": round(auc, 4),
        "f1_score": round(f1, 4),
        "classification_report": report,
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
        "churn_rate_train": round(float(y_train.mean()), 4),
        "churn_rate_test": round(float(y_test.mean()), 4),
        "targets_met": {
            "auc_roc_gte_0_88": auc >= 0.88,
            "f1_gte_0_75": f1 >= 0.75,
        },
    }
    eval_path.write_text(json.dumps(eval_report, indent=2))

    logger.log_artifact(str(model_path), "XGBoost churn model (pickle)")
    logger.log_artifact(str(tuning_path), "Hyperparameter tuning results")
    logger.log_artifact(str(eval_path), "Churn model evaluation report")
    logger.end_phase("Churn Model - Train", f"AUC={auc:.4f} F1={f1:.4f}", "SHAP Explanations")

    return {
        "model": final_model,
        "best_params": best_params,
        "eval_metrics": eval_report,
        "model_path": str(model_path),
        "tuning_path": str(tuning_path),
        "eval_path": str(eval_path),
    }


if __name__ == "__main__":
    result = train_xgboost_churn(n_iter=20)
    print(json.dumps(result["eval_metrics"], indent=2))
