"""
shap_explanations.py — SHAP explanation generator for the XGBoost churn model.

Produces per-account (per-user) factor tables showing which features
drove their churn probability — enabling targeted interventions in Salesforce.

Output:
  data/artifacts/churn/shap_explanations.parquet  — per-user SHAP values
  data/artifacts/churn/shap_summary.png           — global importance bar chart

Phase: Churn Prediction (deliverable 4)
"""

import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import ARTIFACTS_DIR, PROCESSED_DATA_DIR
from src.logging_utils import get_logger

logger = get_logger("churn.shap", "churn")

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


def generate_shap_explanations(
    model=None,
    features_df: Optional[pd.DataFrame] = None,
    flagged_only: bool = True,
    churn_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Generate SHAP explanations for all (or only flagged) users.

    Args:
        model: Fitted XGBClassifier. Loaded from disk if None.
        features_df: Feature DataFrame. Loaded from disk if None.
        flagged_only: If True, only compute SHAP for users with churn_prob >= threshold.
        churn_threshold: Probability cutoff for "flagged" users.

    Returns:
        DataFrame with columns: user_id, churn_probability, shap_<feature> × N, top_factor
    """
    try:
        import shap
    except ImportError:
        logger.warning("shap not installed. Installing...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "shap", "-q"], check=True)
        import shap

    logger.start_phase("SHAP Explanations", "Generate per-user factor tables")
    t0 = time.time()

    # Load model
    if model is None:
        import pickle
        model_path = CHURN_MODELS_DIR / "xgboost_churn_model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Churn model not found at {model_path}. Run train_churn_model.py first."
            )
        with open(str(model_path), "rb") as f:
            model = pickle.load(f)

    # Load features
    if features_df is None:
        feat_path = FEATURE_STORE_DIR / "user_churn_features.parquet"
        features_df = pd.read_parquet(str(feat_path))

    X = features_df[FEATURE_COLS].fillna(0).values
    user_ids = features_df["user_id"].values

    # Score all users
    probs = model.predict_proba(X)[:, 1]

    # Filter to flagged users if requested
    if flagged_only:
        mask = probs >= churn_threshold
        X_explain = X[mask]
        user_ids_explain = user_ids[mask]
        probs_explain = probs[mask]
        logger.info(f"Flagged users (>= {churn_threshold}): {mask.sum():,} of {len(mask):,}")
    else:
        X_explain = X
        user_ids_explain = user_ids
        probs_explain = probs

    # Compute SHAP values using TreeExplainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_explain)

    # Build result DataFrame
    shap_df = pd.DataFrame(
        shap_values,
        columns=[f"shap_{c}" for c in FEATURE_COLS],
    )
    shap_df.insert(0, "user_id", user_ids_explain)
    shap_df.insert(1, "churn_probability", np.round(probs_explain, 5))

    # Identify top (most positive) SHAP factor per user
    shap_cols = [f"shap_{c}" for c in FEATURE_COLS]
    shap_df["top_factor"] = shap_df[shap_cols].idxmax(axis=1).str.replace("shap_", "")

    # Save
    CHURN_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CHURN_MODELS_DIR / "shap_explanations.parquet"
    shap_df.to_parquet(str(out_path), index=False)
    logger.log_artifact(str(out_path), "SHAP per-user factor table (Parquet)")

    # Global importance summary (text-based; avoid matplotlib Agg backend issues)
    importance = np.abs(shap_values).mean(axis=0)
    importance_dict = dict(zip(FEATURE_COLS, importance.tolist()))
    importance_sorted = dict(
        sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    )
    summary_path = CHURN_MODELS_DIR / "shap_global_importance.json"
    summary_path.write_text(json.dumps(importance_sorted, indent=2))
    logger.log_artifact(str(summary_path), "SHAP global feature importance")

    # Try to save bar chart
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        names = list(importance_sorted.keys())
        vals = list(importance_sorted.values())
        ax.barh(names[::-1], vals[::-1], color="#4C72B0")
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title("XGBoost Churn Model — Global Feature Importance (SHAP)")
        fig.tight_layout()
        chart_path = CHURN_MODELS_DIR / "shap_summary.png"
        fig.savefig(str(chart_path), dpi=120)
        plt.close(fig)
        logger.log_artifact(str(chart_path), "SHAP summary bar chart")
    except Exception as e:
        logger.warning(f"Could not save SHAP chart: {e}")

    elapsed = time.time() - t0
    coverage_pct = 100.0 * len(shap_df) / len(probs) if flagged_only else 100.0
    logger.log_metric("shap_coverage_pct", round(coverage_pct, 1), "SHAP")
    logger.log_metric("n_users_explained", len(shap_df), "SHAP")
    logger.log_runtime("shap_explanations", elapsed, "SHAP")
    logger.end_phase(
        "SHAP Explanations",
        f"Explained {len(shap_df):,} users ({coverage_pct:.0f}% of flagged)",
        "Salesforce Webhook",
    )

    return shap_df


if __name__ == "__main__":
    df = generate_shap_explanations(flagged_only=True)
    print(df.head())
    print(f"\nTop factors distribution:\n{df['top_factor'].value_counts()}")
