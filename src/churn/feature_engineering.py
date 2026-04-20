"""
feature_engineering.py — Feature engineering pipeline for XGBoost churn model.

Generates a feature store Parquet compatible with S3 storage conventions.
Features are computed from the MovieLens interaction log, treating
"no activity in last N days" as a churn proxy.

Phase: Churn Prediction (deliverable 2)
"""

import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import PROCESSED_DATA_DIR, RATINGS_PROCESSED, TRAIN_FILE
from src.logging_utils import get_logger

logger = get_logger("churn.features", "churn")

CHURN_WINDOW_DAYS = 180   # users inactive for >180 days = churned
REFERENCE_PERCENTILE = 0.8  # use 80th-percentile timestamp as reference date
FEATURE_STORE_DIR = PROCESSED_DATA_DIR / "feature_store"


def build_churn_features(
    ratings_df: Optional[pd.DataFrame] = None,
    churn_window_days: int = CHURN_WINDOW_DAYS,
) -> pd.DataFrame:
    """
    Build per-user churn feature table.

    Features:
      - total_ratings         : total number of ratings by user
      - avg_rating            : mean rating across all interactions
      - std_rating            : std of ratings
      - rating_sessions       : number of distinct activity days
      - days_since_first      : days between first and reference date
      - days_since_last       : days since most recent interaction (recency)
      - avg_session_gap_days  : mean days between consecutive sessions
      - pct_high_rating       : fraction of ratings >= 4
      - genre_diversity       : number of distinct genres rated
      - churned               : binary label (1 = inactive > churn_window_days)

    Args:
        ratings_df: Pre-loaded ratings DataFrame. Loaded from disk if None.
        churn_window_days: Inactivity threshold in days.

    Returns:
        DataFrame indexed by user_id with feature columns + churned label.
    """
    logger.start_phase("Churn Features", "Build per-user feature table for churn model")
    start = time.time()

    if ratings_df is None:
        logger.info(f"Loading ratings from {RATINGS_PROCESSED}")
        ratings_df = pd.read_parquet(str(RATINGS_PROCESSED))

    # Work with timestamps
    if not pd.api.types.is_datetime64_any_dtype(ratings_df["timestamp"]):
        ratings_df = ratings_df.copy()
        ratings_df["timestamp"] = pd.to_datetime(ratings_df["timestamp"], unit="s")

    # Use 80th-percentile timestamp as reference to ensure class balance
    reference_date = ratings_df["timestamp"].quantile(REFERENCE_PERCENTILE)
    logger.info(f"Reference date (P{int(REFERENCE_PERCENTILE*100)} timestamp): {reference_date}")

    # Per-user aggregates
    user_stats = (
        ratings_df.groupby("user_id")
        .agg(
            total_ratings=("rating", "count"),
            avg_rating=("rating", "mean"),
            std_rating=("rating", "std"),
            first_activity=("timestamp", "min"),
            last_activity=("timestamp", "max"),
        )
        .reset_index()
    )
    user_stats["std_rating"] = user_stats["std_rating"].fillna(0.0)

    # Days-based features
    user_stats["days_since_first"] = (
        reference_date - user_stats["first_activity"]
    ).dt.days.astype(float)
    user_stats["days_since_last"] = (
        reference_date - user_stats["last_activity"]
    ).dt.days.astype(float)

    # Session count (distinct calendar days with activity)
    ratings_df["activity_date"] = ratings_df["timestamp"].dt.date
    sessions = (
        ratings_df.groupby("user_id")["activity_date"]
        .nunique()
        .rename("rating_sessions")
        .reset_index()
    )
    user_stats = user_stats.merge(sessions, on="user_id", how="left")
    user_stats["rating_sessions"] = user_stats["rating_sessions"].fillna(1)

    # Average session gap
    user_stats["avg_session_gap_days"] = (
        user_stats["days_since_first"] / user_stats["rating_sessions"].clip(lower=1)
    )

    # Fraction high ratings
    high = (
        ratings_df[ratings_df["rating"] >= 4]
        .groupby("user_id")
        .size()
        .rename("high_ratings")
        .reset_index()
    )
    user_stats = user_stats.merge(high, on="user_id", how="left")
    user_stats["high_ratings"] = user_stats["high_ratings"].fillna(0)
    user_stats["pct_high_rating"] = (
        user_stats["high_ratings"] / user_stats["total_ratings"]
    )

    # Genre diversity (requires movie features)
    try:
        movies_path = PROCESSED_DATA_DIR / "movies.parquet"
        if movies_path.exists():
            movies_df = pd.read_parquet(str(movies_path))
            genre_cols = [
                c for c in movies_df.columns
                if c.startswith("genre_") and c != "genre_list"
            ]
            if genre_cols:
                movies_df["genre_count"] = movies_df[genre_cols].sum(axis=1)
                merged = ratings_df[["user_id", "movie_id"]].merge(
                    movies_df[["movie_id", "genre_count"]], on="movie_id", how="left"
                )
                genre_div = (
                    merged.groupby("user_id")["genre_count"]
                    .mean()
                    .rename("genre_diversity")
                    .reset_index()
                )
                user_stats = user_stats.merge(genre_div, on="user_id", how="left")
                user_stats["genre_diversity"] = user_stats["genre_diversity"].fillna(0.0)
            else:
                user_stats["genre_diversity"] = 0.0
        else:
            user_stats["genre_diversity"] = 0.0
    except Exception as e:
        logger.warning(f"Genre diversity skipped: {e}")
        user_stats["genre_diversity"] = 0.0

    # Churn label
    user_stats["churned"] = (
        user_stats["days_since_last"] > churn_window_days
    ).astype(int)

    feature_cols = [
        "user_id",
        "total_ratings",
        "avg_rating",
        "std_rating",
        "rating_sessions",
        "days_since_first",
        "days_since_last",
        "avg_session_gap_days",
        "pct_high_rating",
        "genre_diversity",
        "churned",
    ]
    result = user_stats[feature_cols].copy()

    churn_rate = result["churned"].mean()
    elapsed = time.time() - start
    logger.log_metric("churn_rate", round(float(churn_rate), 4), "Churn Features")
    logger.log_metric("n_users_features", len(result), "Churn Features")
    logger.log_runtime("build_churn_features", elapsed, "Churn Features")
    logger.end_phase(
        "Churn Features",
        f"Built features for {len(result):,} users (churn rate={churn_rate:.1%})",
        "Train XGBoost",
    )

    return result


def save_feature_store(features_df: pd.DataFrame) -> Path:
    """Save features to the feature store directory in Parquet format."""
    FEATURE_STORE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FEATURE_STORE_DIR / "user_churn_features.parquet"
    features_df.to_parquet(str(out_path), index=False)
    logger.log_artifact(str(out_path), "Churn feature store (Parquet)")
    return out_path


if __name__ == "__main__":
    features = build_churn_features()
    out = save_feature_store(features)
    print(f"Feature store saved: {out}")
    print(features.describe())
