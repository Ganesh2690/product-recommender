"""
split.py — Time-aware train/validation/test split for MovieLens 1M.

Splits each user's interaction history chronologically:
  - Train: oldest 60% of interactions
  - Validation: next 20%
  - Test: most recent 20%

Users with insufficient interactions fall back to random split.
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import (
    COLD_START_THRESHOLD,
    PROCESSED_DATA_DIR,
    RATINGS_PROCESSED,
    RANDOM_SEED,
    SPLIT_STRATEGY,
    TEST_FILE,
    TEST_RATIO,
    TRAIN_FILE,
    TRAIN_RATIO,
    VAL_FILE,
    VAL_RATIO,
)
from src.logging_utils import data_logger as logger


def time_aware_split(ratings: pd.DataFrame) -> tuple:
    """
    Split ratings chronologically per user.

    For each user:
      - Sort by timestamp ascending
      - First TRAIN_RATIO → train
      - Next VAL_RATIO → validation
      - Remaining TEST_RATIO → test

    Returns:
        (train_df, val_df, test_df)
    """
    train_parts = []
    val_parts = []
    test_parts = []

    # Sort by user and timestamp
    ratings_sorted = ratings.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    for user_id, group in ratings_sorted.groupby("user_id"):
        n = len(group)
        if n < COLD_START_THRESHOLD:
            # Too few ratings — put all in train (cold-start user)
            train_parts.append(group)
            continue

        n_train = max(1, int(n * TRAIN_RATIO))
        n_val = max(1, int(n * VAL_RATIO))
        # remaining goes to test (handles rounding)
        n_test = n - n_train - n_val

        if n_test < 1:
            # Not enough for test — adjust
            n_val = max(0, n_val - 1)
            n_test = n - n_train - n_val

        train_parts.append(group.iloc[:n_train])
        val_parts.append(group.iloc[n_train:n_train + n_val])
        if n_test > 0:
            test_parts.append(group.iloc[n_train + n_val:])

    train = pd.concat(train_parts)
    val = pd.concat(val_parts) if val_parts else pd.DataFrame(columns=ratings.columns)
    test = pd.concat(test_parts) if test_parts else pd.DataFrame(columns=ratings.columns)

    return train, val, test


def random_split(ratings: pd.DataFrame) -> tuple:
    """
    Fallback random train/val/test split.
    """
    shuffled = ratings.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    n = len(shuffled)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    train = shuffled.iloc[:n_train]
    val = shuffled.iloc[n_train:n_train + n_val]
    test = shuffled.iloc[n_train + n_val:]
    return train, val, test


def run_split() -> bool:
    """
    Run the split pipeline and persist train/val/test parquet files.

    Returns:
        True on success.
    """
    logger.start_phase("Phase 3 - Split", f"Generate train/val/test splits (strategy: {SPLIT_STRATEGY})")

    if not RATINGS_PROCESSED.exists():
        logger.error("Preprocessed ratings not found. Run preprocess.py first.")
        return False

    try:
        ratings = pd.read_parquet(str(RATINGS_PROCESSED))
        logger.info(f"Loaded {len(ratings):,} preprocessed ratings")

        if SPLIT_STRATEGY == "time_aware":
            train, val, test = time_aware_split(ratings)
        else:
            train, val, test = random_split(ratings)

        # Log split sizes
        logger.log_metric("train_size", len(train), "split")
        logger.log_metric("val_size", len(val), "split")
        logger.log_metric("test_size", len(test), "split")
        logger.info(f"Split: train={len(train):,} | val={len(val):,} | test={len(test):,}")

        # Log unique user counts per split
        train_users = train["user_id"].nunique()
        val_users = val["user_id"].nunique()
        test_users = test["user_id"].nunique()
        logger.info(f"Unique users: train={train_users:,} | val={val_users:,} | test={test_users:,}")

        # Verify no test leakage (by timestamp)
        if SPLIT_STRATEGY == "time_aware" and len(test) > 0:
            # For any overlapping user, max train timestamp should < min test timestamp
            # (approximate check — full check would be O(n_users))
            sample_users = train["user_id"].unique()[:100]
            leakage = 0
            for uid in sample_users:
                max_train_ts = train[train["user_id"] == uid]["timestamp"].max()
                test_user = test[test["user_id"] == uid]
                if len(test_user) > 0:
                    min_test_ts = test_user["timestamp"].min()
                    if max_train_ts >= min_test_ts:
                        leakage += 1
            if leakage > 0:
                logger.warning(f"Potential temporal leakage detected for {leakage}/100 sampled users")
            else:
                logger.info("Temporal ordering verified for sample of 100 users — no leakage detected")

        # Persist
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        train.to_parquet(str(TRAIN_FILE), index=False)
        val.to_parquet(str(VAL_FILE), index=False)
        test.to_parquet(str(TEST_FILE), index=False)

        logger.log_artifact(str(TRAIN_FILE), "Training set (parquet)")
        logger.log_artifact(str(VAL_FILE), "Validation set (parquet)")
        logger.log_artifact(str(TEST_FILE), "Test set (parquet)")

        # Persist split summary
        summary = {
            "strategy": SPLIT_STRATEGY,
            "train_size": len(train),
            "val_size": len(val),
            "test_size": len(test),
            "train_users": int(train_users),
            "val_users": int(val_users),
            "test_users": int(test_users),
        }
        split_summary_file = PROCESSED_DATA_DIR / "split_summary.json"
        split_summary_file.write_text(json.dumps(summary, indent=2))

        logger.end_phase("Phase 3 - Split", f"Split complete: {SPLIT_STRATEGY}", "Phase 4 - Baseline model")
        return True

    except Exception as e:
        logger.log_error(e, "Split pipeline failed")
        return False


if __name__ == "__main__":
    success = run_split()
    sys.exit(0 if success else 1)
