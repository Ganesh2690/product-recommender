"""
validate.py — Dataset validation for MovieLens 1M.

Verifies file existence, column schemas, row counts, and data quality.
Logs all findings to data_pipeline.log and run_log.jsonl.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import (
    EXPECTED_NUM_MOVIES_MIN,
    EXPECTED_NUM_USERS_MIN,
    EXPECTED_TOTAL_RATINGS_MAX,
    EXPECTED_TOTAL_RATINGS_MIN,
    MIN_RATINGS_PER_USER,
    MOVIES_COLS,
    MOVIES_FILE,
    RATINGS_COLS,
    RATINGS_FILE,
    USERS_COLS,
    USERS_FILE,
)
from src.logging_utils import data_logger as logger


def load_ratings() -> pd.DataFrame:
    """Load ratings.dat with correct separator and column names."""
    df = pd.read_csv(
        str(RATINGS_FILE),
        sep="::",
        names=RATINGS_COLS,
        engine="python",
        encoding="latin-1",
    )
    return df


def load_movies() -> pd.DataFrame:
    """Load movies.dat with correct separator and column names."""
    df = pd.read_csv(
        str(MOVIES_FILE),
        sep="::",
        names=MOVIES_COLS,
        engine="python",
        encoding="latin-1",
    )
    return df


def load_users() -> pd.DataFrame:
    """Load users.dat with correct separator and column names."""
    df = pd.read_csv(
        str(USERS_FILE),
        sep="::",
        names=USERS_COLS,
        engine="python",
        encoding="latin-1",
    )
    return df


def validate_dataset() -> bool:
    """
    Run all validation checks on the MovieLens 1M dataset.

    Returns:
        True if all checks pass, False if any critical check fails.
    """
    logger.start_phase("Phase 2 - Validate", "Validate MovieLens 1M dataset integrity and schema")
    failures = []
    warnings = []

    # --- File existence ---
    for f in [RATINGS_FILE, MOVIES_FILE, USERS_FILE]:
        if not f.exists():
            failures.append(f"Missing file: {f}")
            logger.error(f"Missing required file: {f}")
        else:
            logger.info(f"File exists: {f.name} ({f.stat().st_size:,} bytes)")

    if failures:
        logger.error(f"Critical validation failures: {failures}")
        logger.end_phase("Phase 2 - Validate", "FAILED — missing files", "Fix dataset")
        return False

    # --- Load data ---
    try:
        ratings = load_ratings()
        movies = load_movies()
        users = load_users()
    except Exception as e:
        logger.log_error(e, "Failed to load dataset files")
        return False

    # --- Schema checks ---
    for col in RATINGS_COLS:
        if col not in ratings.columns:
            failures.append(f"ratings.dat missing column: {col}")
    for col in MOVIES_COLS:
        if col not in movies.columns:
            failures.append(f"movies.dat missing column: {col}")
    for col in USERS_COLS:
        if col not in users.columns:
            failures.append(f"users.dat missing column: {col}")

    # --- Row count checks ---
    n_ratings = len(ratings)
    n_movies = len(movies)
    n_users = len(users)

    logger.info(f"Ratings: {n_ratings:,} rows")
    logger.info(f"Movies: {n_movies:,} rows")
    logger.info(f"Users: {n_users:,} rows")

    logger.log_metric("total_ratings", n_ratings, "raw dataset")
    logger.log_metric("total_movies", n_movies, "raw dataset")
    logger.log_metric("total_users", n_users, "raw dataset")

    if not (EXPECTED_TOTAL_RATINGS_MIN <= n_ratings <= EXPECTED_TOTAL_RATINGS_MAX):
        failures.append(f"Unexpected number of ratings: {n_ratings}")
    if n_users < EXPECTED_NUM_USERS_MIN:
        warnings.append(f"Fewer users than expected: {n_users} < {EXPECTED_NUM_USERS_MIN}")
    if n_movies < EXPECTED_NUM_MOVIES_MIN:
        warnings.append(f"Fewer movies than expected: {n_movies} < {EXPECTED_NUM_MOVIES_MIN}")

    # --- Data type checks ---
    if ratings["rating"].min() < 1 or ratings["rating"].max() > 5:
        failures.append(f"Ratings out of expected range [1,5]: {ratings['rating'].min()}–{ratings['rating'].max()}")

    # --- Missing value checks ---
    for df, name in [(ratings, "ratings"), (movies, "movies"), (users, "users")]:
        null_counts = df.isnull().sum()
        total_nulls = null_counts.sum()
        if total_nulls > 0:
            warnings.append(f"{name} has {total_nulls} null values: {null_counts.to_dict()}")
            logger.warning(f"{name} null values: {null_counts.to_dict()}")

    # --- Rating distribution ---
    rating_dist = ratings["rating"].value_counts().sort_index().to_dict()
    mean_rating = ratings["rating"].mean()
    logger.log_metric("mean_rating", round(mean_rating, 4), "raw ratings")
    logger.log_metric("rating_distribution", rating_dist, "raw ratings")

    # --- Sparsity ---
    n_unique_users = ratings["user_id"].nunique()
    n_unique_movies = ratings["movie_id"].nunique()
    matrix_size = n_unique_users * n_unique_movies
    sparsity = 1.0 - (n_ratings / matrix_size)
    logger.log_metric("matrix_sparsity", round(sparsity, 6), f"{n_unique_users} users × {n_unique_movies} movies")
    logger.info(f"Matrix sparsity: {sparsity:.4%} ({n_unique_users} users × {n_unique_movies} movies)")

    # --- Per-user rating counts ---
    user_rating_counts = ratings.groupby("user_id").size()
    users_below_threshold = (user_rating_counts < MIN_RATINGS_PER_USER).sum()
    if users_below_threshold > 0:
        warnings.append(f"{users_below_threshold} users have < {MIN_RATINGS_PER_USER} ratings (will be filtered)")
        logger.warning(f"{users_below_threshold} users below minimum ratings threshold")

    # --- Report ---
    if warnings:
        for w in warnings:
            logger.warning(f"VALIDATION WARNING: {w}")

    if failures:
        for f in failures:
            logger.error(f"VALIDATION FAILURE: {f}")
        logger.end_phase("Phase 2 - Validate", f"Validation FAILED with {len(failures)} errors", "Fix data issues")
        return False

    logger.info(f"Validation PASSED. Warnings: {len(warnings)}")
    logger.end_phase(
        "Phase 2 - Validate",
        f"Dataset validated: {n_ratings:,} ratings, {n_users:,} users, {n_movies:,} movies",
        "Phase 3 - Preprocess",
    )
    return True


if __name__ == "__main__":
    success = validate_dataset()
    sys.exit(0 if success else 1)
