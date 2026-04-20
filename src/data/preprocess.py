"""
preprocess.py — Data preprocessing pipeline for MovieLens 1M.

Normalizes user/item IDs, parses genres, encodes demographic features,
builds user-item interaction matrix, and persists processed outputs.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import (
    ITEM_FEATURES,
    MIN_RATINGS_PER_USER,
    MOVIES_PROCESSED,
    PROCESSED_DATA_DIR,
    RATINGS_PROCESSED,
    USER_ITEM_MATRIX,
    USERS_PROCESSED,
)
from src.data.validate import load_movies, load_ratings, load_users
from src.logging_utils import data_logger as logger

# All genres in MovieLens 1M
ALL_GENRES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


def preprocess_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and normalize the ratings dataframe.

    - Filter users with < MIN_RATINGS_PER_USER ratings
    - Convert timestamp to datetime
    - Ensure correct dtypes
    """
    logger.info(f"Preprocessing ratings: {len(ratings):,} rows")

    # Filter cold-start users
    user_counts = ratings.groupby("user_id").size()
    valid_users = user_counts[user_counts >= MIN_RATINGS_PER_USER].index
    before = len(ratings)
    ratings = ratings[ratings["user_id"].isin(valid_users)].copy()
    after = len(ratings)
    if before != after:
        logger.warning(f"Filtered {before - after:,} ratings from users with < {MIN_RATINGS_PER_USER} ratings")

    # Convert timestamp
    ratings["datetime"] = pd.to_datetime(ratings["timestamp"], unit="s")

    # Ensure dtypes
    ratings["user_id"] = ratings["user_id"].astype(int)
    ratings["movie_id"] = ratings["movie_id"].astype(int)
    ratings["rating"] = ratings["rating"].astype(float)

    logger.log_metric("ratings_after_preprocess", len(ratings))
    logger.log_metric("unique_users_after_filter", ratings["user_id"].nunique())
    return ratings


def preprocess_movies(movies: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and enrich the movies dataframe.

    - Parse genres (pipe-separated) into list and one-hot columns
    - Extract year from title
    """
    logger.info(f"Preprocessing movies: {len(movies):,} rows")

    # Parse genres
    movies["genres_list"] = movies["genres"].str.split("|")
    # Keep genre_list as alias for backward compatibility
    movies["genre_list"] = movies["genres_list"]

    # One-hot encode genres
    for genre in ALL_GENRES:
        safe = genre.lower().replace('-', '_').replace("'", '')
        col = f"genre_{safe}"
        movies[col] = movies["genres_list"].apply(lambda g: int(genre in g))

    # Extract year from title (e.g., "Toy Story (1995)")
    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)$").astype(float)

    # Normalize year to [0, 1]
    yr_min = movies["year"].min()
    yr_max = movies["year"].max()
    if yr_max > yr_min:
        movies["year_norm"] = (movies["year"] - yr_min) / (yr_max - yr_min)
    else:
        movies["year_norm"] = 0.5

    return movies


def preprocess_users(users: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and encode user demographic features.

    - Encode gender as binary (M=1, F=0)
    - Age is already a categorical code (1, 18, 25, 35, 45, 50, 56)
    """
    logger.info(f"Preprocessing users: {len(users):,} rows")
    users["gender_binary"] = (users["gender"] == "M").astype(int)
    return users


def build_user_item_matrix(ratings: pd.DataFrame) -> tuple:
    """
    Build a scipy sparse CSR user-item interaction matrix.

    Returns:
        (matrix, user_id_map, movie_id_map)
        - matrix: scipy.sparse.csr_matrix of shape (n_users, n_movies)
        - user_id_map: dict original_user_id -> matrix_row_index
        - movie_id_map: dict original_movie_id -> matrix_col_index
    """
    unique_users = sorted(ratings["user_id"].unique())
    unique_movies = sorted(ratings["movie_id"].unique())
    user_id_map = {uid: i for i, uid in enumerate(unique_users)}
    movie_id_map = {mid: i for i, mid in enumerate(unique_movies)}

    rows = ratings["user_id"].map(user_id_map).values
    cols = ratings["movie_id"].map(movie_id_map).values
    data = ratings["rating"].values.astype(np.float32)

    matrix = sp.csr_matrix(
        (data, (rows, cols)),
        shape=(len(unique_users), len(unique_movies)),
        dtype=np.float32,
    )
    logger.info(f"User-item matrix: {matrix.shape} — {matrix.nnz:,} non-zeros ({1-matrix.nnz/np.prod(matrix.shape):.2%} sparse)")  # noqa: E501
    logger.log_metric("matrix_shape", matrix.shape, "user-item interaction matrix")
    logger.log_metric("matrix_nnz", matrix.nnz, "non-zero entries")
    return matrix, user_id_map, movie_id_map


def build_item_features(movies: pd.DataFrame) -> pd.DataFrame:
    """
    Build item feature matrix from movie genres and year.
    Used by the hybrid model for content-based scoring.
    """
    genre_cols = [c for c in movies.columns if c.startswith("genre_")]
    feature_cols = genre_cols + ["year"]
    item_features = movies[["movie_id"] + feature_cols].copy()
    # Normalize year to [0,1] range
    fy = item_features["year"].fillna(item_features["year"].median())
    item_features["year_norm"] = (fy - fy.min()) / (fy.max() - fy.min() + 1e-8)
    return item_features


def run_preprocessing() -> bool:
    """
    Run the full preprocessing pipeline and persist results.

    Returns:
        True on success.
    """
    logger.start_phase("Phase 3 - Preprocess", "Preprocess all dataset files, build user-item matrix")
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # Load raw data
        ratings_raw = load_ratings()
        movies_raw = load_movies()
        users_raw = load_users()

        # Preprocess
        ratings = preprocess_ratings(ratings_raw)
        movies = preprocess_movies(movies_raw)
        users = preprocess_users(users_raw)

        # Build user-item matrix
        matrix, user_id_map, movie_id_map = build_user_item_matrix(ratings)

        # Build item features
        item_features = build_item_features(movies)

        # Persist processed files
        ratings.to_parquet(str(RATINGS_PROCESSED), index=False)
        movies.to_parquet(str(MOVIES_PROCESSED), index=False)
        users.to_parquet(str(USERS_PROCESSED), index=False)
        item_features.to_parquet(str(ITEM_FEATURES), index=False)
        sp.save_npz(str(USER_ITEM_MATRIX), matrix)

        # Save ID maps
        import json
        maps_file = PROCESSED_DATA_DIR / "id_maps.json"
        maps_file.write_text(json.dumps({
            "user_id_map": {str(k): int(v) for k, v in user_id_map.items()},
            "movie_id_map": {str(k): int(v) for k, v in movie_id_map.items()},
            "inverse_user_map": {str(v): int(k) for k, v in user_id_map.items()},
            "inverse_movie_map": {str(v): int(k) for k, v in movie_id_map.items()},
        }, indent=2))

        logger.log_artifact(str(RATINGS_PROCESSED), "Preprocessed ratings (parquet)")
        logger.log_artifact(str(USER_ITEM_MATRIX), "User-item interaction matrix (scipy sparse NPZ)")
        logger.end_phase("Phase 3 - Preprocess", "All preprocessing complete", "Phase 3 - Split")
        return True

    except Exception as e:
        logger.log_error(e, "Preprocessing pipeline failed")
        return False


if __name__ == "__main__":
    success = run_preprocessing()
    sys.exit(0 if success else 1)
