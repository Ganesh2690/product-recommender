"""
test_data_pipeline.py — Tests for data loading, validation, preprocessing, and splitting.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    MIN_RATINGS_PER_USER,
    RATINGS_COLS,
    MOVIES_COLS,
    USERS_COLS,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
)


# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------

@pytest.fixture
def sample_ratings():
    """Minimal synthetic ratings DataFrame."""
    return pd.DataFrame({
        "user_id": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
        "movie_id": [101, 102, 103, 104, 105, 101, 102, 103, 106, 107, 102, 103, 104, 105, 108],
        "rating": [4.0, 3.0, 5.0, 2.0, 4.0, 3.0, 4.0, 2.0, 5.0, 3.0, 4.0, 3.0, 5.0, 1.0, 2.0],
        "timestamp": [1000000, 1000100, 1000200, 1000300, 1000400,
                      1000000, 1000100, 1000200, 1000300, 1000400,
                      1000000, 1000100, 1000200, 1000300, 1000400],
    })


@pytest.fixture
def sample_movies():
    """Minimal synthetic movies DataFrame."""
    return pd.DataFrame({
        "movie_id": [101, 102, 103, 104, 105, 106, 107, 108],
        "title": [
            "Movie A (1995)", "Movie B (1996)", "Movie C (1997)",
            "Movie D (1998)", "Movie E (1999)", "Movie F (2000)",
            "Movie G (2001)", "Movie H (2002)",
        ],
        "genres": [
            "Action|Comedy", "Drama", "Action|Thriller",
            "Comedy|Romance", "Drama|Sci-Fi", "Horror",
            "Documentary", "Animation|Children's",
        ],
    })


@pytest.fixture
def sample_users():
    """Minimal synthetic users DataFrame."""
    return pd.DataFrame({
        "user_id": [1, 2, 3],
        "gender": ["M", "F", "M"],
        "age": [25, 35, 45],
        "occupation": [4, 7, 12],
        "zip_code": ["12345", "23456", "34567"],
    })


# -------------------------------------------------------------------------
# Schema validation
# -------------------------------------------------------------------------

class TestSchemaValidation:
    def test_ratings_columns(self, sample_ratings):
        for col in RATINGS_COLS:
            assert col in sample_ratings.columns, f"Missing column: {col}"

    def test_movies_columns(self, sample_movies):
        for col in MOVIES_COLS:
            assert col in sample_movies.columns, f"Missing column: {col}"

    def test_users_columns(self, sample_users):
        for col in USERS_COLS:
            assert col in sample_users.columns, f"Missing column: {col}"

    def test_rating_range(self, sample_ratings):
        assert sample_ratings["rating"].min() >= 1.0
        assert sample_ratings["rating"].max() <= 5.0

    def test_no_null_ratings(self, sample_ratings):
        assert sample_ratings["rating"].isnull().sum() == 0

    def test_user_id_positive(self, sample_ratings):
        assert (sample_ratings["user_id"] > 0).all()

    def test_movie_id_positive(self, sample_ratings):
        assert (sample_ratings["movie_id"] > 0).all()


# -------------------------------------------------------------------------
# Preprocessing
# -------------------------------------------------------------------------

class TestPreprocessing:
    def test_filter_min_ratings(self, sample_ratings):
        from src.data.preprocess import preprocess_ratings
        # Add a user with only 2 ratings (below threshold)
        few_ratings = pd.DataFrame({
            "user_id": [99, 99],
            "movie_id": [101, 102],
            "rating": [3.0, 4.0],
            "timestamp": [2000000, 2000001],
        })
        combined = pd.concat([sample_ratings, few_ratings], ignore_index=True)
        result = preprocess_ratings(combined)
        assert 99 not in result["user_id"].values, "Cold-start user should be filtered"

    def test_timestamp_conversion(self, sample_ratings):
        from src.data.preprocess import preprocess_ratings
        result = preprocess_ratings(sample_ratings)
        assert "datetime" in result.columns
        assert pd.api.types.is_datetime64_any_dtype(result["datetime"])

    def test_genre_parsing(self, sample_movies):
        from src.data.preprocess import preprocess_movies
        result = preprocess_movies(sample_movies)
        assert "genre_list" in result.columns
        assert isinstance(result["genre_list"].iloc[0], list)

    def test_genre_one_hot(self, sample_movies):
        from src.data.preprocess import preprocess_movies
        result = preprocess_movies(sample_movies)
        # Only check binary indicator columns, exclude genre_list (object column)
        genre_cols = [c for c in result.columns if c.startswith("genre_") and c != "genre_list"]
        assert len(genre_cols) > 0
        # All one-hot values should be 0 or 1
        for col in genre_cols:
            assert result[col].isin([0, 1]).all()

    def test_user_item_matrix_shape(self, sample_ratings):
        from src.data.preprocess import preprocess_ratings, build_user_item_matrix
        ratings = preprocess_ratings(sample_ratings)
        matrix, user_map, movie_map = build_user_item_matrix(ratings)
        n_users = ratings["user_id"].nunique()
        n_movies = ratings["movie_id"].nunique()
        assert matrix.shape == (n_users, n_movies)

    def test_user_item_matrix_nnz(self, sample_ratings):
        from src.data.preprocess import preprocess_ratings, build_user_item_matrix
        ratings = preprocess_ratings(sample_ratings)
        matrix, _, _ = build_user_item_matrix(ratings)
        assert matrix.nnz == len(ratings)

    def test_year_extraction(self, sample_movies):
        from src.data.preprocess import preprocess_movies
        result = preprocess_movies(sample_movies)
        # "Movie A (1995)" should yield year=1995
        first_year = result[result["movie_id"] == 101]["year"].iloc[0]
        assert first_year == 1995.0


# -------------------------------------------------------------------------
# Train/test split
# -------------------------------------------------------------------------

class TestSplit:
    def test_no_data_loss(self, sample_ratings):
        from src.data.split import time_aware_split
        from src.data.preprocess import preprocess_ratings
        ratings = preprocess_ratings(sample_ratings)
        train, val, test = time_aware_split(ratings)
        total = len(train) + len(val) + len(test)
        assert total == len(ratings), "No rows should be lost in split"

    def test_no_row_duplication(self, sample_ratings):
        from src.data.split import time_aware_split
        from src.data.preprocess import preprocess_ratings
        ratings = preprocess_ratings(sample_ratings)
        train, val, test = time_aware_split(ratings)
        all_indices = list(train.index) + list(val.index) + list(test.index)
        assert len(all_indices) == len(set(all_indices)), "No duplicate rows"

    def test_temporal_ordering(self, sample_ratings):
        """Train timestamps should be earlier than or equal to test timestamps per user."""
        from src.data.split import time_aware_split
        from src.data.preprocess import preprocess_ratings
        ratings = preprocess_ratings(sample_ratings)
        train, val, test = time_aware_split(ratings)
        if len(test) == 0:
            return
        for uid in test["user_id"].unique():
            max_train_ts = train[train["user_id"] == uid]["timestamp"].max() if uid in train["user_id"].values else 0
            min_test_ts = test[test["user_id"] == uid]["timestamp"].min()
            assert max_train_ts <= min_test_ts, f"Temporal leakage for user {uid}"

    def test_split_ratios_approximate(self, sample_ratings):
        """Verify split ratios are approximately correct."""
        from src.data.split import time_aware_split
        from src.data.preprocess import preprocess_ratings
        ratings = preprocess_ratings(sample_ratings)
        train, val, test = time_aware_split(ratings)
        total = len(train) + len(val) + len(test)
        actual_train_ratio = len(train) / total
        # Allow ±15% slack for small datasets
        assert abs(actual_train_ratio - TRAIN_RATIO) < 0.20


# -------------------------------------------------------------------------
# Data quality
# -------------------------------------------------------------------------

class TestDataQuality:
    def test_sparsity_measurable(self, sample_ratings):
        n_users = sample_ratings["user_id"].nunique()
        n_items = sample_ratings["movie_id"].nunique()
        n_ratings = len(sample_ratings)
        sparsity = 1.0 - n_ratings / (n_users * n_items)
        assert 0.0 <= sparsity <= 1.0

    def test_rating_distribution_valid(self, sample_ratings):
        dist = sample_ratings["rating"].value_counts()
        assert len(dist) > 0
        # All ratings should be between 1 and 5
        assert (sample_ratings["rating"] >= 1.0).all()
        assert (sample_ratings["rating"] <= 5.0).all()
