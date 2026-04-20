"""
test_models.py — Smoke tests and correctness tests for all recommender models.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ratings_df():
    """Synthetic ratings large enough for model smoke tests."""
    rng = np.random.default_rng(42)
    n_users, n_items, n_interactions = 50, 100, 800
    user_ids = rng.integers(1, n_users + 1, size=n_interactions)
    item_ids = rng.integers(1, n_items + 1, size=n_interactions)
    ratings = rng.uniform(1.0, 5.0, size=n_interactions).round(1)
    timestamps = rng.integers(900000000, 1100000000, size=n_interactions)
    df = pd.DataFrame({
        "user_id": user_ids,
        "movie_id": item_ids,
        "rating": ratings,
        "timestamp": timestamps,
    })
    # Keep unique (user, movie) pairs — take last occurrence
    df = df.sort_values("timestamp").drop_duplicates(subset=["user_id", "movie_id"], keep="last")
    return df.reset_index(drop=True)


@pytest.fixture(scope="module")
def movies_df():
    """Synthetic movies."""
    genres_pool = ["Action", "Drama", "Comedy", "Thriller", "Romance", "Sci-Fi", "Horror"]
    rng = np.random.default_rng(42)
    records = []
    for i in range(1, 101):
        g1, g2 = rng.choice(genres_pool, 2, replace=False)
        records.append({
            "movie_id": i,
            "title": f"Movie {i} ({1990 + i % 20})",
            "genres": f"{g1}|{g2}",
        })
    return pd.DataFrame(records)


@pytest.fixture(scope="module")
def processed_data(ratings_df, movies_df):
    from src.data.preprocess import preprocess_ratings, preprocess_movies
    proc_ratings = preprocess_ratings(ratings_df)
    proc_movies = preprocess_movies(movies_df)
    return proc_ratings, proc_movies


# -------------------------------------------------------------------------
# Popularity model tests
# -------------------------------------------------------------------------

class TestPopularityRecommender:
    def test_fit_and_recommend(self, ratings_df):
        from src.models.popularity import PopularityRecommender
        model = PopularityRecommender()
        model.fit(ratings_df)
        recs = model.recommend(user_id=1, n=10, seen_items=set())
        assert isinstance(recs, list)
        assert len(recs) <= 10

    def test_output_structure(self, ratings_df):
        from src.models.popularity import PopularityRecommender
        model = PopularityRecommender()
        model.fit(ratings_df)
        recs = model.recommend(user_id=1, n=5, seen_items=set())
        for item in recs:
            assert "movie_id" in item
            assert "score" in item

    def test_popularity_order(self, ratings_df):
        from src.models.popularity import PopularityRecommender
        model = PopularityRecommender()
        model.fit(ratings_df)
        recs = model.recommend(user_id=1, n=20, seen_items=set())
        scores = [r["score"] for r in recs]
        assert scores == sorted(scores, reverse=True), "Should be sorted descending"

    def test_seen_items_excluded(self, ratings_df):
        from src.models.popularity import PopularityRecommender
        model = PopularityRecommender()
        model.fit(ratings_df)
        seen = {1, 2, 3, 4, 5}
        recs = model.recommend(user_id=1, n=10, seen_items=seen)
        rec_ids = {r["movie_id"] for r in recs}
        assert rec_ids.isdisjoint(seen)

    def test_recommend_batch(self, ratings_df):
        from src.models.popularity import PopularityRecommender
        model = PopularityRecommender()
        model.fit(ratings_df)
        user_ids = [1, 2, 3]
        batch = model.recommend_batch(user_ids, n=5)
        for uid in user_ids:
            assert uid in batch
            assert len(batch[uid]) <= 5

    def test_save_load(self, ratings_df, tmp_path):
        from src.models.popularity import PopularityRecommender
        model = PopularityRecommender()
        model.fit(ratings_df)
        save_path = tmp_path / "pop_model.pkl"
        model.save(str(save_path))
        loaded = PopularityRecommender.load(str(save_path))
        recs1 = model.recommend(user_id=1, n=5, seen_items=set())
        recs2 = loaded.recommend(user_id=1, n=5, seen_items=set())
        assert [r["movie_id"] for r in recs1] == [r["movie_id"] for r in recs2]


# -------------------------------------------------------------------------
# Item-CF model tests
# -------------------------------------------------------------------------

class TestItemCFRecommender:
    def test_fit_and_recommend(self, ratings_df):
        from src.models.item_cf import ItemCFRecommender
        model = ItemCFRecommender(k=10)
        model.fit(ratings_df)
        # Use a user with enough ratings
        active_user = int(ratings_df["user_id"].value_counts().index[0])
        recs = model.recommend(user_id=active_user, n=10, seen_items=set())
        assert isinstance(recs, list)

    def test_output_structure(self, ratings_df):
        from src.models.item_cf import ItemCFRecommender
        model = ItemCFRecommender(k=10)
        model.fit(ratings_df)
        active_user = int(ratings_df["user_id"].value_counts().index[0])
        recs = model.recommend(user_id=active_user, n=5, seen_items=set())
        for item in recs:
            assert "movie_id" in item
            assert "score" in item

    def test_cold_start_returns_empty_or_list(self, ratings_df):
        from src.models.item_cf import ItemCFRecommender
        model = ItemCFRecommender(k=10)
        model.fit(ratings_df)
        # User 99999 is cold-start
        recs = model.recommend(user_id=99999, n=10, seen_items=set())
        assert isinstance(recs, list)

    def test_similar_items(self, ratings_df):
        from src.models.item_cf import ItemCFRecommender
        model = ItemCFRecommender(k=10)
        model.fit(ratings_df)
        items = sorted(ratings_df["movie_id"].unique())
        if len(items) > 0:
            similar = model.similar_items(item_id=items[0], n=5)
            assert isinstance(similar, list)


# -------------------------------------------------------------------------
# Matrix factorization (SVD) tests
# -------------------------------------------------------------------------

class TestSVDRecommender:
    def test_fit_and_recommend(self, ratings_df):
        from src.models.matrix_factorization import SVDRecommender
        model = SVDRecommender(n_factors=10, n_epochs=5, use_svdpp=False)
        model.fit(ratings_df)
        active_user = int(ratings_df["user_id"].value_counts().index[0])
        recs = model.recommend(user_id=active_user, n=10, seen_items=set())
        assert isinstance(recs, list)
        assert len(recs) > 0

    def test_output_structure(self, ratings_df):
        from src.models.matrix_factorization import SVDRecommender
        model = SVDRecommender(n_factors=10, n_epochs=5)
        model.fit(ratings_df)
        active_user = int(ratings_df["user_id"].value_counts().index[0])
        recs = model.recommend(user_id=active_user, n=5, seen_items=set())
        for rec in recs:
            assert "movie_id" in rec
            assert "score" in rec
            assert -1.0 <= rec["score"] <= 6.0, "Score should be in plausible rating range"

    def test_cold_start_returns_list(self, ratings_df):
        from src.models.matrix_factorization import SVDRecommender
        model = SVDRecommender(n_factors=10, n_epochs=5)
        model.fit(ratings_df)
        recs = model.recommend(user_id=99999, n=10, seen_items=set())
        assert isinstance(recs, list)

    def test_predict_rating(self, ratings_df):
        from src.models.matrix_factorization import SVDRecommender
        model = SVDRecommender(n_factors=10, n_epochs=5)
        model.fit(ratings_df)
        active_user = int(ratings_df["user_id"].value_counts().index[0])
        item_id = int(ratings_df["movie_id"].iloc[0])
        pred = model.predict_rating(user_id=active_user, item_id=item_id)
        assert isinstance(pred, float)

    def test_save_load(self, ratings_df, tmp_path):
        from src.models.matrix_factorization import SVDRecommender
        model = SVDRecommender(n_factors=10, n_epochs=5)
        model.fit(ratings_df)
        save_path = tmp_path / "svd.pkl"
        model.save(str(save_path))
        loaded = SVDRecommender.load(str(save_path))
        active_user = int(ratings_df["user_id"].value_counts().index[0])
        recs1 = model.recommend(user_id=active_user, n=5, seen_items=set())
        recs2 = loaded.recommend(user_id=active_user, n=5, seen_items=set())
        assert len(recs1) == len(recs2)


# -------------------------------------------------------------------------
# Hybrid model tests
# -------------------------------------------------------------------------

class TestHybridRecommender:
    def test_fit_and_recommend(self, ratings_df, movies_df):
        from src.models.matrix_factorization import SVDRecommender
        from src.models.popularity import PopularityRecommender
        from src.models.hybrid import HybridRecommender
        from src.data.preprocess import preprocess_movies

        proc_movies = preprocess_movies(movies_df)
        cf_model = SVDRecommender(n_factors=10, n_epochs=5)
        cf_model.fit(ratings_df)
        fallback = PopularityRecommender()
        fallback.fit(ratings_df)

        hybrid = HybridRecommender(cf_model=cf_model, fallback_model=fallback, alpha=0.8)
        hybrid.fit(ratings_df, proc_movies)

        active_user = int(ratings_df["user_id"].value_counts().index[0])
        recs = hybrid.recommend(user_id=active_user, n=10, seen_items=set())
        assert isinstance(recs, list)

    def test_alpha_blend_scores(self, ratings_df, movies_df):
        """All returned scores should be finite numbers."""
        from src.models.matrix_factorization import SVDRecommender
        from src.models.popularity import PopularityRecommender
        from src.models.hybrid import HybridRecommender
        from src.data.preprocess import preprocess_movies

        proc_movies = preprocess_movies(movies_df)
        cf_model = SVDRecommender(n_factors=10, n_epochs=5)
        cf_model.fit(ratings_df)
        fallback = PopularityRecommender()
        fallback.fit(ratings_df)

        hybrid = HybridRecommender(cf_model=cf_model, fallback_model=fallback, alpha=0.8)
        hybrid.fit(ratings_df, proc_movies)

        active_user = int(ratings_df["user_id"].value_counts().index[0])
        recs = hybrid.recommend(user_id=active_user, n=5, seen_items=set())
        for r in recs:
            assert np.isfinite(r["score"])


# -------------------------------------------------------------------------
# Model registry tests
# -------------------------------------------------------------------------

class TestModelRegistry:
    def test_register_and_list(self, ratings_df, tmp_path):
        from src.models.popularity import PopularityRecommender
        from src.models.registry import ModelRegistry

        registry = ModelRegistry(registry_path=str(tmp_path / "registry.json"))
        model = PopularityRecommender()
        model.fit(ratings_df)

        save_path = tmp_path / "pop.pkl"
        model.save(str(save_path))

        registry.register(
            model_name="popularity",
            model_path=str(save_path),
            metrics={"hr_at_10": 0.30},
        )
        entries = registry.list_models()
        assert len(entries) >= 1
        names = [e["model_name"] for e in entries]
        assert "popularity" in names
