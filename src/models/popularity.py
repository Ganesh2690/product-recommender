"""
popularity.py — Popularity Baseline Recommender.

Recommends the globally most-rated items. Non-personalized.
Serves as:
1. Performance baseline (any personalized model must beat this)
2. Cold-start fallback for users with insufficient interaction history

Phase 4 — Baseline Model
"""

import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import (
    MODELS_DIR,
    TOP_N,
    TRAIN_FILE,
)
from src.logging_utils import model_logger as logger


class PopularityRecommender:
    """
    Popularity-based non-personalized recommender.

    Ranks items by number of ratings in the training set.
    Optionally weighted by mean rating (rating-weighted popularity).
    """

    def __init__(self, use_rating_weight: bool = False):
        self.use_rating_weight = use_rating_weight
        self.popular_items: List[int] = []
        self.item_scores: Dict[int, float] = {}
        self.is_fitted = False
        self.training_stats: Dict = {}

    def fit(self, train_df: pd.DataFrame) -> "PopularityRecommender":
        """
        Fit the model by computing item popularity scores.

        Args:
            train_df: DataFrame with columns [user_id, movie_id, rating, timestamp]
        """
        logger.start_phase("Phase 4 - Popularity Fit", "Train popularity baseline on interaction counts")
        start = time.time()

        item_stats = train_df.groupby("movie_id").agg(
            count=("rating", "count"),
            mean_rating=("rating", "mean"),
        ).reset_index()

        if self.use_rating_weight:
            # Score = count × mean_rating (down-weight items with many low ratings)
            item_stats["score"] = item_stats["count"] * item_stats["mean_rating"]
        else:
            # Score = raw count (classic popularity)
            item_stats["score"] = item_stats["count"].astype(float)

        item_stats = item_stats.sort_values("score", ascending=False)
        self.item_scores = dict(zip(item_stats["movie_id"], item_stats["score"]))
        self.popular_items = item_stats["movie_id"].tolist()
        self.is_fitted = True

        elapsed = time.time() - start
        self.training_stats = {
            "n_items": len(self.popular_items),
            "top_item": self.popular_items[0] if self.popular_items else None,
            "use_rating_weight": self.use_rating_weight,
            "fit_time_s": round(elapsed, 3),
        }

        logger.log_metric("popularity_items_trained", len(self.popular_items), "Phase 4")
        logger.log_runtime("popularity_fit", elapsed, "Phase 4")
        logger.end_phase("Phase 4 - Popularity Fit", f"Fitted on {len(self.popular_items)} items", "Phase 4 - Evaluate")
        return self

    def recommend(
        self,
        user_id: int,
        n: int = TOP_N,
        seen_items: Optional[set] = None,
    ) -> List[Dict]:
        """
        Return top-N popular items, excluding already-seen items.

        Args:
            user_id: User ID (ignored — popularity is non-personalized)
            n: Number of recommendations to return
            seen_items: Set of already-seen item IDs to exclude

        Returns:
            List of {movie_id, score} dicts sorted by score descending
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling recommend()")

        exclude = set(seen_items or [])
        recs = [
            {"movie_id": item, "score": float(self.item_scores[item])}
            for item in self.popular_items
            if item not in exclude
        ]
        return recs[:n]

    def recommend_batch(
        self,
        user_ids: List[int],
        n: int = TOP_N,
        user_seen_items: Optional[Dict[int, set]] = None,
    ) -> Dict[int, List[Dict]]:
        """
        Generate recommendations for multiple users at once.
        """
        user_seen = user_seen_items or {}
        return {
            uid: self.recommend(uid, n=n, seen_items=user_seen.get(uid, set()))
            for uid in user_ids
        }

    def get_item_score(self, movie_id: int) -> float:
        """Return the popularity score for an item."""
        return self.item_scores.get(movie_id, 0.0)

    def save(self, path=None) -> Path:
        """Save model artifact."""
        save_path = Path(path) if path is not None else (MODELS_DIR / "popularity_model.pkl")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(save_path), "wb") as f:
            pickle.dump(self, f)
        logger.log_artifact(str(save_path), "Popularity baseline model (pickle)")
        return save_path

    @classmethod
    def load(cls, path: Path) -> "PopularityRecommender":
        """Load a previously saved model."""
        with open(str(path), "rb") as f:
            model = pickle.load(f)
        logger.info(f"Loaded popularity model from {path}")
        return model


def train_popularity_baseline() -> PopularityRecommender:
    """
    Full training routine for the popularity baseline.
    Loads training data, fits model, saves artifact.
    """
    logger.start_phase("Phase 4 - Popularity Baseline", "Train and save popularity baseline")

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Training data not found: {TRAIN_FILE}")

    train_df = pd.read_parquet(str(TRAIN_FILE))
    logger.info(f"Loaded training set: {len(train_df):,} ratings")

    model = PopularityRecommender(use_rating_weight=False)
    model.fit(train_df)
    save_path = model.save()

    logger.end_phase("Phase 4 - Popularity Baseline", f"Saved to {save_path}", "Phase 4 - Evaluate popularity")
    return model


if __name__ == "__main__":
    model = train_popularity_baseline()
    # Quick sanity check
    sample_recs = model.recommend(user_id=1, n=10)
    print(f"Sample recommendations: {sample_recs}")
    sys.exit(0)
