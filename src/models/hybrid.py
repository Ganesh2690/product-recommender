"""
hybrid.py — Hybrid Recommender combining CF (SVD) and Content signals.

Strategy: Weighted blending
  score(user, item) = α × CF_score(user, item) + (1-α) × Content_score(user, item)

Content score is based on:
  - Genre similarity between user's historical preference profile and candidate items
  - Normalized item popularity (log-dampened)

α is tunable and defaults to HYBRID_ALPHA (0.8) — heavily weighted toward CF.

Phase 7 — Hybrid / Neural Re-ranking
"""

import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import (
    HYBRID_ALPHA,
    ITEM_FEATURES,
    MODELS_DIR,
    TOP_N,
    TRAIN_FILE,
)
from src.logging_utils import model_logger as logger


class HybridRecommender:
    """
    Weighted hybrid recommender: combines SVD-based CF score with content-based score.

    Content score = cosine similarity between user's weighted genre profile and item genres,
                    augmented by log-normalized popularity.

    The alpha parameter controls the CF vs content tradeoff:
    - alpha=1.0: pure CF
    - alpha=0.0: pure content
    - alpha=0.8 (default): 80% CF, 20% content
    """

    def __init__(
        self,
        cf_model,
        alpha: float = HYBRID_ALPHA,
        fallback_model=None,
    ):
        self.cf_model = cf_model
        self.alpha = alpha
        self.fallback_model = fallback_model  # optional popularity fallback

        # Content state
        self.item_genre_matrix: Optional[np.ndarray] = None   # (n_items, n_genres)
        self.item_ids: Optional[np.ndarray] = None
        self.item_id_to_idx: Dict[int, int] = {}
        self.genre_cols: List[str] = []
        self.item_popularity: Dict[int, float] = {}

        # User genre profiles (learning)
        self.user_profiles: Dict[int, np.ndarray] = {}
        self.user_rated_items: Dict[int, List[int]] = {}
        self.user_ratings: Dict[int, Dict[int, float]] = {}

        self.is_fitted = False

    def fit(self, train_df: pd.DataFrame, item_features_df: pd.DataFrame) -> "HybridRecommender":
        """
        Fit the hybrid model.

        Args:
            train_df: Training ratings DataFrame [user_id, movie_id, rating]
            item_features_df: Item features DataFrame [movie_id, genre_*, year_norm]
        """
        logger.start_phase("Phase 7 - Hybrid Fit", f"Train hybrid model (alpha={self.alpha})")
        start = time.time()

        # --- Build item feature matrix ---
        # Exclude 'genre_list' (object column with lists) — only keep binary genre indicator cols
        self.genre_cols = [
            c for c in item_features_df.columns
            if c.startswith("genre_") and c != "genre_list"
        ]
        # Use year_norm if present, otherwise just genre indicators
        if "year_norm" in item_features_df.columns:
            feature_cols = self.genre_cols + ["year_norm"]
        else:
            feature_cols = self.genre_cols

        self.item_ids = item_features_df["movie_id"].values
        self.item_id_to_idx = {mid: i for i, mid in enumerate(self.item_ids)}

        feature_df = item_features_df[feature_cols].fillna(0.0)
        self.item_genre_matrix = feature_df.values.astype(np.float32)

        # Normalize rows to unit vectors for cosine similarity
        norms = np.linalg.norm(self.item_genre_matrix, axis=1, keepdims=True) + 1e-8
        self.item_genre_matrix_normalized = self.item_genre_matrix / norms

        # --- Build item popularity scores (log-normalized) ---
        item_counts = train_df.groupby("movie_id")["rating"].count()
        max_count = float(item_counts.max())
        self.item_popularity = {
            mid: float(np.log1p(count) / np.log1p(max_count))
            for mid, count in item_counts.items()
        }

        # --- Build user rating histories for profile computation ---
        for _, row in train_df.iterrows():
            uid = int(row["user_id"])
            mid = int(row["movie_id"])
            r = float(row["rating"])
            if uid not in self.user_rated_items:
                self.user_rated_items[uid] = []
                self.user_ratings[uid] = {}
            self.user_rated_items[uid].append(mid)
            self.user_ratings[uid][mid] = r

        # --- Build user genre profiles ---
        # User profile = weighted average of genre vectors of rated items (weight = rating)
        for uid, rated_items in self.user_rated_items.items():
            profile = np.zeros(len(feature_cols), dtype=np.float32)
            total_weight = 0.0
            for mid in rated_items:
                if mid in self.item_id_to_idx:
                    idx = self.item_id_to_idx[mid]
                    weight = self.user_ratings[uid].get(mid, 3.0)
                    profile += weight * self.item_genre_matrix[idx]
                    total_weight += weight
            if total_weight > 0:
                profile /= total_weight
            # Normalize
            norm = np.linalg.norm(profile) + 1e-8
            self.user_profiles[uid] = profile / norm

        # --- Ensure CF model has all user/item data ---
        if not self.cf_model.is_fitted:
            self.cf_model.fit(train_df)

        self.is_fitted = True
        elapsed = time.time() - start
        logger.log_metric("hybrid_alpha", self.alpha, "Phase 7")
        logger.log_runtime("hybrid_fit", elapsed, "Phase 7")
        logger.end_phase(
            "Phase 7 - Hybrid Fit",
            f"Hybrid model fitted (alpha={self.alpha}, {len(self.user_profiles)} user profiles)",
            "Phase 7 - Evaluate",
        )
        return self

    def _content_score(self, user_id: int, candidate_items: List[int]) -> Dict[int, float]:
        """
        Compute content-based scores for candidate items for a given user.
        Score = cosine_similarity(user_profile, item_genres) + 0.1 * log_popularity
        """
        if user_id not in self.user_profiles:
            # No profile — return uniform scores
            return {mid: 0.5 for mid in candidate_items}

        user_vec = self.user_profiles[user_id]  # (n_features,)

        scores = {}
        for mid in candidate_items:
            if mid not in self.item_id_to_idx:
                scores[mid] = 0.0
                continue
            idx = self.item_id_to_idx[mid]
            item_vec = self.item_genre_matrix_normalized[idx]
            cos_sim = float(np.dot(user_vec, item_vec))
            pop = self.item_popularity.get(mid, 0.0)
            scores[mid] = 0.9 * cos_sim + 0.1 * pop

        return scores

    def _cf_score(self, user_id: int, candidate_items: List[int]) -> Dict[int, float]:
        """Get CF predicted ratings for candidate items."""
        scores = {}
        for mid in candidate_items:
            try:
                pred = self.cf_model.predict_rating(user_id, mid)
                scores[mid] = pred
            except Exception:
                scores[mid] = 3.0  # global mean fallback
        return scores

    def recommend(
        self,
        user_id: int,
        n: int = TOP_N,
        seen_items: Optional[set] = None,
    ) -> List[Dict]:
        """
        Generate hybrid recommendations.

        Gets top candidates from CF, then rescores with blended score.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")

        seen = set(seen_items or []) | set(self.user_rated_items.get(user_id, []))

        # Get more candidates from CF than needed (to allow re-ranking)
        n_candidates = min(n * 5, 200)
        cf_results = self.cf_model.recommend(user_id, n=n_candidates, seen_items=seen)

        if not cf_results:
            return []

        cf_candidates = [r["movie_id"] for r in cf_results]

        # Get CF scores (predicted ratings, normalized to [0,1] roughly)
        cf_scores = {}
        for mid in cf_candidates:
            try:
                pred = self.cf_model.predict_rating(user_id, mid)
                cf_scores[mid] = (pred - 1.0) / 4.0  # scale [1,5] → [0,1]
            except Exception:
                cf_scores[mid] = 0.5

        # Get content scores
        content_scores = self._content_score(user_id, cf_candidates)

        # Blend
        blended = {}
        for mid in cf_candidates:
            blended[mid] = (
                self.alpha * cf_scores.get(mid, 0.5)
                + (1 - self.alpha) * content_scores.get(mid, 0.5)
            )

        # Sort and return top-N
        sorted_items = sorted(blended.items(), key=lambda x: x[1], reverse=True)
        return [{"movie_id": mid, "score": round(score, 4)} for mid, score in sorted_items[:n]]

    def similar_items(self, movie_id: int, n: int = 10) -> List[int]:
        """Delegate to CF model."""
        return self.cf_model.similar_items(movie_id, n=n)

    def save(self, path: Optional[Path] = None) -> Path:
        """Save model artifact."""
        save_path = path or (MODELS_DIR / "hybrid_model.pkl")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(save_path), "wb") as f:
            pickle.dump(self, f)
        logger.log_artifact(str(save_path), "Hybrid recommender model (pickle)")
        return save_path

    @classmethod
    def load(cls, path: Path) -> "HybridRecommender":
        """Load a previously saved model."""
        with open(str(path), "rb") as f:
            model = pickle.load(f)
        logger.info(f"Loaded hybrid model from {path}")
        return model


def train_hybrid(cf_model=None, alpha: float = HYBRID_ALPHA) -> HybridRecommender:
    """
    Train and save the hybrid model.

    Args:
        cf_model: Pre-trained CF model (SVDRecommender). If None, loads from disk.
        alpha: CF vs content blending weight.
    """
    logger.start_phase("Phase 7 - Hybrid", f"Train hybrid model (alpha={alpha})")

    if cf_model is None:
        from src.models.matrix_factorization import SVDRecommender
        svd_path = MODELS_DIR / "svd_model.pkl"
        if svd_path.exists():
            cf_model = SVDRecommender.load(svd_path)
        else:
            from src.models.matrix_factorization import train_svd
            cf_model = train_svd()

    train_df = pd.read_parquet(str(TRAIN_FILE))
    item_features_df = pd.read_parquet(str(ITEM_FEATURES))

    model = HybridRecommender(cf_model=cf_model, alpha=alpha)
    model.fit(train_df, item_features_df)
    save_path = model.save()

    logger.end_phase("Phase 7 - Hybrid", f"Hybrid model saved to {save_path}", "Phase 8 - Serving")
    return model


if __name__ == "__main__":
    model = train_hybrid()
    recs = model.recommend(user_id=1, n=10)
    print(f"Hybrid recommendations for user 1: {recs}")
    sys.exit(0)
