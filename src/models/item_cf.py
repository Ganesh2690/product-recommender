"""
item_cf.py — Item-Item Collaborative Filtering Recommender.

Computes item-item cosine similarity on the user-item rating matrix.
For a given user, scores unrated items as a weighted average of rated-item similarities.

Phase 5 — Collaborative Filtering Baseline
"""

import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import (
    CF_MIN_SUPPORT,
    CF_N_SIMILAR_ITEMS,
    MODELS_DIR,
    TOP_N,
    TRAIN_FILE,
)
from src.logging_utils import model_logger as logger


class ItemCFRecommender:
    """
    Item-item cosine similarity collaborative filtering.

    For each item, stores the top K most similar items.
    At inference time, scores unrated items using the user's rating history
    as a weighted combination of similar items.
    """

    def __init__(self, n_similar: int = CF_N_SIMILAR_ITEMS, min_support: int = CF_MIN_SUPPORT, k: int = None):
        # k is an alias for n_similar (for test compatibility)
        self.n_similar = k if k is not None else n_similar
        self.min_support = min_support

        # Fitted state
        self.item_similarity: Optional[np.ndarray] = None
        self.item_index: Dict[int, int] = {}     # movie_id → matrix col index
        self.index_item: Dict[int, int] = {}     # matrix col index → movie_id
        self.user_index: Dict[int, int] = {}     # user_id → matrix row index
        self.item_means: Optional[np.ndarray] = None
        self.user_item_matrix: Optional[sp.csr_matrix] = None
        self.is_fitted = False

    def fit(self, train_df: pd.DataFrame) -> "ItemCFRecommender":
        """
        Build item-item similarity matrix.

        Process:
        1. Build user-item rating matrix (mean-centered per item)
        2. Compute cosine similarity between all item pairs
        3. Keep only top-K similar items per item (memory efficiency)

        Args:
            train_df: DataFrame with [user_id, movie_id, rating]
        """
        logger.start_phase("Phase 5 - ItemCF Fit", "Build item-item cosine similarity matrix")
        start = time.time()

        # Build index maps
        unique_users = sorted(train_df["user_id"].unique())
        unique_movies = sorted(train_df["movie_id"].unique())
        self.user_index = {uid: i for i, uid in enumerate(unique_users)}
        self.item_index = {mid: i for i, mid in enumerate(unique_movies)}
        self.index_item = {i: mid for mid, i in self.item_index.items()}

        n_users = len(unique_users)
        n_items = len(unique_movies)

        # Build sparse user-item matrix
        rows = train_df["user_id"].map(self.user_index).values
        cols = train_df["movie_id"].map(self.item_index).values
        data = train_df["rating"].values.astype(np.float32)

        self.user_item_matrix = sp.csr_matrix(
            (data, (rows, cols)),
            shape=(n_users, n_items),
            dtype=np.float32,
        )

        logger.info(f"User-item matrix: {n_users} users × {n_items} items")

        # Mean-center items (subtract item mean for cosine similarity)
        # Convert to item-user matrix for pairwise item similarity
        item_user_matrix = self.user_item_matrix.T  # (n_items, n_users)

        # Compute item means for mean-centering
        item_user_dense = item_user_matrix.toarray()
        # Only consider rated entries (non-zero) for mean
        rated_mask = item_user_dense != 0
        with np.errstate(divide="ignore", invalid="ignore"):
            self.item_means = np.where(
                rated_mask.sum(axis=1) > 0,
                item_user_dense.sum(axis=1) / np.maximum(rated_mask.sum(axis=1), 1),
                0.0,
            )

        # Mean-center: subtract item mean from rated entries
        centered = item_user_dense.copy()
        centered[rated_mask] -= self.item_means[
            np.where(rated_mask)[0]
        ]

        # Apply minimum support filter — items with < min_support ratings
        # have their row zeroed to avoid noisy similarities
        support = rated_mask.sum(axis=1)
        low_support = support < self.min_support
        centered[low_support] = 0.0

        logger.info(f"Computing {n_items}×{n_items} cosine similarity matrix...")
        # Compute cosine similarity (batched to manage memory)
        # For MovieLens 1M: ~3700 items → 3700×3700 matrix is manageable (~110MB float32)
        sim_matrix = cosine_similarity(sp.csr_matrix(centered), dense_output=True)

        # Set diagonal to 0 (item not similar to itself for recommendation purposes)
        np.fill_diagonal(sim_matrix, 0.0)

        # Zero out similarities from low-support items
        sim_matrix[low_support] = 0.0
        sim_matrix[:, low_support] = 0.0

        self.item_similarity = sim_matrix.astype(np.float32)

        elapsed = time.time() - start
        logger.log_metric("item_cf_n_items", n_items, "Phase 5")
        logger.log_metric("item_cf_matrix_mb", round(sim_matrix.nbytes / 1e6, 2), "similarity matrix size")
        logger.log_runtime("item_cf_fit", elapsed, "Phase 5")
        self.is_fitted = True

        logger.end_phase(
            "Phase 5 - ItemCF Fit",
            f"Similarity matrix built: {n_items}×{n_items}, "
            f"n_similar_kept={self.n_similar}, elapsed={elapsed:.1f}s",
            "Phase 5 - Evaluate",
        )
        return self

    def _get_user_seen_items(self, user_id: int) -> List[int]:
        """Get list of movie_ids rated by user in training set."""
        if user_id not in self.user_index:
            return []
        u_idx = self.user_index[user_id]
        row = self.user_item_matrix.getrow(u_idx)
        cols = row.nonzero()[1]
        return [self.index_item[c] for c in cols]

    def _get_user_ratings(self, user_id: int) -> Dict[int, float]:
        """Get {movie_id: rating} dict for a user."""
        if user_id not in self.user_index:
            return {}
        u_idx = self.user_index[user_id]
        row = self.user_item_matrix.getrow(u_idx)
        cols = row.nonzero()[1]
        vals = row.data
        return {self.index_item[c]: float(v) for c, v in zip(cols, vals)}

    def recommend(
        self,
        user_id: int,
        n: int = TOP_N,
        seen_items: Optional[set] = None,
    ) -> List[Dict]:
        """
        Generate top-N item recommendations for a user.

        Scoring: For each unrated item j, score(j) = Σ(sim(i,j) * r_ui) / Σ|sim(i,j)|
        where i ∈ user's rated items.

        Args:
            user_id: User to recommend for
            n: Number of recommendations
            seen_items: Set of already-seen item IDs to exclude

        Returns:
            List of {movie_id, score} dicts sorted by score descending
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")

        user_ratings = self._get_user_ratings(user_id)
        if not user_ratings:
            return []  # cold start — caller should use popularity fallback

        exclude = set(seen_items or []) | set(user_ratings.keys())
        rated_indices = []
        ratings_array = []
        for mid, r in user_ratings.items():
            if mid in self.item_index:
                rated_indices.append(self.item_index[mid])
                ratings_array.append(r - self.item_means[self.item_index[mid]])

        if not rated_indices:
            return []

        rated_indices = np.array(rated_indices)
        ratings_array = np.array(ratings_array, dtype=np.float32)

        # Score all items as weighted sum of similarities to rated items
        sim_to_rated = self.item_similarity[:, rated_indices]  # (n_items, n_rated)
        numerator = sim_to_rated.dot(ratings_array)
        denominator = np.abs(sim_to_rated).sum(axis=1) + 1e-8
        scores = numerator / denominator

        # Build recommendation list
        item_scores = []
        for i, score in enumerate(scores):
            mid = self.index_item[i]
            if mid not in exclude:
                item_scores.append((mid, float(score)))

        item_scores.sort(key=lambda x: x[1], reverse=True)
        return [{"movie_id": mid, "score": score} for mid, score in item_scores[:n]]

    def similar_items(self, movie_id: int = 0, n: int = 10, item_id: int = None) -> List[int]:
        """Return the top-N most similar items to the given movie."""
        # Support item_id as alias for movie_id
        effective_id = item_id if item_id is not None else movie_id
        if not self.is_fitted or effective_id not in self.item_index:
            return []
        idx = self.item_index[effective_id]
        sims = self.item_similarity[idx].copy()
        sims[idx] = -1  # exclude self
        top_indices = np.argpartition(sims, -n)[-n:]
        top_indices = top_indices[np.argsort(sims[top_indices])[::-1]]
        return [self.index_item[i] for i in top_indices]

    def save(self, path: Optional[Path] = None) -> Path:
        """Save model artifact."""
        save_path = path or (MODELS_DIR / "item_cf_model.pkl")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(save_path), "wb") as f:
            pickle.dump(self, f)
        logger.log_artifact(str(save_path), "Item-item CF model (pickle)")
        return save_path

    @classmethod
    def load(cls, path: Path) -> "ItemCFRecommender":
        """Load a previously saved model."""
        with open(str(path), "rb") as f:
            model = pickle.load(f)
        logger.info(f"Loaded item-CF model from {path}")
        return model


def train_item_cf() -> ItemCFRecommender:
    """Full training routine for item-item CF."""
    logger.start_phase("Phase 5 - Item-Item CF", "Train item-item CF model on MovieLens 1M")

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Training data not found: {TRAIN_FILE}")

    train_df = pd.read_parquet(str(TRAIN_FILE))
    logger.info(f"Loaded training set: {len(train_df):,} ratings")

    model = ItemCFRecommender(n_similar=CF_N_SIMILAR_ITEMS)
    model.fit(train_df)
    save_path = model.save()

    logger.end_phase("Phase 5 - Item-Item CF", f"Model saved to {save_path}", "Phase 5 - Evaluate")
    return model


if __name__ == "__main__":
    model = train_item_cf()
    sys.exit(0)
