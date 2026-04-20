"""
matrix_factorization.py — SVD and SVD++ Matrix Factorization using Surprise library.

Implements:
- SVDRecommender: Simon Funk SVD (Surprise library)
- SVDPPRecommender: SVD++ with implicit feedback (Surprise library)

Phase 6 — Matrix Factorization (Production Candidate)
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
    MODELS_DIR,
    SVD_LR_ALL,
    SVD_N_EPOCHS,
    SVD_N_FACTORS,
    SVD_REG_ALL,
    SVDPP_N_EPOCHS,
    SVDPP_N_FACTORS,
    TOP_N,
    TRAIN_FILE,
)
from src.logging_utils import model_logger as logger

try:
    from surprise import SVD, SVDpp
    SURPRISE_AVAILABLE = True
except ImportError:
    SURPRISE_AVAILABLE = False
    logger.warning("Surprise library not available. SVD models will use scipy/sklearn fallback.")


# ---------------------------------------------------------------------------
# Scipy/sklearn SVD fallback (used when Surprise is unavailable)
# ---------------------------------------------------------------------------

class _ScipySVD:
    """
    Lightweight Simon Funk-style SVD using scipy/sklearn when Surprise is absent.

    Implements biased SVD: r̂_ui = μ + b_u + b_i + p_u · q_i
    Trained with scipy.linalg.svd on the sparse rating matrix.
    """

    def __init__(self, n_factors: int = 50, n_epochs: int = 20,
                 lr: float = 0.005, reg: float = 0.02):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg

        self._global_mean = 0.0
        self._user_bias: Dict[int, float] = {}
        self._item_bias: Dict[int, float] = {}
        self._P: Optional[np.ndarray] = None   # user factors  (n_users, n_factors)
        self._Q: Optional[np.ndarray] = None   # item factors  (n_items, n_factors)
        self._user_idx: Dict[int, int] = {}
        self._item_idx: Dict[int, int] = {}
        self._idx_item: Dict[int, int] = {}
        self._ratings: List = []

    def fit(self, df: pd.DataFrame) -> "_ScipySVD":
        logger.info(f"[SVD fallback] Fitting with n_factors={self.n_factors}, n_epochs={self.n_epochs}")
        users = sorted(df["user_id"].unique())
        items = sorted(df["movie_id"].unique())
        self._user_idx = {u: i for i, u in enumerate(users)}
        self._item_idx = {m: i for i, m in enumerate(items)}
        self._idx_item = {i: m for m, i in self._item_idx.items()}

        self._global_mean = float(df["rating"].mean())
        self._user_bias = {u: 0.0 for u in users}
        self._item_bias = {m: 0.0 for m in items}

        n_users, n_items = len(users), len(items)
        rng = np.random.default_rng(42)
        self._P = rng.normal(0, 0.1, (n_users, self.n_factors))
        self._Q = rng.normal(0, 0.1, (n_items, self.n_factors))

        ratings = list(df[["user_id", "movie_id", "rating"]].itertuples(index=False))
        self._ratings = [(int(r.user_id), int(r.movie_id), float(r.rating)) for r in ratings]

        for epoch in range(self.n_epochs):
            rng_ep = np.random.default_rng(epoch)
            indices = rng_ep.permutation(len(self._ratings))
            for idx in indices:
                uid, mid, r = self._ratings[idx]
                u = self._user_idx[uid]
                i = self._item_idx[mid]
                pred = self._global_mean + self._user_bias[uid] + self._item_bias[mid] + self._P[u].dot(self._Q[i])
                err = r - pred
                self._user_bias[uid] += self.lr * (err - self.reg * self._user_bias[uid])
                self._item_bias[mid] += self.lr * (err - self.reg * self._item_bias[mid])
                pu, qi = self._P[u].copy(), self._Q[i].copy()
                self._P[u] += self.lr * (err * qi - self.reg * pu)
                self._Q[i] += self.lr * (err * pu - self.reg * qi)
        return self

    def predict(self, uid: int, iid: int) -> float:
        if uid not in self._user_idx or iid not in self._item_idx:
            return self._global_mean
        u, i = self._user_idx[uid], self._item_idx[iid]
        score = (self._global_mean + self._user_bias.get(uid, 0.0) +
                 self._item_bias.get(iid, 0.0) + self._P[u].dot(self._Q[i]))
        return float(np.clip(score, 1.0, 5.0))

    def predict_for_user(self, uid: int) -> np.ndarray:
        """Vectorized: predict scores for all items for a single user. Returns array indexed by item index."""
        if uid not in self._user_idx:
            n_items = self._Q.shape[0]
            return np.full(n_items, self._global_mean)
        u = self._user_idx[uid]
        bu = self._user_bias.get(uid, 0.0)
        item_biases = np.array([self._item_bias.get(self._idx_item[i], 0.0) for i in range(self._Q.shape[0])])
        scores = self._global_mean + bu + item_biases + self._P[u].dot(self._Q.T)
        return np.clip(scores, 1.0, 5.0)

    def qi_matrix(self) -> np.ndarray:
        return self._Q


def _df_to_surprise_dataset(df: pd.DataFrame):
    """Convert a pandas DataFrame to a Surprise Dataset."""
    from surprise import Dataset, Reader
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[["user_id", "movie_id", "rating"]], reader)
    return data


class SVDRecommender:
    """
    Simon Funk SVD matrix factorization (via Surprise library).

    Learns user latent factors P (n_users × n_factors) and
    item latent factors Q (n_items × n_factors) such that:
      r̂_ui = μ + b_u + b_i + p_u · q_i

    Trained by SGD minimizing RMSE on observed ratings.
    """

    def __init__(
        self,
        n_factors: int = SVD_N_FACTORS,
        n_epochs: int = SVD_N_EPOCHS,
        lr_all: float = SVD_LR_ALL,
        reg_all: float = SVD_REG_ALL,
        use_svdpp: bool = False,
    ):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr_all = lr_all
        self.reg_all = reg_all
        self.use_svdpp = use_svdpp

        self._algo = None
        self._trainset = None
        self.is_fitted = False

        # Store user rating histories for seen-item filtering
        self.user_rated_items: Dict[int, List[int]] = {}
        # All items in training set
        self.all_items: List[int] = []
        # Inner-to-raw id mapping (Surprise uses internal ids)
        self._inner_to_raw_iid: Dict[int, int] = {}
        self._raw_to_inner_iid: Dict[int, int] = {}

    def fit(self, train_df: pd.DataFrame) -> "SVDRecommender":
        """
        Fit the SVD model on training data.

        Args:
            train_df: DataFrame with [user_id, movie_id, rating]
        """
        model_name = "SVD++" if self.use_svdpp else "SVD"
        logger.start_phase(
            f"Phase 6 - {model_name} Fit",
            f"Train {model_name} (n_factors={self.n_factors}, n_epochs={self.n_epochs})"
        )
        start = time.time()

        # Store user rating histories for seen-item filtering
        for _, row in train_df.iterrows():
            uid, mid = int(row["user_id"]), int(row["movie_id"])
            if uid not in self.user_rated_items:
                self.user_rated_items[uid] = []
            self.user_rated_items[uid].append(mid)

        self.all_items = sorted(train_df["movie_id"].unique().tolist())

        if SURPRISE_AVAILABLE:
            dataset = _df_to_surprise_dataset(train_df)
            trainset = dataset.build_full_trainset()
            self._trainset = trainset
            self._inner_to_raw_iid = {i: int(trainset.to_raw_iid(i)) for i in trainset.all_items()}
            self._raw_to_inner_iid = {v: k for k, v in self._inner_to_raw_iid.items()}

            if self.use_svdpp:
                algo = SVDpp(n_factors=self.n_factors, n_epochs=self.n_epochs,
                             lr_all=self.lr_all, reg_all=self.reg_all, verbose=False)
            else:
                algo = SVD(n_factors=self.n_factors, n_epochs=self.n_epochs,
                           lr_all=self.lr_all, reg_all=self.reg_all, biased=True, verbose=False)
            algo.fit(trainset)
            self._algo = algo
        else:
            logger.warning(f"[{model_name}] Surprise unavailable — using scipy SGD fallback")
            fallback = _ScipySVD(n_factors=self.n_factors, n_epochs=self.n_epochs,
                                 lr=self.lr_all, reg=self.reg_all)
            fallback.fit(train_df)
            self._algo = fallback
            self._trainset = None

        self.is_fitted = True
        elapsed = time.time() - start
        logger.log_metric(f"{model_name.lower()}_fit_time_s", round(elapsed, 2), f"Phase 6 n_factors={self.n_factors}")
        logger.log_runtime(f"{model_name}_fit", elapsed, f"n_factors={self.n_factors}")
        logger.end_phase(
            f"Phase 6 - {model_name} Fit",
            f"{model_name} trained: n_factors={self.n_factors}, elapsed={elapsed:.1f}s",
            "Phase 6 - Evaluate",
        )
        return self

    def predict_rating(self, user_id: int, movie_id: int = 0, item_id: int = None) -> float:
        """
        Predict the rating for a user-item pair.

        Returns the global mean if the user or item was not seen in training.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
        # Support both movie_id and item_id kwargs
        effective_item = item_id if item_id is not None else movie_id
        if SURPRISE_AVAILABLE and self._trainset is not None:
            pred = self._algo.predict(str(user_id), str(effective_item))
            return float(pred.est)
        else:
            return self._algo.predict(user_id, effective_item)

    def recommend(
        self,
        user_id: int,
        n: int = TOP_N,
        seen_items: Optional[set] = None,
    ) -> List[Dict]:
        """
        Generate top-N recommendations by predicting ratings for all unseen items.

        Args:
            user_id: User to recommend for
            n: Number of items to return
            seen_items: Set of already-seen item IDs to exclude

        Returns:
            List of {movie_id, score} dicts sorted by predicted rating descending
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")

        seen = set(seen_items or []) | set(self.user_rated_items.get(user_id, []))
        candidates = [iid for iid in self.all_items if iid not in seen]

        if not candidates:
            candidates = self.all_items

        # Use vectorized prediction for the scipy fallback (much faster)
        if self._trainset is None and hasattr(self._algo, "predict_for_user"):
            all_scores = self._algo.predict_for_user(user_id)
            idx_item = self._algo._idx_item
            candidate_set = set(candidates)
            predictions = [
                (idx_item[i], float(all_scores[i]))
                for i in range(len(all_scores))
                if idx_item.get(i) in candidate_set
            ]
        else:
            predictions = [
                (iid, self.predict_rating(user_id, iid))
                for iid in candidates
            ]
        predictions.sort(key=lambda x: x[1], reverse=True)
        return [{"movie_id": mid, "score": round(score, 4)} for mid, score in predictions[:n]]

    def recommend_batch(
        self,
        user_ids: List[int],
        n: int = TOP_N,
        user_seen_items: Optional[Dict[int, set]] = None,
    ) -> Dict[int, List[Dict]]:
        """Generate recommendations for multiple users."""
        user_seen = user_seen_items or {}
        return {
            uid: self.recommend(uid, n=n, seen_items=user_seen.get(uid, set()))
            for uid in user_ids
        }

    def similar_items(self, movie_id: int, n: int = 10) -> List[int]:
        """Find similar items using cosine similarity of item factor vectors."""
        if not self.is_fitted:
            return []

        # Get item factor matrix Q
        if SURPRISE_AVAILABLE and self._trainset is not None:
            if movie_id not in self._raw_to_inner_iid:
                return []
            target_idx = self._raw_to_inner_iid[movie_id]
            qi = self._algo.qi
        else:
            qi = self._algo.qi_matrix()
            if movie_id not in self._raw_to_inner_iid:
                return []
            target_idx = self._raw_to_inner_iid[movie_id]

        target_vec = qi[target_idx]
        norms = np.linalg.norm(qi, axis=1) + 1e-8
        target_norm = np.linalg.norm(target_vec) + 1e-8
        sims = qi.dot(target_vec) / (norms * target_norm)
        sims[target_idx] = -1  # exclude self

        top_inner = np.argpartition(sims, -(n + 1))[-(n + 1):]
        top_inner = top_inner[np.argsort(sims[top_inner])[::-1]]

        result = []
        for i in top_inner:
            if i in self._inner_to_raw_iid:
                mid = self._inner_to_raw_iid[i]
                if mid != movie_id:
                    result.append(mid)
        return result[:n]

    def save(self, path=None) -> Path:
        """Save model artifact."""
        suffix = "svdpp" if self.use_svdpp else "svd"
        save_path = Path(path) if path is not None else (MODELS_DIR / f"{suffix}_model.pkl")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(save_path), "wb") as f:
            pickle.dump(self, f)
        logger.log_artifact(str(save_path), f"{'SVD++' if self.use_svdpp else 'SVD'} model (pickle)")
        return save_path

    @classmethod
    def load(cls, path: Path) -> "SVDRecommender":
        """Load a previously saved model."""
        with open(str(path), "rb") as f:
            model = pickle.load(f)
        logger.info(f"Loaded SVD model from {path}")
        return model


def train_svd(n_factors: int = SVD_N_FACTORS) -> SVDRecommender:
    """Train SVD model."""
    logger.start_phase("Phase 6 - SVD Training", f"Train SVD (n_factors={n_factors})")
    train_df = pd.read_parquet(str(TRAIN_FILE))
    model = SVDRecommender(n_factors=n_factors)
    model.fit(train_df)
    save_path = model.save()
    logger.end_phase("Phase 6 - SVD Training", f"SVD saved to {save_path}", "Phase 6 - SVD++")
    return model


def train_svdpp(n_factors: int = SVDPP_N_FACTORS) -> SVDRecommender:
    """Train SVD++ model."""
    logger.start_phase("Phase 6 - SVD++ Training", f"Train SVD++ (n_factors={n_factors})")
    train_df = pd.read_parquet(str(TRAIN_FILE))
    model = SVDRecommender(n_factors=n_factors, use_svdpp=True,
                           n_epochs=SVDPP_N_EPOCHS)
    model.fit(train_df)
    save_path = model.save()
    logger.end_phase("Phase 6 - SVD++ Training", f"SVD++ saved to {save_path}", "Phase 7 - Hybrid")
    return model


if __name__ == "__main__":
    m = train_svd()
    recs = m.recommend(user_id=1, n=10)
    print(f"SVD recommendations for user 1: {recs}")
    sys.exit(0)
