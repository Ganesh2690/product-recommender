"""
registry.py — Model Registry for the Recommender System.

Manages artifact versioning, loading the active model, and providing
a unified RecommendationEngine abstraction with fallback logic.

Phase 8 — Recommendation Service
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import (
    COLD_START_THRESHOLD,
    MAX_MODEL_VERSIONS,
    MODEL_HISTORY_FILE,
    MODEL_VERSION_FILE,
    MODELS_DIR,
    TOP_N,
    TRAIN_FILE,
)
from src.logging_utils import get_logger

logger = get_logger("model_registry", "master")


class ModelRegistry:
    """
    Manages model artifacts with timestamp-based versioning.

    Versioning scheme: model_{name}_{YYYYMMDD}_{version}.pkl
    Current active model pointer: data/artifacts/current_model_version.txt
    History: data/artifacts/model_history.jsonl
    """

    def __init__(self, registry_path: Optional[str] = None):
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        # registry_path allows tests to use a custom JSON file
        self._registry_path = Path(registry_path) if registry_path else MODEL_HISTORY_FILE

    def register_model(
        self,
        model_obj,
        model_name: str,
        metrics: Optional[Dict] = None,
        promoted: bool = False,
    ) -> Path:
        """
        Save a model artifact with a versioned filename.

        Args:
            model_obj: Any model object with a .save(path) method
            model_name: Short name (e.g., 'svd', 'svdpp', 'hybrid')
            metrics: Dict of evaluation metrics
            promoted: Whether this model is being promoted to production

        Returns:
            Path to saved artifact
        """
        import pickle

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        artifact_path = MODELS_DIR / f"{model_name}_{timestamp}.pkl"

        # Save artifact
        with open(str(artifact_path), "wb") as f:
            pickle.dump(model_obj, f)

        # Update history
        entry = {
            "model_name": model_name,
            "artifact_path": str(artifact_path),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics or {},
            "promoted": promoted,
        }
        with open(str(self._registry_path), "a") as f:
            f.write(json.dumps(entry) + "\n")

        if promoted:
            MODEL_VERSION_FILE.write_text(str(artifact_path))
            logger.info(f"Model promoted to production: {artifact_path}")

        logger.log_artifact(str(artifact_path), f"Registered model: {model_name}")
        logger.log_metrics(metrics or {}, f"Registered {model_name}")

        # Prune old versions (keep MAX_MODEL_VERSIONS per model name)
        self._prune_old_versions(model_name)
        return artifact_path

    def get_current_model_path(self) -> Optional[Path]:
        """Return the path of the currently active (promoted) model."""
        if MODEL_VERSION_FILE.exists():
            path = Path(MODEL_VERSION_FILE.read_text().strip())
            if path.exists():
                return path
        return None

    def load_model(self, model_name: Optional[str] = None) -> object:
        """
        Load the active model.

        Args:
            model_name: If specified, load the latest version of this model.
                        If None, load the currently promoted model.
        """
        import pickle

        if model_name:
            candidates = sorted(MODELS_DIR.glob(f"{model_name}_*.pkl"), reverse=True)
            if not candidates:
                raise FileNotFoundError(f"No saved model found for: {model_name}")
            path = candidates[0]
        else:
            path = self.get_current_model_path()
            if path is None:
                raise FileNotFoundError("No production model registered. Run training pipeline first.")

        with open(str(path), "rb") as f:
            model = pickle.load(f)
        logger.info(f"Loaded model from {path}")
        return model

    def list_models(self) -> List[Dict]:
        """Return list of all registered model entries from history."""
        history_file = self._registry_path
        if not history_file.exists():
            return []
        entries = []
        with open(str(history_file)) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return entries

    def register(
        self,
        model_name: str,
        model_path: str,
        metrics: Optional[Dict] = None,
    ) -> None:
        """Lightweight alias used by tests: record model entry without saving artifact."""
        entry = {
            "model_name": model_name,
            "artifact_path": str(model_path),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics or {},
            "promoted": False,
        }
        history_file = self._registry_path
        history_file.parent.mkdir(parents=True, exist_ok=True)
        with open(str(history_file), "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _prune_old_versions(self, model_name: str) -> None:
        """Remove old artifact files for a model, keeping the newest MAX_MODEL_VERSIONS."""
        versions = sorted(MODELS_DIR.glob(f"{model_name}_*.pkl"), reverse=True)
        for old in versions[MAX_MODEL_VERSIONS:]:
            old.unlink(missing_ok=True)
            logger.info(f"Pruned old model artifact: {old.name}")


class RecommendationEngine:
    """
    Unified recommendation engine with fallback logic.

    Supports:
    - Personalized recommendations via the active model
    - Cold-start fallback via popularity model
    - Missing user/item handling
    - Seen-item filtering
    - Similar-items lookup
    """

    def __init__(self, registry: Optional[ModelRegistry] = None):
        self.registry = registry or ModelRegistry()
        self._model = None
        self._popularity_model = None
        self._is_loaded = False

        # Cache of user rated items for filtering
        self._user_seen_items: Dict[int, List[int]] = {}
        self._all_items: List[int] = []

    def load(self) -> "RecommendationEngine":
        """Load the active production model and popularity fallback."""
        try:
            self._model = self.registry.load_model()
            logger.info("Primary model loaded successfully")
        except Exception as e:
            logger.log_error(e, "Failed to load primary model")
            self._model = None

        # Load popularity fallback
        try:
            from src.models.popularity import PopularityRecommender
            pop_path = MODELS_DIR / "popularity_model.pkl"
            if pop_path.exists():
                self._popularity_model = PopularityRecommender.load(pop_path)
                logger.info("Popularity fallback model loaded")
            else:
                logger.warning("Popularity model not found. Cold-start fallback unavailable.")
        except Exception as e:
            logger.log_error(e, "Failed to load popularity model")

        # Load user seen items from training data for filtering
        if TRAIN_FILE.exists():
            try:
                import pandas as pd
                train = pd.read_parquet(str(TRAIN_FILE))
                for _, group in train.groupby("user_id"):
                    uid = int(group["user_id"].iloc[0])
                    self._user_seen_items[uid] = group["movie_id"].tolist()
                self._all_items = list(set(train["movie_id"].tolist()))
            except Exception as e:
                logger.log_error(e, "Failed to load user-seen items from training data")

        self._is_loaded = True
        return self

    def get_user_seen_items(self, user_id: int) -> List[int]:
        """Return list of items the user has already rated/consumed."""
        if hasattr(self._model, "user_rated_items"):
            return self._model.user_rated_items.get(user_id, [])
        return self._user_seen_items.get(user_id, [])

    def is_cold_start_user(self, user_id: int) -> bool:
        """Check if user has insufficient interaction history for personalization."""
        return len(self.get_user_seen_items(user_id)) < COLD_START_THRESHOLD

    def recommend(
        self,
        user_id: int,
        n: int = TOP_N,
        exclude_seen: bool = True,
    ) -> Tuple[List[Dict], str]:
        """
        Get top-N recommendations for a user.

        Args:
            user_id: User ID
            n: Number of recommendations
            exclude_seen: Whether to exclude already-seen items

        Returns:
            (recommendations, source) where source is 'personalized', 'cold_start', or 'fallback'
        """
        if not self._is_loaded:
            self.load()

        seen_set = set(self.get_user_seen_items(user_id)) if exclude_seen else set()

        # Cold-start fallback
        if self._model is None or self.is_cold_start_user(user_id):
            source = "cold_start"
            if self._popularity_model is not None:
                recs = self._popularity_model.recommend(user_id, n=n, seen_items=seen_set)
            else:
                recs = [{"movie_id": mid, "score": 0.0} for mid in self._all_items[:n]]
            logger.debug(f"Cold-start fallback for user {user_id}: {len(recs)} recs")
            return recs, source

        try:
            start = time.time()
            recs = self._model.recommend(user_id, n=n, seen_items=seen_set)
            elapsed_ms = (time.time() - start) * 1000
            logger.log_metric("recommend_latency_ms", round(elapsed_ms, 2), f"user={user_id}")

            if not recs and self._popularity_model is not None:
                recs = self._popularity_model.recommend(user_id, n=n, seen_items=seen_set)
                return recs, "fallback"

            return recs, "personalized"

        except Exception as e:
            logger.log_error(e, f"Recommendation failed for user {user_id}")
            if self._popularity_model is not None:
                return self._popularity_model.recommend(user_id, n=n, seen_items=seen_set), "fallback"
            return [{"movie_id": mid, "score": 0.0} for mid in self._all_items[:n]], "fallback"

    def similar_items(self, movie_id: int, n: int = 10) -> Tuple[List[int], str]:
        """
        Get similar items for a given movie.

        Returns:
            (similar_items, source)
        """
        if not self._is_loaded:
            self.load()

        if self._model is None:
            return [], "unavailable"

        if not hasattr(self._model, "similar_items"):
            return [], "not_supported"

        try:
            items = self._model.similar_items(movie_id, n=n)
            return items, "model"
        except Exception as e:
            logger.log_error(e, f"similar_items failed for movie {movie_id}")
            return [], "error"


# Module-level singleton for use by the API
_engine: Optional[RecommendationEngine] = None


def get_engine(force_reload: bool = False) -> RecommendationEngine:
    """Get or create the global RecommendationEngine singleton."""
    global _engine
    if _engine is None or force_reload:
        _engine = RecommendationEngine()
        _engine.load()
    return _engine
