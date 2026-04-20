"""
benchmark.py — Offline evaluation benchmark runner.

Evaluates all trained models on the test set and produces comparison reports.
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import (
    K_VALUES,
    MODELS_DIR,
    PRIMARY_K,
    TARGET_HR_AT_10,
    TARGET_PRECISION_AT_10,
    TARGET_RECALL_AT_10,
    TEST_FILE,
    TRAIN_FILE,
)
from src.evaluation.metrics import compute_ranking_metrics, compute_rating_metrics
from src.logging_utils import eval_logger as logger


def build_ground_truth(test_df: pd.DataFrame) -> Dict[int, List[int]]:
    """
    Build ground truth dict from test set.

    Returns:
        {user_id: [list of movie_ids in test set for this user]}
    """
    gt = {}
    for user_id, group in test_df.groupby("user_id"):
        gt[int(user_id)] = group["movie_id"].tolist()
    return gt


def build_user_seen_items(train_df: pd.DataFrame) -> Dict[int, List[int]]:
    """Build dict of items seen by each user in training."""
    seen = {}
    for user_id, group in train_df.groupby("user_id"):
        seen[int(user_id)] = group["movie_id"].tolist()
    return seen


def evaluate_model(
    model,
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    model_name: str,
    k: int = PRIMARY_K,
    n_users: Optional[int] = None,
) -> Dict:
    """
    Evaluate a recommender model on the test set.

    Args:
        model: Fitted recommender model with .recommend(user_id, n=k) method
        test_df: Test set DataFrame
        train_df: Training set (for seen-item filtering)
        model_name: Label for logging
        k: Evaluation cutoff
        n_users: If set, only evaluate on first N users (for speed)

    Returns:
        Dict of metrics
    """
    logger.start_phase(f"Evaluate {model_name}", f"Offline evaluation of {model_name} @ k={k}")
    start = time.time()

    ground_truth = build_ground_truth(test_df)
    user_seen = build_user_seen_items(train_df)

    eval_users = list(ground_truth.keys())
    if n_users is not None:
        eval_users = eval_users[:n_users]

    recommendations = {}
    failed_users = []

    for user_id in eval_users:
        try:
            seen = set(user_seen.get(user_id, []))
            raw_recs = model.recommend(user_id, n=k, seen_items=seen)
            # Handle both list[dict] and list[int] return formats
            recs = [r["movie_id"] if isinstance(r, dict) else r for r in raw_recs]
            recommendations[user_id] = recs
        except Exception as e:
            failed_users.append(user_id)
            logger.warning(f"Recommendation failed for user {user_id}: {e}")

    if failed_users:
        logger.warning(f"{len(failed_users)} users failed recommendation generation")

    # Compute ranking metrics
    metrics = {}
    for k_val in K_VALUES:
        m = compute_ranking_metrics(recommendations, ground_truth, k=k_val)
        metrics.update(m)

    # Rating RMSE (if model has predict_rating)
    if hasattr(model, "predict_rating"):
        sample = test_df.sample(min(5000, len(test_df)), random_state=42)
        y_true = []
        y_pred = []
        for _, row in sample.iterrows():
            try:
                pred = model.predict_rating(int(row["user_id"]), int(row["movie_id"]))
                y_true.append(float(row["rating"]))
                y_pred.append(pred)
            except Exception:
                pass
        if y_true:
            rating_metrics = compute_rating_metrics(y_true, y_pred)
            metrics.update(rating_metrics)

    elapsed = time.time() - start
    metrics["eval_time_s"] = round(elapsed, 2)
    metrics["model_name"] = model_name
    metrics["n_users_evaluated"] = len(recommendations)

    # Check against success thresholds
    hr10 = metrics.get(f"hr@{PRIMARY_K}", 0.0)
    p10 = metrics.get(f"precision@{PRIMARY_K}", 0.0)
    r10 = metrics.get(f"recall@{PRIMARY_K}", 0.0)

    threshold_status = {
        "hr@10_met": hr10 >= TARGET_HR_AT_10,
        "precision@10_met": p10 >= TARGET_PRECISION_AT_10,
        "recall@10_met": r10 >= TARGET_RECALL_AT_10,
    }
    metrics["threshold_status"] = threshold_status

    if not threshold_status["hr@10_met"]:
        logger.warning(
            f"{model_name}: HR@10={hr10:.4f} below target {TARGET_HR_AT_10}. "
            f"Gap: {TARGET_HR_AT_10 - hr10:.4f}"
        )
    if not threshold_status["precision@10_met"]:
        logger.warning(f"{model_name}: Precision@10={p10:.4f} below target {TARGET_PRECISION_AT_10}")

    logger.log_metrics(
        {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
        model_name
    )
    logger.end_phase(
        f"Evaluate {model_name}",
        f"HR@10={hr10:.4f} | P@10={p10:.4f} | R@10={r10:.4f} | elapsed={elapsed:.1f}s",
        "Next model evaluation",
    )
    return metrics


def run_full_benchmark() -> Dict[str, Dict]:
    """
    Run evaluation on all available models and return comparison.
    """
    logger.start_phase("Full Benchmark", "Evaluate all models and compare performance")

    if not TEST_FILE.exists() or not TRAIN_FILE.exists():
        raise FileNotFoundError("Train/test splits not found. Run data pipeline first.")

    test_df = pd.read_parquet(str(TEST_FILE))
    train_df = pd.read_parquet(str(TRAIN_FILE))
    logger.info(f"Test set: {len(test_df):,} ratings, {test_df['user_id'].nunique():,} users")

    results = {}

    # Load and evaluate each model
    model_files = {
        "popularity": MODELS_DIR / "popularity_model.pkl",
        "item_cf": MODELS_DIR / "item_cf_model.pkl",
        "svd": MODELS_DIR / "svd_model.pkl",
        "svdpp": MODELS_DIR / "svdpp_model.pkl",
        "hybrid": MODELS_DIR / "hybrid_model.pkl",
    }

    import pickle
    for model_name, model_path in model_files.items():
        if not model_path.exists():
            logger.warning(f"Model not found, skipping: {model_path}")
            continue
        try:
            with open(str(model_path), "rb") as f:
                model = pickle.load(f)
            logger.info(f"Evaluating: {model_name}")
            metrics = evaluate_model(model, test_df, train_df, model_name)
            results[model_name] = metrics
        except Exception as e:
            logger.log_error(e, f"Failed to evaluate {model_name}")
            results[model_name] = {"error": str(e)}

    # Save full results
    results_path = MODELS_DIR / "benchmark_results.json"
    results_path.write_text(json.dumps(results, indent=2, default=str))
    logger.log_artifact(str(results_path), "Full benchmark results")
    logger.end_phase("Full Benchmark", f"Evaluated {len(results)} models", "Generate report")

    return results
