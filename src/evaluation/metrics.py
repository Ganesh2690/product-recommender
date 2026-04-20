"""
metrics.py — Evaluation metrics for recommender systems.

Implements:
- Precision@K
- Recall@K
- Hit Rate@K (HR@K)
- NDCG@K
- MAP@K
- RMSE / MAE (for explicit rating prediction)

All metrics follow the standard definitions used in CF research papers (2006–2018).
"""

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.logging_utils import eval_logger as logger


def precision_at_k(recommended: List[int], relevant: List[int], k: int) -> float:
    """
    Precision@K = |recommended[:K] ∩ relevant| / K

    Args:
        recommended: Ordered list of recommended item IDs
        relevant: Set of ground-truth relevant items (e.g., test set for this user)
        k: Cutoff

    Returns:
        Precision@K score in [0, 1]
    """
    if not recommended or not relevant:
        return 0.0
    recommended_k = recommended[:k]
    relevant_set = set(relevant)
    hits = sum(1 for item in recommended_k if item in relevant_set)
    return hits / k


def recall_at_k(recommended: List[int], relevant: List[int], k: int) -> float:
    """
    Recall@K = |recommended[:K] ∩ relevant| / |relevant|

    Args:
        recommended: Ordered list of recommended item IDs
        relevant: Ground-truth relevant items
        k: Cutoff

    Returns:
        Recall@K score in [0, 1]
    """
    if not recommended or not relevant:
        return 0.0
    recommended_k = recommended[:k]
    relevant_set = set(relevant)
    hits = sum(1 for item in recommended_k if item in relevant_set)
    return hits / len(relevant_set)


def hit_rate_at_k(recommended: List[int], relevant: List[int], k: int) -> float:
    """
    Hit Rate@K = 1 if at least one relevant item appears in recommended[:K], else 0.

    Also called Recall@K with binary relevance and |relevant|=1.
    Commonly used for implicit feedback evaluation.

    Args:
        recommended: Ordered list of recommended item IDs
        relevant: Ground-truth relevant items
        k: Cutoff

    Returns:
        1.0 if there is at least one hit in top-K, else 0.0
    """
    if not recommended or not relevant:
        return 0.0
    recommended_k = set(recommended[:k])
    relevant_set = set(relevant)
    return 1.0 if recommended_k & relevant_set else 0.0


def ndcg_at_k(recommended: List[int], relevant: List[int], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain @ K.

    DCG@K = Σ(i=1 to K) rel_i / log2(i+1)
    Ideal DCG = computed assuming all relevant items are ranked first
    NDCG@K = DCG@K / IDCG@K

    Args:
        recommended: Ordered list of recommended item IDs
        relevant: Ground-truth relevant items (assumed binary relevance)
        k: Cutoff

    Returns:
        NDCG@K score in [0, 1]
    """
    if not recommended or not relevant:
        return 0.0
    relevant_set = set(relevant)
    recommended_k = recommended[:k]

    # DCG
    dcg = 0.0
    for i, item in enumerate(recommended_k, start=1):
        if item in relevant_set:
            dcg += 1.0 / np.log2(i + 1)

    # Ideal DCG (all relevant items would be ranked at top)
    n_rel = min(len(relevant_set), k)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, n_rel + 1))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def average_precision_at_k(recommended: List[int], relevant: List[int], k: int) -> float:
    """
    Average Precision @ K.

    AP@K = (1/|relevant|) × Σ(k: item_k is relevant) Precision@k

    Args:
        recommended: Ordered list of recommended item IDs
        relevant: Ground-truth relevant items
        k: Cutoff

    Returns:
        AP@K score in [0, 1]
    """
    if not recommended or not relevant:
        return 0.0
    relevant_set = set(relevant)
    hits = 0
    sum_precisions = 0.0
    for i, item in enumerate(recommended[:k], start=1):
        if item in relevant_set:
            hits += 1
            sum_precisions += hits / i
    return sum_precisions / min(len(relevant_set), k)


def rmse(y_true: List[float], y_pred: List[float]) -> float:
    """Root Mean Squared Error for explicit rating prediction."""
    if not y_true or not y_pred:
        return float("nan")
    y_true_arr = np.array(y_true, dtype=float)
    y_pred_arr = np.array(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))


def mae(y_true: List[float], y_pred: List[float]) -> float:
    """Mean Absolute Error for explicit rating prediction."""
    if not y_true or not y_pred:
        return float("nan")
    y_true_arr = np.array(y_true, dtype=float)
    y_pred_arr = np.array(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true_arr - y_pred_arr)))


def compute_ranking_metrics(
    recommendations: Dict[int, List[int]],
    ground_truth: Dict[int, List[int]],
    k: int = 10,
) -> Dict[str, float]:
    """
    Compute all ranking metrics averaged over users.

    Args:
        recommendations: {user_id: [rec_item_id, ...]} — ordered recommendations
        ground_truth: {user_id: [relevant_item_id, ...]} — held-out test items
        k: Cutoff for all metrics

    Returns:
        Dict of metric_name → mean value over users
    """
    if not recommendations or not ground_truth:
        return {}

    precision_scores = []
    recall_scores = []
    hr_scores = []
    ndcg_scores = []
    ap_scores = []

    users_evaluated = 0
    for user_id, recs in recommendations.items():
        if user_id not in ground_truth or not ground_truth[user_id]:
            continue
        relevant = ground_truth[user_id]
        precision_scores.append(precision_at_k(recs, relevant, k))
        recall_scores.append(recall_at_k(recs, relevant, k))
        hr_scores.append(hit_rate_at_k(recs, relevant, k))
        ndcg_scores.append(ndcg_at_k(recs, relevant, k))
        ap_scores.append(average_precision_at_k(recs, relevant, k))
        users_evaluated += 1

    if users_evaluated == 0:
        return {}

    metrics = {
        f"precision@{k}": round(float(np.mean(precision_scores)), 6),
        f"recall@{k}": round(float(np.mean(recall_scores)), 6),
        f"hr@{k}": round(float(np.mean(hr_scores)), 6),
        f"ndcg@{k}": round(float(np.mean(ndcg_scores)), 6),
        f"map@{k}": round(float(np.mean(ap_scores)), 6),
        "users_evaluated": users_evaluated,
    }

    for name, val in metrics.items():
        logger.log_metric(name, val, f"k={k}")

    return metrics


def compute_rating_metrics(y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
    """Compute RMSE and MAE for rating prediction."""
    metrics = {
        "rmse": round(rmse(y_true, y_pred), 6),
        "mae": round(mae(y_true, y_pred), 6),
        "n_predictions": len(y_true),
    }
    logger.log_metric("rmse", metrics["rmse"], "rating prediction")
    logger.log_metric("mae", metrics["mae"], "rating prediction")
    return metrics
