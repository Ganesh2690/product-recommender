"""
test_metrics.py — Unit tests for all evaluation metrics.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    hit_rate_at_k,
    ndcg_at_k,
    average_precision_at_k,
    rmse,
    mae,
    compute_ranking_metrics,
    compute_rating_metrics,
)


# -------------------------------------------------------------------------
# Precision@K
# -------------------------------------------------------------------------

class TestPrecisionAtK:
    def test_perfect_recall(self):
        recs = [1, 2, 3, 4, 5]
        ground_truth = {1, 2, 3, 4, 5}
        assert precision_at_k(recs, ground_truth, k=5) == pytest.approx(1.0)

    def test_zero_overlap(self):
        recs = [10, 20, 30]
        ground_truth = {1, 2, 3}
        assert precision_at_k(recs, ground_truth, k=3) == pytest.approx(0.0)

    def test_partial_overlap(self):
        recs = [1, 10, 2, 20, 3]
        ground_truth = {1, 2, 3}
        # 3 hits in top 5 → 0.6
        assert precision_at_k(recs, ground_truth, k=5) == pytest.approx(0.6)

    def test_k_truncation(self):
        recs = [1, 2, 3, 4, 5]
        ground_truth = {4, 5}
        # Only looking at top 3, no hits → 0.0
        assert precision_at_k(recs, ground_truth, k=3) == pytest.approx(0.0)

    def test_empty_recommendations(self):
        assert precision_at_k([], {1, 2}, k=10) == pytest.approx(0.0)

    def test_empty_ground_truth(self):
        assert precision_at_k([1, 2, 3], set(), k=3) == pytest.approx(0.0)


# -------------------------------------------------------------------------
# Recall@K
# -------------------------------------------------------------------------

class TestRecallAtK:
    def test_perfect_recall(self):
        recs = [1, 2, 3]
        ground_truth = {1, 2, 3}
        assert recall_at_k(recs, ground_truth, k=3) == pytest.approx(1.0)

    def test_zero_recall(self):
        recs = [10, 20, 30]
        ground_truth = {1, 2, 3}
        assert recall_at_k(recs, ground_truth, k=3) == pytest.approx(0.0)

    def test_partial_recall(self):
        recs = [1, 10, 20]
        ground_truth = {1, 2, 3, 4}
        # 1 hit out of 4 relevant → 0.25
        assert recall_at_k(recs, ground_truth, k=3) == pytest.approx(0.25)

    def test_empty_recommendations(self):
        assert recall_at_k([], {1, 2}, k=5) == pytest.approx(0.0)

    def test_empty_ground_truth(self):
        assert recall_at_k([1, 2], set(), k=2) == pytest.approx(0.0)


# -------------------------------------------------------------------------
# Hit Rate@K (HR@K)
# -------------------------------------------------------------------------

class TestHitRateAtK:
    def test_hit(self):
        recs = [5, 10, 15, 20]
        ground_truth = {15}
        assert hit_rate_at_k(recs, ground_truth, k=4) == 1

    def test_miss(self):
        recs = [1, 2, 3]
        ground_truth = {99}
        assert hit_rate_at_k(recs, ground_truth, k=3) == 0

    def test_k_cutoff_matters(self):
        # Item is at position 3 (0-indexed), k=2 should not count it
        recs = [1, 2, 3, 4, 5]
        ground_truth = {3}
        assert hit_rate_at_k(recs, ground_truth, k=2) == 0
        assert hit_rate_at_k(recs, ground_truth, k=3) == 1

    def test_empty_ground_truth(self):
        assert hit_rate_at_k([1, 2], set(), k=2) == 0


# -------------------------------------------------------------------------
# NDCG@K
# -------------------------------------------------------------------------

class TestNDCGAtK:
    def test_perfect_ndcg(self):
        recs = [1, 2, 3]
        ground_truth = {1, 2, 3}
        score = ndcg_at_k(recs, ground_truth, k=3)
        assert score == pytest.approx(1.0)

    def test_zero_ndcg(self):
        recs = [10, 20, 30]
        ground_truth = {1, 2, 3}
        assert ndcg_at_k(recs, ground_truth, k=3) == pytest.approx(0.0)

    def test_order_matters(self):
        """Placing relevant items earlier should give higher NDCG."""
        gt = {1, 2}
        recs_good = [1, 2, 10, 20]
        recs_bad = [10, 20, 1, 2]
        assert ndcg_at_k(recs_good, gt, k=4) > ndcg_at_k(recs_bad, gt, k=4)

    def test_ndcg_range(self):
        recs = [1, 3, 2, 10, 20]
        ground_truth = {1, 2, 5}
        score = ndcg_at_k(recs, ground_truth, k=5)
        assert 0.0 <= score <= 1.0

    def test_empty_recommendations(self):
        assert ndcg_at_k([], {1, 2}, k=10) == pytest.approx(0.0)


# -------------------------------------------------------------------------
# Average Precision@K
# -------------------------------------------------------------------------

class TestAveragePrecisionAtK:
    def test_perfect(self):
        recs = [1, 2, 3]
        ground_truth = {1, 2, 3}
        ap = average_precision_at_k(recs, ground_truth, k=3)
        assert ap == pytest.approx(1.0)

    def test_zero(self):
        recs = [10, 20, 30]
        ground_truth = {1, 2, 3}
        ap = average_precision_at_k(recs, ground_truth, k=3)
        assert ap == pytest.approx(0.0)

    def test_partial(self):
        # Hits at positions 1 and 3 (1-indexed)
        recs = [1, 99, 2, 98]
        ground_truth = {1, 2}
        ap = average_precision_at_k(recs, ground_truth, k=4)
        # AP = (1/1 + 2/3) / 2 = (1.0 + 0.667) / 2 ≈ 0.833
        assert ap == pytest.approx((1.0 + 2 / 3) / 2, abs=1e-3)


# -------------------------------------------------------------------------
# RMSE / MAE
# -------------------------------------------------------------------------

class TestRatingMetrics:
    def test_rmse_perfect(self):
        actual = [4.0, 3.0, 5.0]
        predicted = [4.0, 3.0, 5.0]
        assert rmse(actual, predicted) == pytest.approx(0.0)

    def test_rmse_single_error(self):
        actual = [4.0]
        predicted = [3.0]
        assert rmse(actual, predicted) == pytest.approx(1.0)

    def test_rmse_penalises_large_errors(self):
        r1 = rmse([4.0, 4.0], [3.0, 3.0])     # two small errors
        r2 = rmse([4.0, 4.0], [2.0, 6.0])     # one big error
        assert r2 > r1

    def test_mae_perfect(self):
        assert mae([3.5, 4.0], [3.5, 4.0]) == pytest.approx(0.0)

    def test_mae_value(self):
        actual = [4.0, 2.0]
        predicted = [3.0, 3.0]
        # Mean absolute deviation = (1 + 1) / 2 = 1.0
        assert mae(actual, predicted) == pytest.approx(1.0)

    def test_rmse_nonnegative(self):
        actual = np.random.default_rng(0).uniform(1, 5, 50).tolist()
        predicted = np.random.default_rng(1).uniform(1, 5, 50).tolist()
        assert rmse(actual, predicted) >= 0.0
        assert mae(actual, predicted) >= 0.0


# -------------------------------------------------------------------------
# Batch compute helpers
# -------------------------------------------------------------------------

class TestComputeRankingMetrics:
    @pytest.fixture
    def user_recs_and_gt(self):
        user_recs = {
            1: [1, 2, 3, 10, 11],
            2: [5, 6, 7, 8, 9],
            3: [1, 5, 30, 4, 2],
        }
        ground_truth = {
            1: {1, 2},
            2: {5, 6},
            3: {1, 5, 4},
        }
        return user_recs, ground_truth

    def test_returns_dict(self, user_recs_and_gt):
        user_recs, gt = user_recs_and_gt
        result = compute_ranking_metrics(user_recs, gt, k=5)
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_expected_keys(self, user_recs_and_gt):
        user_recs, gt = user_recs_and_gt
        result = compute_ranking_metrics(user_recs, gt, k=5)
        # Keys are formatted as 'metric@K'
        for key in ["precision@5", "recall@5", "hr@5", "ndcg@5"]:
            assert key in result, f"Missing key: {key}"

    def test_values_in_range(self, user_recs_and_gt):
        user_recs, gt = user_recs_and_gt
        result = compute_ranking_metrics(user_recs, gt, k=5)
        skip_keys = {"users_evaluated"}  # not a score, it's a count
        for key, val in result.items():
            if key in skip_keys:
                continue
            assert 0.0 <= val <= 1.0, f"{key} out of range: {val}"


class TestComputeRatingMetrics:
    def test_returns_dict(self):
        actual = [4.0, 3.0, 5.0]
        predicted = [3.5, 3.5, 4.5]
        result = compute_rating_metrics(actual, predicted)
        assert isinstance(result, dict)
        assert "rmse" in result
        assert "mae" in result

    def test_values_nonneg(self):
        actual = [4.0, 3.0, 5.0]
        predicted = [3.5, 3.5, 4.5]
        result = compute_rating_metrics(actual, predicted)
        assert result["rmse"] >= 0.0
        assert result["mae"] >= 0.0
