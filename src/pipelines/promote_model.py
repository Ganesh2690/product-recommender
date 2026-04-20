"""
promote_model.py — Model evaluation gate and promotion pipeline.

Compares a new candidate model against the currently deployed model.
Promotes the new model only if it improves Precision@10 by at least
PROMOTION_MIN_IMPROVEMENT_PCT (1% by default).

Phase 10 — Model Promotion
"""

import json
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import (
    MODELS_DIR,
    MODEL_VERSION_FILE,
    PROMOTION_MIN_IMPROVEMENT_PCT,
    TEST_FILE,
    TRAIN_FILE,
)
from src.logging_utils import pipeline_logger as logger


def get_production_metrics() -> Optional[dict]:
    """
    Get the metrics for the currently promoted production model.
    Returns None if no production model exists.
    """
    if not MODEL_VERSION_FILE.exists():
        return None

    model_path_str = MODEL_VERSION_FILE.read_text().strip()
    model_path = Path(model_path_str)

    if not model_path.exists():
        logger.warning(f"Production model path not found: {model_path}")
        return None

    # Look for saved metrics alongside the model
    metrics_path = model_path.with_suffix(".metrics.json")
    if metrics_path.exists():
        with open(str(metrics_path)) as f:
            return json.load(f)

    # No saved metrics — evaluate production model
    logger.info("No cached metrics for production model. Re-evaluating...")
    try:
        import pickle
        import pandas as pd
        from src.evaluation.benchmark import evaluate_model

        with open(str(model_path), "rb") as f:
            model = pickle.load(f)

        test_df = pd.read_parquet(str(TEST_FILE))
        train_df = pd.read_parquet(str(TRAIN_FILE))
        metrics = evaluate_model(model, test_df, train_df, "production", n_users=1000)
        return metrics
    except Exception as e:
        logger.log_error(e, "Failed to evaluate production model")
        return None


def evaluate_candidate(model_name: str) -> Optional[dict]:
    """Evaluate a candidate model and return its metrics."""
    import pickle
    import pandas as pd
    from src.evaluation.benchmark import evaluate_model

    model_path = MODELS_DIR / f"{model_name}_model.pkl"
    if not model_path.exists():
        # Try versioned naming
        candidates = sorted(MODELS_DIR.glob(f"{model_name}_*.pkl"), reverse=True)
        if not candidates:
            logger.error(f"Candidate model not found: {model_name}")
            return None
        model_path = candidates[0]

    with open(str(model_path), "rb") as f:
        model = pickle.load(f)

    test_df = pd.read_parquet(str(TEST_FILE))
    train_df = pd.read_parquet(str(TRAIN_FILE))
    metrics = evaluate_model(model, test_df, train_df, model_name, n_users=1000)
    return metrics


def promote_if_better(
    candidate_model_name: str,
    force: bool = False,
) -> bool:
    """
    Evaluate candidate model and promote if it beats the production model.

    Promotion criteria:
    - No production model exists → always promote
    - Candidate Precision@10 > production Precision@10 × (1 + PROMOTION_MIN_IMPROVEMENT_PCT)

    Args:
        candidate_model_name: Model name to evaluate (e.g., 'svd', 'hybrid')
        force: If True, promote regardless of metrics

    Returns:
        True if promoted, False otherwise.
    """
    logger.start_phase(
        "Model Promotion",
        f"Evaluate candidate '{candidate_model_name}' vs production model"
    )

    # Get candidate metrics
    candidate_metrics = evaluate_candidate(candidate_model_name)
    if candidate_metrics is None:
        logger.error(f"Cannot evaluate candidate model: {candidate_model_name}")
        return False

    candidate_p10 = candidate_metrics.get("precision@10", 0.0)
    logger.info(f"Candidate '{candidate_model_name}': Precision@10={candidate_p10:.4f}")

    # Get production metrics
    prod_metrics = get_production_metrics()

    if prod_metrics is None or force:
        # No production model — promote unconditionally
        reason = "No production model exists" if prod_metrics is None else "Force promotion"
        logger.info(f"Promoting '{candidate_model_name}'. Reason: {reason}")
        _do_promote(candidate_model_name, candidate_metrics)
        logger.end_phase("Model Promotion", f"Promoted: {candidate_model_name}", "Flush cache")
        return True

    prod_p10 = prod_metrics.get("precision@10", 0.0)
    threshold = prod_p10 * (1 + PROMOTION_MIN_IMPROVEMENT_PCT)

    logger.info(
        f"Production model: Precision@10={prod_p10:.4f} | "
        f"Candidate: {candidate_p10:.4f} | "
        f"Threshold: {threshold:.4f} (+{PROMOTION_MIN_IMPROVEMENT_PCT:.1%})"
    )

    logger.log_decision(
        topic="Model Promotion Decision",
        options=[
            f"Promote {candidate_model_name} (P@10={candidate_p10:.4f})",
            f"Keep production (P@10={prod_p10:.4f})",
        ],
        selected=(
            f"Promote {candidate_model_name}"
            if candidate_p10 >= threshold
            else "Keep production"
        ),
        rationale=(
            f"Candidate {'meets' if candidate_p10 >= threshold else 'does not meet'} "
            f"minimum improvement threshold of {PROMOTION_MIN_IMPROVEMENT_PCT:.1%}"
        ),
    )

    if candidate_p10 >= threshold:
        _do_promote(candidate_model_name, candidate_metrics)
        logger.end_phase(
            "Model Promotion",
            f"Promoted: {candidate_model_name} "
            f"(P@10 {prod_p10:.4f} -> {candidate_p10:.4f})",
            "Flush recommendation cache",
        )
        return True
    else:
        logger.info(
            f"NOT promoting '{candidate_model_name}': "
            f"improvement {(candidate_p10-prod_p10)/max(prod_p10,1e-8):.2%} < "
            f"threshold {PROMOTION_MIN_IMPROVEMENT_PCT:.1%}"
        )
        logger.end_phase(
            "Model Promotion",
            f"Production model retained (candidate did not meet threshold)",
            "Continue with existing production model",
        )
        return False


def _do_promote(model_name: str, metrics: dict) -> None:
    """Record promotion: update pointer file and save metrics."""
    model_path = MODELS_DIR / f"{model_name}_model.pkl"
    if not model_path.exists():
        candidates = sorted(MODELS_DIR.glob(f"{model_name}_*.pkl"), reverse=True)
        if candidates:
            model_path = candidates[0]

    MODEL_VERSION_FILE.write_text(str(model_path))

    # Save metrics alongside model
    metrics_path = model_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str))

    logger.log_artifact(str(MODEL_VERSION_FILE), f"Production model pointer -> {model_path.name}")
    logger.info(f"Production model updated: {model_path.name}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Promote a model to production")
    parser.add_argument("--model", type=str, default="svd", help="Model name to promote")
    parser.add_argument("--force", action="store_true", help="Force promotion without comparison")
    args = parser.parse_args()
    promoted = promote_if_better(args.model, force=args.force)
    sys.exit(0 if promoted else 1)
