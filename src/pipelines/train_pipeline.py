"""
train_pipeline.py — End-to-end training pipeline for the Recommender System.

Orchestrates:
1. Data loading and validation
2. Preprocessing and split (if not already done)
3. Training all configured models
4. Evaluation of each model
5. Model artifact registration

Phase 10 — Retraining and Promotion Pipeline
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import (
    ITEM_FEATURES,
    MODELS_DIR,
    RATINGS_FILE,
    SVD_N_FACTORS,
    TEST_FILE,
    TRAIN_FILE,
    VAL_FILE,
)
from src.logging_utils import pipeline_logger as logger


def run_data_pipeline_if_needed() -> bool:
    """Ensure preprocessed data exists; run pipeline if not."""
    if TRAIN_FILE.exists() and VAL_FILE.exists() and TEST_FILE.exists():
        logger.info("Preprocessed splits already exist. Skipping data pipeline.")
        return True

    logger.info("Preprocessed splits not found. Running data pipeline...")

    if not RATINGS_FILE.exists():
        from src.data.download import download_movielens
        logger.info("Dataset not found. Downloading...")
        if not download_movielens():
            logger.error("Dataset download failed")
            return False

    from src.data.validate import validate_dataset
    if not validate_dataset():
        logger.error("Dataset validation failed")
        return False

    from src.data.preprocess import run_preprocessing
    if not run_preprocessing():
        logger.error("Preprocessing failed")
        return False

    from src.data.split import run_split
    if not run_split():
        logger.error("Split failed")
        return False

    return True


def train_all_models() -> dict:
    """
    Train all configured models and return {model_name: model_obj} dict.
    """
    import pandas as pd
    train_df = pd.read_parquet(str(TRAIN_FILE))
    logger.info(f"Training dataset: {len(train_df):,} ratings")

    trained_models = {}

    # 1. Popularity baseline (fast)
    try:
        logger.info("Training popularity baseline...")
        from src.models.popularity import PopularityRecommender
        pop = PopularityRecommender()
        pop.fit(train_df)
        pop.save()
        trained_models["popularity"] = pop
        logger.info("Popularity baseline trained")
    except Exception as e:
        logger.log_error(e, "Popularity training failed")

    # 2. Item-item CF
    try:
        logger.info("Training item-item CF...")
        from src.models.item_cf import ItemCFRecommender
        cf = ItemCFRecommender()
        cf.fit(train_df)
        cf.save()
        trained_models["item_cf"] = cf
        logger.info("Item-item CF trained")
    except Exception as e:
        logger.log_error(e, "Item-CF training failed")

    # 3. SVD
    try:
        logger.info(f"Training SVD (n_factors={SVD_N_FACTORS})...")
        from src.models.matrix_factorization import SVDRecommender
        svd = SVDRecommender(n_factors=SVD_N_FACTORS)
        svd.fit(train_df)
        svd.save()
        trained_models["svd"] = svd
        logger.info("SVD trained")
    except Exception as e:
        logger.log_error(e, "SVD training failed")

    # 4. SVD++ (optional — slower)
    try:
        logger.info("Training SVD++...")
        from src.models.matrix_factorization import SVDRecommender
        svdpp = SVDRecommender(n_factors=50, use_svdpp=True)
        svdpp.fit(train_df)
        svdpp.save()
        trained_models["svdpp"] = svdpp
        logger.info("SVD++ trained")
    except Exception as e:
        logger.log_error(e, "SVD++ training failed (non-critical)")

    # 5. Hybrid (requires SVD to be trained)
    if "svd" in trained_models and ITEM_FEATURES.exists():
        try:
            logger.info("Training hybrid model...")
            from src.models.hybrid import HybridRecommender
            item_features = pd.read_parquet(str(ITEM_FEATURES))
            hybrid = HybridRecommender(cf_model=trained_models["svd"])
            hybrid.fit(train_df, item_features)
            hybrid.save()
            trained_models["hybrid"] = hybrid
            logger.info("Hybrid model trained")
        except Exception as e:
            logger.log_error(e, "Hybrid training failed (non-critical)")

    return trained_models


def evaluate_all_models(trained_models: dict) -> dict:
    """Evaluate all trained models on the validation set."""
    import pandas as pd
    test_df = pd.read_parquet(str(TEST_FILE))
    train_df = pd.read_parquet(str(TRAIN_FILE))

    from src.evaluation.benchmark import evaluate_model
    results = {}
    for name, model in trained_models.items():
        try:
            logger.info(f"Evaluating {name}...")
            metrics = evaluate_model(model, test_df, train_df, name)
            results[name] = metrics
        except Exception as e:
            logger.log_error(e, f"Evaluation of {name} failed")
            results[name] = {"error": str(e)}

    return results


def run_training_pipeline() -> dict:
    """
    Main training pipeline entry point.

    Returns:
        results dict with model metrics
    """
    logger.start_phase("Phase 10 - Train Pipeline", "End-to-end training, evaluation, and artifact registration")
    start = time.time()

    # Step 1: Ensure data is ready
    if not run_data_pipeline_if_needed():
        logger.error("Data pipeline failed. Cannot proceed with training.")
        return {}

    # Step 2: Train all models
    logger.info("Starting model training...")
    trained_models = train_all_models()

    if not trained_models:
        logger.error("No models trained successfully")
        return {}

    # Step 3: Evaluate
    logger.info("Starting evaluation...")
    results = evaluate_all_models(trained_models)

    # Step 4: Save results
    results_path = MODELS_DIR / "benchmark_results.json"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2, default=str))
    logger.log_artifact(str(results_path), "Training pipeline benchmark results")

    elapsed = time.time() - start
    logger.log_runtime("full_training_pipeline", elapsed, f"{len(trained_models)} models")
    logger.end_phase(
        "Phase 10 - Train Pipeline",
        f"Pipeline complete: {len(trained_models)} models trained, results saved",
        "Phase 10 - Promote model",
    )
    return results


if __name__ == "__main__":
    results = run_training_pipeline()
    for model_name, metrics in results.items():
        hr10 = metrics.get("hr@10", "N/A")
        p10 = metrics.get("precision@10", "N/A")
        print(f"  {model_name}: HR@10={hr10}, P@10={p10}")
    sys.exit(0)
