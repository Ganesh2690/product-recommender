"""
batch_recommend.py — Batch recommendation generation pipeline.

Generates top-N recommendations for all users in the training set,
saves to a batch output file, and optionally pre-warms the cache.

Designed to be run nightly via cron job or CI scheduler.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import ARTIFACTS_DIR, MODELS_DIR, TOP_N, TRAIN_FILE
from src.logging_utils import pipeline_logger as logger


def run_batch_recommend(n: int = TOP_N, max_users: int = None) -> Path:
    """
    Generate top-N recommendations for all training users.

    Args:
        n: Number of recommendations per user
        max_users: If set, only process first N users (for testing)

    Returns:
        Path to output file
    """
    logger.start_phase("Batch Recommend", f"Generate top-{n} recs for all users")
    start = time.time()

    # Load model
    import pickle
    model_path = None
    for model_name in ["hybrid", "svdpp", "svd", "item_cf", "popularity"]:
        candidate = MODELS_DIR / f"{model_name}_model.pkl"
        if candidate.exists():
            model_path = candidate
            logger.info(f"Using model: {model_name} ({candidate.name})")
            break

    if model_path is None:
        logger.error("No trained model found. Run training pipeline first.")
        raise FileNotFoundError("No trained model artifact found")

    with open(str(model_path), "rb") as f:
        model = pickle.load(f)

    # Load training data for user IDs and seen-item filtering
    import pandas as pd
    train_df = pd.read_parquet(str(TRAIN_FILE))
    user_ids = sorted(train_df["user_id"].unique().tolist())
    user_seen = {
        int(uid): grp["movie_id"].tolist()
        for uid, grp in train_df.groupby("user_id")
    }

    if max_users is not None:
        user_ids = user_ids[:max_users]

    logger.info(f"Generating recommendations for {len(user_ids):,} users...")

    batch_results = {}
    failed = 0
    for i, uid in enumerate(user_ids):
        try:
            seen = set(user_seen.get(uid, []))
            raw_recs = model.recommend(uid, n=n, seen_items=seen)
            # Normalise to list of ints
            recs = [r["movie_id"] if isinstance(r, dict) else r for r in raw_recs]
            batch_results[int(uid)] = recs
        except Exception as e:
            failed += 1
            if failed <= 10:
                logger.warning(f"Failed for user {uid}: {e}")

        if (i + 1) % 500 == 0:
            logger.info(f"Progress: {i+1}/{len(user_ids)} users ({failed} failed)")

    elapsed = time.time() - start
    logger.info(f"Batch complete: {len(batch_results):,} users, {failed} failed, {elapsed:.1f}s")
    logger.log_metric("batch_recommend_users", len(batch_results))
    logger.log_metric("batch_recommend_failed", failed)
    logger.log_runtime("batch_recommend", elapsed, f"n={n} users={len(batch_results)}")

    # Save output
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = ARTIFACTS_DIR / "batch_recommendations.json"
    output_path.write_text(json.dumps(batch_results, separators=(",", ":")))

    logger.log_artifact(str(output_path), f"Batch recommendations: {len(batch_results)} users")
    logger.end_phase("Batch Recommend", f"Saved to {output_path}", "Cache pre-warm")
    return output_path


def prewarm_cache(batch_file: Path, n: int = TOP_N) -> None:
    """
    Pre-warm the recommendation cache with batch results.

    Args:
        batch_file: Path to batch_recommendations.json
        n: Recommendation count (used as cache key component)
    """
    logger.info("Pre-warming recommendation cache...")
    from src.serving.cache import get_cache

    with open(str(batch_file)) as f:
        batch_results = json.load(f)

    cache = get_cache()
    count = 0
    for uid_str, recs in batch_results.items():
        try:
            cache.set_recommendations(int(uid_str), n, recs)
            count += 1
        except Exception:
            pass

    logger.info(f"Cache pre-warmed: {count:,} user recommendation lists")
    logger.log_metric("cache_prewarm_users", count, "batch pre-warm")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Batch recommendation generation")
    parser.add_argument("--n", type=int, default=TOP_N, help="Recommendations per user")
    parser.add_argument("--max-users", type=int, default=None, help="Limit to first N users")
    parser.add_argument("--prewarm", action="store_true", help="Pre-warm cache after generation")
    args = parser.parse_args()

    output = run_batch_recommend(n=args.n, max_users=args.max_users)
    if args.prewarm:
        prewarm_cache(output, n=args.n)
    sys.exit(0)
