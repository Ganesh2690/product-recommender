"""
config.py — Centralised configuration for the Recommender System.

All paths, hyperparameters, and feature flags are defined here.
Scripts import from this module rather than hardcoding values.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = DATA_DIR / "artifacts"
LOGS_DIR = PROJECT_ROOT / "logs"
DOCS_DIR = PROJECT_ROOT / "docs"
MODELS_DIR = ARTIFACTS_DIR / "models"
CACHE_DIR = ARTIFACTS_DIR / "cache"

# Ensure directories exist
for _d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, ARTIFACTS_DIR, LOGS_DIR, MODELS_DIR, CACHE_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
MOVIELENS_ZIP = RAW_DATA_DIR / "ml-1m.zip"
MOVIELENS_DIR = RAW_DATA_DIR / "ml-1m"

RATINGS_FILE = MOVIELENS_DIR / "ratings.dat"
MOVIES_FILE = MOVIELENS_DIR / "movies.dat"
USERS_FILE = MOVIELENS_DIR / "users.dat"

RATINGS_COLS = ["user_id", "movie_id", "rating", "timestamp"]
MOVIES_COLS = ["movie_id", "title", "genres"]
USERS_COLS = ["user_id", "gender", "age", "occupation", "zip_code"]

# Processed data paths
RATINGS_PROCESSED = PROCESSED_DATA_DIR / "ratings.parquet"
MOVIES_PROCESSED = PROCESSED_DATA_DIR / "movies.parquet"
USERS_PROCESSED = PROCESSED_DATA_DIR / "users.parquet"
TRAIN_FILE = PROCESSED_DATA_DIR / "train.parquet"
VAL_FILE = PROCESSED_DATA_DIR / "val.parquet"
TEST_FILE = PROCESSED_DATA_DIR / "test.parquet"
USER_ITEM_MATRIX = PROCESSED_DATA_DIR / "user_item_matrix.npz"
ITEM_FEATURES = PROCESSED_DATA_DIR / "item_features.parquet"

# ---------------------------------------------------------------------------
# Data quality thresholds
# ---------------------------------------------------------------------------
MIN_RATINGS_PER_USER = 5
MIN_RATINGS_PER_ITEM = 1
EXPECTED_TOTAL_RATINGS_MIN = 900_000
EXPECTED_TOTAL_RATINGS_MAX = 1_100_000
EXPECTED_NUM_USERS_MIN = 6000
EXPECTED_NUM_MOVIES_MIN = 3700

# ---------------------------------------------------------------------------
# Split configuration
# ---------------------------------------------------------------------------
SPLIT_STRATEGY = "time_aware"   # "time_aware" or "random"
TRAIN_RATIO = 0.60
VAL_RATIO = 0.20
TEST_RATIO = 0.20
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Recommendation parameters
# ---------------------------------------------------------------------------
TOP_N = 10
MAX_RECOMMENDATIONS = 50
COLD_START_THRESHOLD = 5  # users with fewer interactions get popularity fallback

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------
# SVD
SVD_N_FACTORS = int(os.getenv("SVD_N_FACTORS", "100"))
SVD_N_EPOCHS = int(os.getenv("SVD_N_EPOCHS", "20"))
SVD_LR_ALL = float(os.getenv("SVD_LR_ALL", "0.005"))
SVD_REG_ALL = float(os.getenv("SVD_REG_ALL", "0.02"))

# SVD++ (slower but usually better ranking)
SVDPP_N_FACTORS = int(os.getenv("SVDPP_N_FACTORS", "50"))
SVDPP_N_EPOCHS = int(os.getenv("SVDPP_N_EPOCHS", "20"))

# Item-item CF
CF_N_SIMILAR_ITEMS = int(os.getenv("CF_N_SIMILAR_ITEMS", "50"))
CF_MIN_SUPPORT = int(os.getenv("CF_MIN_SUPPORT", "5"))

# Hybrid blending
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.8"))  # weight of CF vs content

# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------
K_VALUES = [5, 10, 20]
PRIMARY_K = 10

# Success thresholds (from project spec)
TARGET_HR_AT_10 = 0.35
TARGET_PRECISION_AT_10 = 0.10
TARGET_RECALL_AT_10 = 0.20
TARGET_API_P95_MS = 150.0

# ---------------------------------------------------------------------------
# API configuration
# ---------------------------------------------------------------------------
FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.getenv("FLASK_PORT", "5000"))
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "false").lower() == "true"

# ---------------------------------------------------------------------------
# Cache configuration
# ---------------------------------------------------------------------------
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "86400"))  # 24 hours
USE_REDIS = os.getenv("USE_REDIS", "true").lower() == "true"

# ---------------------------------------------------------------------------
# Model registry / versioning
# ---------------------------------------------------------------------------
MODEL_VERSION_FILE = ARTIFACTS_DIR / "current_model_version.txt"
MODEL_HISTORY_FILE = ARTIFACTS_DIR / "model_history.jsonl"
MAX_MODEL_VERSIONS = 3  # keep last N versions

# Promotion thresholds
PROMOTION_MIN_IMPROVEMENT_PCT = 0.01  # 1% improvement on Precision@10

# ---------------------------------------------------------------------------
# MLflow (modernization add-on — post-2017, clearly labeled)
# ---------------------------------------------------------------------------
MLFLOW_TRACKING = os.getenv("MLFLOW_TRACKING", "false").lower() == "true"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", str(ARTIFACTS_DIR / "mlruns"))
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "recommender-system")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ---------------------------------------------------------------------------
# Notebook-compatible aliases
# ---------------------------------------------------------------------------
PROCESSED_DIR = PROCESSED_DATA_DIR
N_FACTORS = SVD_N_FACTORS
N_EPOCHS = SVD_N_EPOCHS
ALPHA = HYBRID_ALPHA
SUCCESS_HR_AT_10 = TARGET_HR_AT_10
SUCCESS_PRECISION_AT_10 = TARGET_PRECISION_AT_10
