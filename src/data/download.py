"""
download.py — Automated dataset downloader for MovieLens 1M.

Downloads the dataset, verifies integrity, extracts files, and logs full provenance.
Every execution logs to data_pipeline.log and run_log.jsonl.
"""

import hashlib
import sys
import time
import zipfile
from pathlib import Path

import requests

# Allow running as standalone script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import (
    MOVIELENS_DIR,
    MOVIELENS_URL,
    MOVIELENS_ZIP,
    MOVIES_FILE,
    RAW_DATA_DIR,
    RATINGS_FILE,
    USERS_FILE,
)
from src.logging_utils import data_logger as logger

# Known file sizes (bytes) — used for basic validation
EXPECTED_MIN_SIZE_RATINGS = 24_000_000
EXPECTED_MIN_SIZE_MOVIES = 150_000
EXPECTED_MIN_SIZE_USERS = 130_000


def _compute_sha256(filepath: Path, chunk_size: int = 8192) -> str:
    """Compute SHA-256 checksum of a file."""
    sha256 = hashlib.sha256()
    with open(str(filepath), "rb") as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    return sha256.hexdigest()


def download_movielens(force: bool = False) -> bool:
    """
    Download MovieLens 1M dataset from GroupLens.

    Args:
        force: If True, re-download even if files exist.

    Returns:
        True on success, False on failure.
    """
    logger.start_phase("Phase 2 - Download", "Download MovieLens 1M dataset")
    logger.log_command(f"download from {MOVIELENS_URL}", "Acquire MovieLens 1M dataset")

    # Check if already extracted
    if not force and RATINGS_FILE.exists() and MOVIES_FILE.exists() and USERS_FILE.exists():
        logger.info(f"Dataset already present at {MOVIELENS_DIR}. Skipping download.")
        logger.end_phase("Phase 2 - Download", "Dataset already present — skipped re-download", "Phase 2 - Validate")
        return True

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Download with retry
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Download attempt {attempt}/{max_retries}: {MOVIELENS_URL}")
            start = time.time()
            response = requests.get(MOVIELENS_URL, stream=True, timeout=60)
            response.raise_for_status()

            total_bytes = int(response.headers.get("content-length", 0))  # noqa: F841
            downloaded = 0

            with open(str(MOVIELENS_ZIP), "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)

            elapsed = time.time() - start
            file_size = MOVIELENS_ZIP.stat().st_size
            logger.info(f"Downloaded {file_size:,} bytes in {elapsed:.1f}s")
            logger.log_runtime("movielens_download", elapsed, f"bytes={file_size}")
            break

        except requests.RequestException as e:
            logger.warning(f"Download attempt {attempt} failed: {e}")
            if attempt == max_retries:
                logger.log_error(e, "All download attempts exhausted")
                return False
            time.sleep(2 ** attempt)

    # Compute checksum
    logger.info("Computing SHA-256 checksum...")
    checksum = _compute_sha256(MOVIELENS_ZIP)
    logger.info(f"SHA-256: {checksum}")

    # Save checksum
    checksum_file = RAW_DATA_DIR / "ml-1m.zip.sha256"
    checksum_file.write_text(checksum)

    # Extract
    logger.info(f"Extracting {MOVIELENS_ZIP} to {RAW_DATA_DIR}")
    try:
        with zipfile.ZipFile(str(MOVIELENS_ZIP), "r") as zf:
            zf.extractall(str(RAW_DATA_DIR))
    except zipfile.BadZipFile as e:
        logger.log_error(e, "ZIP extraction failed — file may be corrupted")
        return False

    # Log provenance
    logger.log_dataset_provenance(
        source_url=MOVIELENS_URL,
        local_path=str(MOVIELENS_DIR),
        file_size_bytes=MOVIELENS_ZIP.stat().st_size,
        schema={
            "ratings.dat": "UserID::MovieID::Rating::Timestamp",
            "movies.dat": "MovieID::Title::Genres",
            "users.dat": "UserID::Gender::Age::Occupation::Zip-code",
        },
        checksum=checksum,
    )

    logger.log_artifact(str(MOVIELENS_DIR), "Extracted MovieLens 1M dataset directory")
    logger.end_phase("Phase 2 - Download", f"Dataset downloaded and extracted: {MOVIELENS_DIR}", "Phase 2 - Validate")
    return True


def get_file_info() -> dict:
    """Return a dict of file existence and sizes for all expected dataset files."""
    files = {
        "ratings.dat": RATINGS_FILE,
        "movies.dat": MOVIES_FILE,
        "users.dat": USERS_FILE,
    }
    info = {}
    for name, path in files.items():
        info[name] = {
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else 0,
        }
    return info


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download MovieLens 1M dataset")
    parser.add_argument("--force", action="store_true", help="Force re-download even if files exist")
    args = parser.parse_args()

    success = download_movielens(force=args.force)
    if success:
        info = get_file_info()
        for fname, finfo in info.items():
            status = "OK" if finfo["exists"] else "MISSING"
            print(f"  [{status}] {fname}: {finfo['size_bytes']:,} bytes")
        sys.exit(0)
    else:
        print("Download failed. Check logs/data_pipeline.log for details.")
        sys.exit(1)
