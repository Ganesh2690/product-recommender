"""
salesforce_webhook.py — Salesforce webhook integration for churn alerts.

Pushes high-risk churn users (churn_probability >= threshold) to Salesforce
CRM as Tasks/Cases, enabling the sales team to run targeted retention campaigns.

Authentication: OAuth 2.0 Connected App (Client Credentials flow).
All credentials must be supplied via environment variables — never hardcoded.

Phase: Churn Prediction (deliverable 5)
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import ARTIFACTS_DIR, PROCESSED_DATA_DIR
from src.logging_utils import get_logger

logger = get_logger("churn.salesforce", "churn")

CHURN_MODELS_DIR = ARTIFACTS_DIR / "churn"
DEFAULT_CHURN_THRESHOLD = 0.7
BATCH_SIZE = 200  # Salesforce Composite API max


# ---------------------------------------------------------------------------
# Salesforce OAuth
# ---------------------------------------------------------------------------

def _get_sf_token(
    instance_url: str,
    client_id: str,
    client_secret: str,
    username: str,
    password: str,
    security_token: str,
) -> tuple[str, str]:
    """
    Obtain a Salesforce OAuth 2.0 access token via username-password flow.

    Returns:
        (access_token, instance_url)
    """
    import urllib.parse
    import urllib.request

    payload = urllib.parse.urlencode({
        "grant_type":    "password",
        "client_id":     client_id,
        "client_secret": client_secret,
        "username":      username,
        "password":      password + security_token,
    }).encode()

    token_url = f"{instance_url}/services/oauth2/token"
    req = urllib.request.Request(token_url, data=payload, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")

    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.loads(resp.read())

    return body["access_token"], body["instance_url"]


# ---------------------------------------------------------------------------
# Push to Salesforce
# ---------------------------------------------------------------------------

def push_churn_alerts(
    shap_df: Optional[pd.DataFrame] = None,
    churn_threshold: float = DEFAULT_CHURN_THRESHOLD,
    dry_run: bool = False,
) -> dict:
    """
    Push high-risk churn users to Salesforce as Tasks for the retention team.

    Reads credentials from environment variables:
      SF_INSTANCE_URL, SF_CLIENT_ID, SF_CLIENT_SECRET,
      SF_USERNAME, SF_PASSWORD, SF_SECURITY_TOKEN

    Args:
        shap_df: SHAP explanation DataFrame (includes churn_probability).
                 Loaded from artifacts if None.
        churn_threshold: Minimum probability to include in alert batch.
        dry_run: If True, build the payload but do NOT send to Salesforce.

    Returns:
        dict with counts: total_flagged, sent, failed, dry_run
    """
    logger.start_phase("Salesforce Webhook", f"Push churn alerts (threshold={churn_threshold})")
    t0 = time.time()

    # Load SHAP explanations
    if shap_df is None:
        shap_path = CHURN_MODELS_DIR / "shap_explanations.parquet"
        if not shap_path.exists():
            raise FileNotFoundError(
                "SHAP explanations not found. Run shap_explanations.py first."
            )
        shap_df = pd.read_parquet(str(shap_path))

    flagged = shap_df[shap_df["churn_probability"] >= churn_threshold].copy()
    logger.info(f"Flagged users for Salesforce: {len(flagged):,} (threshold={churn_threshold})")

    if len(flagged) == 0:
        logger.info("No users above threshold. Nothing to send.")
        return {"total_flagged": 0, "sent": 0, "failed": 0, "dry_run": dry_run}

    # Resolve SF credentials from env
    sf_instance_url    = os.getenv("SF_INSTANCE_URL", "https://login.salesforce.com")
    sf_client_id       = os.getenv("SF_CLIENT_ID", "")
    sf_client_secret   = os.getenv("SF_CLIENT_SECRET", "")
    sf_username        = os.getenv("SF_USERNAME", "")
    sf_password        = os.getenv("SF_PASSWORD", "")
    sf_security_token  = os.getenv("SF_SECURITY_TOKEN", "")

    creds_present = all([sf_client_id, sf_client_secret, sf_username, sf_password])

    sent = 0
    failed = 0

    if dry_run or not creds_present:
        if not dry_run:
            logger.warning(
                "Salesforce credentials not set in environment. Running in dry-run mode."
            )
        # Write payload to disk for inspection
        payload_path = CHURN_MODELS_DIR / "sf_payload_dry_run.json"
        records = _build_sf_records(flagged, sf_instance_url)
        payload_path.write_text(json.dumps(records[:10], indent=2))
        logger.info(f"Dry-run payload (first 10 records) saved to {payload_path}")
        sent = len(flagged)
    else:
        try:
            access_token, resolved_url = _get_sf_token(
                sf_instance_url, sf_client_id, sf_client_secret,
                sf_username, sf_password, sf_security_token,
            )
            logger.info("Salesforce OAuth token obtained")

            records = _build_sf_records(flagged, resolved_url)
            sent, failed = _composite_upsert(records, resolved_url, access_token)
        except Exception as e:
            logger.log_error(e, "Salesforce push failed")
            failed = len(flagged)

    elapsed = time.time() - t0
    result = {
        "total_flagged": int(len(flagged)),
        "sent": int(sent),
        "failed": int(failed),
        "dry_run": dry_run or not creds_present,
        "elapsed_s": round(elapsed, 2),
    }

    # Persist result log
    result_path = CHURN_MODELS_DIR / "sf_push_result.json"
    result_path.write_text(json.dumps(result, indent=2))

    logger.log_metric("sf_sent", sent, "Salesforce")
    logger.log_metric("sf_failed", failed, "Salesforce")
    logger.log_artifact(str(result_path), "Salesforce push result")
    logger.end_phase("Salesforce Webhook", f"sent={sent} failed={failed}", "Drift Monitoring")
    return result


def _build_sf_records(flagged: pd.DataFrame, instance_url: str) -> list:
    """Build Salesforce Task record payloads from flagged churn users."""
    records = []
    for _, row in flagged.iterrows():
        uid = int(row["user_id"])
        prob = float(row["churn_probability"])
        top_factor = str(row.get("top_factor", "unknown"))
        records.append({
            "attributes": {"type": "Task"},
            "Subject": f"Churn Risk Alert — User {uid}",
            "Description": (
                f"User {uid} has a churn probability of {prob:.1%}. "
                f"Top contributing factor: {top_factor}. "
                "Review engagement history and initiate retention campaign."
            ),
            "Priority": "High" if prob >= 0.85 else "Normal",
            "Status": "Not Started",
            "ActivityDate": _today_iso(),
            # External_User_ID__c maps to the recommender user_id — custom SF field
            "External_User_ID__c": str(uid),
            "Churn_Probability__c": round(prob, 4),
            "Top_Churn_Factor__c": top_factor,
        })
    return records


def _today_iso() -> str:
    from datetime import date
    return date.today().isoformat()


def _composite_upsert(records: list, instance_url: str, access_token: str) -> tuple[int, int]:
    """
    Send records to Salesforce using the Composite batch API (200 per request).
    Returns (sent, failed).
    """
    import urllib.request

    api_url = f"{instance_url}/services/data/v58.0/composite/sobjects"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    sent = 0
    failed = 0

    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i: i + BATCH_SIZE]
        payload = json.dumps({"allOrNone": False, "records": batch}).encode()
        req = urllib.request.Request(api_url, data=payload, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                results = json.loads(resp.read())
                batch_sent = sum(1 for r in results if r.get("success"))
                batch_failed = len(results) - batch_sent
                sent += batch_sent
                failed += batch_failed
                logger.info(
                    f"Batch {i // BATCH_SIZE + 1}: sent={batch_sent} failed={batch_failed}"
                )
        except Exception as e:
            logger.log_error(e, f"Composite batch {i // BATCH_SIZE + 1} failed")
            failed += len(batch)

    return sent, failed


if __name__ == "__main__":
    result = push_churn_alerts(churn_threshold=0.7, dry_run=True)
    print(json.dumps(result, indent=2))
