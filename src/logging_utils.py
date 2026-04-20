"""
logging_utils.py — Project-wide logging framework for the Recommender System.

This module provides a reusable ProjectLogger that every script, model, and pipeline
imports. It supports console logging, file logging, JSONL structured logs, phase-level
checkpoint logging, command logging, exception logging, model metrics logging,
dataset provenance logging, and runtime measurement logging.
"""

import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILES = {
    "master": LOGS_DIR / "master_execution_log.md",
    "decision": LOGS_DIR / "decision_log.md",
    "jsonl": LOGS_DIR / "run_log.jsonl",
    "data_pipeline": LOGS_DIR / "data_pipeline.log",
    "model_training": LOGS_DIR / "model_training.log",
    "evaluation": LOGS_DIR / "evaluation.log",
    "api": LOGS_DIR / "api.log",
    "test": LOGS_DIR / "test.log",
    "checkpoint": LOGS_DIR / "checkpoint_status.md",
}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# ProjectLogger
# ---------------------------------------------------------------------------
class ProjectLogger:
    """
    Centralised logger for the Recommender System project.

    Usage:
        from src.logging_utils import get_logger
        logger = get_logger("data_pipeline")
        logger.start_phase("Phase 2", "Download MovieLens 1M dataset")
        ...
        logger.end_phase("Phase 2", "Dataset downloaded and validated", next_step="Phase 3")
    """

    def __init__(self, component: str, log_file_key: str = "master"):
        self.component = component
        self.log_file_key = log_file_key
        self._phase_start_times: Dict[str, float] = {}

        # Standard Python logger (console + file)
        self._logger = logging.getLogger(f"recommender.{component}")
        if not self._logger.handlers:
            self._logger.setLevel(logging.DEBUG)
            # Console handler
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.INFO)
            ch.setFormatter(
                logging.Formatter(
                    "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            self._logger.addHandler(ch)

            # File handler (component-specific)
            file_key = log_file_key if log_file_key in LOG_FILES else "master"
            log_path = LOG_FILES.get(file_key, LOGS_DIR / f"{component}.log")
            fh = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(
                logging.Formatter(
                    "[%(asctime)s] [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            self._logger.addHandler(fh)

    # -----------------------------------------------------------------------
    # Core log methods
    # -----------------------------------------------------------------------
    def info(self, msg: str) -> None:
        self._logger.info(msg)
        self._write_jsonl({"level": "INFO", "component": self.component, "message": msg})

    def warning(self, msg: str) -> None:
        self._logger.warning(msg)
        self._write_jsonl({"level": "WARNING", "component": self.component, "message": msg})

    def error(self, msg: str) -> None:
        self._logger.error(msg)
        self._write_jsonl({"level": "ERROR", "component": self.component, "message": msg})

    def debug(self, msg: str) -> None:
        self._logger.debug(msg)

    # -----------------------------------------------------------------------
    # Phase lifecycle helpers
    # -----------------------------------------------------------------------
    def start_phase(self, phase_name: str, objective: str) -> None:
        """Log the start of a named phase."""
        self._phase_start_times[phase_name] = time.time()
        msg = f"[PHASE START] {phase_name} | Objective: {objective}"
        self.info(msg)
        self._write_jsonl({
            "event": "phase_start",
            "phase": phase_name,
            "objective": objective,
            "timestamp": _now_iso(),
        })
        self._append_to_log(
            "master",
            f"\n### PHASE START — {phase_name}\n"
            f"- **Timestamp:** {_now()}\n"
            f"- **Objective:** {objective}\n"
            f"- **Status:** STARTED\n",
        )

    def end_phase(self, phase_name: str, outcome: str, next_step: str = "") -> None:
        """Log the successful completion of a named phase."""
        elapsed = ""
        if phase_name in self._phase_start_times:
            secs = time.time() - self._phase_start_times.pop(phase_name)
            elapsed = f" (elapsed: {secs:.1f}s)"
        msg = f"[PHASE COMPLETE] {phase_name} | Outcome: {outcome}{elapsed}"
        self.info(msg)
        self._write_jsonl({
            "event": "phase_complete",
            "phase": phase_name,
            "outcome": outcome,
            "next_step": next_step,
            "timestamp": _now_iso(),
        })
        self._append_to_log(
            "master",
            f"\n### PHASE COMPLETE — {phase_name}\n"
            f"- **Timestamp:** {_now()}\n"
            f"- **Outcome:** {outcome}\n"
            f"- **Next Step:** {next_step}\n",
        )

    # -----------------------------------------------------------------------
    # Decision logging
    # -----------------------------------------------------------------------
    def log_decision(
        self,
        topic: str,
        options: List[str],
        selected: str,
        rationale: str,
        risks: Optional[str] = None,
        mitigation: Optional[str] = None,
    ) -> None:
        """Record an engineering decision with full context."""
        self.info(f"[DECISION] {topic} -> Selected: {selected}")
        entry = {
            "event": "decision",
            "topic": topic,
            "options": options,
            "selected": selected,
            "rationale": rationale,
            "risks": risks,
            "mitigation": mitigation,
            "timestamp": _now_iso(),
        }
        self._write_jsonl(entry)
        decision_md = (
            f"\n### DECISION — {topic}\n"
            f"- **Timestamp:** {_now()}\n"
            f"- **Options:** {', '.join(options)}\n"
            f"- **Selected:** {selected}\n"
            f"- **Rationale:** {rationale}\n"
        )
        if risks:
            decision_md += f"- **Risks:** {risks}\n"
        if mitigation:
            decision_md += f"- **Mitigation:** {mitigation}\n"
        self._append_to_log("decision", decision_md)

    # -----------------------------------------------------------------------
    # Command logging
    # -----------------------------------------------------------------------
    def log_command(self, cmd: str, purpose: str) -> None:
        """Log a shell command that was run."""
        self.info(f"[CMD] {cmd} | Purpose: {purpose}")
        self._write_jsonl({
            "event": "command",
            "command": cmd,
            "purpose": purpose,
            "timestamp": _now_iso(),
        })

    # -----------------------------------------------------------------------
    # Metric logging
    # -----------------------------------------------------------------------
    def log_metric(self, metric_name: str, value: Any, context: str = "") -> None:
        """Log a measured metric value."""
        self.info(f"[METRIC] {metric_name}={value} | {context}")
        self._write_jsonl({
            "event": "metric",
            "metric": metric_name,
            "value": value,
            "context": context,
            "timestamp": _now_iso(),
        })

    def log_metrics(self, metrics: Dict[str, Any], context: str = "") -> None:
        """Log multiple metrics at once."""
        for name, value in metrics.items():
            self.log_metric(name, value, context)

    # -----------------------------------------------------------------------
    # Artifact logging
    # -----------------------------------------------------------------------
    def log_artifact(self, path: str, description: str) -> None:
        """Log the creation or modification of a file artifact."""
        self.info(f"[ARTIFACT] {path} | {description}")
        self._write_jsonl({
            "event": "artifact",
            "path": path,
            "description": description,
            "timestamp": _now_iso(),
        })

    # -----------------------------------------------------------------------
    # Dataset provenance logging
    # -----------------------------------------------------------------------
    def log_dataset_provenance(
        self,
        source_url: str,
        local_path: str,
        file_size_bytes: int,
        schema: Dict[str, Any],
        checksum: Optional[str] = None,
    ) -> None:
        """Log full dataset provenance information."""
        self.info(f"[PROVENANCE] Dataset downloaded from {source_url} -> {local_path}")
        self._write_jsonl({
            "event": "dataset_provenance",
            "source_url": source_url,
            "local_path": local_path,
            "file_size_bytes": file_size_bytes,
            "schema": schema,
            "checksum_sha256": checksum,
            "download_timestamp": _now_iso(),
        })
        self._append_to_log(
            "data_pipeline",
            f"\n[{_now()}] [PROVENANCE] source={source_url}"
            f" size_bytes={file_size_bytes}"
            f" sha256={checksum or 'N/A'}\n",
        )

    # -----------------------------------------------------------------------
    # Error logging
    # -----------------------------------------------------------------------
    def log_error(self, error: Exception, context: str = "") -> None:
        """Log an exception with full traceback."""
        tb = traceback.format_exc()
        self.error(f"[ERROR] {type(error).__name__}: {error} | Context: {context}\n{tb}")
        self._write_jsonl({
            "event": "error",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": tb,
            "context": context,
            "timestamp": _now_iso(),
        })

    # -----------------------------------------------------------------------
    # Runtime measurement
    # -----------------------------------------------------------------------
    def log_runtime(self, operation: str, seconds: float, context: str = "") -> None:
        """Log the runtime of an operation."""
        self.info(f"[RUNTIME] {operation}: {seconds:.3f}s | {context}")
        self._write_jsonl({
            "event": "runtime",
            "operation": operation,
            "seconds": seconds,
            "context": context,
            "timestamp": _now_iso(),
        })

    # -----------------------------------------------------------------------
    # Cache event logging
    # -----------------------------------------------------------------------
    def log_cache_event(self, event_type: str, key: str, context: str = "") -> None:
        """Log cache hit, miss, or refresh events."""
        self._write_jsonl({
            "event": f"cache_{event_type}",
            "key": key,
            "context": context,
            "timestamp": _now_iso(),
        })
        self.debug(f"[CACHE {event_type.upper()}] key={key}")

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------
    def _write_jsonl(self, record: Dict[str, Any]) -> None:
        """Append a JSON record to the JSONL run log."""
        try:
            jsonl_path = LOG_FILES["jsonl"]
            with open(str(jsonl_path), "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            pass  # Never let logging failures crash the application

    def _append_to_log(self, log_key: str, content: str) -> None:
        """Append text content to a specific log file."""
        try:
            path = LOG_FILES.get(log_key)
            if path:
                with open(str(path), "a", encoding="utf-8") as f:
                    f.write(content)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Logging health check
# ---------------------------------------------------------------------------
def check_logging_health() -> bool:
    """
    Verify that all required log files exist and are writable.
    Returns True if healthy, False otherwise.
    Logs any failures to stdout.
    """
    all_healthy = True
    print(f"[{_now()}] Logging health check starting...")
    for key, path in LOG_FILES.items():
        if not path.exists():
            print(f"  [FAIL] Missing log file: {path}")
            all_healthy = False
        elif not os.access(str(path), os.W_OK):
            print(f"  [FAIL] Log file not writable: {path}")
            all_healthy = False
        else:
            print(f"  [OK]   {path.name}")
    if all_healthy:
        print(f"[{_now()}] All log files healthy.")
    else:
        print(f"[{_now()}] LOGGING HEALTH CHECK FAILED — fix before continuing.")
    return all_healthy


# ---------------------------------------------------------------------------
# Factory / singleton access
# ---------------------------------------------------------------------------
_loggers: Dict[str, ProjectLogger] = {}


def get_logger(component: str, log_file_key: str = "master") -> ProjectLogger:
    """
    Get (or create) a named ProjectLogger.

    Args:
        component: Component name (e.g., 'data_pipeline', 'svd_model', 'api')
        log_file_key: Key from LOG_FILES dict specifying which log file to write to.
                      Common values: 'data_pipeline', 'model_training', 'evaluation',
                      'api', 'test', 'master'

    Returns:
        ProjectLogger instance
    """
    key = f"{component}:{log_file_key}"
    if key not in _loggers:
        _loggers[key] = ProjectLogger(component, log_file_key)
    return _loggers[key]


# Convenience pre-created loggers
data_logger = get_logger("data_pipeline", "data_pipeline")
model_logger = get_logger("model_training", "model_training")
eval_logger = get_logger("evaluation", "evaluation")
api_logger = get_logger("api", "api")
test_logger = get_logger("testing", "test")
pipeline_logger = get_logger("pipeline", "master")
