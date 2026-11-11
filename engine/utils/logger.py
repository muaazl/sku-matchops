import logging
import os
import sys

from engine.config import LOG_DIR, LOG_FILE

# Configure logging with a stream handler and file handler
os.makedirs(LOG_DIR, exist_ok=True)

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Preserve existing test/framework handlers (e.g. pytest LogCaptureHandler)
has_stream = any(
    isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
    for h in root_logger.handlers
)
has_file = any(isinstance(h, logging.FileHandler) for h in root_logger.handlers)

formatter = logging.Formatter(
    "%(asctime)s - [%(levelname)s] - %(name)s - [%(filename)s:%(lineno)d] - %(message)s"
)

if not has_stream:
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    root_logger.addHandler(sh)

if not has_file:
    try:
        fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
        fh.setFormatter(formatter)
        root_logger.addHandler(fh)
    except Exception:
        pass


class LoggerUtils:
    """Utilities for centralized logging and telemetry."""

    @staticmethod
    def log_execution(data: dict):
        """Deprecated: Cloud telemetry to Google Sheets has been disabled."""
        pass

    @staticmethod
    def log_api_job(
        job_id: str,
        task: str,
        domain: str,
        spreadsheet_id: str,
        sheet_name: str,
        status: str,
        total_skus: int = 0,
        high_conf: int = 0,
        med_conf: int = 0,
        low_conf: int = 0,
        match_rate: float = 0.0,
        duration: float = 0.0,
        error: str = "",
        escalated_count: int = 0,
    ):
        """Deprecated: Google Sheets logging has been disabled."""
        pass
