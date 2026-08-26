import logging
import os
import sys

from engine import config

# Configure logging with a stream handler and file handler
DB_DIR = config.DB_DIR
os.makedirs(DB_DIR, exist_ok=True)
LOG_FILE = os.path.join(DB_DIR, "app.log")

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")

sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(formatter)
root_logger.addHandler(sh)

fh = logging.FileHandler(LOG_FILE, encoding='utf-8')
fh.setFormatter(formatter)
root_logger.addHandler(fh)

class LoggerUtils:
    """Utilities for centralized logging and telemetry."""

    @staticmethod
    def log_execution(data: dict):
        """Deprecated: Cloud telemetry to Google Sheets has been disabled."""
        pass

    @staticmethod
    def log_api_job(job_id: str, task: str, domain: str, spreadsheet_id: str, sheet_name: str, status: str, 
                    total_skus: int = 0, high_conf: int = 0, med_conf: int = 0, low_conf: int = 0,
                    match_rate: float = 0.0, duration: float = 0.0, error: str = "",
                    escalated_count: int = 0):
        """Deprecated: Google Sheets logging has been disabled."""
        pass

