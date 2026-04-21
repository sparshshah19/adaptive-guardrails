# logger.py
# Appends one DecisionLog per line to a daily JSONL file in ./logs/.
#
# Why JSONL instead of individual JSON files?
# eval.py needs to read all records sequentially. JSONL = one file open, one line per record.
# Individual JSON files would require directory listing, sorting, and multiple file opens.
# Pydantic's model_dump_json() and model_validate_json() handle serialisation cleanly.
#
# Why daily files (decisions_YYYYMMDD.jsonl)?
# Keeps logs bounded — one file per day. Easy to diff runs by date.
# eval.py defaults to today, but accepts a date_str argument for historical analysis.

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from models import DecisionLog

LOGS_DIR = Path("./logs")


def _ensure_logs_dir() -> None:
    LOGS_DIR.mkdir(exist_ok=True)


def write_decision(log: DecisionLog) -> None:
    """Append one decision to today's JSONL log. Creates the file if it doesn't exist."""
    _ensure_logs_dir()
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    log_file = LOGS_DIR / f"decisions_{date_str}.jsonl"
    with open(log_file, "a") as f:
        f.write(log.model_dump_json() + "\n")


def load_decisions(date_str: Optional[str] = None) -> list[DecisionLog]:
    """
    Load all DecisionLog records from a given date's JSONL file.
    Defaults to today if date_str is None.
    Returns an empty list if the file doesn't exist yet.
    """
    _ensure_logs_dir()
    if date_str is None:
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    log_file = LOGS_DIR / f"decisions_{date_str}.jsonl"
    if not log_file.exists():
        return []
    decisions = []
    with open(log_file) as f:
        for line in f:
            line = line.strip()
            if line:
                decisions.append(DecisionLog.model_validate_json(line))
    return decisions
