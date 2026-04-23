# logger.py
# Appends one DecisionLog per line to a daily JSONL file in ./logs/.
#
# Why JSONL (JSON Lines) format?
# eval.py reads all records sequentially. JSONL = one file open, one line per record.
# Compared to individual JSON files: no directory listing, no sorting, no multiple opens.
# Pydantic's model_dump_json() and model_validate_json() handle serialisation cleanly.
#
# Why daily rotation (decisions_YYYYMMDD.jsonl)?
# Keeps files bounded. Easy to compare runs by date. eval.py defaults to today
# but accepts a date_str argument for historical analysis.

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config import LOGS_DIR
from models import DecisionLog

_logs_dir = Path(LOGS_DIR)


def _ensure_logs_dir() -> None:
    _logs_dir.mkdir(exist_ok=True)


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def write_decision(log: DecisionLog) -> None:
    """Append one DecisionLog to today's JSONL file. Creates the file if needed."""
    _ensure_logs_dir()
    log_file = _logs_dir / f"decisions_{_today()}.jsonl"
    with open(log_file, "a") as f:
        f.write(log.model_dump_json() + "\n")


def load_decisions(date_str: Optional[str] = None) -> list[DecisionLog]:
    """
    Load all DecisionLog records from a given date's JSONL file.
    Defaults to today. Returns [] if the file doesn't exist or has no valid lines.
    Skips malformed lines rather than crashing on a partial write.
    """
    _ensure_logs_dir()
    if date_str is None:
        date_str = _today()
    log_file = _logs_dir / f"decisions_{date_str}.jsonl"
    if not log_file.exists():
        return []
    decisions = []
    with open(log_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                decisions.append(DecisionLog.model_validate_json(line))
            except Exception:
                continue   # skip corrupt lines from partial writes
    return decisions
