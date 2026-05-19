"""
Audit trail for tender extraction runs.
Appends structured records to a JSONL file for traceability.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from src.logging_setup import get_logger

logger = get_logger("audit")

AUDIT_DIR = Path("audit_logs")
AUDIT_FILE = AUDIT_DIR / "extractions.jsonl"


def compute_file_hash(file_path: Path) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def log_extraction_run(
    filename: str,
    file_path: Path | None,
    page_count: int,
    candidate_count: int,
    llm_items_count: int,
    verification_summary: dict | None = None,
    model_id: str = "",
    errors: list[str] | None = None,
) -> None:
    """
    Log an extraction run to the audit trail.
    """
    AUDIT_DIR.mkdir(exist_ok=True)

    file_hash = ""
    if file_path and file_path.exists():
        try:
            file_hash = compute_file_hash(file_path)
        except Exception:
            file_hash = "HASH_FAILED"

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "filename": filename,
        "file_hash_sha256": file_hash,
        "page_count": page_count,
        "candidate_count": candidate_count,
        "llm_items_count": llm_items_count,
        "verification_summary": verification_summary or {},
        "model_id": model_id,
        "errors": errors or [],
    }

    try:
        with open(AUDIT_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info("Audit record written for %s", filename)
    except Exception as e:
        logger.error("Failed to write audit record: %s", str(e))
