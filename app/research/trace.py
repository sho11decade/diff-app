import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from app.core.config import TRACE_OUTPUT_DIR


def new_trace_id() -> str:
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid4().hex[:8]}"


def save_trace_log(trace_id: str, payload: dict) -> str | None:
    date_dir = datetime.utcnow().strftime("%Y%m%d")
    out_dir = Path(TRACE_OUTPUT_DIR) / date_dir

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{trace_id}.json"
        out_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return str(out_file)
    except Exception:
        # Trace logging is best-effort and should not fail the API itself.
        return None


def load_trace_log(trace_id: str) -> dict | None:
    root = Path(TRACE_OUTPUT_DIR)
    if not root.exists():
        return None

    for file_path in root.glob(f"**/{trace_id}.json"):
        try:
            return json.loads(file_path.read_text(encoding="utf-8"))
        except Exception:
            return None

    return None
