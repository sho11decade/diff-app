import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from PIL import Image

from app.core.config import TRACE_OUTPUT_DIR


def new_trace_id() -> str:
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid4().hex[:8]}"


def request_dir_for(trace_id: str) -> Path:
    date_dir = datetime.utcnow().strftime("%Y%m%d")
    return Path(TRACE_OUTPUT_DIR) / date_dir / trace_id


def save_trace_log(trace_id: str, payload: dict) -> str | None:
    out_dir = request_dir_for(trace_id)

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "trace.json"
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

    direct_path = root / datetime.utcnow().strftime("%Y%m%d") / trace_id / "trace.json"
    if direct_path.exists():
        try:
            return json.loads(direct_path.read_text(encoding="utf-8"))
        except Exception:
            return None

    new_style = list(root.glob(f"**/{trace_id}/trace.json"))
    if new_style:
        try:
            return json.loads(new_style[0].read_text(encoding="utf-8"))
        except Exception:
            return None

    for file_path in root.glob(f"**/{trace_id}.json"):
        try:
            return json.loads(file_path.read_text(encoding="utf-8"))
        except Exception:
            return None

    return None


def save_request_artifacts(
    trace_id: str,
    request_params: dict,
    images: list[tuple[str, Image.Image]],
) -> str | None:
    out_dir = request_dir_for(trace_id)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)

        params_path = out_dir / "params.json"
        params_path.write_text(
            json.dumps(request_params, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        for name, image in images:
            image.save(out_dir / f"{name}.png", format="PNG")

        return str(out_dir)
    except Exception:
        # Artifact persistence is best-effort and should not fail the API itself.
        return None
