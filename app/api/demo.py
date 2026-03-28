import json
import time
from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile
from fastapi import HTTPException, status
from fastapi.responses import FileResponse, HTMLResponse

from app.pipeline.generator import (
    generate_differences,
    image_to_base64_png,
    load_image,
    validate_difficulty,
)

ROOT_DIR = Path(__file__).resolve().parents[2]
DEMO_DIR = ROOT_DIR / "demo"
INDEX_HTML = DEMO_DIR / "index.html"
RESULT_HTML = DEMO_DIR / "result.html"

router = APIRouter()


@router.get("/demo", response_class=FileResponse)
def demo_index() -> FileResponse:
    return FileResponse(INDEX_HTML)


@router.post("/demo/process", response_class=HTMLResponse)
async def demo_process(
    image: UploadFile | None = File(default=None),
    num_differences: int = Form(4, ge=1, le=10),
    difficulty: str = Form("medium"),
    seed: str | None = Form(default=None),
) -> HTMLResponse:
    validate_difficulty(difficulty)

    if image is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "image is required. Send multipart/form-data with the 'image' file field."
            ),
        )

    parsed_seed: int | None = None
    if seed is not None and seed.strip() != "":
        try:
            parsed_seed = int(seed)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="seed must be an integer.",
            ) from exc

    started = time.perf_counter()
    source = await load_image(image)

    output = generate_differences(
        image=source,
        num_differences=num_differences,
        difficulty=difficulty,
        seed=parsed_seed,
    )
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    payload = {
        "difficulty": difficulty,
        "num_differences": num_differences,
        "processing_time_ms": elapsed_ms,
        "image_width": source.width,
        "image_height": source.height,
        "source_image_base64": output.source_image_base64,
        "puzzle_image_base64": output.puzzle_image_base64,
        "answer_image_base64": output.answer_image_base64,
        "positions": [position.model_dump() for position in output.positions],
        "difference_cards": [card.model_dump() for card in output.difference_cards],
        "step_images": [
            {
                "name": step_name,
                "image_base64": image_to_base64_png(step_image),
            }
            for step_name, step_image in output.step_images
        ],
    }

    html = RESULT_HTML.read_text(encoding="utf-8")
    safe_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return HTMLResponse(content=html.replace("__RESULT_JSON__", safe_json))
