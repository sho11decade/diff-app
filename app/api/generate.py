import time

from fastapi import APIRouter, File, Form, Header, Query, UploadFile
from fastapi import HTTPException, status

from app.api.schemas import GenerateResponse
from app.core.security import check_api_key, check_rate_limit
from app.pipeline.generator import generate_differences, load_image, validate_difficulty
from app.research.trace import load_trace_log, new_trace_id, save_request_artifacts, save_trace_log

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/generate", response_model=GenerateResponse)
async def generate(
    image: UploadFile | None = File(default=None),
    num_differences: int = Form(3, ge=1, le=10),
    difficulty: str = Form("medium"),
    seed: int | None = Query(default=None),
    trace: bool = Query(default=False),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> GenerateResponse:
    client_id = check_api_key(x_api_key)
    check_rate_limit(client_id)
    validate_difficulty(difficulty)

    if image is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "image is required. Send multipart/form-data with the 'image' file field."
            ),
        )

    started = time.perf_counter()
    source = await load_image(image)

    output = generate_differences(
        image=source,
        num_differences=num_differences,
        difficulty=difficulty,
        seed=seed,
    )
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    trace_id = new_trace_id()
    trace_log_path: str | None = None
    artifact_dir: str | None = None
    cards = None

    artifact_images = [
        ("source", output.source_image),
        ("puzzle", output.puzzle_image),
        ("answer", output.answer_image),
    ]
    artifact_images.extend(output.step_images)

    artifact_params = {
        "trace_id": trace_id,
        "seed": seed,
        "difficulty": difficulty,
        "num_differences": num_differences,
        "processing_time_ms": elapsed_ms,
        "positions": [position.model_dump() for position in output.positions],
        "difference_cards": [card.model_dump() for card in output.difference_cards],
    }
    artifact_dir = save_request_artifacts(
        trace_id=trace_id,
        request_params=artifact_params,
        images=artifact_images,
    )

    if trace:
        cards = output.difference_cards
        trace_payload = {
            "trace_id": trace_id,
            "seed": seed,
            "difficulty": difficulty,
            "num_differences": num_differences,
            "positions": [position.model_dump() for position in output.positions],
            "difference_cards": [card.model_dump() for card in output.difference_cards],
            "processing_time_ms": elapsed_ms,
        }
        trace_log_path = save_trace_log(trace_id=trace_id, payload=trace_payload)

    return GenerateResponse(
        puzzle_image_base64=output.puzzle_image_base64,
        answer_image_base64=output.answer_image_base64,
        positions=output.positions,
        processing_time_ms=elapsed_ms,
        trace_id=trace_id,
        seed=seed,
        difference_cards=cards,
        trace_log_path=trace_log_path,
        artifact_dir=artifact_dir,
    )


@router.get("/experiments/{trace_id}")
def get_trace(trace_id: str) -> dict:
    payload = load_trace_log(trace_id)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Trace log not found.",
        )
    return payload
