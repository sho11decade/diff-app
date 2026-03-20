import base64
import io
import os
import random
import time
from collections import defaultdict, deque

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile, status
from pydantic import BaseModel
from PIL import Image, ImageDraw, ImageEnhance, ImageOps


MAX_UPLOAD_BYTES = 5 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}

API_KEY = os.getenv("DIFF_APP_API_KEY", "dev-api-key")
MAX_REQUESTS_PER_MINUTE = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "30"))

_request_log: dict[str, deque[float]] = defaultdict(deque)

app = FastAPI(
    title="Diff Generator API",
    description="Input image from user and generate spot-the-difference outputs.",
    version="0.1.0",
)


class DifferencePosition(BaseModel):
    x: int
    y: int
    width: int
    height: int


class GenerateResponse(BaseModel):
    puzzle_image_base64: str
    answer_image_base64: str
    positions: list[DifferencePosition]
    processing_time_ms: int


def _check_api_key(x_api_key: str | None) -> str:
    if x_api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="X-API-Key header is required.",
        )

    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )

    return x_api_key


def _check_rate_limit(client_id: str) -> None:
    now = time.time()
    window_start = now - 60

    queue = _request_log[client_id]
    while queue and queue[0] < window_start:
        queue.popleft()

    if len(queue) >= MAX_REQUESTS_PER_MINUTE:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please retry later.",
        )

    queue.append(now)


async def _load_image(upload: UploadFile) -> Image.Image:
    if upload.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only JPEG and PNG images are supported.",
        )

    data = await upload.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Image size must be 5MB or smaller.",
        )

    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image data.",
        ) from exc

    return image


def _image_to_base64_png(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _difficulty_factor(difficulty: str) -> float:
    mapping = {
        "easy": 1.3,
        "medium": 1.0,
        "hard": 0.75,
    }
    return mapping[difficulty]


def _apply_random_edit(region: Image.Image) -> Image.Image:
    mode = random.choice(["brightness", "color", "flip"])

    if mode == "brightness":
        factor = random.uniform(0.65, 1.35)
        return ImageEnhance.Brightness(region).enhance(factor)

    if mode == "color":
        factor = random.uniform(0.5, 1.7)
        return ImageEnhance.Color(region).enhance(factor)

    return ImageOps.mirror(region)


def _generate_differences(
    image: Image.Image, num_differences: int, difficulty: str
) -> tuple[Image.Image, Image.Image, list[DifferencePosition]]:
    edited = image.copy()
    answer = image.copy()
    draw = ImageDraw.Draw(answer)

    width, height = image.size
    factor = _difficulty_factor(difficulty)
    min_side = max(24, int(min(width, height) * 0.06 * factor))
    max_side = max(min_side + 1, int(min(width, height) * 0.16 * factor))

    positions: list[DifferencePosition] = []

    for _ in range(num_differences):
        box_w = random.randint(min_side, max_side)
        box_h = random.randint(min_side, max_side)

        x = random.randint(0, max(0, width - box_w))
        y = random.randint(0, max(0, height - box_h))

        region = edited.crop((x, y, x + box_w, y + box_h))
        edited_region = _apply_random_edit(region)
        edited.paste(edited_region, (x, y))

        positions.append(DifferencePosition(x=x, y=y, width=box_w, height=box_h))

        cx = x + box_w // 2
        cy = y + box_h // 2
        radius = max(box_w, box_h) // 2 + 8
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline="red", width=4)

    return edited, answer, positions


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateResponse)
async def generate(
    image: UploadFile = File(...),
    num_differences: int = Form(3, ge=1, le=10),
    difficulty: str = Form("medium"),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> GenerateResponse:
    client_id = _check_api_key(x_api_key)
    _check_rate_limit(client_id)

    if difficulty not in VALID_DIFFICULTIES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="difficulty must be one of: easy, medium, hard.",
        )

    started = time.perf_counter()
    source = await _load_image(image)

    puzzle_image, answer_image, positions = _generate_differences(
        source, num_differences, difficulty
    )
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    return GenerateResponse(
        puzzle_image_base64=_image_to_base64_png(puzzle_image),
        answer_image_base64=_image_to_base64_png(answer_image),
        positions=positions,
        processing_time_ms=elapsed_ms,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
