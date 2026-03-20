import base64
import io
import random
from dataclasses import dataclass

from fastapi import HTTPException, UploadFile, status
from PIL import Image, ImageDraw

from app.api.schemas import DifferenceCard, DifferencePosition
from app.core.config import ALLOWED_CONTENT_TYPES, MAX_UPLOAD_BYTES, VALID_DIFFICULTIES
from app.pipeline.editors import apply_random_edit, difficulty_factor


@dataclass
class GenerationOutput:
    source_image: Image.Image
    puzzle_image: Image.Image
    answer_image: Image.Image
    step_images: list[tuple[str, Image.Image]]
    puzzle_image_base64: str
    answer_image_base64: str
    positions: list[DifferencePosition]
    difference_cards: list[DifferenceCard]


async def load_image(upload: UploadFile) -> Image.Image:
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


def image_to_base64_png(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def validate_difficulty(difficulty: str) -> None:
    if difficulty not in VALID_DIFFICULTIES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="difficulty must be one of: easy, medium, hard.",
        )


def _difficulty_score_breakdown(
    box_w: int,
    box_h: int,
    image_w: int,
    image_h: int,
    factor: float,
    edit_type: str,
) -> dict[str, float]:
    area_ratio = (box_w * box_h) / max(1, image_w * image_h)
    edit_complexity = {
        "brightness": 0.20,
        "color": 0.30,
        "flip": 0.40,
    }[edit_type]

    # Rule-based initial score for explainable trace output.
    return {
        "area_ratio": round(area_ratio, 6),
        "difficulty_factor": round(factor, 3),
        "edit_complexity": edit_complexity,
        "composite_score": round((1.0 - min(area_ratio, 1.0)) * 0.4 + edit_complexity * 0.6, 6),
    }


def generate_differences(
    image: Image.Image,
    num_differences: int,
    difficulty: str,
    seed: int | None,
) -> GenerationOutput:
    rng = random.Random(seed)

    edited = image.copy()
    answer = image.copy()
    draw = ImageDraw.Draw(answer)

    width, height = image.size
    factor = difficulty_factor(difficulty)
    min_side = max(24, int(min(width, height) * 0.06 * factor))
    max_side = max(min_side + 1, int(min(width, height) * 0.16 * factor))

    positions: list[DifferencePosition] = []
    cards: list[DifferenceCard] = []
    step_images: list[tuple[str, Image.Image]] = [("step_00_source", image.copy())]

    for idx in range(num_differences):
        box_w = rng.randint(min_side, max_side)
        box_h = rng.randint(min_side, max_side)

        x = rng.randint(0, max(0, width - box_w))
        y = rng.randint(0, max(0, height - box_h))

        region = edited.crop((x, y, x + box_w, y + box_h))
        edited_region, edit_type, edit_strength = apply_random_edit(region, rng)
        edited.paste(edited_region, (x, y))

        positions.append(DifferencePosition(x=x, y=y, width=box_w, height=box_h))

        score_breakdown = _difficulty_score_breakdown(
            box_w=box_w,
            box_h=box_h,
            image_w=width,
            image_h=height,
            factor=factor,
            edit_type=edit_type,
        )
        cards.append(
            DifferenceCard(
                index=idx,
                edit_type=edit_type,
                edit_strength=round(edit_strength, 3),
                region_area=box_w * box_h,
                difficulty_factor=round(factor, 3),
                score_breakdown=score_breakdown,
            )
        )

        cx = x + box_w // 2
        cy = y + box_h // 2
        radius = max(box_w, box_h) // 2 + 8
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline="red", width=4)
        step_images.append((f"step_{idx + 1:02d}_{edit_type}", edited.copy()))

    return GenerationOutput(
        source_image=image.copy(),
        puzzle_image=edited,
        answer_image=answer,
        step_images=step_images,
        puzzle_image_base64=image_to_base64_png(edited),
        answer_image_base64=image_to_base64_png(answer),
        positions=positions,
        difference_cards=cards,
    )
