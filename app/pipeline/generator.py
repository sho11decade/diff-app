import base64
import io
import random
from dataclasses import dataclass

from fastapi import HTTPException, UploadFile, status
from PIL import Image, ImageDraw

from app.api.schemas import DifferenceCard, DifferencePosition
from app.core.config import (
    ALLOWED_CONTENT_TYPES,
    DIFFICULTY_PROFILES,
    DIFF_SIDE_RATIO_MAX,
    DIFF_SIDE_RATIO_MIN,
    IMPROVEMENT_ATTEMPTS,
    MAX_UPLOAD_BYTES,
    MIN_DIFF_SIDE,
    VALID_DIFFICULTIES,
)
from app.pipeline.editors import apply_random_edit, blend_region_with_feather, difficulty_factor
from app.pipeline.editors import create_natural_edit_mask
from app.pipeline.naturalness import NaturalnessMetrics, evaluate_naturalness


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
    naturalness: NaturalnessMetrics,
    attempts: int,
) -> dict[str, float]:
    area_ratio = (box_w * box_h) / max(1, image_w * image_h)
    edit_complexity = {
        "brightness": 0.20,
        "color": 0.30,
        "flip": 0.40,
    }[edit_type]

    # Rule-based initial score for explainable trace output.
    composite = (1.0 - min(area_ratio, 1.0)) * 0.25 + edit_complexity * 0.35 + naturalness.score * 0.40

    return {
        "area_ratio": round(area_ratio, 6),
        "difficulty_factor": round(factor, 3),
        "edit_complexity": edit_complexity,
        "naturalness_score": naturalness.score,
        "mean_abs_diff": naturalness.mean_abs_diff,
        "edge_delta": naturalness.edge_delta,
        "change_score": naturalness.change_score,
        "edge_score": naturalness.edge_score,
        "attempts": float(attempts),
        "composite_score": round(composite, 6),
    }


def _feather_radius(box_w: int, box_h: int, difficulty: str, edit_type: str) -> int:
    base = max(2, int(min(box_w, box_h) * 0.10))
    diff_mul = {
        "easy": 1.00,
        "medium": 1.15,
        "hard": 1.25,
    }[difficulty]
    mode_mul = {
        "brightness": 1.00,
        "color": 1.10,
        "flip": 1.35,
    }[edit_type]
    return int(base * diff_mul * mode_mul)


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
    profile = DIFFICULTY_PROFILES[difficulty]
    factor = difficulty_factor(difficulty) * profile["size_multiplier"]
    min_side = max(MIN_DIFF_SIDE, int(min(width, height) * DIFF_SIDE_RATIO_MIN * factor))
    max_side = max(min_side + 1, int(min(width, height) * DIFF_SIDE_RATIO_MAX * factor))

    positions: list[DifferencePosition] = []
    cards: list[DifferenceCard] = []
    step_images: list[tuple[str, Image.Image]] = [("step_00_source", image.copy())]

    for idx in range(num_differences):
        box_w = rng.randint(min_side, max_side)
        box_h = rng.randint(min_side, max_side)

        x = rng.randint(0, max(0, width - box_w))
        y = rng.randint(0, max(0, height - box_h))

        region = edited.crop((x, y, x + box_w, y + box_h))
        max_attempts = IMPROVEMENT_ATTEMPTS[difficulty]
        naturalness_threshold = profile["naturalness_threshold"]

        best_region = region
        best_mode = "flip"
        best_strength = 1.0
        best_mask_coverage = 0.0
        best_metrics = NaturalnessMetrics(0.0, 0.0, 1.0, 0.0, 0.0)
        chosen_attempts = 1

        preferred_mode = rng.choice(["brightness", "color", "flip"])
        strength_scale = profile["initial_strength"]

        for attempt in range(1, max_attempts + 1):
            candidate_region, edit_type, edit_strength = apply_random_edit(
                region=region,
                rng=rng,
                strength_scale=strength_scale,
                preferred_mode=preferred_mode,
            )
            edit_mask, mask_coverage = create_natural_edit_mask(region.size, rng)
            feather_radius = _feather_radius(box_w, box_h, difficulty, edit_type)
            blended_candidate = blend_region_with_feather(
                base_region=region,
                edited_region=candidate_region,
                feather_radius=feather_radius,
                edit_mask=edit_mask,
            )
            metrics = evaluate_naturalness(
                region,
                blended_candidate,
                target_change=profile["target_change"],
                tolerance=profile["change_tolerance"],
            )

            if metrics.score > best_metrics.score:
                best_region = blended_candidate
                best_mode = edit_type
                best_strength = edit_strength
                best_mask_coverage = mask_coverage
                best_metrics = metrics
                chosen_attempts = attempt

            if metrics.score >= naturalness_threshold:
                break

            # Self-improvement: if too strong, reduce strength; if too weak, increase slightly.
            if metrics.mean_abs_diff > 0.22:
                strength_scale = max(0.45, strength_scale * 0.8)
            elif metrics.mean_abs_diff < 0.08:
                strength_scale = min(1.35, strength_scale * 1.12)
            else:
                strength_scale = max(0.55, strength_scale * 0.93)

        edited.paste(best_region, (x, y))

        positions.append(DifferencePosition(x=x, y=y, width=box_w, height=box_h))

        score_breakdown = _difficulty_score_breakdown(
            box_w=box_w,
            box_h=box_h,
            image_w=width,
            image_h=height,
            factor=factor,
            edit_type=best_mode,
            naturalness=best_metrics,
            attempts=chosen_attempts,
        )
        score_breakdown["feather_radius"] = float(_feather_radius(box_w, box_h, difficulty, best_mode))
        score_breakdown["mask_coverage"] = best_mask_coverage
        score_breakdown["target_change"] = float(profile["target_change"])
        cards.append(
            DifferenceCard(
                index=idx,
                edit_type=best_mode,
                edit_strength=round(best_strength, 3),
                region_area=box_w * box_h,
                difficulty_factor=round(factor, 3),
                naturalness_score=best_metrics.score,
                improvement_attempts=chosen_attempts,
                score_breakdown=score_breakdown,
            )
        )

        cx = x + box_w // 2
        cy = y + box_h // 2
        radius = max(box_w, box_h) // 2 + 8
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline="red", width=4)
        step_images.append((f"step_{idx + 1:02d}_{best_mode}", edited.copy()))

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
