import base64
import io
import random
from dataclasses import dataclass

import numpy as np
from fastapi import HTTPException, UploadFile, status
from PIL import Image, ImageDraw

from app.api.schemas import DifferenceCard, DifferencePosition
from app.core.config import (
    AB_DENSITY_CONSTRAINT_ENABLED,
    ALLOWED_CONTENT_TYPES,
    DIFFICULTY_PROFILES,
    DIFF_SIDE_RATIO_MAX,
    DIFF_SIDE_RATIO_MIN,
    IMPROVEMENT_ATTEMPTS,
    MAX_UPLOAD_BYTES,
    MIN_DIFF_SIDE,
    SEGMENTATION_MIN_FOREGROUND_RATIO,
    SEGMENTATION_REGION_BOOST,
    VALID_DIFFICULTIES,
)
from app.models.segmentation import get_segmentation_service
from app.pipeline.editors import apply_random_edit, blend_region_with_feather, difficulty_factor
from app.pipeline.editors import create_natural_edit_mask
from app.pipeline.editors import apply_force_visible_edit
from app.pipeline.naturalness import NaturalnessMetrics, evaluate_naturalness


@dataclass
class GenerationOutput:
    source_image: Image.Image
    puzzle_image: Image.Image
    answer_image: Image.Image
    step_images: list[tuple[str, Image.Image]]
    source_image_base64: str
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
        "contrast": 0.26,
        "shift": 0.44,
        "flip": 0.40,
        "fallback_visible_contrast": 0.24,
        "fallback_visible_tint": 0.30,
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
        "contrast": 1.06,
        "shift": 1.28,
        "flip": 1.35,
        "fallback_visible_contrast": 1.05,
        "fallback_visible_tint": 1.15,
    }[edit_type]
    return int(base * diff_mul * mode_mul)


def _choose_edit_mode(
    rng: random.Random,
    mode_counts: dict[str, int],
    photo_mode: bool,
    region_features: dict[str, float],
) -> str:
    mean_sat = region_features.get("region_mean_saturation", 0.3)
    bright_ratio = region_features.get("region_bright_ratio", 0.3)

    if photo_mode:
        base_weights = {
            "brightness": 0.24,
            "color": 0.12,
            "contrast": 0.12,
            "shift": 0.26,
            "flip": 0.26,
        }
    else:
        base_weights = {
            "brightness": 0.22,
            "color": 0.30,
            "contrast": 0.10,
            "shift": 0.18,
            "flip": 0.20,
        }

    # Low-saturation / bright regions tend to look more natural with luminance edits.
    if mean_sat < 0.22:
        base_weights["contrast"] += 0.04
        base_weights["brightness"] += 0.03
        base_weights["shift"] += 0.03
        base_weights["color"] = max(0.14, base_weights["color"] - 0.08)
    if bright_ratio > 0.55:
        base_weights["flip"] = max(0.12, base_weights["flip"] - 0.06)
        base_weights["contrast"] += 0.04

    weighted = []
    total = 0.0
    for mode, w in base_weights.items():
        # Penalize overused mode to avoid one-mode domination.
        adjusted = w / (1.0 + 0.75 * mode_counts.get(mode, 0))
        weighted.append((mode, adjusted))
        total += adjusted

    pick = rng.random() * total
    acc = 0.0
    for mode, w in weighted:
        acc += w
        if pick <= acc:
            return mode
    return weighted[-1][0]


def _passes_quality_gate(
    profile: dict[str, float],
    metrics: NaturalnessMetrics,
    mask_coverage: float,
    photo_mode: bool,
) -> bool:
    effective_change = _effective_visible_change(metrics.mean_abs_diff, mask_coverage)
    min_visible = profile["min_visible_change"] * (0.58 if photo_mode else 0.90)
    max_visible = profile["max_visible_change"] * (1.10 if photo_mode else 1.0)
    min_coverage = profile["min_mask_coverage"] * (1.0 if photo_mode else 0.92)
    return (
        min_visible <= effective_change <= max_visible
        and mask_coverage >= min_coverage
    )


def _effective_visible_change(mean_abs_diff: float, mask_coverage: float) -> float:
    return min(1.0, mean_abs_diff / max(mask_coverage, 0.05))


def _is_photo_like(image: Image.Image) -> bool:
    sample = image.convert("RGB").resize((128, 128))
    arr = np.asarray(sample, dtype=np.uint8)
    flat = arr.reshape(-1, 3)

    unique_ratio = len(np.unique(flat, axis=0)) / float(flat.shape[0])
    gray = np.asarray(sample.convert("L"), dtype=np.float32)
    texture = float(gray.std())

    return unique_ratio > 0.45 and texture > 22.0


def _region_features_and_score(region: Image.Image, photo_mode: bool) -> tuple[float, dict[str, float]]:
    rgb = np.asarray(region.convert("RGB"), dtype=np.float32)
    gray = np.asarray(region.convert("L"), dtype=np.float32)
    hsv = np.asarray(region.convert("HSV"), dtype=np.float32)

    lum = gray / 255.0
    sat = hsv[:, :, 1] / 255.0

    mean_lum = float(lum.mean())
    std_lum = float(lum.std())
    mean_sat = float(sat.mean())
    bright_ratio = float((lum > 0.86).mean())

    # Background-like region heuristic: bright + low saturation + low texture.
    background_penalty = 0.0
    if mean_lum > 0.78 and mean_sat < 0.20:
        background_penalty += 0.45
    if bright_ratio > 0.62:
        background_penalty += 0.30
    if std_lum < (0.08 if photo_mode else 0.05):
        background_penalty += 0.25

    # Prefer textured / meaningful regions; avoid flat bright backgrounds.
    score = 0.45 * std_lum + 0.25 * mean_sat + 0.20 * (1.0 - bright_ratio) + 0.10 * (1.0 - mean_lum)
    score = max(0.0, min(1.0, score - background_penalty))

    features = {
        "region_mean_luminance": round(mean_lum, 6),
        "region_std_luminance": round(std_lum, 6),
        "region_mean_saturation": round(mean_sat, 6),
        "region_bright_ratio": round(bright_ratio, 6),
        "region_background_penalty": round(background_penalty, 6),
    }
    return round(score, 6), features


def _select_edit_region(
    image: Image.Image,
    rng: random.Random,
    min_side: int,
    max_side: int,
    min_region_score: float,
    photo_mode: bool,
    existing_positions: list[DifferencePosition],
    segmentation_map: np.ndarray | None,
    attempts: int = 36,
) -> tuple[int, int, int, int, float, dict[str, float]]:
    width, height = image.size

    best = None
    best_score = -1.0
    best_features: dict[str, float] = {}

    def _is_too_close_or_overlap(x: int, y: int, w: int, h: int) -> bool:
        if not AB_DENSITY_CONSTRAINT_ENABLED:
            return False

        cx = x + w / 2.0
        cy = y + h / 2.0

        for pos in existing_positions:
            ex = pos.x
            ey = pos.y
            ew = pos.width
            eh = pos.height
            ecx = ex + ew / 2.0
            ecy = ey + eh / 2.0

            # Keep a minimum center distance so neighboring differences are separable.
            min_center_dist = 0.95 * ((max(w, h) + max(ew, eh)) / 2.0)
            center_dist = ((cx - ecx) ** 2 + (cy - ecy) ** 2) ** 0.5
            if center_dist < min_center_dist:
                return True

            # Reject heavy overlap to avoid being perceived as one single difference.
            ix0 = max(x, ex)
            iy0 = max(y, ey)
            ix1 = min(x + w, ex + ew)
            iy1 = min(y + h, ey + eh)
            inter_w = max(0, ix1 - ix0)
            inter_h = max(0, iy1 - iy0)
            inter = inter_w * inter_h
            if inter <= 0:
                continue

            union = (w * h) + (ew * eh) - inter
            iou = inter / max(1, union)
            if iou > 0.06:
                return True

        return False

    fallback_best = None
    fallback_best_score = -1.0
    fallback_best_features: dict[str, float] = {}

    for _ in range(attempts):
        box_w = rng.randint(min_side, max_side)
        box_h = rng.randint(min_side, max_side)

        x = rng.randint(0, max(0, width - box_w))
        y = rng.randint(0, max(0, height - box_h))

        if _is_too_close_or_overlap(x, y, box_w, box_h):
            continue

        region = image.crop((x, y, x + box_w, y + box_h))
        score, features = _region_features_and_score(region, photo_mode=photo_mode)

        if segmentation_map is not None:
            roi = segmentation_map[y:y + box_h, x:x + box_w]
            if roi.size > 0:
                fg_ratio = float(roi.mean())
                if fg_ratio < SEGMENTATION_MIN_FOREGROUND_RATIO:
                    score *= 0.80
                score = min(1.0, score + (SEGMENTATION_REGION_BOOST * fg_ratio))
                features["region_foreground_ratio"] = round(fg_ratio, 6)
                features["segmentation_boost"] = round(SEGMENTATION_REGION_BOOST * fg_ratio, 6)

        if score > best_score:
            best = (x, y, box_w, box_h)
            best_score = score
            best_features = features

        if score >= min_region_score:
            return x, y, box_w, box_h, score, features

        if score > fallback_best_score:
            fallback_best = (x, y, box_w, box_h)
            fallback_best_score = score
            fallback_best_features = features

    # Fallback priority: distance-constrained best, then globally best.
    if fallback_best is not None:
        return (
            fallback_best[0],
            fallback_best[1],
            fallback_best[2],
            fallback_best[3],
            fallback_best_score,
            fallback_best_features,
        )

    # Last resort if constraints are too strict on tiny images.
    assert best is not None
    return best[0], best[1], best[2], best[3], best_score, best_features


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
    photo_mode = _is_photo_like(image)
    factor = difficulty_factor(difficulty) * profile["size_multiplier"]
    min_side = max(MIN_DIFF_SIDE, int(min(width, height) * DIFF_SIDE_RATIO_MIN * factor))
    max_side = max(min_side + 1, int(min(width, height) * DIFF_SIDE_RATIO_MAX * factor))

    positions: list[DifferencePosition] = []
    cards: list[DifferenceCard] = []
    step_images: list[tuple[str, Image.Image]] = [("step_00_source", image.copy())]
    mode_counts = {"brightness": 0, "color": 0, "contrast": 0, "shift": 0, "flip": 0}

    segmentation_map = None
    segmentation_result = get_segmentation_service().predict_foreground_map(image)
    if segmentation_result is not None:
        segmentation_map = segmentation_result.foreground_map

    for idx in range(num_differences):
        x, y, box_w, box_h, region_score, region_features = _select_edit_region(
            image=edited,
            rng=rng,
            min_side=min_side,
            max_side=max_side,
            min_region_score=profile["min_region_score"],
            photo_mode=photo_mode,
            existing_positions=positions,
            segmentation_map=segmentation_map,
        )

        region = edited.crop((x, y, x + box_w, y + box_h))
        max_attempts = IMPROVEMENT_ATTEMPTS[difficulty]
        naturalness_threshold = profile["naturalness_threshold"]

        best_region = region
        best_mode = "flip"
        best_strength = 1.0
        best_mask_coverage = 0.0
        best_metrics = NaturalnessMetrics(0.0, 0.0, 1.0, 0.0, 0.0)
        chosen_attempts = 1
        best_candidate_score = -1.0

        strength_scale = profile["initial_strength"]

        for attempt in range(1, max_attempts + 1):
            preferred_mode = _choose_edit_mode(
                rng,
                mode_counts=mode_counts,
                photo_mode=photo_mode,
                region_features=region_features,
            )
            candidate_region, edit_type, edit_strength = apply_random_edit(
                region=region,
                rng=rng,
                strength_scale=strength_scale,
                preferred_mode=preferred_mode,
            )
            edit_mask, mask_coverage = create_natural_edit_mask(
                region.size,
                rng,
                min_coverage=profile["min_mask_coverage"],
                max_coverage=min(0.82, profile["min_mask_coverage"] + 0.34),
            )
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

            gate_ok = _passes_quality_gate(profile, metrics, mask_coverage, photo_mode=photo_mode)
            effective_change = _effective_visible_change(metrics.mean_abs_diff, mask_coverage)

            candidate_score = metrics.score
            if not gate_ok:
                candidate_score -= 0.30
            if gate_ok:
                candidate_score += 0.08

            if candidate_score > best_candidate_score:
                best_region = blended_candidate
                best_mode = edit_type
                best_strength = edit_strength
                best_mask_coverage = mask_coverage
                best_metrics = metrics
                chosen_attempts = attempt
                best_candidate_score = candidate_score

            if metrics.score >= naturalness_threshold and gate_ok:
                break

            # Self-improvement: if too strong, reduce strength; if too weak, increase slightly.
            if effective_change > profile["max_visible_change"]:
                strength_scale = max(0.45, strength_scale * 0.8)
            elif effective_change < profile["min_visible_change"] or mask_coverage < profile["min_mask_coverage"]:
                strength_scale = min(1.35, strength_scale * 1.12)
            else:
                strength_scale = max(0.55, strength_scale * 0.93)

        if not _passes_quality_gate(profile, best_metrics, best_mask_coverage, photo_mode=photo_mode):
            # Fallback: force a detectable but still blended change.
            fallback_best_score = -1.0
            fallback_steps = [1.0, 1.3, 1.65, 2.0]
            mean_sat = region_features.get("region_mean_saturation", 0.3)
            fallback_order = ["contrast", "tint"] if mean_sat < 0.24 else ["tint", "contrast"]

            for i, visibility_boost in enumerate(fallback_steps, start=1):
                preferred_strategy = fallback_order[(i - 1) % len(fallback_order)]
                fallback_region, fallback_mode, fallback_strength = apply_force_visible_edit(
                    region=region,
                    rng=rng,
                    photo_mode=photo_mode,
                    visibility_boost=visibility_boost,
                    preferred_strategy=preferred_strategy,
                )
                fallback_mask, fallback_coverage = create_natural_edit_mask(
                    region.size,
                    rng,
                    min_coverage=max(profile["min_mask_coverage"], min(0.78, 0.38 + 0.07 * i)),
                    max_coverage=min(0.90, 0.72 + 0.04 * i),
                )
                fallback_blended = blend_region_with_feather(
                    base_region=region,
                    edited_region=fallback_region,
                    feather_radius=_feather_radius(box_w, box_h, difficulty, fallback_mode),
                    edit_mask=fallback_mask,
                )
                fallback_metrics = evaluate_naturalness(
                    region,
                    fallback_blended,
                    target_change=max(profile["target_change"], profile["min_visible_change"] + 0.03),
                    tolerance=profile["change_tolerance"],
                )

                gate_ok = _passes_quality_gate(profile, fallback_metrics, fallback_coverage, photo_mode=photo_mode)
                fallback_score = fallback_metrics.score + (0.22 if gate_ok else -0.10) + (0.08 * fallback_coverage)

                if fallback_score > fallback_best_score:
                    best_region = fallback_blended
                    best_mode = fallback_mode
                    best_strength = fallback_strength
                    best_mask_coverage = fallback_coverage
                    best_metrics = fallback_metrics
                    chosen_attempts = max_attempts + i
                    fallback_best_score = fallback_score

                if gate_ok:
                    break

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
        score_breakdown["effective_visible_change"] = _effective_visible_change(
            best_metrics.mean_abs_diff,
            best_mask_coverage,
        )
        score_breakdown["target_change"] = float(profile["target_change"])
        score_breakdown["min_visible_change"] = float(profile["min_visible_change"])
        score_breakdown["max_visible_change"] = float(profile["max_visible_change"])
        score_breakdown["min_mask_coverage"] = float(profile["min_mask_coverage"])
        score_breakdown["quality_gate_passed"] = float(
            _passes_quality_gate(profile, best_metrics, best_mask_coverage, photo_mode=photo_mode)
        )
        score_breakdown["photo_mode"] = float(photo_mode)
        score_breakdown["region_score"] = region_score
        score_breakdown["min_region_score"] = float(profile["min_region_score"])
        score_breakdown.update(region_features)
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

        if best_mode in mode_counts:
            mode_counts[best_mode] += 1

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
        source_image_base64=image_to_base64_png(image),
        puzzle_image_base64=image_to_base64_png(edited),
        answer_image_base64=image_to_base64_png(answer),
        positions=positions,
        difference_cards=cards,
    )
