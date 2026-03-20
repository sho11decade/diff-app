import random

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


def difficulty_factor(difficulty: str) -> float:
    mapping = {
        "easy": 1.3,
        "medium": 1.0,
        "hard": 0.75,
    }
    return mapping[difficulty]


def apply_random_edit(
    region: Image.Image,
    rng: random.Random,
    strength_scale: float = 1.0,
    preferred_mode: str | None = None,
) -> tuple[Image.Image, str, float]:
    mode = preferred_mode or rng.choice(["brightness", "color", "flip"])

    if mode == "brightness":
        delta = rng.uniform(-0.35, 0.35) * strength_scale
        factor = max(0.65, min(1.35, 1.0 + delta))
        return ImageEnhance.Brightness(region).enhance(factor), mode, factor

    if mode == "color":
        delta = rng.uniform(-0.7, 0.7) * strength_scale
        factor = max(0.5, min(1.7, 1.0 + delta))
        return ImageEnhance.Color(region).enhance(factor), mode, factor

    return ImageOps.mirror(region), mode, 1.0


def blend_region_with_feather(
    base_region: Image.Image,
    edited_region: Image.Image,
    feather_radius: int,
) -> Image.Image:
    if feather_radius <= 0:
        return edited_region

    mask = Image.new("L", base_region.size, 255)
    soft_mask = mask.filter(ImageFilter.GaussianBlur(radius=feather_radius))
    return Image.composite(edited_region, base_region, soft_mask)
